import time
import random
import socket
import collections
import asyncio
import discord
import logging

logger = logging.getLogger(__name__)

async def check_dns_connectivity(timeout: float = 2.0) -> bool:
    """
    Resolve a short list of known domains in a thread so we don't block the event loop.
    Returns True if at least one domain resolves within the timeout.
    """
    domains = ("api.roblox.com", "www.roblox.com")
    async def _resolve(domain):
        try:
            # run blocking getaddrinfo in a thread
            res = await asyncio.wait_for(
                asyncio.to_thread(socket.getaddrinfo, domain, None), #asnico.to_thread adds to a seperate thread instead of freezing bot, socket.getaddrinfo turns domain name into ip address
                timeout=timeout / len(domains)
            )
            return bool(res)
        except Exception:
            return False

    results = await asyncio.gather(*[_resolve(d) for d in domains], return_exceptions=True)
    return any(r is True for r in results)

# --- CLASSES ---
class ConnectionMonitor:
    #Function: initialises variables for class object
    def __init__(self, bot, check_interval: int = 300):
        self.bot = bot
        self.check_interval = check_interval
        self.last_disconnect = None
        self.disconnect_count = 0
        self._task = None
        self._stopping = False

    #Function: checks whether a monitor is running and start it as a background if its not 
    async def start(self):
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._monitor_connection())
    #Function: Simply stops any on-going monitoring tasks
    async def stop(self):
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    # Function: Monitors the bot's connectivity to the discord server periodically in the background
    # and resets disconnect counters every 30 seconds and sends a warning if disconnect counter is high
    async def _monitor_connection(self):
        try:
            while not self._stopping:
                # report status every check_interval
                await asyncio.sleep(self.check_interval)
                logger.info("ConnectionMonitor: bot closed=%s latency=%.0fms",
                            self.bot.is_closed(), getattr(self.bot, "latency", 0.0) * 1000.0)
                # reset old counters after 30 minutes
                if self.last_disconnect and (time.time() - self.last_disconnect) > 1800:
                    self.disconnect_count = 0
                if self.disconnect_count > 5:
                    logger.warning("ConnectionMonitor: high disconnect_count=%d", self.disconnect_count)
                    # we intentionally do NOT attempt to hard-restart the process here. Render will restart
                    # the service if the process exits; a clean restart policy should be applied externally.
        except asyncio.CancelledError:
            return

    # Function: Makes a record of the disconnect
    def record_disconnect(self):
        self.last_disconnect = time.time()
        self.disconnect_count += 1
        logger.warning("ConnectionMonitor: recorded disconnect (count=%d)", self.disconnect_count)

        
class GlobalRateLimiter:
    def __init__(self, limit=30):
        self.semaphore = asyncio.Semaphore(limit)
        self.last_request = 0
        self.min_delay = 1.0 / limit
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self.semaphore.acquire()
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_request
            
            if elapsed < self.min_delay:
                await asyncio.sleep((self.min_delay - elapsed) + random.uniform(0, 0.02))
            
            self.last_request = asyncio.get_event_loop().time()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.semaphore.release()
        if exc:
            logger.error(f"Rate limit block error: {exc}")

class EnhancedRateLimiter:
    def __init__(self, calls_per_minute: int):
        self.interval = 60.0 / calls_per_minute
        self.buckets = {}
        self.locks = collections.defaultdict(asyncio.Lock)
        self.global_lock = asyncio.Lock()
        self.global_limit = 48
        self.global_count = 0
        self.last_reset = time.perf_counter()

    async def wait_if_needed(self, bucket: str = "global"):
        now = time.perf_counter()

        async with self.global_lock:
            if now - self.last_reset >= 1.0:
                self.global_count = 0
                self.last_reset = now
            
            if self.global_count >= self.global_limit:
                await asyncio.sleep(max(0, self.last_reset + 1.0 - now))
                self.global_count = 0
                self.last_reset = time.perf_counter()
            self.global_count += 1

        async with self.locks[bucket]:
            data = self.buckets.get(bucket, {'last_call': 0.0})
            
            elapsed = time.perf_counter() - data['last_call']
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            
            # Update the bucket data
            self.buckets[bucket] = {'last_call': time.perf_counter()}

# Function: Wraps Discord API calls with retry logic - then apply global rate limiting, handling 429 responses with exponetial backoff (using Discord's retry-after header if its available) and log all errors, and stops after 3 failed attempts
class DiscordAPI:
    """Helper class for Discord API requests with retry logic"""
    
    _rate_limiter = None
    
    @classmethod
    def initialize(cls, rate_limiter):
        """Initialize the DiscordAPI with a rate limiter"""
        cls._rate_limiter = rate_limiter
    
    @staticmethod
    async def execute_with_retry(coro, max_retries=3, initial_delay=1.0):
        if DiscordAPI._rate_limiter is None:
            # Fallback - create a temporary one
            from .network import GlobalRateLimiter
            DiscordAPI._rate_limiter = GlobalRateLimiter()
            logger.warning("DiscordAPI using fallback rate limiter")
        
        for attempt in range(max_retries):
            try:
                async with DiscordAPI._rate_limiter:
                    return await coro
            except discord.errors.HTTPException as e:
                status = getattr(e, "status", None)
                if status == 429:  # Too Many Requests
                    retry_after = initial_delay * (2 ** attempt)
                    try:
                        header_ra = e.response.headers.get('Retry-After')
                        if header_ra:
                            retry_after = float(header_ra)
                    except Exception:
                        pass
                    logger.warning(f"Rate limited. Attempt {attempt+1}/{max_retries}. Waiting {retry_after:.2f}s")
                    await asyncio.sleep(retry_after)
                    continue
                logger.error(f"Discord HTTPException (non-429) on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(initial_delay * (2 ** attempt))
            except Exception as e:
                logger.error(f"API Error on attempt {attempt+1}: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(initial_delay * (2 ** attempt))
        
        raise Exception(f"Failed after {max_retries} attempts")
