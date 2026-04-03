import asyncio
import discord
import socket
import time
import logging
import aiohttp
from aiohttp.resolver import AsyncResolver
import random
from time import monotonic as now

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
    #Function: initialises variables for class object
    def __init__(self):
        self.semaphore = asyncio.Semaphore(30) # think it like a bouncer that only allows 30 people in and makes other wait 
        self.last_request = 0
        self.min_delay = 1.0 / 30
        self.request_count = 0
        self.last_reset = time.time()
        self.total_limited = 0
        self._lock = asyncio.Lock() # only things to occur one at a time think as Semaphore(1)

    # Function: Called automatically on "async with [CLASS OBJECT e.g. rate_limiter]" and accquires
    # a Semaphore slot and enforces rate limiting before alloing the request
    async def __aenter__(self):
        await self.semaphore.acquire()
        
        async with self._lock:
            current_time = time.time()
            self.request_count += 1
            
            # Reset counter every minute
            if current_time - self.last_reset > 60:
                if self.request_count > 1000:
                    logger.warning(f"High request rate: {self.request_count}/min")
                    self.total_limited += 1
                self.request_count = 0
                self.last_reset = current_time
            
            # Apply backoff if we've been rate limited frequently
            base_delay = self.min_delay
            if self.total_limited > 3:
                base_delay *= 1.5
                
            # Wait if needed
            elapsed = current_time - self.last_request
            if elapsed < base_delay:
                wait_time = base_delay - elapsed
                # Add small jitter but don't block for too long
                await asyncio.sleep(min(wait_time + random.uniform(0, 0.15), 1.0))
            
            self.last_request = time.time()
        return self

    # Function: Called automatically on 'async with' exit - releases the
    # accquired Semaphore slot (from __aenter__) and logs any errors that occurs
    async def __aexit__(self, exc_type, exc, tb):
        self.semaphore.release()
        if exc:
            logger.error(f"Error in rate limited block: {exc}")
        return False
        
class EnhancedRateLimiter:
    # Function: Intialising variables for class object
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute

        self.buckets = {}
        self.locks = {}

        self.global_lock = asyncio.Lock()
        self.global_count = 0
        self.last_global_reset = now()

        self.last_bucket_cleanup = now()

    # Function: When called it checks has passed and if it has then finds all buckets that haven't been used in 24 hours and remove them and their associated locks

    async def _cleanup_old_buckets(self):
        async with self.global_lock:
            current_time = now()
            if current_time - self.last_bucket_cleanup < 3600:
                return  # Cleanup only once per hour

            stale = [
                key for key, data in self.buckets.items()
                if current_time - data['last_call'] > 86400  # 24h
            ]
            for key in stale:
                self.buckets.pop(key, None) #Another example of defensive programming, if key is not found it returns none inststead of an erro
                self.locks.pop(key, None)

            self.last_bucket_cleanup = current_time

    # Function: 1% chance of trigerring a background bucket cleanup and then checks if a full second has passed and resets the global counter if so. If 48 requests have already been made this second, waitrs until the second is up before allowing more. 
    async def wait_if_needed(self, bucket: str = "global"):
        current_time = now()

        # Chance-based cleanup
        if random.random() < 0.01:
            asyncio.create_task(self._cleanup_old_buckets())

        # === GLOBAL RATE LIMITING ===
        async with self.global_lock:
            if current_time - self.last_global_reset >= 1.0:
                self.global_count = 0
                self.last_global_reset = current_time

            if self.global_count >= 48:
                wait = self.last_global_reset + 1.0 - current_time
                if wait > 0:
                    await asyncio.sleep(wait)
                self.global_count = 0
                self.last_global_reset = now()

            self.global_count += 1

        #Function: Creates a per-bucket tracker if one doesn't exist, then enforces that requests to that specific bucket are spaced out evenly and don't exceed the per-minute limit.

        # === LOCAL BUCKET RATE LIMITING ===
        if bucket not in self.buckets:
            async with self.global_lock:  # Prevent race on init
                if bucket not in self.buckets:
                    self.buckets[bucket] = {
                        'last_call': 0.0,
                        'count': 0,
                        'window_start': current_time
                    }
                    self.locks[bucket] = asyncio.Lock()

        async with self.locks[bucket]:
            data = self.buckets[bucket]

            if current_time - data['window_start'] > 60:
                data['count'] = 0
                data['window_start'] = current_time

            elapsed = current_time - data['last_call']
            wait = self.interval - elapsed

            if wait > 0:
                await asyncio.sleep(wait + random.uniform(0, 0.01))  # Optional jitter

            data['last_call'] = now()
            data['count'] += 1

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
