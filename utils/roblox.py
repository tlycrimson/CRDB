import os
import io
import ssl
import time
import random
import aiohttp
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Any, Dict, Tuple

load_dotenv()
logger = logging.getLogger(__name__)


class TokenBucket:
    def __init__(self, capacity: float, rate: float):
        self._capacity = capacity
        self._rate = rate
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock: Optional[asyncio.Lock] = None  

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self):
        wait_time = 0
        async with self._get_lock():
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self._rate
                self._tokens = 0
            else:
                self._tokens -= 1

        if wait_time > 0:
            logger.debug(f"TokenBucket throttling for {wait_time:.2f}s (tokens exhausted)")
            await asyncio.sleep(wait_time)

class TTLCache:
    def __init__(self, ttl_seconds: float = 45, max_size: int = 1000):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Tuple[bool, Any]:
        entry = self._store.get(key)
        if entry is None:
            return False, None
        value, expiry = entry
        if time.monotonic() > expiry:
            self._store.pop(key, None)
            return False, None
        return True, value

    def set(self, key: str, value: Any):
        if len(self._store) >= self._max_size:
            self._store.pop(next(iter(self._store)))
        self._store[key] = (value, time.monotonic() + self._ttl)


class RequestDeduplicator:
    def __init__(self):
        self._inflight: Dict[str, asyncio.Future] = {}

    async def get_or_fetch(self, key: str, fetch_func, *args, **kwargs):
        if key in self._inflight:
            # Wait on the existing future rather than shielding a task
            try:
                return await asyncio.shield(self._inflight[key])
            except Exception:
                raise

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._inflight[key] = future

        try:
            result = await fetch_func(*args, **kwargs)
            future.set_result(result)
            return result
        except Exception as e:
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            self._inflight.pop(key, None)


class RobloxClient:
    PROXY_LIST = [
        os.getenv("WORKER_URL"),
        "rotunnel.com",
        "roproxy.com",
    ]

    MAX_RETRIES = 5
    BASE_DELAY = 1.0

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._proxy_index = 0
        # Instantiated per-object so locks are always on the correct event loop
        self._bucket = TokenBucket(capacity=20, rate=10)
        self._cache = TTLCache(ttl_seconds=60)
        self._dedup = RequestDeduplicator()

    async def start(self):
        if self._session and not self._session.closed:
            return
        connector = aiohttp.TCPConnector(
            limit=25,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": "MPRbxClient"},
        )
        
        ssl_context = ssl.create_default_context()
        oc_connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
        )

        self._oc_session = aiohttp.ClientSession(
            connector=oc_connector,
            headers={"User-Agent": "MPRbxClient"},
        )

    async def close(self):
        for s in (self._session, self._oc_session):
               if s:
                await s.close()
        
        self._session = None
        self._oc_session = None

    def _get_url(self, subdomain: str, path: str) -> str:
        domain = self.PROXY_LIST[self._proxy_index]
        if "workers.dev" in domain:
            return f"https://{domain}/{subdomain}{path}"
        return f"https://{subdomain}.{domain}{path}"

    async def _fetch(
        self, method: str, subdomain: str, path: str, **kwargs
    ) -> Tuple[Optional[Any], int]:
        if not self._session or self._session.closed:
            logger.error("_fetch called but session is not ready!")
            return None, 500

        request_timeout = aiohttp.ClientTimeout(total=12, sock_read=10)
        logger.debug(f"_fetch called: {method} {subdomain}{path}")
        for attempt in range(self.MAX_RETRIES):
            url = self._get_url(subdomain, path)
            await self._bucket.acquire()
            try:
                async with self._session.request(
                    method, url, timeout=request_timeout, **kwargs
                ) as resp:
                    status = resp.status

                    if status == 200:
                        return await resp.json(), 200

                    if status == 403:
                        return None, 403

                    if status == 429:
                        self._proxy_index = (self._proxy_index + 1) % len(self.PROXY_LIST)
                        retry_after = float(resp.headers.get("Retry-After", 0))
                        backoff = self.BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                        sleep_time = max(retry_after, backoff)
                        logger.warning(f"Rate limited (429) on {url}. Waiting {sleep_time:.2f}s")
                        await asyncio.sleep(sleep_time)
                        continue

                    if status in (400, 401, 404):
                        error_data = await resp.text()
                        logger.debug(f"Permanent error {status} for {url}: {error_data}")
                        return None, status

                    # Transient server error — backoff and retry
                    await asyncio.sleep(self.BASE_DELAY * (2**attempt))

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed for {url}: {e}")
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"Giving up after {self.MAX_RETRIES} attempts: {url}")
                    return None, 500
                self._proxy_index = (self._proxy_index + 1) % len(self.PROXY_LIST)
                await asyncio.sleep(self.BASE_DELAY * (2**attempt))

        return None, 500

    # ── Public Helpers ────────────────────────────────────────

    def extract_award_dates(self, badges: list[dict]) -> list[str]:
        """
        The Open Cloud Inventory API already returns the award timestamp as
        `addTime` on each item, so no second API call is needed.
        Returns a list of ISO 8601 date strings.
        """
        dates = []
        for item in badges:
            date_val = item.get("addTime") or item.get("created")
            if date_val:
                dates.append(date_val)
        return dates

    def convert_date_to_datetime(self, date: str) -> datetime:
        """
        Convert an ISO 8601 timestamp string to a datetime object.
        Handles variable fractional-second precision and trailing 'Z'.
        """
        return datetime.fromisoformat(date.replace("Z", "+00:00"))

    async def get_badges(self, user_id: int, stop_at: int = 10_000) -> tuple[int, list[dict]]:
        cache_key = f"badges:{user_id}"
        
        hit, val = self._cache.get(cache_key)
        if hit:
            return 200, val

        async def _fetch_all_badges():
            api_key = os.getenv("RBX_API_KEY")
            if not api_key:
                logger.error("RBX_API_KEY is not set in environment")
                return 500, []

            await self.start()

            url = f"https://apis.roblox.com/cloud/v2/users/{user_id}/inventory-items"
            headers = {"x-api-key": api_key}
            timeout = aiohttp.ClientTimeout(total=10, sock_connect=4, sock_read=8)
            semaphore = asyncio.Semaphore(5)  # Limit concurrent page fetches

            async def fetch_page(page_token: str | None) -> tuple[dict | None, int]:
                params: dict = {"filter": "badges=true", "maxPageSize": 100}
                if page_token:
                    params["pageToken"] = page_token

                async with semaphore:
                    for attempt in range(self.MAX_RETRIES):
                        try:
                            async with self._oc_session.request(
                                "GET", url, params=params, headers=headers, timeout=timeout
                            ) as resp:
                                if resp.status == 200:
                                    return await resp.json(), 200
                                if resp.status == 429:
                                    retry_after = float(resp.headers.get("Retry-After", self.BASE_DELAY * (2 ** attempt)))
                                    await asyncio.sleep(retry_after)
                                    continue
                                return None, resp.status
                        except (asyncio.TimeoutError, aiohttp.ClientError):
                            await asyncio.sleep(self.BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))

                return None, 500

            first_data, first_status = await fetch_page(None)
            if first_status != 200 or not first_data:
                return first_status, []

            badges = first_data.get("inventoryItems", [])
            pending_tokens = []
            if token := first_data.get("nextPageToken"):
                pending_tokens.append(token)

            while pending_tokens and len(badges) < stop_at:
                current_batch = pending_tokens[:10]
                pending_tokens = pending_tokens[10:]
                
                results = await asyncio.gather(*[fetch_page(t) for t in current_batch], return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception) or result[0] is None:
                        continue
                    
                    data, status = result
                    if status == 200:
                        new_items = data.get("inventoryItems", [])
                        badges.extend(new_items)
                        if next_t := data.get("nextPageToken"):
                            pending_tokens.append(next_t)
                
                if len(badges) >= stop_at:
                    break

            processed_badges = badges[:stop_at]
            self._cache.set(cache_key, processed_badges)
            return 200, processed_badges

        return await self._dedup.get_or_fetch(cache_key, _fetch_all_badges)

    def plot_cumulative_badges(self, username: str, user_id: str, dates: list[str]):
        """Graph the cumulative total of badges earned over time."""
        parsed = sorted(self.convert_date_to_datetime(d) for d in dates)

        cumulative_counts = list(range(1, len(parsed) + 1))

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(parsed, cumulative_counts, marker="o", alpha=0.4, color="#7289da", s=10)
        ax.plot(parsed, cumulative_counts, color="#7289da", linewidth=1, alpha=0.6)

        ax.set_xlabel("Badge Earned Date")
        ax.set_ylabel("Total Badges")
        ax.set_title(f"Badges Growth: {username} ({user_id})")

        fig.autofmt_xdate()
        plt.tight_layout()


        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    async def get_user_id(self, username: str) -> Optional[int]:
        cache_key = f"uid:{username.lower()}"
        hit, val = self._cache.get(cache_key)
        if hit:
            return val

        async def _do():
            data, _ = await self._fetch(
                "POST",
                "users",
                "/v1/usernames/users",
                json={"usernames": [username], "excludeBannedUsers": False},
            )
            if data and data.get("data"):
                uid = data["data"][0].get("id")
                self._cache.set(cache_key, uid)
                return uid
            return None

        return await self._dedup.get_or_fetch(cache_key, _do)

    async def get_group_rank(self, user_id: int, group_id: int) -> Optional[str]:
        cache_key = f"rank:{user_id}:{group_id}"
        hit, val = self._cache.get(cache_key)
        if hit:
            return val

        async def _do_fetch():
            data, _ = await self._fetch("GET", "groups", f"/v2/users/{user_id}/groups/roles")
            if not data:
                return None
            for group in data.get("data", []):
                if group.get("group", {}).get("id") == group_id:
                    return group.get("role", {}).get("name")
            return None

        result = await self._dedup.get_or_fetch(cache_key, _do_fetch)
        self._cache.set(cache_key, result)
        return result

    async def get_user_info(self, user_id: int) -> Optional[dict]:
        cache_key = f"info:{user_id}"
        hit, val = self._cache.get(cache_key)
        if hit:
            return val
        result, _ = await self._fetch("GET", "users", f"/v1/users/{user_id}")
        if result:
            self._cache.set(cache_key, result)
        return result

    async def get_friends_count(self, user_id: int) -> Optional[int]:
        cache_key = f"friends:{user_id}"
        hit, val = self._cache.get(cache_key)
        if hit:
            return val
        data, _ = await self._fetch("GET", "friends", f"/v1/users/{user_id}/friends/count")
        result = data.get("count") if data else None
        if result is not None:
            self._cache.set(cache_key, result)
        return result

    async def get_avatar_url(self, user_id: int) -> Optional[str]:
        cache_key = f"avatar:{user_id}"
        hit, val = self._cache.get(cache_key)
        if hit:
            return val
        data, _ = await self._fetch(
            "GET",
            "thumbnails",
            "/v1/users/avatar",
            params={"userIds": user_id, "size": "150x150", "format": "Png"},
        )
        url = data["data"][0].get("imageUrl") if data and data.get("data") else None
        if url:
            self._cache.set(cache_key, url)
        return url

    async def get_groups(self, user_id: int) -> list:
        cache_key = f"groups:{user_id}"
        hit, val = self._cache.get(cache_key)
        if hit:
            return val
        data, _ = await self._fetch("GET", "groups", f"/v2/users/{user_id}/groups/roles")
        result = data.get("data", []) if data else []
        self._cache.set(cache_key, result)
        return result

    async def is_inventory_private(self, user_id: int) -> int:
        cache_key = f"inv_private:{user_id}"
        hit, val = self._cache.get(cache_key)
        if hit:
            return val
        data, status = await self._fetch(
            "GET",
            "inventory",
            f"/v1/users/{user_id}/assets/classic-clothing?limit=1",
        )
        if status == 403:
            is_private_status = -1
        elif status != 200 or data is None:
            is_private_status = -2
        else:
            is_private_status = 0
        self._cache.set(cache_key, is_private_status)
        return is_private_status

    async def user_exists(self, user_id: int) -> bool:
        result = await self.get_user_info(user_id)
        return bool(result and "id" in result)

    # ── Convenience: fetch everything for /sc in parallel ─────────────
    async def fetch_sc_data(self, user_id: int, group_id: int, include_badges: bool = False) -> dict:
        keys = ("user_info", "rank", "groups", "friends_count", "avatar_url", "badges" if include_badges else "badge_count")
        final_data = {k: None for k in keys}
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    self.get_user_info(user_id),
                    self.get_group_rank(user_id, group_id),
                    self.get_groups(user_id),
                    self.get_friends_count(user_id),
                    self.get_avatar_url(user_id),
                    self.get_badges(user_id),
                    return_exceptions=True,
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            logger.error(f"fetch_sc_data timed out for user {user_id}")
            return final_data

        for k, v in zip(keys, results):
            if isinstance(v, BaseException):
                logger.error("Task [%s] failed: %s", k, v)
            else:
                final_data[k] = v

        if final_data.get("badges") is not None:
            status, badge_list = final_data["badges"]
            final_data["badges"] = badge_list
            final_data["badge_count"] = len(badge_list)
        elif final_data.get("badge_count") is not None:
            status, badge_list = final_data["badge_count"]
            final_data["badge_count"] = len(badge_list)

        return final_data
