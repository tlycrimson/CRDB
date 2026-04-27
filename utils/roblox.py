import os
import aiohttp
import asyncio
import time
import random
import logging
from dotenv import load_dotenv
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
            timeout=aiohttp.ClientTimeout(total=15, sock_connect=5, sock_read=10),
            headers={"User-Agent": "MPRbxClient"},
        )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

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

    async def get_badge_count(self, user_id: int, stop_at: int = 500) -> int:
        cache_key = f"badges:{user_id}"
        hit, val = self._cache.get(cache_key)
        if hit:
            return val

        total = 0
        cursor = None
        MAX_PAGES = 50  

        for _ in range(MAX_PAGES):
            params = {"limit": 100}
            if cursor:
                params["cursor"] = cursor

            data, status = await self._fetch(
                "GET", "badges", f"/v1/users/{user_id}/badges", params=params
            )

            if status == 403:
                return -1  # Private inventory

            if status != 200 or data is None:
                return -2  # API error

            badge_list = data.get("data", [])

            if not badge_list and cursor is None:
                return -1  # Private inventory (secondary detection)

            if not badge_list:
                break

            total += len(badge_list)
            if total >= stop_at:
                self._cache.set(cache_key, total)
                return total

            cursor = data.get("nextPageCursor")
            if not cursor:
                break

        self._cache.set(cache_key, total)
        return total

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

    async def fetch_sc_data(self, user_id: int, group_id: int) -> dict:
        keys = ("user_info", "rank", "groups", "friends_count", "avatar_url", "badge_count")
        final_data = {k: None for k in keys}

        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    self.get_user_info(user_id),
                    self.get_group_rank(user_id, group_id),
                    self.get_groups(user_id),
                    self.get_friends_count(user_id),
                    self.get_avatar_url(user_id),
                    self.get_badge_count(user_id),
                    return_exceptions=True,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.error(f"fetch_sc_data timed out for user {user_id}")
            return final_data

        for k, v in zip(keys, results):
            if isinstance(v, BaseException):
                logger.error("Task [%s] failed: %s", k, v)
            else:
                final_data[k] = v

        return final_data
