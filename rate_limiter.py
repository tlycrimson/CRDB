import time

class RateLimiter:
    """Handles rate limiting for API calls"""
    def __init__(self, calls_per_minute: int = 50):
        self.calls_per_minute = calls_per_minute
        self.last_calls: List[float] = []

    async def wait_if_needed(self):
        now = time.time()
        self.last_calls = [t for t in self.last_calls if now - t < 60]
        
        if len(self.last_calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.last_calls[0])
            await asyncio.sleep(wait_time)
            
        self.last_calls.append(now)
