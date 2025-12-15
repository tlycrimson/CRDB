import os
import re
import io
import time
import asyncio
import threading
import requests
import aiohttp
import discord
import logging
import random
import aiodns
import socket
import mimetypes
import functools
import numpy as np
from typing import Optional, Set, Dict, List, Tuple, Any, Literal
from decorators import min_rank_required, has_allowed_role, has_allowed_role_2
from rate_limiter import RateLimiter
from discord import app_commands
from config import Config
from discord.ext import commands
from discord.utils import escape_markdown
from dotenv import load_dotenv
from roblox_commands import create_sc_command
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from time import monotonic as now
from collections import deque
from aiohttp.resolver import AsyncResolver
from functools import lru_cache

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
TIMEOUT = aiohttp.ClientTimeout(total=10)

# XP Limit Configuration
MAX_XP_PER_ACTION =  20  # Maximum XP that can be given/taken in a single action
MAX_EVENT_XP_PER_USER = 20 # Maximum XP per user in event distributions
MAX_EVENT_TOTAL_XP = 5000  # Maximum total XP for entire event distribution

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- SOME FUNCTIONS & STUFF ---
# Username Cleaner
def clean_nickname(nickname: str) -> str:
    """Remove tags like [INS] from nicknames and clean whitespace"""
    if not nickname:  # Handle None or empty string
        return "Unknown"
    cleaned = re.sub(r'\[.*?\]', '', nickname).strip()
    return cleaned or nickname  # Fallback to original if empty after cleaning

# XP Tier configuration
TIERS = [
    ("🌟 Platinum", 800),
    ("💎 Diamond", 400),
    ("🥇 Gold", 200),
    ("🥈 Silver", 135),
    ("🥉 Bronze", 100),
    ("⚪ Unranked", 0),
]
# Pre-sort tiers from highest to lowest threshold
TIERS_SORTED = sorted(TIERS, key=lambda x: x[1], reverse=True)

def get_tier_info(xp: int) -> Tuple[str, int, Optional[int]]:
    """Return (tier_name, current_threshold, next_threshold)"""
    # Use pre-sorted tiers
    for i, (name, threshold) in enumerate(TIERS_SORTED):
        if xp >= threshold:
            # Get next tier's threshold (if exists)
            next_threshold = TIERS_SORTED[i-1][1] if i > 0 else None
            return name, threshold, next_threshold
    
    # Fallback (should never reach here with 0 threshold)
    return "⚪ Unranked", 0, TIERS_SORTED[-2][1] if len(TIERS_SORTED) > 1 else 100
    
def get_tier_name(xp: int) -> str:
    """Return just the tier name"""
    tier_name, _, _ = get_tier_info(xp)
    return tier_name

async def get_user_rank(user_id: int, sorted_users: list) -> Optional[int]:
    """Get a user's rank position based on XP (lower number = higher rank)"""
    try:
        # Create a lookup dictionary for this specific call
        rank_lookup = {user_id_in_db: index + 1 
                      for index, (user_id_in_db, _) in enumerate(sorted_users)}
        
        user_id_str = str(user_id)
        return rank_lookup.get(user_id_str)
    except Exception:
        return None


# Progress bar
def make_progress_bar(xp: int, current: int, next_threshold: Optional[int]) -> str:
    if not next_threshold:  # Already at max tier
        return "🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨 (MAX)"

    total_needed = next_threshold - current
    gained = xp - current
    filled = int((gained / total_needed) * 10)
    filled = min(filled, 10)

    bar = "🟩" * filled + "⬛" * (10 - filled)
    return f"{bar} ({gained}/{total_needed} XP)"

async def send_hr_welcome_test(member: discord.Member):
    """Send a test HR welcome message fetched from Supabase."""
    logger.info(f"📬 Preparing to send HR welcome message to {member.display_name} ({member.id})")

    template = test_messages.get("hr_welcome")
    if not template:
        logger.warning("⚠️ No 'hr_welcome' message found in test_messages dict.")
        template = "Welcome to HR, {user_mention}! (default fallback)"

    message = template.format(user_mention=member.mention)

    try:
        await member.send(message)
        logger.info(f"✅ Successfully sent HR welcome message to {member.display_name}")
    except discord.Forbidden:
        logger.error(f"🚫 Cannot send DM to {member.display_name} — they likely have DMs disabled.")
    except discord.HTTPException as e:
        logger.error(f"❌ Discord HTTP error while sending HR welcome: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error sending HR welcome: {e}")

    


# Global rate limiter configuration
GLOBAL_RATE_LIMIT = 15  # requests per minute
COMMAND_COOLDOWN = 10    # seconds between command uses per user


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
                asyncio.to_thread(socket.getaddrinfo, domain, None),
                timeout=timeout / len(domains)
            )
            return bool(res)
        except Exception:
            return False

    results = await asyncio.gather(*[_resolve(d) for d in domains], return_exceptions=True)
    return any(r is True for r in results)

# --- CLASSES ---
class ConnectionMonitor:
    def __init__(self, bot, check_interval: int = 300):
        self.bot = bot
        self.check_interval = check_interval
        self.last_disconnect = None
        self.disconnect_count = 0
        self._task = None
        self._stopping = False

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._monitor_connection())

    async def stop(self):
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

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

    def record_disconnect(self):
        self.last_disconnect = time.time()
        self.disconnect_count += 1
        logger.warning("ConnectionMonitor: recorded disconnect (count=%d)", self.disconnect_count)

        
class GlobalRateLimiter:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(30)
        self.last_request = 0
        self.min_delay = 1.0 / 30
        self.request_count = 0
        self.last_reset = time.time()
        self.total_limited = 0
        self._lock = asyncio.Lock()

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

    async def __aexit__(self, exc_type, exc, tb):
        self.semaphore.release()
        if exc:
            logger.error(f"Error in rate limited block: {exc}")
        return False
        
class EnhancedRateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute

        self.buckets = {}
        self.locks = {}

        self.global_lock = asyncio.Lock()
        self.global_count = 0
        self.last_global_reset = now()

        self.last_bucket_cleanup = now()

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
                self.buckets.pop(key, None)
                self.locks.pop(key, None)

            self.last_bucket_cleanup = current_time

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

class DiscordAPI:
    """Helper class for Discord API requests with retry logic"""
    @staticmethod
    async def execute_with_retry(coro, max_retries=3, initial_delay=1.0):
        for attempt in range(max_retries):
            try:
                async with global_rate_limiter:
                    return await coro
            except discord.errors.HTTPException as e:
                status = getattr(e, "status", None)
                if status == 429:
                    retry_after = initial_delay * (2 ** attempt)
                    # try to get header if available
                    try:
                        header_ra = e.response.headers.get('Retry-After')
                        if header_ra:
                            retry_after = float(header_ra)
                    except Exception:
                        pass
                    logger.warning(f"Rate limited. Attempt {attempt+1}/{max_retries}. Waiting {retry_after:.2f}s")
                    await asyncio.sleep(retry_after)
                    continue
                # other HTTP errors - rethrow after last attempt
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


#supabase set-up
class DatabaseHandler:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            logger.warning("Supabase credentials not provided; DatabaseHandler will be inert.")
            self.supabase = None
        else:
            # create sync client once (cheap); all sync calls are run in threads
            self.supabase = create_client(url, key)

    async def _run_sync(self, fn, *args, **kwargs):
        """Helper to run sync functions in the default threadpool."""
        return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))

    # Example: get user xp (was blocking)
    async def get_user_xp(self, user_id: str) -> int:
        if not self.supabase:
            return 0
        def _work():
            res = self.supabase.table('users').select("xp").eq("user_id", str(user_id)).execute()
            return res.data[0].get('xp', 0) if getattr(res, 'data', None) else 0
        try:
            return await self._run_sync(_work)
        except Exception as e:
            logger.exception("get_user_xp failed: %s", e)
            return 0

    async def add_xp(self, user_id: str, username: str, xp: int) -> Tuple[bool, int]:
        if not self.supabase:
            return False, 0
        def _work():
            # fetch current
            res = self.supabase.table('users').select("xp").eq("user_id", str(user_id)).execute()
            current = res.data[0].get('xp', 0) if getattr(res, 'data', None) else 0
            new_total = current + xp
            # upsert new value
            self.supabase.table('users').upsert({
                "user_id": str(user_id),
                "xp": new_total,
                "username": clean_nickname(username)
            }).execute()
            return True, new_total
        try:
            return await self._run_sync(_work)
        except Exception as e:
            logger.exception("add_xp failed: %s", e)
            return False, 0

    async def remove_xp(self, user_id: str, xp: int) -> Tuple[bool, int]:
        if not self.supabase:
            return False, 0
        def _work():
            res = self.supabase.table('users').select("xp").eq("user_id", str(user_id)).execute()
            current = res.data[0].get('xp', 0) if getattr(res, 'data', None) else 0
            new_total = max(0, current - xp)
            self.supabase.table('users').update({"xp": new_total}).eq("user_id", str(user_id)).execute()
            return True, new_total
        try:
            return await self._run_sync(_work)
        except Exception as e:
            logger.exception("remove_xp failed: %s", e)
            return False, current

    # Generic helper to run arbitrary table queries when needed from command handlers
    async def run_query(self, fn):
        """Run a provided sync function that performs supabase queries"""
        if not self.supabase:
            raise RuntimeError("Supabase not configured")
        try:
            return await self._run_sync(fn)
        except Exception as e:
            logger.exception("run_query failed: %s", e)
            raise

    async def increment_points(self, table: str, member: discord.Member, points_awarded: int):
        try:
            # Use the _run_sync helper to execute the supabase operations
            def _work():
                # Fetch existing record (if any)
                res = self.supabase.table(table).select("points").eq("user_id", str(member.id)).execute()
                current_points = res.data[0]["points"] if res.data and len(res.data) > 0 else 0
                
                # Upsert with new total
                self.supabase.table(table).upsert({
                    "user_id": str(member.id),
                    "username": clean_nickname(member.display_name),
                    "points": current_points + points_awarded
                }).execute()
                
                return current_points, current_points + points_awarded
            
            # Run the synchronous operation in a thread
            old_points, new_points = await self._run_sync(_work)
            
            logger.info(f"📊 Updated {table} points for {member.display_name} ({member.id}): {old_points} ➝ {new_points}")
            
        except Exception as e:
            logger.error(f"❌ Failed to increment points in {table}: {e}")
            
    async def get_all_users_sorted_by_xp(self) -> list:
        """Get all users sorted by XP in descending order"""
        if not self.supabase:
            logger.warning("Supabase not configured, returning empty list")
            return []
        
        try:
            def _work():
                # Use Supabase query to get all users sorted by XP descending
                res = self.supabase.table('users').select("user_id, xp").order("xp", desc=True).execute()
                return res.data if hasattr(res, 'data') else []
            
            # Get the sorted user data
            user_data = await self._run_sync(_work)
            
            # Convert to the expected format: list of (user_id, xp) tuples
            return [(user['user_id'], user['xp']) for user in user_data]
            
        except Exception as e:
            logger.error(f"Error getting sorted users from Supabase: {e}")
            return []


    async def discharge_user(self, user_id: str, username: str, guild: discord.Guild) -> None:
        """Delete a user from all relevant tables, and log the result to the default channel."""
        tables = ["users", "HRs", "LRs"]
        success = True
    
        def _work():
            nonlocal success
            for table in tables:
                try:
                    self.supabase.table(table).delete().eq("user_id", str(user_id)).execute()
                    logger.info(f"Deleted {username} ({user_id}) from {table}")
                except Exception as e:
                    logger.error(f"Failed to delete {username} ({user_id}) from {table}: {e}")
                    success = False

        try:
            await self._run_sync(_work)
        except Exception as e:
            logger.error(f"Discharge operation failed for {username} ({user_id}): {e}")
            success = False

        # Prepare embed
        if success:
            color = discord.Color.green()
            title = "✅ Removed from Database"
            description = f"**{username}** (`{user_id}`) has been successfully removed from all tables."
        else:
            color = discord.Color.red()
            title = "❌ Database Removal Failed"
            description = f"An error occurred while removing **{username}** (`{user_id}`). Check logs for details."
    
        embed = discord.Embed(title=title, description=description, color=color)
        embed.set_footer(text="Automated database removal log")

        # Correct location for logging
        try:
            log_channel = guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
            if log_channel:
                await log_channel.send(embed=embed)
            else:
                logger.warning("Default log channel not found.")
        except Exception as e:
            logger.error(f"Failed to send database removal embed: {e}")


    async def save_user_roles(self, user_id: str, username: str, role_ids: list[int]):
        """Save a user's tracked roles into the 'user_roles' table."""
        if not self.supabase:
            logger.warning("Supabase not configured; save_user_roles aborted.")
            return False

        def _work():
            try:
                self.supabase.table("user_roles").upsert({
                    "user_id": str(user_id),
                    "username": clean_nickname(username),
                    "roles": role_ids
                }).execute()
                return True
            except Exception as e:
                logger.error(f"save_user_roles failed for {user_id}: {e}")
                return False

        return await self._run_sync(_work)


    async def get_user_roles(self, user_id: str):
        """Retrieve saved roles for a user."""
        if not self.supabase:
            logger.warning("Supabase not configured; get_user_roles aborted.")
            return None

        def _work():
            try:
                res = self.supabase.table("user_roles").select("roles").eq("user_id", str(user_id)).execute()
                if res.data:
                    return res.data[0].get("roles", [])
                return None
            except Exception as e:
                logger.error(f"get_user_roles failed for {user_id}: {e}")
                return None

        return await self._run_sync(_work)

        # Send to logging channel
        try:
            log_channel = guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
            if log_channel:
                await log_channel.send(embed=embed)
            else:
                logger.warning("Default log channel not found.")
        except Exception as e:
            logger.error(f"Failed to send database removal embed: {e}")

# Clean up old processed reactions and messages in db
async def start_cleanup_task(self):
    """Clean up old database entries periodically"""
    async def cleanup_loop():
        while True:
            await asyncio.sleep(86400)  # Cleanup daily
            try:
                # Clean entries older than 30 days
                def _cleanup():
                    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                    self.bot.db.supabase.table('processed_messages')\
                        .delete()\
                        .lt('processed_at', cutoff.isoformat())\
                        .execute()
                    self.bot.db.supabase.table('processed_reactions')\
                        .delete()\
                        .lt('processed_at', cutoff.isoformat())\
                        .execute()
                await self.bot.db.run_query(_cleanup)
                logger.info("Cleaned up old processed messages/reactions")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    self._cleanup_task = asyncio.create_task(cleanup_loop())

    
# Logs XP changes in logging channel
async def log_xp_to_discord(
    admin: discord.User,
    user: discord.User,
    xp_change: int,
    new_total: int,
    reason: str
):
    """Log XP changes to a Discord channel instead of Supabase."""
    log_channel = bot.get_channel(Config.DEFAULT_LOG_CHANNEL)  # Replace with your channel ID
    if not log_channel:
        logger.error("XP log channel not found!")
        return False

    embed = discord.Embed(
        title="📊 XP Change Logged",
        color=discord.Color.green() if xp_change > 0 else discord.Color.red(),
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(name="Staff", value=f"{admin.mention} (`{admin.id}`)", inline=True)
    embed.add_field(name="User", value=f"{user.mention} (`{user.id}`)", inline=True)
    embed.add_field(name="XP Change", value=f"`{xp_change:+}`", inline=True)
    embed.add_field(name="New Total", value=f"`{new_total}`", inline=True)
    embed.add_field(name="Reason", value=f"```{reason}```", inline=False)
    
    try:
        await log_channel.send(embed=embed)
        return True
    except Exception as e:
        logger.error(f"Failed to log XP to Discord: {str(e)}")
        return False


# Roblox API helper functions
class RobloxAPI:
    @staticmethod
    async def get_user_id(username: str) -> Optional[int]:
        """Get Roblox user ID from username using shared session"""
        try:
            async with bot.rate_limiter:  # Use global rate limiter
                async with bot.shared_session.get(
                    f"https://api.roblox.com/users/get-by-username?username={username}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("Id"):
                            return data["Id"]
                        else:
                            logger.warning(f"Roblox username not found: {username}")
                            return None
                    elif response.status == 429:
                        logger.warning("Roblox API rate limited - consider adding delay")
                        return None
                    else:
                        logger.warning(f"Roblox API error: HTTP {response.status} for {username}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"Roblox API timeout for username: {username}")
            return None
        except Exception as e:
            logger.error(f"Failed to get Roblox user ID for {username}: {e}")
            return None

    @staticmethod
    async def get_group_rank(user_id: int, group_id: int = 4972920) -> Optional[str]:
        """Get user's rank in specific group using shared session"""
        try:
            async with bot.rate_limiter:  # Use global rate limiter
                async with bot.shared_session.get(
                    f"https://groups.roblox.com/v1/users/{user_id}/groups/roles",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for group in data.get("data", []):
                            if group["group"]["id"] == group_id:
                                return group["role"]["name"]
                        logger.warning(f"User {user_id} not in group {group_id}")
                        return None
                    elif response.status == 429:
                        logger.warning("Roblox groups API rate limited")
                        return None
                    else:
                        logger.warning(f"Groups API error: HTTP {response.status} for user {user_id}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"Groups API timeout for user: {user_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get group rank for user {user_id}: {e}")
            return None

    @staticmethod
    async def get_roblox_rank(discord_name: str, group_id: int = 4972920) -> str:
        """Get Roblox rank from Discord display name with better error handling"""
        try:
            # Clean the nickname and extract potential Roblox username
            cleaned_name = clean_nickname(discord_name)
            
            if not cleaned_name or cleaned_name.lower() == "unknown":
                logger.warning(f"Invalid Discord name for Roblox lookup: {discord_name}")
                return "Invalid Name"
            
            # Try to find Roblox user ID
            user_id = await RobloxAPI.get_user_id(cleaned_name)
            if not user_id:
                logger.warning(f"Roblox user not found for Discord name: {cleaned_name}")
                return "Not Found"
            
            # Get group rank with retry logic for rate limits
            rank = await RobloxAPI.get_group_rank(user_id, group_id)
            if not rank:
                logger.info(f"User {cleaned_name} (ID: {user_id}) has no specific rank in group")
                return "Member"  # Default to "Member" if no specific rank found
            
            return rank
            
        except Exception as e:
            logger.error(f"Failed to get Roblox rank for {discord_name}: {e}")
            return "Error"

# Reaction Logger 
class ReactionLogger:
    POINTS_PER_ACTIVITY = 0.5
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.monitor_channel_ids = set(Config.DEFAULT_MONITOR_CHANNELS)
        self.log_channel_id = Config.DEFAULT_LOG_CHANNEL
        self.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)
        self.event_channel_ids = [Config.W_EVENT_LOG_CHANNEL_ID, Config.EVENT_LOG_CHANNEL_ID]
        self.phase_log_channel_id = Config.PHASE_LOG_CHANNEL_ID
        self.tryout_log_channel_id = Config.TRYOUT_LOG_CHANNEL_ID
        self.course_log_channel_id = Config.COURSE_LOG_CHANNEL_ID
        self.activity_log_channel_id = Config.ACTIVITY_LOG_CHANNEL_ID
        

    async def setup(self, interaction=None, log_channel=None, monitor_channels=None):
        """Setup reaction logger with optional parameters"""
        if interaction and log_channel and monitor_channels:
            # Setup from command
            channel_ids = [int(cid.strip()) for cid in monitor_channels.split(',')]
            self.monitor_channel_ids = set(channel_ids)
            self.log_channel_id = log_channel.id
            await interaction.followup.send("✅ Reaction tracking setup complete", ephemeral=True)
            
        """Register event listeners and validate DB connection."""
        # Supabase connection check
        try:
            await self.bot.db.run_query(lambda: self.bot.db.supabase.table("LD").select("count").limit(1).execute())
            logger.info("✅ Supabase connection validated")
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")

    
    async def on_ready_setup(self):
        """Verify configured channels when bot starts"""
        guild = self.bot.guilds[0]
        valid_channels = set()
        for channel_id in self.monitor_channel_ids:
            if channel := guild.get_channel(channel_id):
                valid_channels.add(channel.id)
        
        self.monitor_channel_ids = valid_channels
        
        log_channel = guild.get_channel(self.log_channel_id)
        if not log_channel:
            logger.warning(f"⚠️ Default log channel {self.log_channel_id} not found! Trying other.")
            log_channel = guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
        else:
            logger.info("Default Log channel configured.")
        
        if not log_channel:
            logger.error("❌ No valid log channel found for ReactionLogger!")
            
    async def is_reaction_processed(self, message_id: int, user_id: int) -> bool:
        """Check if reaction was already processed"""
        try:
            def _check():
                res = self.bot.db.supabase.table('processed_reactions')\
                    .select('id')\
                    .eq('message_id', message_id)\
                    .eq('user_id', user_id)\
                    .execute()
                return len(res.data) > 0
            return await self.bot.db.run_query(_check)
        except Exception as e:
            logger.error(f"Error checking processed reaction: {e}")
            return False

    async def mark_reaction_processed(self, message_id: int, user_id: int):
        """Mark reaction as processed"""
        try:
            def _insert():
                self.bot.db.supabase.table('processed_reactions')\
                    .insert({
                        'message_id': message_id,
                        'user_id': user_id
                    })\
                    .execute()
            await self.bot.db.run_query(_insert)
        except Exception as e:
            logger.error(f"Error marking reaction processed: {e}")

    async def health_check(self):
        """Check the health of the reaction logger"""
        try:
            # Check if listeners are registered
            add_listeners = [
                l for l in self.bot.extra_events.get('on_raw_reaction_add', [])
                if getattr(l, '__self__', None) is self
            ]
            remove_listeners = [
                l for l in self.bot.extra_events.get('on_raw_reaction_remove', []) 
                if getattr(l, '__self__', None) is self
            ]   
            
            status = {
                "add_listeners": len(add_listeners),
                "remove_listeners": len(remove_listeners),
                "monitor_channels": len(self.monitor_channel_ids),
                "log_channel_id": self.log_channel_id,
            }
            logger.info(f"🔧 ReactionLogger Health Check: {status}")
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'error': str(e)}

    
    async def log_reaction(self, payload: discord.RawReactionActionEvent):
        """Main reaction handler that routes to specific loggers with DB + memory duplicate checks."""
        if payload.event_type == "REACTION_REMOVE":
            return
        
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
    
        member = guild.get_member(payload.user_id)
        if not member:
            try:
                member = await guild.fetch_member(payload.user_id)
            except discord.NotFound:
                return
    
        # === DUPLICATE GUARD (silent) ===
        if await self.is_reaction_processed(payload.message_id, payload.user_id):
            logger.debug(
                f"Duplicate reaction ignored | "
                f"msg={payload.message_id} user={payload.user_id}"
            )
            return
            
        
        logger.info(f"🔍 Reaction detected: {payload.emoji} in channel {payload.channel_id} by user {payload.user_id}")

        # === For Background Checkers ===
        if payload.channel_id == Config.BGC_LOGS_CHANNEL:
            emoji = str(payload.emoji)
            if emoji == "☑️" or emoji not in Config.TRACKED_REACTIONS:
                return
        
            hicom_role = guild.get_role(Config.HIGH_COMMAND_ROLE_ID)
            if not hicom_role or hicom_role not in member.roles:
                return
        
            channel = guild.get_channel(payload.channel_id)
            if not channel:
                return
        
            message = await channel.fetch_message(payload.message_id)
            log_author = message.author
        
        
            match = re.search(
                r"Security\s*Check(?:\(s\)|s)?:\s*(\d+)",
                message.content,
                re.IGNORECASE
            )
            
            if not match:
                logger.warning(
                    f"No Security Check count found in message {payload.message_id}"
                )
                return
            
            security_checks = int(match.group(1))
        
            log_channel = guild.get_channel(self.log_channel_id)
            if not log_channel:
                return
            
            points = self.POINTS_PER_ACTIVITY * security_checks
            embed = discord.Embed(
                title="🪪 Security Check Log Approved",
                color=discord.Color.blue()
            )
            embed.add_field(name="Approved by", value=f"{member.mention} ({member.id})", inline=False)
            embed.add_field(name="Logger", value=f"{log_author.mention} ({log_author.id})", inline=False)
            embed.add_field(name="Points Awarded", value=points, inline=True)
            embed.add_field(name="Log ID", value=f"`{payload.message_id}`", inline=False)
        
            await log_channel.send(embed=embed)
            await self._update_hr_record(log_author, {"courses": points})
        
            logger.info(
                f"🪪 Security Check log Approved: ApprovedBy={member.id}, Logger={log_author.id}"
            )
            # Prevents it from going to other log pipelines
            await self.mark_reaction_processed(payload.message_id, payload.user_id)
            return
        


        # === For Exam graders ===
        if payload.channel_id in Config.EXAM_MONITOR_CHANNELS:
            emoji = str(payload.emoji)
            if emoji == "☑️" or emoji not in Config.TRACKED_REACTIONS:
                return
        
            hicom_role = guild.get_role(Config.HIGH_COMMAND_ROLE_ID)
            inductor_role = guild.get_role(Config.LA_ROLE_ID)
            
            if not (
                (hicom_role and hicom_role in member.roles) or
                (inductor_role and inductor_role in member.roles)
            ):
                return
                    
            channel = guild.get_channel(payload.channel_id)
            if not channel:
                return
        
            message = await channel.fetch_message(payload.message_id)
            examiner = message.author
        
            log_channel = guild.get_channel(self.log_channel_id)
            if not log_channel:
                return
            
            if payload.channel_id == Config.LA_INDUCTION_CHANNEL_ID: 
                embed = discord.Embed(
                    title="📝 Inductor Activity Logged",
                    color=discord.Color.pink()
                )
                
                embed.add_field(name="Inductor", value=f"{member.mention} ({member.id})", inline=False)
                embed.add_field(name="Message Request", value=f"<#{payload.channel_id}>", inline=True)
                embed.add_field(name="Status", value=f"{emoji}", inline=True)
                embed.add_field(name="Points Awarded", value=self.POINTS_PER_ACTIVITY, inline=True)
                await log_channel.send(embed=embed)
                logger.info(
                f"📝 Inductor Activity logged: Inductor={member.id}, Channel={payload.channel_id}"
                ) 
                await self._update_hr_record(member, {"courses": self.POINTS_PER_ACTIVITY})
            else:
                points = self.POINTS_PER_ACTIVITY*2
                embed = discord.Embed(
                    title="📝 Examiner Activity Logged",
                    color=discord.Color.pink()
                )
                embed.add_field(name="Examiner", value=f"{examiner.mention} ({examiner.id})", inline=False)
                embed.add_field(name="Exam Type", value=f"<#{payload.channel_id}>", inline=True)
                embed.add_field(name="Exam Message", value=f"`{payload.message_id}`", inline=False)
                embed.add_field(name="Points Awarded", value=points, inline=True)
                await log_channel.send(embed=embed)
                logger.info(
                f"📝 Examiner logged: Examiner={examiner.id}, Channel={payload.channel_id}"
                )
                await self._update_hr_record(examiner, {"courses": points})
        
            await self.mark_reaction_processed(payload.message_id, payload.user_id)      
            return
        
        # === Route to handlers ===
        try:
            await self.rate_limiter.wait_if_needed(bucket="reaction_log")
            await self._log_reaction_impl(payload)
            await self._log_event_reaction_impl(payload, member)
            await self._log_training_reaction_impl(payload, member)
            await self._log_activity_reaction_impl(payload, member)
            await self.mark_reaction_processed(payload.message_id, payload.user_id)
        except Exception as e:
            logger.error(f"Failed to log reaction: {type(e).__name__}: {e}")
            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                error_embed = discord.Embed(
                    title="❌ Reaction Logging Error",
                    description="An error occured while logging this reaction.",
                    color=discord.Color.red(),
                )
                await log_channel.send(content=member.mention, embed=error_embed)

     
    async def _log_reaction_impl(self, payload: discord.RawReactionActionEvent):
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
    
        if (payload.channel_id not in self.monitor_channel_ids or 
            str(payload.emoji) not in Config.TRACKED_REACTIONS):
            return
    
        if (payload.channel_id in Config.IGNORED_CHANNELS and 
            str(payload.emoji) in Config.IGNORED_EMOJI):
            return
    
        member = guild.get_member(payload.user_id)
        if not member:
            try:
                member = await guild.fetch_member(payload.user_id)
            except discord.NotFound:
                logger.info("Member not found for reaction event; skipping")
                return
            except Exception:
                logger.exception("Failed to fetch member for reaction", exc_info=True)
                return
    
        if Config.DB_LOGGER_ROLE_ID:
            monitor_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
            if not monitor_role or monitor_role not in member.roles:
                return
    
        channel = guild.get_channel(payload.channel_id)
        log_channel = guild.get_channel(self.log_channel_id)
    
        if not all((channel, member, log_channel)):
            return
        
        try:
            await asyncio.sleep(0.5)
            message = await channel.fetch_message(payload.message_id)

            content = (message.content[:100] + "...") if len(message.content) > 100 else message.content

            embed = discord.Embed(
                title="🧑‍💻 DB Logger Activity Recorded",
                description=f"{member.mention} reacted with {payload.emoji}",
                color=discord.Color.purple()
            )
            embed.add_field(name="Channel", value=channel.mention)
            embed.add_field(name="Author", value=message.author.mention)
            embed.add_field(name="Message", value=content, inline=False)
            embed.add_field(name="Points Awarded", value=self.POINTS_PER_ACTIVITY, inline=False)
            embed.add_field(name="Jump to", value=f"[Click here]({message.jump_url})", inline=False)

            await log_channel.send(embed=embed)

            logger.info(f"Attempting to update points for: {member.display_name}")
            await self._update_hr_record(member, {"courses": self.POINTS_PER_ACTIVITY})
            logger.info(f"✅ Added {self.POINTS_PER_ACTIVITY} points to {member.display_name} for activity.")


        except discord.NotFound:
            return
        except Exception as e:
            logger.error(f"Reaction log error: {type(e).__name__}: {str(e)}")
            if log_channel:
                error_embed = discord.Embed(
                    title="❌ Error",
                    description=f"Failed to log reaction: {str(e)}",
                    color=discord.Color.red()
                )
                await log_channel.send(embed=error_embed)


   
    async def _log_event_reaction_impl(self, payload: discord.RawReactionActionEvent, member: discord.Member):
        """Handle event logging without confirmation"""
        if payload.channel_id not in self.event_channel_ids or str(payload.emoji) != "✅":
            return

        guild = member.guild
        if not guild:
            return

        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return

        channel = guild.get_channel(payload.channel_id)
        log_channel = guild.get_channel(self.log_channel_id)
        if not channel or not log_channel:
            return

        try:
            await asyncio.sleep(0.5)  # Prevent bursts
            message = await channel.fetch_message(payload.message_id)

            # Extract host
            host_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
            host_id = int(host_mention.group(1)) if host_mention else message.author.id
            host_member = guild.get_member(host_id) or await guild.fetch_member(host_id)

            cleaned_host_name = clean_nickname(host_member.display_name)

            # Determine event type
            hr_update_field = "events"
            if payload.channel_id == Config.W_EVENT_LOG_CHANNEL_ID:
                if re.search(r'\bjoint\b', message.content, re.IGNORECASE):
                    hr_update_field = "joint_events"
                elif re.search(r'\b(inspection|pi|Inspection)\b', message.content, re.IGNORECASE):
                    hr_update_field = "inspections"

            event_name_match = re.search(r'Event:\s*(.*?)(?:\n|$)', message.content, re.IGNORECASE)
            event_name = event_name_match.group(1).strip() if event_name_match else hr_update_field.replace("_", " ").title()

            # Update HR table
            await self._update_hr_record(host_member, {hr_update_field: 1})
            logger.info(f"✅ Logged host {cleaned_host_name} to HR table")

            # Process attendees
            attendees_section = re.search(r'(?:Attendees:|Passed:)\s*((?:<@!?\d+>\s*)+)', message.content, re.IGNORECASE)
            if not attendees_section:
                return
            attendee_mentions = re.findall(r'<@!?(\d+)>', attendees_section.group(1))

            hr_role = guild.get_role(Config.HR_ROLE_ID)
            hr_attendees, success_count = [], 0

            for attendee_id in attendee_mentions:
                attendee_member = guild.get_member(int(attendee_id)) or await guild.fetch_member(int(attendee_id))
                if not attendee_member:
                    continue
                if hr_role and hr_role in attendee_member.roles:
                    hr_attendees.append(attendee_id)
                    continue
                await self._update_lr_record(attendee_member, {"events_attended": 1})
                success_count += 1

            # Embed
            done_embed = discord.Embed(title="✅ Event Logged Successfully", color=discord.Color.green())
            done_embed.add_field(name="Host", value=host_member.mention, inline=True)
            done_embed.add_field(name="Attendees Recorded", value=str(success_count), inline=True)
            if hr_attendees:
                done_embed.add_field(name="HR Attendees Excluded", value=str(len(hr_attendees)), inline=False)
            done_embed.add_field(name="Logged By", value=member.mention, inline=False)
            done_embed.add_field(name="Event Type", value=event_name, inline=True)
            done_embed.add_field(name="Message", value=f"[Jump to Event]({message.jump_url})", inline=False)

            await log_channel.send(content=member.mention, embed=done_embed)
            logger.info(
                f"✅ Event logged successfully: "
                f"Host={host_member} ({host_member.id}), "
                f"Attendees={success_count}, "
                f"HR_Excluded={len(hr_attendees) if hr_attendees else 0}, "
                f"Logged_By={member} ({member.id}), "
                f"EventType={event_name}, "
                f"MessageID={message.id}"
            )


        except Exception as e:
            logger.error(f"Error processing event reaction: {e}")
            await log_channel.send(embed=discord.Embed(title="❌ Event Log Error", description=str(e), color=discord.Color.red()))



    async def _log_training_reaction_impl(self, payload: discord.RawReactionActionEvent, member: discord.Member):
        """Handle training logs (phases, tryouts, courses)"""
        mapping = {
            self.phase_log_channel_id: "phases",
            self.tryout_log_channel_id: "tryouts",
            self.course_log_channel_id: "phases",
        }
        column_to_update = mapping.get(payload.channel_id)
        if not column_to_update or str(payload.emoji) != "✅":
            return

        guild = member.guild
        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return

        try:
            await asyncio.sleep(0.5)
            message = await guild.get_channel(payload.channel_id).fetch_message(payload.message_id)

            user_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
            user_id = int(user_mention.group(1)) if user_mention else message.author.id
            user_member = guild.get_member(user_id) or await guild.fetch_member(user_id)

            await self._update_hr_record(user_member, {column_to_update: 1})

            title = {
                "phases": "📊 Phase Logged",
                "tryouts": "📊 Tryout Logged",
                "courses": "📊 Course Logged",
            }.get(column_to_update, "📊 Training Logged")

            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                embed = discord.Embed(title=title, color=discord.Color.blue())
                embed.add_field(name="Host", value=user_member.mention)
                embed.add_field(name="Logged By", value=member.mention)
                await log_channel.send(embed=embed)
                logger.info(
                    f"📘 Event log embed sent: "
                    f"Title='{title}', "
                    f"Host={user_member} ({user_member.id}), "
                    f"Logged_By={member} ({member.id}), "
                    f"Channel=#{log_channel.name}"
                )

        except Exception as e:
            logger.error(f"Error processing {column_to_update} reaction: {e}")
            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                await log_channel.send(embed=discord.Embed(title="❌ Log Error", description=str(e), color=discord.Color.red()))


    async def _log_activity_reaction_impl(self, payload: discord.RawReactionActionEvent, member: discord.Member):
        """Handle activity logs (time guarded and activity)"""
        if payload.channel_id != self.activity_log_channel_id or str(payload.emoji) != "✅":
            return

        guild = member.guild
        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return

        try:
            await asyncio.sleep(0.5)
            message = await guild.get_channel(payload.channel_id).fetch_message(payload.message_id)

            user_mention = re.search(r'<@!?(\d+)>', message.content)
            user_id = int(user_mention.group(1)) if user_mention else message.author.id
            user_member = guild.get_member(user_id) or await guild.fetch_member(user_id)

            updates = {}
            time_match = re.search(r'Time:\s*(\d+)', message.content)
            if time_match:
                if "Guarded:" in message.content:
                    updates["time_guarded"] = int(time_match.group(1))
                else:
                    updates["activity"] = int(time_match.group(1))

            if updates:
                #Update LR record first
                await self._update_lr_record(user_member, updates)
                
                # 🟩 XP logic: 1 XP per 30 mins (activity or guarded)
                total_minutes = 0
                if "activity" in updates:
                    total_minutes += updates["activity"]
                if "time_guarded" in updates:
                    total_minutes += updates["time_guarded"]
            
                xp_to_award = total_minutes // 30
                if xp_to_award > 0:
                    success, new_xp = await self.bot.db.add_xp(
                        str(user_member.id),
                        user_member.display_name,
                        xp_to_award
                    )
                    if success:
                        logger.info(f"⭐ Gave {xp_to_award} XP to {user_member.display_name} ({user_member.id}) for {total_minutes} mins activity")

                log_channel = guild.get_channel(self.log_channel_id)
                if log_channel:
                    embed = discord.Embed(title="⏱ Activity Logged", color=discord.Color.green())
                    embed.add_field(name="Member", value=user_member.mention)
                    if "activity" in updates:
                        embed.add_field(name="Activity Time", value=f"{updates['activity']} mins")
                    if "time_guarded" in updates:
                        embed.add_field(name="Guarded Time", value=f"{updates['time_guarded']} mins")
                    if xp_to_award > 0:
                        embed.add_field(name="XP Awarded", value=f"+{xp_to_award} XP")
                    embed.add_field(name="Logged By", value=member.mention)
                    embed.add_field(name="Message", value=f"[Jump to Log]({message.jump_url})")
                    await log_channel.send(content=member.mention, embed=embed)

        except Exception as e:
            logger.error(f"Error processing activity reaction: {e}")
            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                await log_channel.send(embed=discord.Embed(title="❌ Activity Log Error", description=str(e), color=discord.Color.red()))

                
    async def _update_hr_record(self, member: discord.Member, updates: dict):
        u_str = str(member.id)
    
        def _work():
            sup = self.bot.db.supabase
            row = sup.table('HRs').select('*').eq('user_id', u_str).execute()
            
            FLOAT_COLUMNS = {"courses"} 

            if getattr(row, "data", None):
                existing = row.data[0]
                incremented = {}
            
                for key, value in updates.items():
                    if isinstance(value, (int, float)):
                        current = existing.get(key, 0) or 0
            
                        # If this column is meant to store floats
                        if key in FLOAT_COLUMNS:
                            incremented[key] = float(current) + float(value)
                        else:
                            incremented[key] = int(current) + int(value)
                    else:
                        incremented[key] = value
            
                return sup.table("HRs").update({
                    **incremented,
                    "username": clean_nickname(member.display_name)
                }).eq("user_id", u_str).execute()
            
            else:
                payload = {
                    "user_id": u_str,
                    "username": clean_nickname(member.display_name),
                    **updates
                }
                return sup.table("HRs").insert(payload).execute()
    
        try:
            await self.bot.db.run_query(_work)
        except Exception:
            logger.exception("ReactionLogger._update_hr_record failed")
    
    
    async def _update_lr_record(self, member: discord.Member, updates: dict):
        u_str = str(member.id)
    
        def _work():
            sup = self.bot.db.supabase
            row = sup.table('LRs').select('*').eq('user_id', u_str).execute()
    
            if getattr(row, "data", None):
                existing = row.data[0]
                # Increment numerical fields
                incremented = {}
                for key, value in updates.items():
                    if isinstance(value, int):
                        incremented[key] = existing.get(key, 0) + value
                    else:
                        incremented[key] = value
                return sup.table('LRs').update({
                    **incremented,
                    "username": clean_nickname(member.display_name)
                }).eq('user_id', u_str).execute()
            else:
                payload = {
                    'user_id': u_str,
                    "username": clean_nickname(member.display_name),
                    **updates
                }
                return sup.table('LRs').insert(payload).execute()
    
        try:
            await self.bot.db.run_query(_work)
        except Exception:
            logger.exception("ReactionLogger._update_lr_record failed")
            
    async def update_hr(self, member: discord.Member, updates: dict):
        """Public method to update HR record from other classes"""
        try:
            await self._update_hr_record(member, updates)
            logger.info(f"HR record updated for {member.display_name}: {updates}")
        except Exception as e:
            logger.error(f"Failed to update HR record for {member.display_name}: {e}")

        
                        
      # --- Event listener ---
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Triggered when a reaction is added anywhere the bot can see."""
        try:
            await self.log_reaction(payload)
        except Exception as e:
            logger.error(f"ReactionLogger.on_raw_reaction_add failed: {e}", exc_info=True)

          


class ConfirmView(discord.ui.View):
    def __init__(self, *, timeout: float = 30.0):
        super().__init__(timeout=timeout)
        self.value = None

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = True
        await interaction.response.defer()
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = False
        await interaction.response.defer()
        self.stop()

    async def on_timeout(self):
        """Handle when the view times out"""
        self.value = False
        self.stop()



    

# --- Bot Initialization ---
intents = discord.Intents.default()
intents.guilds = True
intents.members = True               # Requires privileged intent in Developer Portal
intents.message_content = True       # Requires privileged intent in Developer Portal
intents.reactions = True

bot = commands.Bot(
    command_prefix="!.",
    intents=intents,
    activity=discord.Activity(
        type=discord.ActivityType.watching,
        name="out for RMP"
    ),
    max_messages=5000,              # Reduced from None to limit memory usage
    heartbeat_timeout=60.0,
    guild_ready_timeout=2.0,        # Faster guild readiness
    member_cache_flush_time=None,   #  Does not flush cache 
    chunk_guilds_at_startup=True,  # Chunks all members at once
    status=discord.Status.online
)

# --- Intents validation logging ---
if not intents.members:
    logger.warning("Privileged intent 'members' is disabled in code. Some checks may fail.")
if not intents.message_content:
    logger.warning("Privileged intent 'message_content' is disabled in code.")
logger.info("Bot initialized with intents: members=%s, message_content=%s, reactions=%s, guilds=%s",
            intents.members, intents.message_content, intents.reactions, intents.guilds)

global_rate_limiter = GlobalRateLimiter()
bot.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)
bot.reaction_logger = ReactionLogger(bot)
bot.api = DiscordAPI()
bot.db = DatabaseHandler()

# --- Custom Message Test Loader ---
test_messages = {}

async def load_test_messages():
    """Fetches test messages (like HR welcome) from Supabase for configurable templates."""
    global test_messages
    if not bot.db.supabase:
        logger.warning("Supabase not configured; skipping message load.")
        return

    try:
        result = bot.db.supabase.table("Messages_Test").select("*").execute()
        test_messages = {item["key"]: item["content"] for item in result.data}
        logger.info(f"✅ Loaded {len(test_messages)} test messages from Supabase.")
    except Exception as e:
        logger.error(f"❌ Failed to load test messages: {e}")


# --- Command Error Handler ---
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"⏳ This command is on cooldown. Try again in {error.retry_after:.1f} seconds.", ephemeral=True)
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("🔒 You don't have permission to use this command.", ephemeral=True)
    elif isinstance(error, discord.errors.HTTPException) and error.status == 429:
        retry_after = error.response.headers.get('Retry-After', 5)
        await ctx.send(f"⚠️ Too many requests. Please wait {retry_after} seconds before trying again.", ephemeral=True)
    else:
        logger.error(f"Command error: {type(error).__name__}: {str(error)}")
        await ctx.send("❌ An error occurred while processing your command.", ephemeral=True)

       
# Creating Sc command
create_sc_command(bot)

AVATAR_PATH = "avatar.gif"
BANNER_PATH = "banner.gif"

@bot.event
async def on_ready():
    logger.info("Logged in as %s (ID: %s)", bot.user, getattr(bot.user, "id", "unknown"))
    logger.info("Connected to %d guild(s)", len(bot.guilds))

    
    if not os.path.exists(AVATAR_PATH):
        logger.info("❌ GIF not found at {AVATAR_PATH}")
        return

    if os.path.exists(BANNER_PATH):
        try:
            with open(BANNER_PATH, "rb") as f:
                banner_bytes = f.read()
            await bot.user.edit(banner=banner_bytes)
            logger.info("✅ Bot Banner updated successfully!")
        except discord.HTTPException as e:
            logger.info("❌ Failed to set banner: {e}")
    else:
        logger.info("❌ Banner not found at {BANNER_PATH}")


    try:
        with open(AVATAR_PATH, "rb") as f:
            gif_bytes = f.read()

        await bot.user.edit(avatar=gif_bytes)
        logger.info("✅ Bot avatar updated successfully!")

    except discord.HTTPException as e:
        logger.info("❌ Failed to set avatar: {e}")


    # Close old session if it somehow exists
    if hasattr(bot, "shared_session") and bot.shared_session and not bot.shared_session.closed:
        await bot.shared_session.close()
        
    connector = aiohttp.TCPConnector(
        limit=15,
        limit_per_host=4,
        enable_cleanup_closed=True,
        force_close=False
    )
    bot.shared_session = aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT},
        timeout=aiohttp.ClientTimeout(total=12, connect=5, sock_connect=3, sock_read=6),
        connector=connector,
        trust_env=True,
    )
    logger.info("Created new shared aiohttp.ClientSession for this session")

    # Attach/start connection monitor and component setup
    bot.connection_monitor = ConnectionMonitor(bot)
    await bot.connection_monitor.start()

    try:
        await load_test_messages()
        logger.info("✅ Test messages loaded successfully from Supabase.")
    except Exception as e:
        logger.error(f"❌ Failed to load test messages: {e}")

    # === ROBUST ReactionLogger listener registration ===
    try:
        # Remove any existing listeners first to prevent duplicates
        for event_name in ['on_raw_reaction_add']:
            if event_name in bot.extra_events:
                bot.extra_events[event_name] = [
                    listener for listener in bot.extra_events[event_name]
                    if getattr(listener, '__self__', None) != bot.reaction_logger
                ]
        
        # Register fresh listeners
        bot.add_listener(bot.reaction_logger.on_raw_reaction_add, "on_raw_reaction_add")
        
        # Verify registration
        reaction_listeners = len([
            l for l in bot.extra_events.get('on_raw_reaction_add', [])
            if getattr(l, '__self__', None) is bot.reaction_logger
        ])
        
        logger.info(f"✅ ReactionLogger setup complete - {reaction_listeners} listeners active")
        
    except Exception as e:
        logger.error(f"❌ ReactionLogger listener registration failed: {e}")

    try:
        await bot.reaction_logger.setup()  
        logger.info("✅ ReactionLogger database setup complete")
    except Exception as e:
        logger.error(f"❌ ReactionLogger setup failed: {e}")
        
    # Setup reaction logger channels
    try:
        await bot.reaction_logger.on_ready_setup()
        logger.info("✅ ReactionLogger channel validation complete")
    except Exception as e:
        logger.error(f"❌ ReactionLogger channel validation failed: {e}")


    # Sync commands but don't block startup indefinitely
    try:
        await asyncio.wait_for(bot.tree.sync(), timeout=15.0)
    except asyncio.TimeoutError:
        logger.warning("command sync timed out (continuing)")


@bot.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord")
    if hasattr(bot.reaction_logger, "stop_cleanup_task"):
        await bot.reaction_logger.stop_cleanup_task()

    # Close the HTTP session
    if hasattr(bot, "shared_session") and bot.shared_session and not bot.shared_session.closed:
        await bot.shared_session.close()
    bot.shared_session = None
    logger.info("Closed shared aiohttp.ClientSession on disconnect")

@bot.event
async def on_resumed():
    logger.info("Bot successfully resumed (session resumption). Restoring state...")
    
    await asyncio.sleep(1.0)  # give Discord some breathing room

    # Recreate HTTP session if needed
    if not getattr(bot, "shared_session", None) or bot.shared_session.closed:
        connector = aiohttp.TCPConnector(limit=15, limit_per_host=4, enable_cleanup_closed=True)
        bot.shared_session = aiohttp.ClientSession(
            headers={"User-Agent": USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=12, connect=5, sock_connect=3, sock_read=6),
            connector=connector,
            trust_env=True,
        )
        logger.info("Recreated shared aiohttp.ClientSession after resume")

    # === COMPLETELY REBIND ReactionLogger listeners ===
    try:
        # Remove any existing listeners to prevent duplicates
        bot.extra_events['on_raw_reaction_add'] = [l for l in bot.extra_events.get('on_raw_reaction_add', []) 
                                                  if getattr(l, '__self__', None) != bot.reaction_logger]
        bot.extra_events['on_raw_reaction_remove'] = [l for l in bot.extra_events.get('on_raw_reaction_remove', []) 
                                                     if getattr(l, '__self__', None) != bot.reaction_logger]
        
        # Add fresh listeners
        bot.add_listener(bot.reaction_logger.on_raw_reaction_add, "on_raw_reaction_add")
        bot.add_listener(bot.reaction_logger.on_raw_reaction_remove, "on_raw_reaction_remove")
        
        logger.info("🔄 Re-attached ReactionLogger event listeners after resume")
    except Exception as e:
        logger.error(f"❌ Failed to reattach ReactionLogger listeners: {e}")

    # Restart channel + cleanup setups
    try:
        await asyncio.sleep(0.5)
        await bot.reaction_logger.on_ready_setup()

        if not getattr(bot.reaction_logger, "_cleanup_task", None) or bot.reaction_logger._cleanup_task.done():
            await bot.reaction_logger.start_cleanup_task()
            
        # Reinitialize reaction logger's rate limiter
        bot.reaction_logger.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)

        logger.info("✅ Restored ReactionLogger after resume.")
    except Exception as e:
        logger.error(f"❌ Failed to restore tracker after resume: {e}")



        
# --- COMMANDS --- 
# /addxp Command
@bot.tree.command(name="add-xp", description="Add XP to a user")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.CSM_ROLE_ID)
async def add_xp(interaction: discord.Interaction, user: discord.User, xp: int):
    async with global_rate_limiter:
        # Validate XP amount
        if xp <= 0:
            await interaction.response.send_message(
                "❌ XP amount must be positive.",
                ephemeral=True
            )
            return
        if xp > MAX_XP_PER_ACTION:
            await interaction.response.send_message(
                f"❌ Cannot give more than {MAX_XP_PER_ACTION} XP at once.",
                ephemeral=True
            )
            return
    
        cleaned_name = clean_nickname(user.display_name)
        current_xp = await bot.db.get_user_xp(user.id)
        
        # Additional safety check
        if current_xp > 100000:  # Extreme value check
            await interaction.response.send_message(
                "❌ User has unusually high XP. Contact admin.",
                ephemeral=True
            )
            return
        
        success, new_total = await bot.db.add_xp(user.id, cleaned_name, xp)
        
        if success:
            await interaction.response.send_message(
                f"✅ Added {xp} XP to {cleaned_name}. New total: {new_total} XP"
            )
            # Log the XP change
            await log_xp_to_discord(interaction.user, user, xp, new_total, "Manual Addition")
             
        else:
            await interaction.response.send_message(
                "❌ Failed to add XP. Notify admin.",
                ephemeral=True
            )


# /take-xp Command
@bot.tree.command(name="take-xp", description="Takes XP from user")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.CSM_ROLE_ID)
async def take_xp(interaction: discord.Interaction, user: discord.User, xp: int):
    async with global_rate_limiter:
        if xp <= 0:
            await interaction.response.send_message(
                "❌ XP amount must be positive. Use /addxp to give XP.",
                ephemeral=True
            )
            return
        if xp > MAX_XP_PER_ACTION:
            await interaction.response.send_message(
                f"❌ Cannot remove more than {MAX_XP_PER_ACTION} XP at once.",
                ephemeral=True
            )
            return
    
        cleaned_name = clean_nickname(user.display_name)
        current_xp = await bot.db.get_user_xp(user.id)
        
        if xp > current_xp:
            await interaction.response.send_message(
                f"❌ User only has {current_xp} XP. Cannot take {xp}.",
                ephemeral=True
            )
            return
        
        success, new_total = await bot.db.remove_xp(user.id, xp)
        
        if success:
            message = f"✅ Removed {xp} XP from {cleaned_name}. New total: {new_total} XP"
            if new_total == 0:
                message += "\n⚠️ User's XP has reached 0"
            await interaction.response.send_message(message)
            # Log the XP change
            await log_xp_to_discord(interaction.user, user, -xp, new_total, "Manual Removal")
            
        else:
            await interaction.response.send_message(
                "❌ Failed to take XP. Notify admin.",
                ephemeral=True
            )


# /xp Command
@bot.tree.command(name="xp", description="Check your XP or someone else's XP")
@app_commands.describe(user="The user to look up (leave empty to view your own XP)")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.RMP_ROLE_ID)
async def xp_command(interaction: discord.Interaction, user: Optional[discord.User] = None):
    try:
        await interaction.response.defer(ephemeral=True)
        target_user = user or interaction.user
        cleaned_name = clean_nickname(target_user.display_name)

        xp = await bot.db.get_user_xp(target_user.id)
        tier, current_threshold, next_threshold = get_tier_info(xp)  # Use fixed function

        result = await asyncio.to_thread(
            lambda: bot.db.supabase.table('users')
            .select("user_id", "xp")
            .order("xp", desc=True)
            .execute()
        )

        position = next(
            (idx for idx, entry in enumerate(result.data, 1)
             if str(entry['user_id']) == str(target_user.id)),
            None
        )

        progress = make_progress_bar(xp, current_threshold, next_threshold)

        embed = discord.Embed(
            title=f"📊 XP Profile: {cleaned_name}",
            color=discord.Color.green()
        ).set_thumbnail(url=target_user.display_avatar.url)

        embed.add_field(name="Current XP", value=f"```{xp}```", inline=True)
        embed.add_field(name="Tier", value=f"```{tier}```", inline=True)

        if position:
            embed.add_field(name="Leaderboard Position", value=f"```#{position}```", inline=True)

        embed.add_field(name="Progression", value=progress, inline=False)

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"XP command error: {str(e)}")
        await interaction.followup.send("❌ Failed to fetch XP data.", ephemeral=True)

async def handle_command_error(interaction: discord.Interaction, error: Exception):
    """Centralized error handling for commands"""
    try:
        if isinstance(error, discord.NotFound):
            await interaction.followup.send(
                "⚠️ Operation timed out. Please try again.",
                ephemeral=True
            )
        else:
            await interaction.followup.send(
                "❌ An error occurred. Please try again later.",
                ephemeral=True
            )
    except:
        if interaction.channel:
            await interaction.channel.send(
                f"{interaction.user.mention} ❌ Command failed. Please try again.",
                delete_after=10
            )

# Leadebaord Command
@bot.tree.command(name="leaderboard", description="View the top 15 users by XP")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.RMP_ROLE_ID)  
async def leaderboard(interaction: discord.Interaction):
    try:
        async with global_rate_limiter:
            # Rate limiting for Supabase
            await bot.rate_limiter.wait_if_needed(bucket="supabase_query")
            
            # Fetch top 15 users from Supabase
            result = bot.db.supabase.table('users') \
                .select("user_id", "username", "xp") \
                .order("xp", desc=True) \
                .limit(15) \
                .execute()
            
            data = result.data
            
            if not data:
                await interaction.response.send_message("❌ No leaderboard data available.", ephemeral=True)
                return
            
            leaderboard_lines = []
            
            for idx, entry in enumerate(data, start=1):
                try:
                    user_id = int(entry['user_id'])
                    user = interaction.guild.get_member(user_id) or await bot.fetch_user(user_id)
                    display_name = clean_nickname(user.display_name) if user else entry.get('username', f"Unknown ({user_id})")
                    xp = entry['xp']
                    leaderboard_lines.append(f"**#{idx}** - {display_name}: `{xp} XP`")
                except Exception as e:
                    logger.error(f"Error processing leaderboard entry {idx}: {str(e)}")
                    continue
            
            embed = discord.Embed(
                title="🏆 XP Leaderboard (Top 15)",
                description="\n".join(leaderboard_lines) or "No data available",
                color=discord.Color.gold()
            )
            
            embed.set_footer(text=f"Requested by {interaction.user.display_name}")
            
            await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in /leaderboard command: {str(e)}")
        await interaction.response.send_message(
            "❌ Failed to fetch leaderboard data. Please try again later.",
            ephemeral=True
        )


# Give Event XP Command
@bot.tree.command(name="give-event-xp", description="Give XP to attendees mentioned in an event log message")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.CSM_ROLE_ID)
async def give_event_xp(
    interaction: discord.Interaction,
    message_link: str,
    xp_amount: int,
    attendees_section: Literal["Attendees:", "Passed:"] = "Attendees:"
):
    async with global_rate_limiter:
        # Rate Limit
        await bot.rate_limiter.wait_if_needed(bucket="global_xp_update")
        # Validate XP amount first
        if xp_amount <= 0:
            await interaction.response.send_message(
                "❌ XP amount must be positive.",
                ephemeral=True
            )
            return
        if xp_amount > MAX_EVENT_XP_PER_USER:
            await interaction.response.send_message(
                f"❌ Cannot give more than {MAX_EVENT_XP_PER_USER} XP per user in events.",
                ephemeral=True
            )
            return
    
        # Defer the response immediately to prevent timeout
        await interaction.response.defer()
        initial_message = await interaction.followup.send("⏳ Attempting to give XP...", wait=True)
        
        try:
            # Add timeout for the entire operation
            async with asyncio.timeout(60):  # 60 second timeout for entire operation
                # Rate limiting check with timeout
                try:
                    await asyncio.wait_for(
                        bot.rate_limiter.wait_if_needed(bucket=f"give_xp_{interaction.user.id}"),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    await initial_message.edit(content="⌛ Rate limit check timed out. Please try again.")
                    return
    
                # Parse and validate message link
                if not message_link.startswith('https://discord.com/channels/'):
                    await initial_message.edit(content="❌ Invalid message link format")
                    return
                    
                try:
                    parts = message_link.split('/')
                    guild_id = int(parts[4])
                    channel_id = int(parts[5])
                    message_id = int(parts[6])
                except (IndexError, ValueError):
                    await initial_message.edit(content="❌ Invalid message link format")
                    return
                
                if guild_id != interaction.guild.id:
                    await initial_message.edit(content="❌ Message must be from this server")
                    return
                    
                # Fetch the message with timeout
                try:
                    channel = interaction.guild.get_channel(channel_id)
                    if not channel:
                        await initial_message.edit(content="❌ Channel not found")
                        return
                        
                    try:
                        message = await asyncio.wait_for(
                            channel.fetch_message(message_id),
                            timeout=10.0
                        )
                    except discord.NotFound:
                        await initial_message.edit(content="❌ Message not found")
                        return
                    except discord.Forbidden:
                        await initial_message.edit(content="❌ No permission to read that channel")
                        return
                except asyncio.TimeoutError:
                    await initial_message.edit(content="⌛ Timed out fetching message")
                    return
                    
                # Process attendees section
                content = message.content
                section_index = content.find(attendees_section)
                if section_index == -1:
                    await initial_message.edit(content=f"❌ Could not find '{attendees_section}' in the message")
                    return
                    
                mentions_section = content[section_index + len(attendees_section):]
                mentions = re.findall(r'<@!?(\d+)>', mentions_section)
                
                if not mentions:
                    await initial_message.edit(content=f"❌ No user mentions found after '{attendees_section}'")
                    return
                    
                # Process users with progress updates
                unique_mentions = list(set(mentions))
                total_potential_xp = xp_amount * len(unique_mentions)
                
                if total_potential_xp > MAX_EVENT_TOTAL_XP:
                    await initial_message.edit(
                        content=f"❌ Event would give {total_potential_xp} XP total (max is {MAX_EVENT_TOTAL_XP}). Reduce XP or attendees."
                    )
                    return
                    
                await initial_message.edit(content=f"🎯 Processing XP for {len(unique_mentions)} users...")
                
                success_count = 0
                failed_users = []
                processed_users = 0
                
                for i, user_id in enumerate(unique_mentions, 1):
                    try:
                        # Update progress every 5 users
                        if i % 5 == 0 or i == len(unique_mentions):
                            await initial_message.edit(
                                content=f"⏳ Processing {i}/{len(unique_mentions)} users ({success_count} successful)..."
                            )
                        
                        # Rate limit between users
                        if i > 1:
                            await asyncio.sleep(0.75)  # Slightly longer delay
                            
                        member = interaction.guild.get_member(int(user_id))
                        if not member:
                            try:
                                member = await interaction.guild.fetch_member(int(user_id))
                            except discord.NotFound:
                                failed_users.append(f"User {user_id} (not in guild)")
                                continue
                            
                        try:
                            current_xp = await asyncio.wait_for(
                                bot.db.get_user_xp(member.id),
                                timeout=5.0
                            )
                            if current_xp + xp_amount > 100000:
                                failed_users.append(f"{clean_nickname(member.display_name)} (would exceed max XP)")
                                continue
                                
                            success, new_total = await asyncio.wait_for(
                                bot.db.add_xp(member.id, member.display_name, xp_amount),
                                timeout=5.0
                            )
                            
                            if success:
                                success_count += 1
                                await interaction.followup.send(
                                    f"✨ **{clean_nickname(interaction.user.display_name)}** gave {xp_amount} XP to {member.mention} (New total: {new_total} XP)",
                                    silent=True
                                )
                                await log_xp_to_discord(interaction.user, member, xp_amount, new_total, f"Event: {message.jump_url}")
            
                            else:
                                failed_users.append(clean_nickname(member.display_name))
                                
                        except asyncio.TimeoutError:
                            failed_users.append(f"{clean_nickname(member.display_name)} (timeout)")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error processing user {user_id}: {str(e)}")
                        failed_users.append(f"User {user_id} (error)")
                        continue
                        
                # Final summary
                result_message = [
                    f"✅ **XP Distribution Complete**",
                    f"**Given by:** {interaction.user.mention}",
                    f"**XP per user:** {xp_amount}",
                    f"**Successful distributions:** {success_count}",
                    f"**Total XP given:** {xp_amount * success_count}"
                ]
                
                if failed_users:
                    result_message.append(f"\n**Failed distributions:** {len(failed_users)}")
                    for chunk in [failed_users[i:i + 10] for i in range(0, len(failed_users), 10)]:
                        await interaction.followup.send("• " + "\n• ".join(chunk), ephemeral=True)
                
                await interaction.followup.send("\n".join(result_message))
                
        except asyncio.TimeoutError:
            await initial_message.edit(content="⌛ Command timed out. Some XP may have been awarded.")
        except Exception as e:
            logger.error(f"Error in give_event_xp: {str(e)}", exc_info=True)
            await initial_message.edit(content="❌ An unexpected error occurred. Please check logs.")

# Edit database command
@bot.tree.command(name="edit-db", description="Edit a specific user's record in the HR or LR table.")
@has_allowed_role()
async def edit_db(interaction: discord.Interaction, user: discord.User, column: str, value: str):
    await interaction.response.defer(ephemeral=True)
    guild = interaction.guild
    if not guild:
        await interaction.followup.send("❌ This command can only be used in a server.")
        return
        
    member = guild.get_member(user.id)
    if not member:
        try:
            member = await guild.fetch_member(user.id)
        except discord.NotFound:
            await interaction.followup.send(f"❌ {user.mention} not found in this server.")
            return
            
    hr_role = guild.get_role(Config.HR_ROLE_ID)
    is_hr = hr_role and hr_role in member.roles
    table = "HRs" if is_hr else "LRs"
    user_id = str(user.id)

    # Define available columns based on role
    hr_columns = ["tryouts", "events", "phases", "courses", "inspections", "joint_events"]
    lr_columns = ["activity", "time_guarded", "events_attended"]
    
    # Validate column based on role
    available_columns = hr_columns if is_hr else lr_columns
    if column not in available_columns:
        await interaction.followup.send(
            f"❌ Invalid column `{column}` for {table} table. "
            f"Available columns for {'HRs' if is_hr else 'LRs'}: {', '.join(available_columns)}"
        )
        return

    def _work():
        sup = bot.db.supabase
        res = sup.table(table).select("*").eq("user_id", user_id).execute()
        return res

    try:
        res = await bot.db.run_query(_work)
        if not res.data:
            await interaction.followup.send(f"❌ No record found for {user.mention} in `{table}` table.")
            return
        if len(res.data) > 1:
            await interaction.followup.send(f"❌ Multiple records found for {user.mention} in `{table}` table.")
            return

        old_value = res.data[0].get(column, "N/A")
        try:
            value_converted = int(value)
        except ValueError:
            try:
                value_converted = float(value)
            except ValueError:
                value_converted = value

        def _update_work():
            return bot.db.supabase.table(table).update({column: value_converted}).eq("user_id", user_id).execute()

        await bot.db.run_query(_update_work)
        await interaction.followup.send(
            f"✅ Updated `{column}` for {user.mention} from `{old_value}` to `{value_converted}`."
        )
    except Exception as e:
        logger.exception("edit_db failed: %s", e)
        await interaction.followup.send(f"❌ Failed to update data: `{e}`")


# Add autocomplete for the column parameter
@edit_db.autocomplete('column')
async def edit_db_column_autocomplete(interaction: discord.Interaction, current: str):
    # Get the user parameter from the current interaction
    user_option = next((opt for opt in interaction.data.get('options', []) if opt['name'] == 'user'), None)
    
    if not user_option:
        return []
    
    user_id = int(user_option['value'])
    guild = interaction.guild
    
    if not guild:
        return []
    
    # Get the member to check their role
    member = guild.get_member(user_id)
    if not member:
        try:
            member = await guild.fetch_member(user_id)
        except discord.NotFound:
            return []
    
    hr_role = guild.get_role(Config.HR_ROLE_ID)
    is_hr = hr_role and hr_role in member.roles
    
    # Define columns based on role
    hr_columns = ["tryouts", "events", "phases", "courses", "inspections", "joint_events"]
    lr_columns = ["activity", "time_guarded", "events_attended"]
    
    available_columns = hr_columns if is_hr else lr_columns
    
    # Filter based on current input
    choices = [column for column in available_columns if current.lower() in column.lower()]
    
    return [discord.app_commands.Choice(name=column, value=column) for column in choices[:25]]


# Reset Database Command
@bot.tree.command(name="reset-db", description="Reset the LR and HR tables.")
async def reset_db(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    guild = interaction.guild
    if not guild:
        await interaction.followup.send("❌ This command can only be used in a server.")
        return

    ld_hicom_roles = [
        guild.get_role(Config.HIGH_COMMAND_ROLE_ID)
    ]
    
    if not any(role in interaction.user.roles for role in ld_hicom_roles if role):
        await interaction.followup.send("❌ You don’t have permission to use this command.", ephemeral=True)
        return



    def _reset_work():
        sup = bot.db.supabase
        sup.table('HRs').update({
            'tryouts': 0, 'events': 0, 'phases': 0,
            'courses': 0, 'inspections': 0, 'joint_events': 0
        }).neq('user_id', 0).execute()
        sup.table('LRs').update({
            'activity': 0, 'time_guarded': 0, 'events_attended': 0
        }).neq('user_id', 0).execute()
        return True

    try:
        await bot.db.run_query(_reset_work)
        await interaction.followup.send("✅ Database reset successfully!", ephemeral=True)
    except Exception as e:
        logger.exception("reset_db failed: %s", e)
        await interaction.followup.send(f"❌ Error resetting database: {e}", ephemeral=True)



# Manual Fallback Log Command (covers all ReactionLogger cases)
@bot.tree.command(name="force-log", description="Force log a message by link (reaction logger fallback)")
@has_allowed_role()
async def force_log(interaction: discord.Interaction, message_link: str):
    """Force log the exact message from a link - acts as fallback for reaction logger"""
    await interaction.response.defer(ephemeral=True)

    try:
        # Parse the link
        match = re.match(r"https://discord\.com/channels/(\d+)/(\d+)/(\d+)", message_link)
        if not match:
            await interaction.followup.send("❌ Invalid message link.", ephemeral=True)
            return

        guild_id, channel_id, message_id = map(int, match.groups())
        
        if guild_id != interaction.guild.id:
            await interaction.followup.send("❌ Message must be from this server.", ephemeral=True)
            return

        channel = interaction.guild.get_channel(channel_id)
        if not channel:
            await interaction.followup.send("❌ Channel not found.", ephemeral=True)
            return

        # Fetch the specific message
        message = await channel.fetch_message(message_id)
        
        logger.info(f"🔧 Force-log: Processing message {message_id} in #{channel.name} by {interaction.user.display_name}")

        processed = False
        results = []

        # 1. EVENT LOGS (event channels)
        if channel_id in bot.reaction_logger.event_channel_ids:
            try:
                # Extract host
                host_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
                host_id = int(host_mention.group(1)) if host_mention else message.author.id
                host_member = interaction.guild.get_member(host_id) or await interaction.guild.fetch_member(host_id)

                # Determine event type
                hr_update_field = "events"
                if channel_id == Config.W_EVENT_LOG_CHANNEL_ID:
                    if re.search(r'\bjoint\b', message.content, re.IGNORECASE):
                        hr_update_field = "joint_events"
                    elif re.search(r'\b(inspection|pi)\b', message.content, re.IGNORECASE):
                        hr_update_field = "inspections"

                # Update HR table for host
                await bot.reaction_logger._update_hr_record(host_member, {hr_update_field: 1})
                
                # Process attendees
                attendees_section = re.search(r'(?:Attendees:|Passed:)\s*((?:<@!?\d+>\s*)+)', message.content, re.IGNORECASE)
                success_count = 0
                hr_attendees = []
                
                if attendees_section:
                    attendee_mentions = re.findall(r'<@!?(\d+)>', attendees_section.group(1))
                    hr_role = interaction.guild.get_role(Config.HR_ROLE_ID)
                    
                    for attendee_id in attendee_mentions:
                        attendee_member = interaction.guild.get_member(int(attendee_id))
                        if not attendee_member:
                            continue
                        if hr_role and hr_role in attendee_member.roles:
                            hr_attendees.append(attendee_id)
                            continue
                        await bot.reaction_logger._update_lr_record(attendee_member, {"events_attended": 1})
                        success_count += 1

                
                results.append(f"✅ Event: {success_count} attendees + host logged")
                processed = True
                logger.info(f"✅ Event force-logged: {success_count} attendees")
                
            except Exception as e:
                logger.error(f"Event logging failed: {e}")
                results.append(f"❌ Event: {str(e)[:50]}")

        # 2. TRAINING LOGS (phases, tryouts, courses)
        elif channel_id in [
            Config.PHASE_LOG_CHANNEL_ID,
            Config.TRYOUT_LOG_CHANNEL_ID, 
            Config.COURSE_LOG_CHANNEL_ID,
        ]:
            try:
                user_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
                user_id = int(user_mention.group(1)) if user_mention else message.author.id
                user_member = interaction.guild.get_member(user_id) or await interaction.guild.fetch_member(user_id)

                column_to_update = {
                    Config.PHASE_LOG_CHANNEL_ID: "phases",
                    Config.TRYOUT_LOG_CHANNEL_ID: "tryouts",
                    Config.COURSE_LOG_CHANNEL_ID: "courses",
                }.get(channel_id)

                if column_to_update:
                    await bot.reaction_logger._update_hr_record(user_member, {column_to_update: 1})
                    
                    results.append(f"✅ {column_to_update.title()} logged")
                    
                    processed = True
                    
                    logger.info(f"✅ {column_to_update} force-logged")
                    
            except Exception as e:
                logger.error(f"Training logging failed: {e}")
                results.append(f"❌ Training: {str(e)[:50]}")

    
        # 3. ACTIVITY LOGS
        elif channel_id == Config.ACTIVITY_LOG_CHANNEL_ID:
            try:
                user_mention = re.search(r'<@!?(\d+)>', message.content)
                user_id = int(user_mention.group(1)) if user_mention else message.author.id
                user_member = interaction.guild.get_member(user_id) or await interaction.guild.fetch_member(user_id)
        
                updates = {}
                time_match = re.search(r'Time:\s*(\d+)', message.content)
                if time_match:
                    minutes = int(time_match.group(1))
                    if "Guarded:" in message.content:
                        updates["time_guarded"] = minutes
                    else:
                        updates["activity"] = minutes
        
                if updates:
                    # Update LR record
                    await bot.reaction_logger._update_lr_record(user_member, updates)
        
                    # 🟩 NEW: Award XP (1 XP per 30 mins total)
                    total_minutes = updates.get("activity", 0) + updates.get("time_guarded", 0)
                    xp_to_award = total_minutes // 30
                    xp_text = ""
                    if xp_to_award > 0:
                        success, new_xp = await bot.db.add_xp(
                            str(user_member.id),
                            user_member.display_name,
                            xp_to_award
                        )
                        if success:
                            xp_text = f" (+{xp_to_award} XP)"
                            logger.info(f"⭐ Gave {xp_to_award} XP to {user_member.display_name} ({user_member.id}) for {total_minutes} mins (force-log)")
        
        
                    # Finish up and include XP info in results
                    field_name = "time_guarded" if "time_guarded" in updates else "activity"
                    results.append(f"✅ Activity: {updates[field_name]} mins logged{xp_text}")
                    processed = True
                    logger.info(f"✅ Activity force-logged: {updates[field_name]} mins{xp_text}")
        
            except Exception as e:
                logger.error(f"Activity logging failed: {e}")
                results.append(f"❌ Activity: {str(e)[:50]}")


        elif channel_id in bot.reaction_logger.monitor_channel_ids:
            try:
                processed = True
                logger.info("✅ Activity force-logged")
                
            except Exception as e:
                logger.error(f"Force activity logging failed: {e}")
                results.append(f"❌ DB_LOGGER: {str(e)[:50]}")

        # Send results
        if processed:
            result_text = "\n".join(results)
            await interaction.followup.send(
                f"✅ **Force-log completed**\n"
                f"**Message:** [Jump to message]({message.jump_url})\n"
                f"**Channel:** #{channel.name}\n"
                f"**Results:**\n{result_text}",
                ephemeral=True
            )
            
            # Also log to the main log channel
            log_channel = interaction.guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
            if log_channel:
                embed = discord.Embed(
                    title="🔄 Manual Log Entry (Force-log)",
                    color=discord.Color.orange(),
                    timestamp=datetime.now(timezone.utc)
                )
                embed.add_field(name="Logged by", value=interaction.user.mention, inline=True)
                embed.add_field(name="Channel", value=channel.mention, inline=True)
                embed.add_field(name="Message Type", value=channel.name, inline=True)
                embed.add_field(name="Results", value=result_text, inline=False)
                embed.add_field(name="Message", value=f"[Jump to message]({message.jump_url})", inline=False)
                
                await log_channel.send(embed=embed)
                
        else:
            await interaction.followup.send(
                f"❌ No applicable log types found for this message.\n"
                f"**Channel:** #{channel.name}\n"
                f"**Supported channels:** Events, Training, Activity logs, or DB_LOGGER-monitored channels",
                ephemeral=True
            )

    except Exception as e:
        logger.error(f"force-log failed: {e}", exc_info=True)
        await interaction.followup.send("❌ Failed to force-log message.", ephemeral=True)





# Discharge Command
@bot.tree.command(name="discharge", description="Notify members of honourable/dishonourable discharge and log it")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def discharge(
    interaction: discord.Interaction,
    members: str,  # Comma-separated user mentions/IDs
    reason: str,
    discharge_type: Literal["Honourable", "Dishonourable"] = "Honourable",
    evidence: Optional[discord.Attachment] = None
):
    view = ConfirmView()
    await interaction.response.send_message("Confirm discharge?", view=view, ephemeral=True)
    await view.wait()
    if not view.value:
        return

    try:
        # Input Sanitization 
        reason = escape_markdown(reason) 
        # Reason character limit check
        if len(reason) > 1000:
            await interaction.followup.send(
                "❌ Reason must be under 1000 characters",
                ephemeral=True
            )
            return

        member_ids = []
        for mention in members.split(','):
            mention = mention.strip()
            try:
                if mention.startswith('<@') and mention.endswith('>'):
                    member_id = int(mention[2:-1].replace('!', ''))  # Handle nicknames
                else:
                    member_id = int(mention)
                member_ids.append(member_id)
            except ValueError:
                logger.warning(f"Invalid member identifier: {mention}")

        processing_msg = await interaction.followup.send(
            "⚙️ Processing discharge...",
            ephemeral=True,
            wait=True
        )

        await bot.rate_limiter.wait_if_needed(bucket="discharge")

        discharged_members = []
        for member_id in member_ids:
            if member := interaction.guild.get_member(member_id):
                discharged_members.append(member)
            else:
                logger.warning(f"Member {member_id} not found in guild")

        if not discharged_members:
            await interaction.followup.send("❌ No valid members found.", ephemeral=True)
            return

        # Embed creation
        color = discord.Color.green() if discharge_type == "Honourable" else discord.Color.red()
        embed = discord.Embed(
            title=f"{discharge_type} Discharge Notification",
            color=color,
            timestamp=datetime.now(timezone.utc)
        )
        embed.add_field(name="Reason", value=reason, inline=False)

        if evidence:
            embed.add_field(name="Evidence", value=f"[Attachment Link]({evidence.url})", inline=False)
            mime, _ = mimetypes.guess_type(evidence.filename)
            if mime and mime.startswith("image/"):
                embed.set_image(url=evidence.url)

        embed.set_footer(text=f"Discharged by {interaction.user.display_name}")

        success_count = 0
        failed_members = []

        for member in discharged_members:
            cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
            try:
                await bot.db.discharge_user(member.id, cleaned_nickname, interaction.guild)

            except Exception as e:
                logger.error(f"Error removing {cleaned_nickname} ({member.id}) from DB: {e}")
            try:
                try:
                    await member.send(embed=embed)
                except discord.Forbidden:
                    if channel := interaction.guild.get_channel(1219410104240050236):
                        await channel.send(f"{member.mention}", embed=embed)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to notify {member.display_name}: {str(e)}")
                failed_members.append(member.mention)

        result_embed = discord.Embed(
            title="Discharge Summary",
            color=color
        )
        result_embed.add_field(name="Action", value=f"{discharge_type} Discharge", inline=False)
        result_embed.add_field(
            name="Results",
            value=f"✅ Successfully notified: {success_count}\n❌ Failed: {len(failed_members)}",
            inline=False
        )
        if failed_members:
            result_embed.add_field(name="Failed Members", value=", ".join(failed_members), inline=False)

        await processing_msg.edit(
            content=None,
            embed=result_embed
        )

        # Log to D_LOG_CHANNEL_ID
        if d_log := interaction.guild.get_channel(Config.D_LOG_CHANNEL_ID):
            log_embed = discord.Embed(
                title=f"Discharge Log",
                color=color,
                timestamp=datetime.now(timezone.utc)
            )
            log_embed.add_field(
                name="Type",
                value=f"🔰 {discharge_type} Discharge" if discharge_type == "Honourable" else f"🚨 {discharge_type} Discharge",
                inline=False
            )
            log_embed.add_field(name="Reason", value=f"```{reason}```", inline=False)
            
            # Format member mentions with their cleaned nicknames
            member_entries = []
            for member in discharged_members:
                # Clean the nickname by removing any tags like [INS]
                cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
                member_entries.append(f"{member.mention} | {cleaned_nickname}")
            
            log_embed.add_field(
                name="Discharged Members",
                value="\n".join(member_entries) or "None",
                inline=False
            )
            
            if evidence:
                log_embed.add_field(name="Evidence", value=f"[View Attachment]({evidence.url})", inline=True)

            log_embed.add_field(name="Discharged By", value=interaction.user.mention, inline=True)
            
            await d_log.send(embed=log_embed)

    except Exception as e:
        logger.error(f"Discharge command failed: {e}")
        await interaction.followup.send("❌ An error occurred while processing the discharge.", ephemeral=True)
    



@bot.tree.command(name="commands", description="List all available commands")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.RMP_ROLE_ID)  
async def command_list(interaction: discord.Interaction):
    embed = discord.Embed(
        title="📜 Available Commands",
        color=discord.Color.blue()
    )
    
    categories = {
        "🛠️ Utility": [
            "/ping - Check bot responsiveness",
            "/commands - Show this help message",
            "/sc - Security Check Roblox user",
            "/discharge - Sends discharge notification to user and logs in discharge logs",
            "/edit-db - Edit a specific user's record in the HR or LR table",
            "/force-log - Force log an event/training/activity manually (fallback if reactions fail)",
            "/report-bug - Report a bug to Crimson"
            
        ],
         "⭐ XP": [
            "/add-xp - Gives xp to user",
            "/take-xp - Takes xp from user",
            "/give-event-xp - Gives xp to attendees/passers in event logs",
            "/xp - Checks amount of xp user has",
            "/leaderboard - View the top 15 users by XP"
        ]
    }
    
    for name, value in categories.items():
        embed.add_field(name=name, value="\n".join(value), inline=False)
    
    await interaction.response.send_message(embed=embed)
    
# Ping Command
@bot.tree.command(name="ping", description="Check bot latency")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@has_allowed_role()
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)
    await interaction.response.send_message(
        f"🏓 Pong! Latency: {latency}ms",
        ephemeral=True
    )

# Report Bug Command
@bot.tree.command(name="report-bug", description="Report a bug to Crimson")
@min_rank_required(Config.RMP_ROLE_ID)  
async def report_bug(interaction: discord.Interaction, description: str):
    """
    Report a bug to the bot developer.
    """
    await interaction.response.defer(ephemeral=True)
    
    DEVELOPER_ID = 353167234698444802
    
    try:
        developer = await bot.fetch_user(DEVELOPER_ID)
        
        embed = discord.Embed(
            title="🐛 New Bug Report",
            color=discord.Color.orange(),
            timestamp=discord.utils.utcnow()
        )
        
        embed.add_field(
            name="Reporter",
            value=f"{interaction.user.mention} ({interaction.user.id})",
            inline=False
        )
        
        if interaction.guild:
            embed.add_field(
                name="Server",
                value=f"{interaction.guild.name} ({interaction.guild.id})",
                inline=False
            )
        
        embed.add_field(
            name="Description",
            value=description,
            inline=False
        )
        
        try:
            await developer.send(embed=embed)
            await interaction.followup.send(
                "✅ Thank you for reporting the bug! Crimson has been notified.",
                ephemeral=True
            )
        except discord.Forbidden:
            await interaction.followup.send(
                "❌ I couldn't send the bug report. Try contacting Crimson (353167234698444802) directly.",
                ephemeral=True
            )
            
    except Exception as e:
        logger.exception("Failed to send bug report: %s", e)
        await interaction.followup.send(
            "❌ An error occurred. Please try again later.",
            ephemeral=True
        )


#Save Roles Command
@bot.tree.command(name="save-roles", description="Save a user's tracked roles to the database.")
@has_allowed_role()
async def save_roles(interaction: discord.Interaction, member: discord.Member):
    # Defer immediately
    await interaction.response.defer(ephemeral=True)

    tracked_ids = Config.TRACKED_ROLE_IDS
    matched_roles = [r for r in member.roles if r.id in tracked_ids]
    role_ids = [r.id for r in matched_roles]

    try:
        success = await bot.db.save_user_roles(
            user_id=str(member.id),
            username=member.display_name,
            role_ids=role_ids
        )
    except Exception as e:
        embed = discord.Embed(
            title="❌ Error Saving Roles",
            description=f"Failed to save roles for {member.mention}.\n{e}",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed, ephemeral=True)
        return

    # Build status embed
    embed = discord.Embed(
        title="✅ Roles Saved" if success else "❌ Error Saving Roles",
        description=(
            f"Saved **{len(role_ids)}** tracked roles for {member.mention}.\n"
            f"**Roles:** {', '.join([r.name for r in matched_roles]) or 'None'}"
        ) if success else f"Failed to save roles for {member.mention}.",
        color=discord.Color.green() if success else discord.Color.red()
    )

    # Log to default channel
    log_channel = interaction.guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
    if log_channel:
        await log_channel.send(embed=embed)

    # Send ephemeral follow-up to the user
    await interaction.followup.send(embed=embed, ephemeral=True)

#Restore Roles Command
@bot.tree.command(name="restore-roles", description="Restore saved roles for a user.")
async def restore_roles(interaction: discord.Interaction, member: discord.Member):
    # Defer the interaction immediately with error handling
    try:
        await interaction.response.defer(ephemeral=True)
        deferred = True
    except discord.NotFound:
        # If deferral fails, the interaction might have timed out
        deferred = False
    except Exception as e:
        print(f"Error deferring interaction: {e}")
        deferred = False

    # Fetch saved roles from Supabase
    try:
        saved_roles = await bot.db.get_user_roles(str(member.id))
    except Exception as e:
        error_msg = f"❌ Failed to fetch saved roles: {e}"
        if deferred:
            await interaction.followup.send(error_msg, ephemeral=True)
        else:
            try:
                await interaction.response.send_message(error_msg, ephemeral=True)
            except discord.InteractionResponded:
                await interaction.followup.send(error_msg, ephemeral=True)
        return

    if not saved_roles:
        msg = f"⚠️ No saved roles found for {member.mention}."
        if deferred:
            await interaction.followup.send(msg, ephemeral=True)
        else:
            try:
                await interaction.response.send_message(msg, ephemeral=True)
            except discord.InteractionResponded:
                await interaction.followup.send(msg, ephemeral=True)
        return

    # Format for Dyno command: ?role <user_id> <role id 1>, <role id 2>, <role id 3>
    roles_string = ", ".join([str(role_id) for role_id in saved_roles])
    dyno_command = f"?role {member.id} {roles_string}"

    # Create embed with Dyno command
    embed = discord.Embed(
        title=f"✅ Roles restored for {member.display_name}",
        description=f"Use the following command in chat to reassign roles:\n`{dyno_command}`",
        color=discord.Color.green()
    )
    
    if deferred:
        await interaction.followup.send(embed=embed, ephemeral=True)
    else:
        try:
            await interaction.response.send_message(embed=embed, ephemeral=True)
        except discord.InteractionResponded:
            await interaction.followup.send(embed=embed, ephemeral=True)



# HR Welcome Message
async def send_hr_welcome(member: discord.Member):
    if not (welcome_channel := member.guild.get_channel(Config.HR_CHAT_CHANNEL_ID)):
        logger.warning("HR welcome channel not found!")
        return

    embed = discord.Embed(
        title="🎉 Welcome to the HR Team!",
        description=(
            f"{member.mention}\n\n"
            "**Please note the following:**\n"
            "• Request for document access in [HR Documents](https://discord.com/channels/1165368311085809717/1165368317532438646).\n"
            "• You are exempted from quota this week only - you start next week ([Quota Info](https://discord.com/channels/1165368311085809717/1206998095552978974)).\n"
            "• Uncomplete quota = strike.\n"
            "• One failed tryout allowed if your try quota portion ≥2.\n"
            "• Ask for help anytime - we're friendly!\n"
            "• Are you Lieutenant+ in BA? Apply for the Education Department!\n"
            "• Are you Captain+ in BA? Apply for both departments: [Applications](https://discord.com/channels/1165368311085809717/1165368316970405916)."
        ),
        color=discord.Color.gold(),  
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.set_footer(text="We're excited to have you on the team!")
    
    try:
        await welcome_channel.send(content=f"{member.mention}", embed=embed)
        logger.info(f"Sent HR welcome to {member.display_name}")
    except discord.Forbidden:
        logger.error(f"Missing permissions to send HR welcome for {member.id}")
    except Exception as e:
        logger.error(f"Failed to send HR welcome: {str(e)}")

# RMP Welcome Message
async def send_rmp_welcome(member: discord.Member):
    # First embed (welcome message)
    embed1 = discord.Embed(
        title=" 👮| Welcome to the Royal Military Police",
        description="Congratulations on passing your security check, you're officially a TRAINING member of the police force. Please be sure to read the information found below.\n\n"
                   "> ** 1.** Make sure to read all of the rules found in <#1165368313925353580> and in the brochure found below.\n\n"
                   "> **2.** You **MUST** read the RMP main guide and MSL before starting your duties.\n\n"
                   "> **3.** You can't use your L85 unless you are doing it for Self-Militia or enforcing the PD rules. (Self-defence)\n\n"
                   "> **4.** Make sure to follow the Chain Of Command. 2nd Lieutenant > Lieutenant > Captain > Major > Lieutenant Colonel > Colonel > Brigadier > Major General\n\n"
                   "> **5.** For phases, you may wait for one to be hosted in <#1207367013698240584> or request the phase you need in <#1270700562433839135>.\n\n"
                   "> **6.** All the information about the Defence School of Policing and Guarding is found in both <#1237062439720452157> and <#1207366893631967262>\n\n"
                   "> **7.** Choose your timezone here https://discord.com/channels/1165368311085809717/1165368313925353578\n\n"
                   "**Besides that, good luck with your phases!**",
        color=discord.Color.from_str("#330000") 
    )

    # Special Role Embed
    special_embed = discord.Embed(
        title="Special Roles",
        description=" > Get your role pings here <#1196085670360404018> and don't forget the Game Night role RMP always hoosts fun events, don't miss out!",
        color=discord.Color.from_str("#330000")
    )
    
    # Second embed (detailed rules)
    embed2 = discord.Embed(
        title="Trainee Constable Brochure",
        color=discord.Color.from_str("#660000")
    )
    
    embed2.add_field(
        name="**TOP 5 RULES**",
        value="> **1**. You **MUST** read the RMP main guide and MSL before starting your duties.\n"
              "> **2**. You **CANNOT** enforce the MSL. Only the Parade Deck (PD) rules **AFTER** you pass your phase 1.\n"
              "> **3**. You **CANNOT** use your bike on the PD or the pavements.\n"
              "> **4**. You **MUST** use good spelling and grammar to the best of your ability.\n"
              "> **5**. You **MUST** remain mature and respectful at all times.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED ON THE PD AT ALL TIMES?__",
        value="> ↠ Royal Army Medical Corps,\n"
              "> ↠ Royal Military Police,\n" 
              "> ↠ Intelligence Corps.\n"
              "> ↠ Royal Family.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED ON THE PD WHEN CARRYING OUT THEIR DUTIES?**",
        value="> ↠ United Kingdom Special Forces,\n"
              "> ↠ Grenadier Guards,\n"
              "> ↠ Foreign Relations,\n"
              "> ↠ Royal Logistic Corps,\n"
              "> ↠ Adjutant General's Corps,\n"
              "> ↠ High Ranks, RSM, CSM and ASM hosting,\n"
              "> ↠ Regimental personnel watching one of their regiment's events inside Pad area.",
        inline=False
    )
    
    embed2.add_field(
        name="**HOW DO I ENFORCE PD RULES ON PEOPLE NOT ALLOWED ON IT?**",
        value="> 1. Give them their first warning to get off the PD, \"W1, off the PD!\"\n"
              "> 2. Wait 3-5 seconds for them to listen; if they don't, give them their second warning, \"W2, off the PD!\"\n"
              "> 3. Wait 3-5 seconds for them to listen; if they don't kill them.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED __ON__ THE ACTUAL STAGE AT ALL TIMES**",
        value="> ↠ Major General and above,\n"
              "> ↠ Royal Family (they should have a purple name tag),\n"
              "> ↠ Those who have been given permission by a Lieutenant General.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED TO PASS THE RED LINE IN-FRONT OF THE STAGE?**",
        value="> ↠ Major General and above,\n"
              "> ↠ Royal Family,\n"
              "> ↠ Those who have been given permission by a Lieutenant General,\n"
              "> ↠ COMBATIVE Home Command Regiments:\n"
              "> - Royal Military Police,\n"
              "> - United Kingdom Forces,\n"
              "> - Household Division.\n"
              "> **Kill those not allowed who touch or pass the red line.**",
        inline=False
    )
    
    embed2.add_field(
        name="\u200b",  
        value="**LASTLY, IF YOU'RE UNSURE ABOUT SOMETHING, ASK SOMEONE USING THE CHAIN OF COMMAND BEFORE TAKING ACTION!**",
        inline=False
    )

    try:
        await member.send(embeds=[embed1, special_embed, embed2])
    except discord.Forbidden:
        if welcome_channel := member.guild.get_channel(722002957738180620):
            await welcome_channel.send(f"{member.mention}", embeds=[embed1, special_embed, embed2])
            logger.info(f"Sending welcome message to {member.display_name} ({member.id})")
    except discord.HTTPException as e:
        logger.error(f"Failed to send welcome message: {e}")

@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    async with global_rate_limiter:
        # Check for HR role
        if hr_role := after.guild.get_role(Config.HR_ROLE_ID):
            if hr_role not in before.roles and hr_role in after.roles:
                await send_hr_welcome(after)
        
        # Check for RMP role
        if rmp_role := after.guild.get_role(Config.RMP_ROLE_ID):
            if rmp_role not in before.roles and rmp_role in after.roles:
                await send_rmp_welcome(after)

@bot.event
async def on_member_remove(member: discord.Member):
    async with global_rate_limiter:
        guild = member.guild
        if not (deserter_role := guild.get_role(Config.RMP_ROLE_ID)):
            return
    
        if deserter_role not in member.roles:
            return
            
        if not (alert_channel := guild.get_channel(Config.HR_CHAT_CHANNEL_ID)):
            return

        # Clean nickname for logs + DB
        cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name

        # 🔑 Remove deserter from all DB tables
        try:
            await bot.db.discharge_user(member.id, cleaned_nickname, interaction.guild)

        except Exception as e:
            logger.error(f"Error removing deserter {cleaned_nickname} ({member.id}) from DB: {e}")

        # 🚨 Alert embed
        embed = discord.Embed(
            title="🚨 Deserter Alert",
            description=f"{member.mention} just deserted! Please log this in BA.",
            color=discord.Color.red()
        )
        embed.set_thumbnail(url=member.display_avatar.url)
        
        await alert_channel.send(
            content=f"<@&{Config.HIGH_COMMAND_ROLE_ID}>",
            embed=embed
        )

        # Discharge log embed
        dishonourable_embed = discord.Embed(
            title="Discharge Log",
            color=discord.Color.red(),
            timestamp=datetime.now(timezone.utc)
        )
        dishonourable_embed.add_field(name="Type", value="🚨 Dishonourable Discharge", inline=False)
        dishonourable_embed.add_field(name="Reason", value="```Desertion.```", inline=False)
        dishonourable_embed.add_field(name="Discharged Members", value=f"{member.mention} | {cleaned_nickname}", inline=False)
        dishonourable_embed.add_field(name="Discharged By", value="None - Unauthorised", inline=True)
        dishonourable_embed.set_footer(text="Desertion Monitor System")

        # Blacklist log embed
        current_date = datetime.now(timezone.utc)    
        hr_role = guild.get_role(Config.HR_ROLE_ID)  
        if hr_role and hr_role in member.roles:
            ending_date = current_date + timedelta(days=180)  
        else:
            ending_date = current_date + timedelta(days=90)   
        blacklist_embed = discord.Embed(
            title="⛔ Blacklist",
            color=discord.Color.red(),
            timestamp=current_date
        )
        blacklist_embed.add_field(name="Issuer:", value="MP Assistant", inline=False)
        blacklist_embed.add_field(name="Name:", value=cleaned_nickname, inline=False)
        blacklist_embed.add_field(name="Starting date", value=f"<t:{int(current_date.timestamp())}:D>", inline=False)
        blacklist_embed.add_field(name="Ending date", value=f"<t:{int(ending_date.timestamp())}:D>", inline=False)
        blacklist_embed.add_field(name="Reason:", value="Desertion.", inline=False)
        blacklist_embed.set_footer(text="Desertion Monitor System")

        try:
            d_log = bot.get_channel(Config.D_LOG_CHANNEL_ID)
            b_log = bot.get_channel(Config.B_LOG_CHANNEL_ID)
            
            if d_log and b_log:
                await d_log.send(embed=dishonourable_embed)
                await b_log.send(embed=blacklist_embed)
                logger.info(f"Logged deserted member, {cleaned_nickname}.")
            else:
                alt_channel = bot.get_channel(1165368316970405917)
                if alt_channel:
                    await alt_channel.send(f"⚠️ Failed to log deserter discharge for {cleaned_nickname}")
                    logger.error(f"Failed to log deserted member {cleaned_nickname} - main channel not found")
        except Exception as e:
            logger.error(f"Error logging deserter discharge: {str(e)}")



@bot.event
async def on_socket_raw_receive(msg):
    # Log WebSocket activity to detect issues
    pass

@bot.event
async def on_socket_raw_send(payload):
    # Log WebSocket activity to detect issues
    pass

# Command Listener
@bot.event
async def on_interaction(interaction: discord.Interaction):
    # Check if this is a command completion
    if interaction.command is not None and interaction.type == discord.InteractionType.application_command:

        if interaction.user.id == 353167234698444802:
            return
            
        guild = interaction.guild
        if not guild:
            return  # Skip DMs

        log_channel = guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
        if not log_channel:
            return

        user = interaction.user
        command = interaction.command
        logger.info(f"⚙️ Command executed: /{command.name} by {user.display_name} ({user.id})")

        embed = discord.Embed(
            title="⚙️ Command Executed",
            description=f"**/{command.qualified_name}**",
            color=discord.Color.blurple(),
            timestamp=discord.utils.utcnow()
        )
        embed.add_field(name="User", value=f"{user.mention} (`{user.id}`)", inline=False)
        embed.add_field(name="Channel", value=interaction.channel.mention, inline=False)

        # Add arguments if present
        if interaction.data and "options" in interaction.data:
            args = ", ".join(
                f"`{opt['name']}`: {opt.get('value', 'N/A')}"
                for opt in interaction.data["options"]
            )
            embed.add_field(name="Arguments", value=args, inline=False)

        await log_channel.send(embed=embed)

async def run_bot():
    while True:
        try:
            await bot.start(TOKEN)
        except discord.errors.HTTPException as e:
            if e.status == 429:
                retry_after = e.response.headers.get('Retry-After', 30)
                logger.warning(f"Rate limited during login. Waiting {retry_after} seconds...")
                await asyncio.sleep(float(retry_after))
                continue
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
        else:
            break

async def run_bot_forever():
    backoff = 1.0
    max_backoff = 300.0
    consecutive_failures = 0
    
    while True:
        try:
            logger.info(f"Starting Discord client (attempt {consecutive_failures + 1})...")
            await bot.start(TOKEN)
            
            # Reset backoff on successful run
            backoff = 1.0
            consecutive_failures = 0
            
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received; shutting down bot.")
            try:
                await bot.close()
            finally:
                break
                
        except discord.errors.HTTPException as e:
            consecutive_failures += 1
            if getattr(e, "status", None) == 429:
                retry_after = float(e.response.headers.get('Retry-After', 30) if getattr(e, "response", None) else 30)
                logger.warning(f"Rate limited during login. Waiting {retry_after}s.")
                await asyncio.sleep(retry_after)
                continue
            logger.error(f"HTTPException in run loop: {e}")
            
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Unexpected error in Discord client: {type(e).__name__}: {e}")

        # Exponential backoff with jitter
        sleep_time = min(max_backoff, backoff * (2 ** consecutive_failures))
        sleep_time *= random.uniform(0.8, 1.2)  # Add jitter
        
        logger.info(f"Discord client stopped; restarting in {sleep_time:.1f}s")
        await asyncio.sleep(sleep_time)
        backoff = sleep_time

if __name__ == '__main__':  
    try:
        asyncio.run(run_bot_forever())
    except Exception as e:
        logger.critical(f"Fatal error running bot: {e}", exc_info=True)
        raise








































































































































