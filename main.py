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

    async def add_to_hr(self, user_id: str, username: str, guild: discord.Guild) -> bool:
        """Add or update a user in the HRs table."""
        if not self.supabase:
            return False
    
        def _work():
            try:
                self.supabase.table("HRs").upsert({
                    "user_id": str(user_id),
                    "username": clean_nickname(username),
                    "guild_id": str(guild.id)
                }).execute()
                return True
            except Exception as e:
                logger.error(f"Failed to add {user_id} to HRs: {e}")
                return False
    
        return await self._run_sync(_work)

    async def remove_from_lr(self, user_id: str) -> bool:
        """Remove a user from the LRs table if they exist."""
        if not self.supabase:
            return False
    
        def _work():
            try:
                self.supabase.table("LRs").delete().eq("user_id", str(user_id)).execute()
                return True
            except Exception as e:
                logger.error(f"Failed to remove {user_id} from LRs: {e}")
                return False
    
        return await self._run_sync(_work)

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
            
    async def get_welcome_message(self, message_type: str):
        """Get welcome message (uses cache)"""
        return await welcome_cache.get(message_type)

    async def update_welcome_message(self, message_type: str, embeds_data: list, updated_by: str):
        """Update welcome message (updates both cache and database)"""
        return await welcome_cache.update(message_type, embeds_data, updated_by)
    
    async def get_welcome_message_history(self, message_type: str, limit: int = 5):
        """Get historical versions of welcome messages"""
        def _work():
            result = bot.db.supabase.table('welcome_messages') \
                .select('*') \
                .eq('message_type', message_type) \
                .order('last_updated', desc=True) \
                .limit(limit) \
                .execute()
            return result.data
        
        try:
            return await self._run_sync(_work)
        except Exception as e:
            logger.error(f"Error getting history for {message_type}: {e}")
            return []


async def create_or_update_user_in_db(
    discord_id: str,
    username: str,
    guild: discord.Guild
) -> bool:
    """
    Create or update a user in the 'users' table with Roblox ID.
    Returns: True if successful, False if failed
    """
    try:
        # Clean the username
        cleaned_username = clean_nickname(username)
        
        # Get Roblox ID
        roblox_id = await get_roblox_id_from_username(cleaned_username)
        
        if not roblox_id:
            logger.warning(f"Could not find Roblox ID for {cleaned_username} ({discord_id})")
            # Still create the row but with null Roblox ID
            roblox_id = None
        
        def _work():
            sup = bot.db.supabase
            # Check if user already exists
            res = sup.table('users').select('*').eq('user_id', discord_id).execute()
            
            if getattr(res, 'data', None) and len(res.data) > 0:
                # Update existing user
                update_data = {
                    "username": cleaned_username,
                    "roblox_id": roblox_id
                }
                return sup.table('users').update(update_data).eq('user_id', discord_id).execute()
            else:
                # Create new user with default XP = 0
                new_user = {
                    "user_id": discord_id,
                    "username": cleaned_username,
                    "roblox_id": roblox_id,
                    "xp": 0  # Default XP
                }
                return sup.table('users').insert(new_user).execute()
        
        result = await bot.db.run_query(_work)
        
        if roblox_id:
            logger.info(f"✅ Created/updated user {cleaned_username} ({discord_id}) with Roblox ID: {roblox_id}")
        else:
            logger.info(f"✅ Created/updated user {cleaned_username} ({discord_id}) without Roblox ID")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create/update user {discord_id} in database: {e}")
        return False

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



# Welcome Message Cache & Functions
class WelcomeMessageCache:
    """In-memory cache for welcome messages with multiple embed support"""
    
    def __init__(self):
        self._cache = {}  # message_type -> {"embeds": [], "metadata": {}}
        self._lock = asyncio.Lock()
        self._initialized = False
        
    async def initialize(self):
        """Load all welcome messages from database on startup"""
        if self._initialized:
            return
            
        async with self._lock:
            try:
                for msg_type in ['hr_welcome', 'rmp_welcome']:
                    data = await self._load_from_database(msg_type)
                    if data:
                        self._cache[msg_type] = data
                
                logger.info(f"✅ Loaded {len(self._cache)} welcome messages into cache")
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize welcome message cache: {e}")
                # Initialize with empty cache, will use original functions as fallback
    
    async def _load_from_database(self, message_type: str):
        """Load a specific message type from database"""
        def _work():
            result = bot.db.supabase.table('welcome_messages') \
                .select('*') \
                .eq('message_type', message_type) \
                .eq('is_active', True) \
                .limit(1) \
                .execute()
            
            if result.data and len(result.data) > 0:
                return {
                    'id': result.data[0]['id'],
                    'message_type': result.data[0]['message_type'],
                    'embeds': result.data[0]['embeds'],
                    'last_updated': result.data[0]['last_updated'],
                    'updated_by': result.data[0]['updated_by'],
                    'version': result.data[0].get('version', 1)
                }
            return None
        
        try:
            return await bot.db._run_sync(_work)
        except Exception as e:
            logger.error(f"Error loading {message_type} from database: {e}")
            return None
    
    async def get(self, message_type: str, refresh: bool = False):
        """
        Get a welcome message from cache.
        If refresh=True or not in cache, load from database.
        """
        async with self._lock:
            if refresh or message_type not in self._cache:
                data = await self._load_from_database(message_type)
                if data:
                    self._cache[message_type] = data
            
            return self._cache.get(message_type)
    
    async def update(self, message_type: str, embeds_data: list, updated_by: str):
        """
        Update welcome message with multiple embeds.
        """
        async with self._lock:
            try:
                # 1. Mark old message as inactive
                def _deactivate_old():
                    bot.db.supabase.table('welcome_messages') \
                        .update({'is_active': False}) \
                        .eq('message_type', message_type) \
                        .eq('is_active', True) \
                        .execute()
                
                await bot.db._run_sync(_deactivate_old)
                
                # 2. Get next version number
                def _get_next_version():
                    result = bot.db.supabase.table('welcome_messages') \
                        .select('version') \
                        .eq('message_type', message_type) \
                        .order('version', desc=True) \
                        .limit(1) \
                        .execute()
                    
                    if result.data and len(result.data) > 0:
                        return result.data[0]['version'] + 1
                    return 1
                
                next_version = await bot.db._run_sync(_get_next_version)
                
                # 3. Insert new active message
                def _insert_new():
                    new_message = {
                        'message_type': message_type,
                        'version': next_version,
                        'embeds': embeds_data,
                        'updated_by': updated_by,
                        'is_active': True
                    }
                    
                    result = bot.db.supabase.table('welcome_messages') \
                        .insert(new_message) \
                        .execute()
                    
                    return result.data[0] if result.data else None
                
                new_message = await bot.db._run_sync(_insert_new)
                
                if new_message:
                    # 4. Update cache
                    self._cache[message_type] = {
                        'id': new_message['id'],
                        'message_type': new_message['message_type'],
                        'embeds': new_message['embeds'],
                        'last_updated': new_message['last_updated'],
                        'updated_by': new_message['updated_by'],
                        'version': new_message['version']
                    }
                    
                    logger.info(f"✅ Updated {message_type} welcome message with {len(embeds_data)} embeds")
                    return True, self._cache[message_type]
                else:
                    logger.error(f"Failed to insert new {message_type} message")
                    return False, None
                    
            except Exception as e:
                logger.error(f"Error updating welcome message {message_type}: {e}")
                return False, None

# Create global instance
welcome_cache = WelcomeMessageCache()


def dict_to_embed(data: dict) -> discord.Embed:
    """Convert dictionary to Discord Embed object"""
    embed = discord.Embed(
        title=data.get('title', ''),
        description=data.get('description', ''),
        color=discord.Color.from_str(data.get('color', '#000000'))
    )
    
    if 'footer' in data:
        embed.set_footer(text=data['footer'])
    
    if 'thumbnail' in data:
        embed.set_thumbnail(url=data['thumbnail'])
    
    if 'image' in data:
        embed.set_image(url=data['image'])
    
    if 'fields' in data:
        for field in data['fields']:
            embed.add_field(
                name=field.get('name', ''),
                value=field.get('value', ''),
                inline=field.get('inline', False)
            )
    
    return embed

# Ranks and Division Tracker
class RankTracker:
    """Tracks and updates member ranks and divisions in the database"""
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(__name__)
        self.startup_completed = False  # Track if startup update is done
        
        # Combine all rank mappings for quick lookup
        self.all_ranks = {
            **Config.SOR_HR_RANKS,
            **Config.PW_HR_RANKS,
            **Config.SOR_LR_RANKS,
            **Config.PW_LR_RANKS
        }
        
        # Caching for performance
        self.member_cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
    def _get_cache_key(self, member_id: int) -> str:
        """Generate cache key for a member"""
        return f"member_{member_id}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.member_cache:
            return False
        data, timestamp = self.member_cache[cache_key]
        return time.time() - timestamp < self.cache_ttl
    
    
    async def get_member_info(self, member: discord.Member, force_refresh: bool = False) -> dict:
        """Get comprehensive member information including rank and division"""
        cache_key = self._get_cache_key(member.id)
        
        if not force_refresh and self._is_cache_valid(cache_key):
            return self.member_cache[cache_key][0]
        
        # Determine division
        division = await self._determine_division(member)
        
        # Determine rank
        rank_name = await self._determine_rank(member, division)
        
        # Check if HR or LR
        hr_role = member.guild.get_role(Config.HR_ROLE_ID)
        is_hr = hr_role and hr_role in member.roles
        
        # Check if trainee
        trainee_role = member.guild.get_role(Config.TRAINEE_ROLE_ID)
        is_trainee = trainee_role and trainee_role in member.roles
        
        # Check if RSM
        rsm_role = member.guild.get_role(Config.RSM_ROLE_ID)
        is_rsm = rsm_role and rsm_role in member.roles
        
        # Check if HQ
        hq_role = member.guild.get_role(Config.HQ_ROLE_ID)
        is_hq = hq_role and hq_role in member.roles
        
        member_info = {
            'member_id': member.id,
            'username': clean_nickname(member.display_name),
            'division': division,
            'rank': rank_name,
            'is_hr': is_hr,
            'is_lr': not is_hr,
            'is_trainee': is_trainee,
            'is_rsm': is_rsm,
            'is_hq': is_hq,
            'roles': [role.id for role in member.roles],
            'timestamp': time.time()
        }
        
        # Cache the result
        self.member_cache[cache_key] = (member_info, time.time())
        
        return member_info
    
    async def _determine_division(self, member: discord.Member) -> str:
        """Determine member's division (HQ, SOR, PW, or Unknown)"""
        # Check for HQ (Provost Marshal)
        hq_role = member.guild.get_role(Config.HQ_ROLE_ID)
        if hq_role and hq_role in member.roles:
            return "HQ"
        
        # Check for SOR role
        sor_role = member.guild.get_role(Config.SOR_ROLE_ID)
        has_sor = sor_role and sor_role in member.roles
        
        # Check for PW roles (any PW rank indicates PW division)
        has_pw = False
        for pw_role_id in Config.PW_HR_RANKS.keys() | Config.PW_LR_RANKS.keys():
            if role := member.guild.get_role(pw_role_id):
                if role in member.roles:
                    has_pw = True
                    break
        
        # Check HQ eligibility
        hq_role = member.guild.get_role(Config.HQ_ROLE_ID)
        rsm_role = member.guild.get_role(Config.RSM_ROLE_ID)
        
        is_hq = (
            (hq_role and hq_role in member.roles) or
            (rsm_role and rsm_role in member.roles)
        )
        
        # Determine division based on roles
        if is_hq:
            return "HQ"
        elif has_sor:
            return "SOR"
        elif has_pw:
            return "PW"
        else:
            return "Unknown"

    
    async def _determine_rank(self, member: discord.Member, division: str) -> str:
        """Determine member's rank name based on roles"""
        # Check for specific ranks in order of hierarchy
        member_role_ids = [role.id for role in member.roles]
        
        # Define rank priority (highest to lowest)
        rank_priority = []
        
        if division == "SOR":
            # SOR HR ranks (highest to lowest)
            for role_id in [
                1368777853235101702,  # SOR Commander
                1368777611001462936,   # SOR Executive
                1368780792842424511,   # Squadron Commander
                1368777380344102912,   # Squadron Executive Officer
                1368777213444624489,   # Tactical Officer
                1368777046003552298,   # Operations Officer
                1368776765270396978    # Junior Operations Officer
            ]:
                if role_id in member_role_ids:
                    return Config.SOR_HR_RANKS.get(role_id, "Unknown SOR HR Rank")
            
            # SOR LR ranks (highest to lowest)
            for role_id in [
                1368776612878876723,   # Operations Sergeant Major
                1368776341289304165,   # Tactical Leader
                1368776344787484802,   # Field Specialist
                1368776092969730149,   # Senior Operator
                1368775864141086770    # Operator
            ]:
                if role_id in member_role_ids:
                    return Config.SOR_LR_RANKS.get(role_id, "Unknown SOR LR Rank")
        
        elif division == "PW":
            # PW HR ranks (highest to lowest)
            for role_id in [
                1165368311840784515,   # PW Commander
                1165368311840784514,   # PW Executive
                1165368311840784512,   # Lieutenant Colonel
                1165368311840784511,   # Major
                1165368311840784510,   # Superintendent
                1309231446258356405,   # Chief Inspector
                1309231448569680078    # Inspector
            ]:
                if role_id in member_role_ids:
                    return Config.PW_HR_RANKS.get(role_id, "Unknown PW HR Rank")
            
            # PW LR ranks (highest to lowest)
            for role_id in [
                1309231451321139200,   # Company Sergeant Major
                1165368311777869933,   # Staff Sergeant
                1165368311777869932,   # Sergeant
                1165368311777869931,   # Senior Constable
                1165368311777869930    # Constable
            ]:
                if role_id in member_role_ids:
                    return Config.PW_LR_RANKS.get(role_id, "Unknown PW LR Rank")
        
        elif division == "HQ":
            # Check for Provost Marshal
            hq_role = member.guild.get_role(Config.HQ_ROLE_ID)
            if hq_role and hq_role in member.roles:
                return "Provost Marshal"
            
            # HQ members might have other ranks too
            # Check all ranks in priority order
            for role_id in [
                # HQ specific
                1165368311874326650,   # Provost Marshal
                # Highest SOR
                1368777853235101702,   # SOR Commander
                1368777611001462936,   # SOR Executive
                # Highest PW
                1165368311840784515,   # PW Commander
                1165368311840784514,   # PW Executive
                # Continue with other ranks...
            ]:
                if role_id in member_role_ids:
                    return self.all_ranks.get(role_id, "Unknown Rank")
        
        # Check for RSM
        rsm_role = member.guild.get_role(Config.RSM_ROLE_ID)
        if rsm_role and rsm_role in member.roles:
            return "Regimental Sergeant Major"
        
        # Check for trainee
        trainee_role = member.guild.get_role(Config.TRAINEE_ROLE_ID)
        if trainee_role and trainee_role in member.roles:
            return "Trainee Constable"
        
        return "Unranked"
    
    async def update_member_in_database(self, member: discord.Member) -> bool:
        """Update member's rank and division in the database"""
        try:
            member_info = await self.get_member_info(member, force_refresh=True)
            
            # Determine which table to update
            table = "HRs" if member_info['is_hr'] else "LRs"
            
            def _update_work():
                sup = self.bot.db.supabase
                
                # Check if record exists
                res = sup.table(table).select('*').eq('user_id', str(member.id)).execute()
                
                update_data = {
                    'username': member_info['username'],
                    'division': member_info['division'],
                    'rank': member_info['rank']
                }
                
                if getattr(res, 'data', None) and len(res.data) > 0:
                    # Update existing record
                    return sup.table(table).update(update_data).eq('user_id', str(member.id)).execute()
                else:
                    # Insert new record
                    return sup.table(table).insert({
                        'user_id': str(member.id),
                        **update_data
                    }).execute()
            
            await self.bot.db.run_query(_update_work)
            self.logger.info(f"Updated {table} for {member_info['username']}: Division={member_info['division']}, Rank={member_info['rank']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update member {member.id} in database: {e}")
            return False
    
    async def update_all_members(self, guild: discord.Guild, batch_size: int = 50) -> dict:
        """Update rank and division for all members in the guild"""
        results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'updated_hr': 0,
            'updated_lr': 0,
            'errors': []
        }
        
        members = [member for member in guild.members if not member.bot]
        results['total'] = len(members)
        
        self.logger.info(f"Starting mass update for {len(members)} members")
        
        # Process in batches to avoid rate limits
        for i in range(0, len(members), batch_size):
            batch = members[i:i + batch_size]
            
            for member in batch:
                try:
                    success = await self.update_member_in_database(member)
                    
                    if success:
                        results['success'] += 1
                        member_info = await self.get_member_info(member)
                        if member_info['is_hr']:
                            results['updated_hr'] += 1
                        else:
                            results['updated_lr'] += 1
                    else:
                        results['failed'] += 1
                        
                    # Rate limiting delay
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"{member.display_name}: {str(e)}")
                    self.logger.error(f"Error updating {member.display_name}: {e}")
            
            self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(members) + batch_size - 1)//batch_size}")
            
            # Wait between batches
            if i + batch_size < len(members):
                await asyncio.sleep(1)
        
        return results
    
    async def check_member_consistency(self, member: discord.Member) -> dict:
        """Check if database record matches current member roles"""
        try:
            current_info = await self.get_member_info(member, force_refresh=True)
            table = "HRs" if current_info['is_hr'] else "LRs"
            
            def _check_work():
                sup = self.bot.db.supabase
                res = sup.table(table).select('division, rank').eq('user_id', str(member.id)).execute()
                
                if getattr(res, 'data', None) and len(res.data) > 0:
                    db_data = res.data[0]
                    return {
                        'db_division': db_data.get('division'),
                        'db_rank': db_data.get('rank'),
                        'current_division': current_info['division'],
                        'current_rank': current_info['rank'],
                        'needs_update': (
                            db_data.get('division') != current_info['division'] or
                            db_data.get('rank') != current_info['rank']
                        )
                    }
                return None
            
            result = await self.bot.db.run_query(_check_work)
            return result or {'error': 'No database record found'}
            
        except Exception as e:
            self.logger.error(f"Consistency check failed for {member.id}: {e}")
            return {'error': str(e)}
    
    async def get_division_stats(self, guild: discord.Guild) -> dict:
        """Get statistics about divisions and ranks"""
        try:
            def _get_stats_work():
                sup = self.bot.db.supabase
                
                # Get HR stats
                hr_res = sup.table('HRs').select('division, rank').execute()
                hr_data = hr_res.data if getattr(hr_res, 'data', None) else []
                
                # Get LR stats
                lr_res = sup.table('LRs').select('division, rank').execute()
                lr_data = lr_res.data if getattr(lr_res, 'data', None) else []
                
                return {
                    'hr': hr_data,
                    'lr': lr_data
                }
            
            raw_data = await self.bot.db.run_query(_get_stats_work)
            
            # Process statistics
            stats = {
                'total_members': len(raw_data['hr']) + len(raw_data['lr']),
                'divisions': {
                    'HQ': {'hr': 0, 'lr': 0, 'total': 0},
                    'SOR': {'hr': 0, 'lr': 0, 'total': 0},
                    'PW': {'hr': 0, 'lr': 0, 'total': 0},
                    'Unknown': {'hr': 0, 'lr': 0, 'total': 0}
                },
                'ranks': {}
            }
            
            # Process HR data
            for record in raw_data['hr']:
                division = record.get('division', 'Unknown')
                rank = record.get('rank', 'Unknown')
                
                if division in stats['divisions']:
                    stats['divisions'][division]['hr'] += 1
                    stats['divisions'][division]['total'] += 1
                
                stats['ranks'][rank] = stats['ranks'].get(rank, 0) + 1
            
            # Process LR data
            for record in raw_data['lr']:
                division = record.get('division', 'Unknown')
                rank = record.get('rank', 'Unknown')
                
                if division in stats['divisions']:
                    stats['divisions'][division]['lr'] += 1
                    stats['divisions'][division]['total'] += 1
                
                stats['ranks'][rank] = stats['ranks'].get(rank, 0) + 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get division stats: {e}")
            return {'error': str(e)}
    
    def _is_rank_related_role(self, role_id: int) -> bool:
        """Check if a role ID is related to rank/division determination"""
        # Check all rank-related role IDs
        return role_id in (
            # Division roles
            Config.SOR_ROLE_ID,
            Config.HQ_ROLE_ID,
            Config.RSM_ROLE_ID,
            Config.TRAINEE_ROLE_ID,
            # All rank roles
            *Config.SOR_HR_RANKS.keys(),
            *Config.PW_HR_RANKS.keys(),
            *Config.SOR_LR_RANKS.keys(),
            *Config.PW_LR_RANKS.keys(),
            # HR/LR role (for determining which table)
            Config.HR_ROLE_ID,
            Config.RMP_ROLE_ID  # If you want to track RMP role changes
        )
    
    def _roles_changed_affect_rank(self, before_roles: list, after_roles: list) -> bool:
        """Check if role changes affect rank/division determination"""
        before_set = set(before_roles)
        after_set = set(after_roles)
        
        # Get roles that were added or removed
        added_roles = after_set - before_set
        removed_roles = before_set - after_set
        
        # Check if any added/removed roles are rank-related
        for role_id in added_roles.union(removed_roles):
            if self._is_rank_related_role(role_id):
                return True
        
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


async def get_roblox_id_from_username(username: str) -> Optional[int]:
    """
    Get Roblox ID from username using the Roblox API.
    Returns: Roblox ID (int) or None if not found
    """
    try:
        url = "https://users.roblox.com/v1/usernames/users"
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': USER_AGENT
        }
        
        payload = {
            "usernames": [username],
            "excludeBannedUsers": False
        }
        
        async with bot.shared_session.post(url, json=payload, headers=headers, timeout=15) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("data") and len(data["data"]) > 0:
                    roblox_id = data["data"][0].get("id")
                    if roblox_id:
                        return int(roblox_id)
            
            # If we get here, user not found or API error
            logger.warning(f"Roblox API returned: {response.status} for username: {username}")
            if response.status == 429:
                logger.warning("⚠ Rate limited! Adding delay...")
                await asyncio.sleep(5)
            return None
            
    except asyncio.TimeoutError:
        logger.warning(f"Timeout connecting to Roblox API for username: {username}")
        return None
    except Exception as e:
        logger.error(f"API Error fetching Roblox ID for {username}: {str(e)[:100]}")
        return None


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
        self.tc_supervision_log_channel_id = Config.TC_SUPERVISION_CHANNEL_ID
        

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
            
            status = {
                "add_listeners": len(add_listeners),
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
            embed.add_field(name="Logger", value=f"{log_author.mention}", inline=False)
            embed.add_field(name="Amount of Checks", value=security_checks, inline=False)
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
                
                embed.add_field(name="Inductor", value=f"{member.mention}", inline=False)
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
    
        if str(payload.emoji) in Config.IGNORED_EMOJI:
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

            try:
                await asyncio.wait_for(
                    log_channel.send(embed=embed),
                    timeout=5
                )
            except asyncio.TimeoutError: 
                logger.error("log_channel.send() timed out")
                return

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
            self.tc_supervision_log_channel_id: "phases",
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
                if payload.channel_id == self.tc_supervision_log_channel_id:
                    embed.add_field(name="Supervisor", value=user_member.mention)
                else:
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
    def __init__(self, author: discord.User, *, timeout: float = 30.0):
        super().__init__(timeout=timeout)
        self.author = author
        self.value: bool | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author.id:
            await interaction.response.send_message(
                "❌ You cannot respond to this confirmation.",
                ephemeral=True
            )
            return False
        return True

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
bot.rank_tracker = RankTracker(bot)

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

    # Initialize welcome message cache
    try:
        await welcome_cache.initialize()
        logger.info("✅ Welcome message cache initialized")
    except Exception as e:
        logger.error(f"Failed to initialize welcome cache: {e}")

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

    # Initialize rank tracker
    bot.rank_tracker = RankTracker(bot)
    logger.info("✅ Rank tracker initialized")
    



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
        
        # Add fresh listeners
        bot.add_listener(bot.reaction_logger.on_raw_reaction_add, "on_raw_reaction_add")
        
        logger.info("🔄 Re-attached ReactionLogger event listener after resume")
    except Exception as e:
        logger.error(f"❌ Failed to reattach ReactionLogger listener: {e}")

    # Restart channel + cleanup setups
    try:
        await asyncio.sleep(0.5)
        await bot.reaction_logger.on_ready_setup()

        # Reinitialize reaction logger's rate limiter
        bot.reaction_logger.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)

        logger.info("✅ Restored ReactionLogger after resume.")
    except Exception as e:
        logger.error(f"❌ Failed to restore reaction logger: {e}")



        
# --- BOT COMMANDS --- 
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
            title=f"{cleaned_name}",
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
async def edit_db_column_autocomplete(
    interaction: discord.Interaction,
    current: str
):
    user_option = next(
        (opt for opt in interaction.data.get('options', []) if opt['name'] == 'user'),
        None
    )
    if not user_option or not interaction.guild:
        return []

    user_id = int(user_option['value'])
    guild = interaction.guild

    member = guild.get_member(user_id)
    if not member:
        try:
            member = await guild.fetch_member(user_id)
        except discord.NotFound:
            return []

    hr_role = guild.get_role(Config.HR_ROLE_ID)
    is_hr = hr_role and hr_role in member.roles

    # Display name -> DB column
    hr_columns = {
        "Tryouts": "tryouts",
        "Events": "events",
        "Phases": "phases",
        "Logistics": "courses",       
        "Inspections": "inspections",
        "Joint Events": "joint_events",
    }

    lr_columns = {
        "Activity": "activity",
        "Time Guarded": "time_guarded",
        "Events Attended": "events_attended",
    }

    available = hr_columns if is_hr else lr_columns

    return [
        discord.app_commands.Choice(name=display, value=db_value)
        for display, db_value in available.items()
        if current.lower() in display.lower()
    ][:25]


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


# Reset Database Command
@bot.tree.command(name="reset-db", description="Reset the LR and HR tables.")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def reset_db(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    guild = interaction.guild
    if not guild:
        await interaction.followup.send(
            "❌ This command can only be used in a server.",
            ephemeral=True
        )
        return

    view = ConfirmView(author=interaction.user)

    await interaction.followup.send(
        "⚠️ **Are you sure you want to reset the database?**\n"
        "This will reset **ALL LR and HR stats**.\n\n"
        "Click **Confirm** to proceed or **Cancel** to abort.",
        view=view,
        ephemeral=True
    )

    # Wait for user input (or timeout)
    await view.wait()

    if view.value is not True:
        await interaction.followup.send(
            "❎ Database reset cancelled.",
            ephemeral=True
        )
        return

    def _reset_work():
        sup = bot.db.supabase
        sup.table('HRs').update({
            'tryouts': 0,
            'events': 0,
            'phases': 0,
            'courses': 0,
            'inspections': 0,
            'joint_events': 0
        }).neq('user_id', 0).execute()

        sup.table('LRs').update({
            'activity': 0,
            'time_guarded': 0,
            'events_attended': 0
        }).neq('user_id', 0).execute()

        return True

    try:
        await bot.db.run_query(_reset_work)
        await interaction.followup.send(
            "✅ **Database reset successfully!**",
            ephemeral=True
        )
    except Exception as e:
        logger.exception("reset_db failed: %s", e)
        await interaction.followup.send(
            f"❌ Error resetting database:\n```{e}```",
            ephemeral=True
        )




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


#Edit-Welcome Messages Commands
@bot.tree.command(name="edit-welcome", description="Edit welcome messages for HR or new RMP members")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def edit_welcome(
    interaction: discord.Interaction,
    message_type: Literal["HR Welcome", "RMP Welcome"],
    action: Literal["View Full", "Edit Title", "Edit Description", "Edit Color", "Add Field", "Remove Field", "Add Embed", "Remove Embed", "Reset to Default"] = "View Full"
):
    """Main command for editing welcome messages"""
    
    # Map display name to database type
    type_mapping = {
        "HR Welcome": "hr_welcome",
        "RMP Welcome": "rmp_welcome"
    }
    db_type = type_mapping.get(message_type)
    
    if not db_type:
        await interaction.response.send_message("❌ Invalid message type.", ephemeral=True)
        return
    
    # Get current message
    message_data = await bot.db.get_welcome_message(db_type)
    if not message_data or 'embeds' not in message_data:
        await interaction.response.send_message(
            f"❌ No {message_type} found in database.", 
            ephemeral=True
        )
        return
    
    embeds_data = message_data['embeds']
    
    # Handle different actions
    if action == "View Full":
        await _view_full_welcome(interaction, db_type, message_type, embeds_data)
    
    elif action == "Edit Title":
        await _edit_title_menu(interaction, db_type, message_type, embeds_data)
    
    elif action == "Edit Description":
        await _edit_description_menu(interaction, db_type, message_type, embeds_data)
    
    elif action == "Edit Color":
        await _edit_color_menu(interaction, db_type, message_type, embeds_data)
    
    elif action == "Add Field":
        await _add_field_menu(interaction, db_type, message_type, embeds_data)
    
    elif action == "Remove Field":
        await _remove_field_menu(interaction, db_type, message_type, embeds_data)
    
    elif action == "Add Embed":
        await _add_embed_menu(interaction, db_type, message_type, embeds_data)
    
    elif action == "Remove Embed":
        await _remove_embed_menu(interaction, db_type, message_type, embeds_data)
    
    elif action == "Reset to Default":
        await _reset_to_default(interaction, db_type, message_type)

async def _view_full_welcome(interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
    """View the full welcome message with all embeds"""
    if not embeds_data:
        await interaction.response.send_message(
            f"❌ {display_name} has no embeds.", 
            ephemeral=True
        )
        return
    
    # Defer since we might send multiple messages
    await interaction.response.defer(ephemeral=True)
    
    # Create Discord embed objects
    discord_embeds = []
    for embed_data in embeds_data:
        discord_embeds.append(dict_to_embed(embed_data))
    
    # Send all embeds (Discord allows up to 10 per message)
    for i, chunk in enumerate([discord_embeds[j:j+10] for j in range(0, len(discord_embeds), 10)]):
        if i == 0:
            content = f"**{display_name} - Full Preview**\nTotal embeds: {len(discord_embeds)}"
        else:
            content = f"**{display_name} - Continued (Part {i+1})**"
        
        await interaction.followup.send(
            content=content,
            embeds=chunk,
            ephemeral=True
        )

async def _edit_title_menu(interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
    """Menu to select which embed title to edit - WORKING VERSION"""
    
    class TitleSelectView(discord.ui.View):
        def __init__(self, db_type, display_name, embeds_data):
            super().__init__(timeout=60)
            self.db_type = db_type
            self.display_name = display_name
            self.embeds_data = embeds_data
        
        @discord.ui.select(
            placeholder="Select embed to edit title...",
            options=[
                discord.SelectOption(
                    label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                    value=str(i),
                    description=embed.get('description', '')[:100] or "No description"
                )
                for i, embed in enumerate(embeds_data)
            ]
        )
        async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
            embed_index = int(select.values[0])
            current_title = self.embeds_data[embed_index].get('title', '')
            
            # Modal is defined INSIDE the callback to capture embed_index
            class TitleModal(discord.ui.Modal, title=f"Edit Title - Embed {embed_index + 1}"):
                def __init__(self, embed_index, embeds_data, db_type):
                    super().__init__()
                    self.embed_index = embed_index
                    self.embeds_data = embeds_data
                    self.db_type = db_type
                    
                    self.title_input = discord.ui.TextInput(
                        label="New Title",
                        default=current_title,
                        max_length=256,
                        required=True
                    )
                    self.add_item(self.title_input)
                
                async def on_submit(self, modal_interaction: discord.Interaction):
                    await modal_interaction.response.defer(ephemeral=True)
                    
                    new_title = self.title_input.value
                    updated_embeds = self.embeds_data.copy()
                    old_title = updated_embeds[self.embed_index].get('title', 'Untitled')
                    updated_embeds[self.embed_index]['title'] = new_title
                    
                    # SIMPLIFIED CALL - remove admin_user and change_details
                    success, new_data = await bot.db.update_welcome_message(
                        self.db_type,
                        updated_embeds,
                        f"{modal_interaction.user.name} ({modal_interaction.user.id})"
                    )
                    
                    if success:
                        await modal_interaction.followup.send(
                            f"✅ Updated title of Embed {self.embed_index + 1} to: **{new_title}**",
                            ephemeral=True
                        )
                    else:
                        await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
            
            # Create and show the modal
            modal = TitleModal(embed_index, self.embeds_data, self.db_type)
            await select_interaction.response.send_modal(modal)
    
    view = TitleSelectView(db_type, display_name, embeds_data)
    await interaction.response.send_message(
        f"Select which embed of **{display_name}** to edit title:",
        view=view,
        ephemeral=True
    )

async def _edit_description_menu(interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
    """Menu to select which embed description to edit"""
    
    class DescriptionSelectView(discord.ui.View):
        def __init__(self, db_type, display_name, embeds_data):
            super().__init__(timeout=60)
            self.db_type = db_type
            self.display_name = display_name
            self.embeds_data = embeds_data
        
        @discord.ui.select(
            placeholder="Select embed to edit description...",
            options=[
                discord.SelectOption(
                    label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                    value=str(i),
                    description="Has description" if embed.get('description') else "No description"
                )
                for i, embed in enumerate(embeds_data)
            ]
        )
        async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
            embed_index = int(select.values[0])
            current_desc = self.embeds_data[embed_index].get('description', '')
            
            class DescriptionModal(discord.ui.Modal, title=f"Edit Description - Embed {embed_index + 1}"):
                def __init__(self, embed_index, embeds_data, db_type):
                    super().__init__()
                    self.embed_index = embed_index
                    self.embeds_data = embeds_data
                    self.db_type = db_type
                    
                    self.desc_input = discord.ui.TextInput(
                        label="New Description",
                        default=current_desc,
                        style=discord.TextStyle.paragraph,
                        max_length=4000,
                        required=False
                    )
                    self.add_item(self.desc_input)
                
                async def on_submit(self, modal_interaction: discord.Interaction):
                    await modal_interaction.response.defer(ephemeral=True)
                    
                    new_desc = self.desc_input.value
                    updated_embeds = self.embeds_data.copy()
                    updated_embeds[self.embed_index]['description'] = new_desc
                    
                    success, _ = await bot.db.update_welcome_message(
                        self.db_type,
                        updated_embeds,
                        f"{modal_interaction.user.name} ({modal_interaction.user.id})"
                    )
                    
                    if success:
                        preview = new_desc[:100] + "..." if len(new_desc) > 100 else new_desc
                        await modal_interaction.followup.send(
                            f"✅ Updated description of Embed {self.embed_index + 1}\n**Preview:** {preview}",
                            ephemeral=True
                        )
                    else:
                        await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
            
            modal = DescriptionModal(embed_index, self.embeds_data, self.db_type)
            await select_interaction.response.send_modal(modal)
    
    view = DescriptionSelectView(db_type, display_name, embeds_data)
    await interaction.response.send_message(
        f"Select which embed of **{display_name}** to edit description:",
        view=view,
        ephemeral=True
    )


async def _edit_color_menu(interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
    """Menu to select which embed color to edit"""
    
    class ColorSelectView(discord.ui.View):
        def __init__(self, db_type, display_name, embeds_data):
            super().__init__(timeout=60)
            self.db_type = db_type
            self.display_name = display_name
            self.embeds_data = embeds_data
        
        @discord.ui.select(
            placeholder="Select embed to edit color...",
            options=[
                discord.SelectOption(
                    label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                    value=str(i),
                    description=f"Color: {embed.get('color', '#000000')}"
                )
                for i, embed in enumerate(embeds_data)
            ]
        )
        async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
            embed_index = int(select.values[0])
            current_color = self.embeds_data[embed_index].get('color', '#000000')
            
            class ColorModal(discord.ui.Modal, title=f"Edit Color - Embed {embed_index + 1}"):
                def __init__(self, embed_index, embeds_data, db_type):
                    super().__init__()
                    self.embed_index = embed_index
                    self.embeds_data = embeds_data
                    self.db_type = db_type
                    
                    self.color_input = discord.ui.TextInput(
                        label="New Color (hex format, e.g., #FFD700)",
                        default=current_color,
                        max_length=7,
                        required=True
                    )
                    self.add_item(self.color_input)
                
                async def on_submit(self, modal_interaction: discord.Interaction):
                    await modal_interaction.response.defer(ephemeral=True)
                    
                    new_color = self.color_input.value.upper()
                    
                    # Validate hex color
                    import re
                    if not re.match(r'^#[0-9A-F]{6}$', new_color):
                        await modal_interaction.followup.send(
                            "❌ Invalid color format. Use hex format like #FFD700",
                            ephemeral=True
                        )
                        return
                    
                    updated_embeds = self.embeds_data.copy()
                    updated_embeds[self.embed_index]['color'] = new_color
                    
                    success, _ = await bot.db.update_welcome_message(
                        self.db_type,
                        updated_embeds,
                        f"{modal_interaction.user.name} ({modal_interaction.user.id})"
                    )
                    
                    if success:
                        await modal_interaction.followup.send(
                            f"✅ Updated color of Embed {self.embed_index + 1} to: **{new_color}**",
                            ephemeral=True
                        )
                    else:
                        await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
            
            modal = ColorModal(embed_index, self.embeds_data, self.db_type)
            await select_interaction.response.send_modal(modal)
    
    view = ColorSelectView(db_type, display_name, embeds_data)
    await interaction.response.send_message(
        f"Select which embed of **{display_name}** to edit color:",
        view=view,
        ephemeral=True
    )


async def _add_field_menu(interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
    """Menu to add a field to an embed"""
    
    class AddFieldSelectView(discord.ui.View):
        def __init__(self, db_type, display_name, embeds_data):
            super().__init__(timeout=60)
            self.db_type = db_type
            self.display_name = display_name
            self.embeds_data = embeds_data
        
        @discord.ui.select(
            placeholder="Select embed to add field to...",
            options=[
                discord.SelectOption(
                    label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                    value=str(i),
                    description=f"Fields: {len(embed.get('fields', []))}"
                )
                for i, embed in enumerate(embeds_data)
            ]
        )
        async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
            embed_index = int(select.values[0])
            
            class AddFieldModal(discord.ui.Modal, title=f"Add Field - Embed {embed_index + 1}"):
                def __init__(self, embed_index, embeds_data, db_type):
                    super().__init__()
                    self.embed_index = embed_index
                    self.embeds_data = embeds_data
                    self.db_type = db_type
                    
                    # Create form fields
                    self.field_name = discord.ui.TextInput(
                        label="Field Name",
                        placeholder="Enter field name...",
                        max_length=256,
                        required=True
                    )
                    self.field_value = discord.ui.TextInput(
                        label="Field Value",
                        placeholder="Enter field value...",
                        style=discord.TextStyle.paragraph,
                        max_length=1024,
                        required=True
                    )
                    self.inline_input = discord.ui.TextInput(
                        label="Inline? (true/false)",
                        placeholder="true or false",
                        default="false",
                        max_length=5,
                        required=False
                    )
                    
                    self.add_item(self.field_name)
                    self.add_item(self.field_value)
                    self.add_item(self.inline_input)
                
                async def on_submit(self, modal_interaction: discord.Interaction):
                    await modal_interaction.response.defer(ephemeral=True)
                    
                    # Parse inline boolean
                    inline_bool = self.inline_input.value.lower() == 'true'
                    
                    # Create new field
                    new_field = {
                        'name': self.field_name.value,
                        'value': self.field_value.value,
                        'inline': inline_bool
                    }
                    
                    # Update embeds
                    updated_embeds = self.embeds_data.copy()
                    if 'fields' not in updated_embeds[self.embed_index]:
                        updated_embeds[self.embed_index]['fields'] = []
                    
                    updated_embeds[self.embed_index]['fields'].append(new_field)
                    
                    success, _ = await bot.db.update_welcome_message(
                        self.db_type,
                        updated_embeds,
                        f"{modal_interaction.user.name} ({modal_interaction.user.id})"
                    )
                    
                    if success:
                        await modal_interaction.followup.send(
                            f"✅ Added field **{self.field_name.value}** to Embed {self.embed_index + 1}",
                            ephemeral=True
                        )
                    else:
                        await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
            
            modal = AddFieldModal(embed_index, self.embeds_data, self.db_type)
            await select_interaction.response.send_modal(modal)
    
    view = AddFieldSelectView(db_type, display_name, embeds_data)
    await interaction.response.send_message(
        f"Select which embed of **{display_name}** to add a field to:",
        view=view,
        ephemeral=True
    )


async def _remove_field_menu(interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
    """Menu to remove a field from an embed - SIMPLIFIED VERSION"""
    
    # First, find embeds that have fields
    embeds_with_fields = []
    for i, embed in enumerate(embeds_data):
        if embed.get('fields'):
            embeds_with_fields.append((i, embed))
    
    if not embeds_with_fields:
        await interaction.response.send_message(
            f"❌ No embeds in {display_name} have fields to remove.",
            ephemeral=True
        )
        return
    
    class RemoveFieldSelectView(discord.ui.View):
        def __init__(self, db_type, display_name, embeds_with_fields, embeds_data):
            super().__init__(timeout=60)
            self.db_type = db_type
            self.display_name = display_name
            self.embeds_with_fields = embeds_with_fields
            self.embeds_data = embeds_data
        
        @discord.ui.select(
            placeholder="Select embed with field to remove...",
            options=[
                discord.SelectOption(
                    label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                    value=str(i),
                    description=f"{len(embed.get('fields', []))} fields"
                )
                for i, embed in embeds_with_fields
            ]
        )
        async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
            embed_index = int(select.values[0])
            
            # Show field selection
            fields = self.embeds_data[embed_index].get('fields', [])
            
            class FieldSelectView(discord.ui.View):
                def __init__(self, embed_index, db_type, embeds_data):
                    super().__init__(timeout=60)
                    self.embed_index = embed_index
                    self.db_type = db_type
                    self.embeds_data = embeds_data
                
                @discord.ui.select(
                    placeholder="Select field to remove...",
                    options=[
                        discord.SelectOption(
                            label=f"Field {j+1}: {field.get('name', f'Field {j+1}')[:100]}",
                            value=str(j),
                            description=field.get('value', '')[:50]
                        )
                        for j, field in enumerate(fields)
                    ]
                )
                async def field_callback(self, field_interaction: discord.Interaction, field_select: discord.ui.Select):
                    field_index = int(field_select.values[0])
                    field_name = fields[field_index].get('name', f'Field {field_index + 1}')
                    
                    # Confirm removal
                    confirm_view = ConfirmView(field_interaction.user)
                    await field_interaction.response.send_message(
                        f"⚠️ Remove field **{field_name}** from Embed {embed_index + 1}?",
                        view=confirm_view,
                        ephemeral=True
                    )
                    
                    await confirm_view.wait()
                    
                    if confirm_view.value:
                        updated_embeds = self.embeds_data.copy()
                        updated_embeds[self.embed_index]['fields'].pop(field_index)
                        
                        # Clean up if no fields left
                        if not updated_embeds[self.embed_index]['fields']:
                            updated_embeds[self.embed_index].pop('fields', None)
                        
                        success, _ = await bot.db.update_welcome_message(
                            self.db_type,
                            updated_embeds,
                            f"{field_interaction.user.name} ({field_interaction.user.id})"
                        )
                        
                        if success:
                            await field_interaction.followup.send(
                                f"✅ Removed field **{field_name}**",
                                ephemeral=True
                            )
                        else:
                            await field_interaction.followup.send("❌ Failed to save.", ephemeral=True)
                    else:
                        await field_interaction.followup.send("❌ Cancelled.", ephemeral=True)
            
            field_view = FieldSelectView(embed_index, self.db_type, self.embeds_data)
            await select_interaction.response.send_message(
                f"Select which field to remove from Embed {embed_index + 1}:",
                view=field_view,
                ephemeral=True
            )
    
    view = RemoveFieldSelectView(db_type, display_name, embeds_with_fields, embeds_data)
    await interaction.response.send_message(
        f"Select which embed of **{display_name}** has the field to remove:",
        view=view,
        ephemeral=True
    )

async def _add_embed_menu(interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
    """Add a new embed to the welcome message"""
    
    class AddEmbedModal(discord.ui.Modal, title=f"Add New Embed to {display_name}"):
        def __init__(self, db_type, embeds_data):
            super().__init__()
            self.db_type = db_type
            self.embeds_data = embeds_data
            
            self.title_input = discord.ui.TextInput(
                label="Embed Title",
                placeholder="Enter embed title...",
                max_length=256,
                required=True
            )
            self.desc_input = discord.ui.TextInput(
                label="Embed Description",
                placeholder="Enter embed description...",
                style=discord.TextStyle.paragraph,
                max_length=4000,
                required=False
            )
            self.color_input = discord.ui.TextInput(
                label="Embed Color (hex)",
                placeholder="#3498db",
                default="#3498db",
                max_length=7,
                required=False
            )
            
            self.add_item(self.title_input)
            self.add_item(self.desc_input)
            self.add_item(self.color_input)
        
        async def on_submit(self, modal_interaction: discord.Interaction):
            await modal_interaction.response.defer(ephemeral=True)
            
            # Validate color
            new_color = self.color_input.value.upper()
            import re
            if not re.match(r'^#[0-9A-F]{6}$', new_color):
                new_color = "#3498db"
            
            # Create new embed
            new_embed = {
                'title': self.title_input.value,
                'description': self.desc_input.value,
                'color': new_color
            }
            
            # Add to existing embeds
            updated_embeds = self.embeds_data.copy()
            updated_embeds.append(new_embed)
            
            success, _ = await bot.db.update_welcome_message(
                self.db_type,
                updated_embeds,
                f"{modal_interaction.user.name} ({modal_interaction.user.id})"
            )
            
            if success:
                await modal_interaction.followup.send(
                    f"✅ Added new embed **{self.title_input.value}** to {display_name}\nTotal embeds: {len(updated_embeds)}",
                    ephemeral=True
                )
            else:
                await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
    
    modal = AddEmbedModal(db_type, embeds_data)
    await interaction.response.send_modal(modal)

async def _remove_embed_menu(interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
    """Remove an embed from the welcome message"""
    
    if len(embeds_data) <= 1:
        await interaction.response.send_message(
            f"❌ Cannot remove embed. {display_name} must have at least 1 embed.",
            ephemeral=True
        )
        return
    
    class RemoveEmbedSelectView(discord.ui.View):
        def __init__(self, db_type, display_name, embeds_data):
            super().__init__(timeout=60)
            self.db_type = db_type
            self.display_name = display_name
            self.embeds_data = embeds_data
        
        @discord.ui.select(
            placeholder="Select embed to remove...",
            options=[
                discord.SelectOption(
                    label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                    value=str(i),
                    description="Click to remove"
                )
                for i, embed in enumerate(embeds_data)
            ]
        )
        async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
            embed_index = int(select.values[0])
            embed_title = self.embeds_data[embed_index].get('title', f'Embed {embed_index + 1}')
            
            # Confirm removal
            confirm_view = ConfirmView(select_interaction.user)
            await select_interaction.response.send_message(
                f"⚠️ Remove **{embed_title}** from {self.display_name}?\nThis will leave {len(self.embeds_data)-1} embeds.",
                view=confirm_view,
                ephemeral=True
            )
            
            await confirm_view.wait()
            
            if confirm_view.value:
                updated_embeds = self.embeds_data.copy()
                updated_embeds.pop(embed_index)
                
                success, _ = await bot.db.update_welcome_message(
                    self.db_type,
                    updated_embeds,
                    f"{select_interaction.user.name} ({select_interaction.user.id})"
                )
                
                if success:
                    await select_interaction.followup.send(
                        f"✅ Removed **{embed_title}**\nRemaining: {len(updated_embeds)} embeds",
                        ephemeral=True
                    )
                else:
                    await select_interaction.followup.send("❌ Failed to save.", ephemeral=True)
            else:
                await select_interaction.followup.send("❌ Cancelled.", ephemeral=True)
    
    view = RemoveEmbedSelectView(db_type, display_name, embeds_data)
    await interaction.response.send_message(
        f"⚠️ **Remove Embed from {display_name}**\nSelect which embed to remove:",
        view=view,
        ephemeral=True
    )
    

async def _reset_to_default(interaction: discord.Interaction, db_type: str, display_name: str):
    """Reset welcome message to default"""
    # Defer first since we're showing a view
    await interaction.response.defer(ephemeral=True)
    
    # Get default messages
    default_messages = await _get_default_messages()
    default_embeds = default_messages.get(db_type, [])
    
    if not default_embeds:
        await interaction.followup.send(
            f"❌ No default found for {display_name}.",
            ephemeral=True
        )
        return
    
    # Show confirmation
    confirm_view = ConfirmView(interaction.user)
    await interaction.followup.send(
        f"⚠️ **Reset {display_name} to Default**\nThis will restore the original welcome message.\n\n**Are you sure?**",
        view=confirm_view,
        ephemeral=True
    )
    
    await confirm_view.wait()
    
    if confirm_view.value:
        # Save defaults to database
        success, new_data = await bot.db.update_welcome_message(
            db_type,
            default_embeds,
            f"{interaction.user.name} ({interaction.user.id}) - Reset to default",
        )
        
        if success:
            await interaction.followup.send(
                f"✅ Reset {display_name} to default configuration\nEmbeds restored: {len(default_embeds)}",
                ephemeral=True
            )
        else:
            await interaction.followup.send("❌ Failed to reset to default.", ephemeral=True)
    else:
        await interaction.followup.send("❌ Reset cancelled.", ephemeral=True)

async def _get_default_messages():
    """Get default welcome messages"""
    # These should match what you inserted into the database
    return {
        'hr_welcome': [
            {
                'title': '🎉 Welcome to the HR Team!',
                'description': '**Please note the following:**\n• Request for document access in [HR Documents](https://discord.com/channels/1165368311085809717/1165368317532438646).\n• You are exempted from quota this week only - you start next week ([Quota Info](https://discord.com/channels/1165368311085809717/1206998095552978974)).\n• Uncomplete quota = strike.\n• One failed tryout allowed if your try quota portion ≥2.\n• Ask for help anytime - we\'re friendly!\n• Are you Lieutenant+ in BA? Apply for the Education Department!\n• Are you Captain+ in BA? Apply for both departments: [Applications](https://discord.com/channels/1165368311085809717/1165368316970405916).',
                'color': '#FFD700',
                'footer': 'We\'re excited to have you on board!'
            }
        ],
        'rmp_welcome': [
            {
                'title': '👮| Welcome to the Royal Military Police',
                'description': 'Congratulations on passing your security check, you\'re officially a TRAINING member of the police force. Please be sure to read the information found below.\n\n> ** 1.** Make sure to read all of the rules found in <#1165368313925353580> and in the brochure found below.\n\n> **2.** You **MUST** read the RMP main guide and MSL before starting your duties.\n\n> **3.** You can\'t use your L85 unless you are doing it for Self-Militia or enforcing the PD rules. (Self-defence)\n\n> **4.** Make sure to follow the Chain Of Command. 2nd Lieutenant > Lieutenant > Captain > Major > Lieutenant Colonel > Colonel > Brigadier > Major General\n\n> **5.** For phases, you may wait for one to be hosted in <#1207367013698240584> or request the phase you need in <#1270700562433839135>.\n\n> **6.** All the information about the Defence School of Policing and Guarding is found in both <#1237062439720452157> and <#1207366893631967262>\n\n> **7.** Choose your timezone here https://discord.com/channels/1165368311085809717/1165368313925353578\n\n**Besides that, good luck with your phases!**',
                'color': '#330000'
            },
            {
                'title': 'Special Roles',
                'description': '> Get your role pings here <#1196085670360404018> and don\'t forget the Game Night role RMP always hosts fun events, don\'t miss out!',
                'color': '#330000'
            },
            {
                'title': 'Trainee Constable Brochure',
                'color': '#660000',
                'fields': [
                    {
                        'name': '**TOP 5 RULES**',
                        'value': '> **1**. You **MUST** read the RMP main guide and MSL before starting your duties.\n> **2**. You **CANNOT** enforce the MSL. Only the Parade Deck (PD) rules **AFTER** you pass your phase 1.\n> **3**. You **CANNOT** use your bike on the PD or the pavements.\n> **4**. You **MUST** use good spelling and grammar to the best of your ability.\n> **5**. You **MUST** remain mature and respectful at all times.'
                    },
                    {
                        'name': '**WHO\'S ALLOWED ON THE PD AT ALL TIMES?**',
                        'value': '> ↠ Royal Army Medical Corps,\n> ↠ Royal Military Police,\n> ↠ Intelligence Corps.\n> ↠ Royal Family.'
                    }
                ]
            }
        ]
    }


# Preview Welcome Message Command
@bot.tree.command(name="preview-welcome", description="Preview welcome messages as they appear to new members")
@min_rank_required(Config.HR_ROLE_ID)
async def preview_welcome(
    interaction: discord.Interaction,
    message_type: Literal["HR Welcome", "RMP Welcome"],
    target_user: discord.Member = None
):
    """Preview welcome message exactly as a new member would see it"""
    
    await interaction.response.defer(ephemeral=True)
    
    type_mapping = {
        "HR Welcome": "hr_welcome",
        "RMP Welcome": "rmp_welcome"
    }
    db_type = type_mapping.get(message_type)
    
    if not db_type:
        await interaction.followup.send("❌ Invalid message type.", ephemeral=True)
        return
    
    message_data = await bot.db.get_welcome_message(db_type)
    if not message_data or 'embeds' not in message_data:
        await interaction.followup.send(f"❌ No {message_type} found.", ephemeral=True)
        return
    
    # Create Discord embeds
    discord_embeds = []
    for embed_data in message_data['embeds']:
        discord_embeds.append(dict_to_embed(embed_data))
    
    # Send preview
    preview_text = f"**Preview: {message_type}**"
    if target_user:
        preview_text += f"\n*Simulating send to {target_user.mention}*"
    
    await interaction.followup.send(
        content=preview_text,
        embeds=discord_embeds,
        ephemeral=True
    )
    
    # Log the preview
    logger.info(f"{interaction.user} previewed {message_type}")

@preview_welcome.autocomplete('message_type')
async def preview_welcome_autocomplete(
    interaction: discord.Interaction,
    current: str
):
    """Autocomplete for preview welcome"""
    types = ["HR Welcome", "RMP Welcome"]
    return [
        discord.app_commands.Choice(name=msg_type, value=msg_type)
        for msg_type in types
        if current.lower() in msg_type.lower()
    ][:25]


#Welcome Message(s) History Command
@bot.tree.command(name="welcome-history", description="View history of welcome message changes")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def welcome_history(
    interaction: discord.Interaction,
    message_type: Literal["HR Welcome", "RMP Welcome"],
    limit: app_commands.Range[int, 1, 10] = 5
):
    """View historical versions of welcome messages"""
    
    await interaction.response.defer(ephemeral=True)
    
    type_mapping = {
        "HR Welcome": "hr_welcome",
        "RMP Welcome": "rmp_welcome"
    }
    db_type = type_mapping.get(message_type)
    
    if not db_type:
        await interaction.followup.send("❌ Invalid message type.", ephemeral=True)
        return
    
    history = await bot.db.get_welcome_message_history(db_type, limit)
    
    if not history:
        await interaction.followup.send(f"❌ No history found for {message_type}.", ephemeral=True)
        return
    
    embed = discord.Embed(
        title=f"📜 History: {message_type}",
        description=f"Last {len(history)} versions (newest first)",
        color=discord.Color.blue()
    )
    
    for i, version in enumerate(history, 1):
        timestamp = version.get('last_updated')
        if isinstance(timestamp, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = f"<t:{int(dt.timestamp())}:R>"
            except:
                time_str = timestamp
        elif hasattr(timestamp, 'timestamp'):
            time_str = f"<t:{int(timestamp.timestamp())}:R>"
        else:
            time_str = "Unknown"
        
        embed_count = len(version.get('embeds', []))
        active_status = "✅ Active" if version.get('is_active') else "📁 Archived"
        
        embed.add_field(
            name=f"Version {version.get('version', i)} - {active_status}",
            value=(
                f"**When:** {time_str}\n"
                f"**By:** {version.get('updated_by', 'Unknown')}\n"
                f"**Embeds:** {embed_count}\n"
                f"**ID:** `{version.get('id')}`"
            ),
            inline=False
        )
    
    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.context_menu(name="Preview Welcome For")
@min_rank_required(Config.HR_ROLE_ID)
async def preview_welcome_context(interaction: discord.Interaction, member: discord.Member):
    """Preview welcome message for a specific member"""
    await interaction.response.defer(ephemeral=True)
    
    # Determine which welcome based on member's roles
    hr_role = interaction.guild.get_role(Config.HR_ROLE_ID)
    rmp_role = interaction.guild.get_role(Config.RMP_ROLE_ID)
    
    if hr_role and hr_role in member.roles:
        message_type = "HR Welcome"
        db_type = "hr_welcome"
    elif rmp_role and rmp_role in member.roles:
        message_type = "RMP Welcome"
        db_type = "rmp_welcome"
    else:
        await interaction.followup.send(
            "This member doesn't have HR or RMP roles.",
            ephemeral=True
        )
        return
    
    message_data = await bot.db.get_welcome_message(db_type)
    if not message_data or 'embeds' not in message_data:
        await interaction.followup.send(f"No {message_type} configured.", ephemeral=True)
        return
    
    # Create preview
    discord_embeds = []
    for embed_data in message_data['embeds']:
        discord_embeds.append(dict_to_embed(embed_data))
    
    await interaction.followup.send(
        f"**Preview for {member.mention}**\n{message_type}",
        embeds=discord_embeds,
        ephemeral=True
    )

#Error handling for Welcome Message Commands
@edit_welcome.error
@preview_welcome.error
@welcome_history.error
async def welcome_commands_error(interaction: discord.Interaction, error):
    """Handle errors in welcome commands"""
    try:
        if isinstance(error, discord.app_commands.errors.MissingPermissions):
            await interaction.response.send_message(
                "❌ You don't have permission to use welcome commands.",
                ephemeral=True
            )
        elif isinstance(error, discord.app_commands.errors.CommandOnCooldown):
            await interaction.response.send_message(
                f"⏳ Please wait {error.retry_after:.1f}s before editing welcome messages again.",
                ephemeral=True
            )
        else:
            logger.error(f"Welcome command error: {type(error).__name__}: {error}")
            await interaction.response.send_message(
                "❌ An error occurred. Please try again or contact admin.",
                ephemeral=True
            )
    except discord.NotFound:
        pass  # Interaction already handled
    except Exception as e:
        logger.error(f"Error handler error: {e}")

# Discharge Command
@bot.tree.command(name="discharge", description="Notify members of honourable/general/dishonourable discharge and log it")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def discharge(
    interaction: discord.Interaction,
    members: str,  # Comma-separated user mentions/IDs
    reason: str,
    discharge_type: Literal["Honourable", "General", "Dishonourable"] = "General",
    evidence: Optional[discord.Attachment] = None
):
    view = ConfirmView(author=interaction.user)
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
        if discharge_type == "Honourable" or discharge_type == "General":
            color = discord.Color.green() 
        else 
            color = discord.Color.red()
            
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
                await bot.db.discharge_user(str(member.id), cleaned_nickname, interaction.guild)

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
                value = f"🔰 {discharge_type} Discharge" if discharge_type in ("Honourable", "General") else f"🚨 {discharge_type} Discharge",
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

@bot.tree.command(name="blacklist", description="Blacklist members with specified duration")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def blacklist(
    interaction: discord.Interaction,
    members: str,  # Comma-separated user mentions/IDs
    reason: str,
    duration_unit: Literal["Permanent", "Years", "Months", "Days"],
    duration_amount: app_commands.Range[int, 1, 100] = 1,
    evidence: Optional[discord.Attachment] = None
):
    view = ConfirmView(author=interaction.user)
    
    # Create confirmation message
    duration_text = "Permanent" if duration_unit == "Permanent" else f"{duration_amount} {duration_unit}"
    confirmation_msg = (
        f"⚠️ **Are you sure you want to blacklist member(s)?**\n"
        f"**Duration:** {duration_text}\n"
        f"**Reason:** {reason[:200]}{'...' if len(reason) > 200 else ''}\n\n"
        f"Click **Confirm** to proceed or **Cancel** to abort."
    )
    
    await interaction.response.send_message(confirmation_msg, view=view, ephemeral=True)
    await view.wait()
    
    if not view.value:
        await interaction.followup.send("❎ Blacklist cancelled.", ephemeral=True)
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

        # Parse member IDs
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
            "⚙️ Processing blacklist...",
            ephemeral=True,
            wait=True
        )

        await bot.rate_limiter.wait_if_needed(bucket="blacklist")

        # Get member objects
        blacklisted_members = []
        for member_id in member_ids:
            if member := interaction.guild.get_member(member_id):
                blacklisted_members.append(member)
            else:
                logger.warning(f"Member {member_id} not found in guild")

        if not blacklisted_members:
            await interaction.followup.send("❌ No valid members found.", ephemeral=True)
            return

        # Calculate blacklist duration
        if duration_unit == "Permanent":
            blacklist_duration = "Permanent"
        elif duration_unit == "Years":
            blacklist_duration = f"{duration_amount} year{'s' if duration_amount > 1 else ''}"
        elif duration_unit == "Months":
            blacklist_duration = f"{duration_amount} month{'s' if duration_amount > 1 else ''}"
        elif duration_unit == "Days":
            blacklist_duration = f"{duration_amount} day{'s' if duration_amount > 1 else ''}"
        
        # Calculate ending date for blacklist
        ending_date = None
        if duration_unit != "Permanent":
            current_date = datetime.now(timezone.utc)
            if duration_unit == "Years":
                ending_date = current_date + timedelta(days=duration_amount * 365)
            elif duration_unit == "Months":
                ending_date = current_date + timedelta(days=duration_amount * 30)  # Approximate
            elif duration_unit == "Days":
                ending_date = current_date + timedelta(days=duration_amount)

        # Create notification embed for users (Dishonourable discharge + blacklist info)
        embed = discord.Embed(
            title="Dishonourable Discharge & Blacklist Notification",
            color=discord.Color.red(),
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.add_field(name="Reason", value=reason, inline=False)
        embed.add_field(name="Blacklist Duration", value=blacklist_duration, inline=False)
        
        if ending_date:
            embed.add_field(
                name="Blacklist Ends", 
                value=f"<t:{int(ending_date.timestamp())}:D> (<t:{int(ending_date.timestamp())}:R>)",
                inline=False
            )
        
        if evidence:
            embed.add_field(name="Evidence", value=f"[Attachment Link]({evidence.url})", inline=False)
            mime, _ = mimetypes.guess_type(evidence.filename)
            if mime and mime.startswith("image/"):
                embed.set_image(url=evidence.url)

        embed.set_footer(text=f"Blacklisted by {interaction.user.display_name}")

        success_count = 0
        failed_members = []

        # Process each member
        for member in blacklisted_members:
            cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
            
            try:
                # Remove from database (dishonourable discharge)
                await bot.db.discharge_user(str(member.id), cleaned_nickname, interaction.guild)
                
                # Try to notify the user
                try:
                    await member.send(embed=embed)
                except discord.Forbidden:
                    # If DMs are closed, try public channel
                    if channel := interaction.guild.get_channel(1219410104240050236):
                        await channel.send(f"{member.mention}", embed=embed)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {member.display_name}: {str(e)}")
                failed_members.append(member.mention)

        # Create result embed
        result_embed = discord.Embed(
            title="Blacklist Summary",
            color=discord.Color.red()
        )
        
        result_embed.add_field(name="Action", value=f"Dishonourable Discharge & Blacklist", inline=False)
        result_embed.add_field(name="Duration", value=blacklist_duration, inline=False)
        result_embed.add_field(
            name="Results",
            value=f"✅ Successfully processed: {success_count}\n❌ Failed: {len(failed_members)}",
            inline=False
        )
        
        if failed_members:
            result_embed.add_field(name="Failed Members", value=", ".join(failed_members), inline=False)

        await processing_msg.edit(
            content=None,
            embed=result_embed
        )

        # ========== LOG TO DISCHARGE CHANNEL (AS DISHONOURABLE) ==========
        if d_log := interaction.guild.get_channel(Config.D_LOG_CHANNEL_ID):
            log_embed = discord.Embed(
                title="Discharge Log",
                color=discord.Color.red(),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Log as dishonourable discharge (same format as /discharge command)
            log_embed.add_field(name="Type", value="🚨 Dishonourable Discharge", inline=False)
            log_embed.add_field(name="Sub-Type", value="⛔ With Blacklist", inline=True)
            log_embed.add_field(name="Blacklist Duration", value=blacklist_duration, inline=True)
            
            if ending_date:
                log_embed.add_field(
                    name="Blacklist Ends", 
                    value=f"<t:{int(ending_date.timestamp())}:D>",
                    inline=True
                )
            
            log_embed.add_field(name="Reason", value=f"```{reason}```", inline=False)
            
            # Format member mentions with their cleaned nicknames (same format as /discharge)
            member_entries = []
            for member in blacklisted_members:
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

        # ========== LOG TO BLACKLIST CHANNEL ==========
        if b_log := interaction.guild.get_channel(Config.B_LOG_CHANNEL_ID):
            blacklist_log_embed = discord.Embed(
                title="⛔ Blacklist Entry",
                color=discord.Color.dark_red(),
                timestamp=datetime.now(timezone.utc)
            )
            
            blacklist_log_embed.add_field(name="Issuer:", value=interaction.user.mention, inline=False)
            
            # List all blacklisted members with Roblox IDs if available
            for member in blacklisted_members:
                cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
                
                # Try to get Roblox ID from database before removing
                roblox_id = None
                try:
                    def _get_roblox_id():
                        sup = bot.db.supabase
                        res = sup.table('users').select('roblox_id').eq('user_id', str(member.id)).execute()
                        if getattr(res, 'data', None) and len(res.data) > 0:
                            return res.data[0].get('roblox_id')
                        return None
                    
                    roblox_id = await bot.db.run_query(_get_roblox_id)
                except Exception as e:
                    logger.error(f"Error getting Roblox ID for {cleaned_nickname}: {e}")
                
                member_info = f"**Name:** {cleaned_nickname}\n**Discord:** {member.mention}"
                if roblox_id:
                    member_info += f"\n**Roblox ID:** {roblox_id}"
                
                blacklist_log_embed.add_field(
                    name=cleaned_nickname,
                    value=member_info,
                    inline=False
                )
            
            blacklist_log_embed.add_field(name="Duration:", value=blacklist_duration, inline=False)
            blacklist_log_embed.add_field(name="Reason:", value=reason, inline=False)
            
            if ending_date:
                blacklist_log_embed.add_field(
                    name="Ending date", 
                    value=f"<t:{int(ending_date.timestamp())}:D>",
                    inline=False
                )
            
            blacklist_log_embed.add_field(name="Starting date", value=f"<t:{int(datetime.now(timezone.utc).timestamp())}:D>", inline=False)
            blacklist_log_embed.set_footer(text=f"Blacklisted by {interaction.user.display_name}")
            
            await b_log.send(embed=blacklist_log_embed)

    except Exception as e:
        logger.error(f"Blacklist command failed: {e}")
        await interaction.followup.send("❌ An error occurred while processing the blacklist.", ephemeral=True)

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
    """Send HR welcome message using cached template"""
    if not (welcome_channel := member.guild.get_channel(Config.HR_CHAT_CHANNEL_ID)):
        logger.warning("HR welcome channel not found!")
        return
    
    # Get message data from cache
    message_data = await welcome_cache.get('hr_welcome')
    
    if not message_data or 'embeds' not in message_data:
        logger.error("No HR welcome message found in cache! Using fallback.")
        await _send_fallback_hr_welcome(member, welcome_channel)
        return
    
    try:
        # Create Discord embed objects
        discord_embeds = []
        for embed_data in message_data['embeds']:
            embed = dict_to_embed(embed_data)
            discord_embeds.append(embed)
        
        await welcome_channel.send(content=member.mention, embeds=discord_embeds)
        logger.info(f"✅ Sent HR welcome with {len(discord_embeds)} embeds to {member.display_name}")
        
    except Exception as e:
        logger.error(f"Failed to send HR welcome: {e}")
        await _send_fallback_hr_welcome(member, welcome_channel)

async def _send_fallback_hr_welcome(member: discord.Member, channel):
    """Fallback HR welcome if cache fails"""
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
    embed.set_footer(text="We're excited to have you on board!")
    await channel.send(content=member.mention, embed=embed)
    

# RMP Welcome Message
async def send_rmp_welcome(member: discord.Member):
    """Send RMP welcome message with multiple embeds using cached template"""
    
    # Get message data from cache (no database hit)
    message_data = await welcome_cache.get('rmp_welcome')
    
    if not message_data or 'embeds' not in message_data:
        logger.error("No RMP welcome message found in cache! Using original function.")
        # Fallback to original function
        await _send_original_rmp_welcome(member)
        return
    
    try:
        # Create Discord embed objects from cached data
        discord_embeds = []
        for embed_data in message_data['embeds']:
            embed = dict_to_embed(embed_data)
            discord_embeds.append(embed)
        
        # Send all embeds
        try:
            await member.send(embeds=discord_embeds)
            logger.info(f"✅ Sent {len(discord_embeds)} RMP welcome embeds to {member.display_name}")
        except discord.Forbidden:
            # Try public channel as fallback
            if welcome_channel := member.guild.get_channel(722002957738180620):
                await welcome_channel.send(content=member.mention, embeds=discord_embeds)
                logger.info(f"✅ Sent RMP welcome to {member.display_name} in public channel")
        
    except Exception as e:
        logger.error(f"Failed to send RMP welcome: {e}")
        # Fallback to original
        await _send_original_rmp_welcome(member)

# Keep your original RMP welcome function but rename it
async def _send_original_rmp_welcome(member: discord.Member):
    """Original RMP welcome function as fallback"""
    embed1 = discord.Embed(
        title="👮| Welcome to the Royal Military Police",
        description="Congratulations on passing your security check, you're officially a TRAINING member of the police force. Please be sure to read the information found below.\n\n> ** 1.** Make sure to read all of the rules found in <#1165368313925353580> and in the brochure found below.\n\n> **2.** You **MUST** read the RMP main guide and MSL before starting your duties.\n\n> **3.** You can't use your L85 unless you are doing it for Self-Militia or enforcing the PD rules. (Self-defence)\n\n> **4.** Make sure to follow the Chain Of Command. 2nd Lieutenant > Lieutenant > Captain > Major > Lieutenant Colonel > Colonel > Brigadier > Major General\n\n> **5.** For phases, you may wait for one to be hosted in <#1207367013698240584> or request the phase you need in <#1270700562433839135>.\n\n> **6.** All the information about the Defence School of Policing and Guarding is found in both <#1237062439720452157> and <#1207366893631967262>\n\n> **7.** Choose your timezone here https://discord.com/channels/1165368311085809717/1165368313925353578\n\n**Besides that, good luck with your phases!**",
        color=discord.Color.from_str("#330000") 
    )

    special_embed = discord.Embed(
        title="Special Roles",
        description="> Get your role pings here <#1196085670360404018> and don't forget the Game Night role RMP always hosts fun events, don't miss out!",
        color=discord.Color.from_str("#330000")
    )
    
    embed2 = discord.Embed(
        title="Trainee Constable Brochure",
        color=discord.Color.from_str("#660000")
    )
    
    embed2.add_field(
        name="**TOP 5 RULES**",
        value="> **1**. You **MUST** read the RMP main guide and MSL before starting your duties.\n> **2**. You **CANNOT** enforce the MSL. Only the Parade Deck (PD) rules **AFTER** you pass your phase 1.\n> **3**. You **CANNOT** use your bike on the PD or the pavements.\n> **4**. You **MUST** use good spelling and grammar to the best of your ability.\n> **5**. You **MUST** remain mature and respectful at all times.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED ON THE PD AT ALL TIMES?**",
        value="> ↠ Royal Army Medical Corps,\n> ↠ Royal Military Police,\n> ↠ Intelligence Corps.\n> ↠ Royal Family.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED ON THE PD WHEN CARRYING OUT THEIR DUTIES?**",
        value="> ↠ United Kingdom Special Forces,\n> ↠ Grenadier Guards,\n> ↠ Foreign Relations,\n> ↠ Royal Logistic Corps,\n> ↠ Adjutant General's Corps,\n> ↠ High Ranks, RSM, CSM and ASM hosting,\n> ↠ Regimental personnel watching one of their regiment's events inside Pad area.",
        inline=False
    )
    
    embed2.add_field(
        name="**HOW DO I ENFORCE PD RULES ON PEOPLE NOT ALLOWED ON IT?**",
        value="> 1. Give them their first warning to get off the PD, \"W1, off the PD!\"\n> 2. Wait 3-5 seconds for them to listen; if they don't, give them their second warning, \"W2, off the PD!\"\n> 3. Wait 3-5 seconds for them to listen; if they don't kill them.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED __ON__ THE ACTUAL STAGE AT ALL TIMES**",
        value="> ↠ Major General and above,\n> ↠ Royal Family (they should have a purple name tag),\n> ↠ Those who have been given permission by a Lieutenant General.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED TO PASS THE RED LINE IN-FRONT OF THE STAGE?**",
        value="> ↠ Major General and above,\n> ↠ Royal Family,\n> ↠ Those who have been given permission by a Lieutenant General,\n> ↠ COMBATIVE Home Command Regiments:\n> - Royal Military Police,\n> - United Kingdom Forces,\n> - Household Division.\n> **Kill those not allowed who touch or pass the red line.**",
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
        # ===== CHECK FOR RMP ROLE REMOVAL (DESERTION/REMOVAL) =====
        rmp_role = after.guild.get_role(Config.RMP_ROLE_ID)
        
        if rmp_role and rmp_role in before.roles and rmp_role not in after.roles:
            # Member had RMP role but lost it - handle desertion/removal
            cleaned_nickname = re.sub(r'\[.*?\]', '', after.display_name).strip() or after.name
            
            try:
                await bot.db.discharge_user(str(after.id), cleaned_nickname, after.guild)
                logger.info(f"Removed {cleaned_nickname} ({after.id}) from database due to RMP role removal")
            except Exception as e:
                logger.error(f"Error removing {cleaned_nickname} ({after.id}) from DB: {e}")
            
            # Optionally send alert about role removal
            try:
                alert_channel = after.guild.get_channel(Config.HR_CHAT_CHANNEL_ID)
                if alert_channel:
                    embed = discord.Embed(
                        title="⚠️ RMP Role Removed",
                        description=f"{after.mention} no longer has the RMP role.",
                        color=discord.Color.orange(),
                        timestamp=datetime.now(timezone.utc)
                    )
                    embed.add_field(name="User", value=f"{cleaned_nickname} ({after.id})", inline=True)
                    embed.add_field(name="Action", value="Database record removed", inline=True)
                    await alert_channel.send(embed=embed)
            except Exception as e:
                logger.error(f"Failed to send RMP removal alert: {e}")
        
        # ===== WELCOME MESSAGES =====

        #HR Welcome
        hr_role = after.guild.get_role(Config.HR_ROLE_ID)
        if hr_role and hr_role not in before.roles and hr_role in after.roles:
            cleaned_nickname = re.sub(r'\[.*?\]', '', after.display_name).strip() or after.name
        
            try:
                await bot.db.remove_from_lr(str(after.id))
         
                await bot.db.add_to_hr(
                    user_id=str(after.id),
                    username=cleaned_nickname,
                    guild=after.guild
                )
        
                logger.info(
                    f"🔁 {cleaned_nickname} ({after.id}) moved from LRs → HRs in database"
                )
        
            except Exception as e:
                logger.error(f"❌ Failed HR DB transfer for {after.id}: {e}")
        
            await send_hr_welcome(after)
        
        
        # Check for RMP role addition
        if rmp_role and rmp_role not in before.roles and rmp_role in after.roles:
            # Create/update user in database with Roblox ID
            await create_or_update_user_in_db(str(after.id), after.display_name, after.guild)

            await send_rmp_welcome(after)

        # ===== RANK TRACKING =====
        # Check if roles changed
        if set(before.roles) == set(after.roles):
            return  # No role changes, exit early
        
        # Get role IDs for comparison
        before_role_ids = [role.id for role in before.roles]
        after_role_ids = [role.id for role in after.roles]
        
        # Check if rank-related roles changed (only update if they did)
        if hasattr(bot, 'rank_tracker') and bot.rank_tracker:
            if not bot.rank_tracker._roles_changed_affect_rank(before_role_ids, after_role_ids):
                return  # Role changes don't affect rank, exit early
        
        # Only update rank if member has RMP or HR role
        rmp_role = after.guild.get_role(Config.RMP_ROLE_ID)  # Re-fetch in case we need it
        hr_role = after.guild.get_role(Config.HR_ROLE_ID)
        
        if not ((rmp_role and rmp_role in after.roles) or (hr_role and hr_role in after.roles)):
            return  # Not an RMP or HR member, exit early
        
        # Update rank in database (with slight delay to prevent rapid updates)
        try:
            await asyncio.sleep(0.5)  # Small delay
            
            if hasattr(bot, 'rank_tracker') and bot.rank_tracker:
                success = await bot.rank_tracker.update_member_in_database(after)
                
                if success:
                    member_info = await bot.rank_tracker.get_member_info(after)
                    logger.info(
                        f"🔄 Auto-updated rank for {after.display_name}: "
                        f"Division={member_info['division']}, Rank={member_info['rank']}"
                    )
                else:
                    logger.warning(f"Failed to auto-update rank for {after.display_name}")
            else:
                logger.warning("Rank tracker not initialized, cannot update rank")
                
        except Exception as e:
            logger.error(f"Error in on_member_update for {after.display_name}: {e}")
            
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
        
        # 🔍 FIRST: Get Roblox ID from database BEFORE removing user
        roblox_id = None
        try:
            def _get_roblox_id():
                sup = bot.db.supabase
                res = sup.table('users').select('roblox_id').eq('user_id', str(member.id)).execute()
                if getattr(res, 'data', None) and len(res.data) > 0:
                    return res.data[0].get('roblox_id')
                return None
            
            roblox_id = await bot.db.run_query(_get_roblox_id)
            logger.info(f"Retrieved Roblox ID for deserter {cleaned_nickname}: {roblox_id}")
        except Exception as e:
            logger.error(f"Error getting Roblox ID for {cleaned_nickname}: {e}")

        # 🚨 Alert embed (send immediately)
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

        # Current date for blacklist calculation
        current_date = datetime.now(timezone.utc)    
        hr_role = guild.get_role(Config.HR_ROLE_ID)  
        
        if hr_role and hr_role in member.roles:
            ending_date = current_date + timedelta(days=180)  
        else:
            ending_date = current_date + timedelta(days=90)

        # 🔧 Dishonourable discharge embed WITH Roblox ID
        dishonourable_embed = discord.Embed(
            title="Discharge Log",
            color=discord.Color.red(),
            timestamp=current_date
        )
        dishonourable_embed.add_field(name="Type", value="🚨 Dishonourable Discharge", inline=False)
        dishonourable_embed.add_field(name="Reason", value="```Desertion.```", inline=False)
        
        # Build member info with Roblox ID if available [Discontinued| will probably readd later on"
        member_info = f"{member.mention} | {cleaned_nickname}"
        
        dishonourable_embed.add_field(name="Discharged Members", value=member_info, inline=False)
        dishonourable_embed.add_field(name="Discharged By", value="None - Unauthorised", inline=True)
        dishonourable_embed.set_footer(text="Desertion Monitor System")

        # ⛔ Blacklist embed
        blacklist_embed = discord.Embed(
            title="⛔ Blacklist",
            color=discord.Color.red(),
            timestamp=current_date
        )
        blacklist_embed.add_field(name="Issuer:", value="MP Assistant", inline=False)
        blacklist_embed.add_field(name="Name:", value=cleaned_nickname, inline=False)
        
        # Add Roblox ID to blacklist embed if available
        if roblox_id:
            blacklist_embed.add_field(name="Roblox ID:", value=str(roblox_id), inline=False)
        
        blacklist_embed.add_field(name="Starting date", value=f"<t:{int(current_date.timestamp())}:D>", inline=False)
        blacklist_embed.add_field(name="Ending date", value=f"<t:{int(ending_date.timestamp())}:D>", inline=False)
        blacklist_embed.add_field(name="Reason:", value="Desertion.", inline=False)
        blacklist_embed.set_footer(text="Desertion Monitor System")

        # 📝 SECOND: Now send the embeds to the logs
        try:
            d_log = bot.get_channel(Config.D_LOG_CHANNEL_ID)
            b_log = bot.get_channel(Config.B_LOG_CHANNEL_ID)
            
            if d_log and b_log:
                await d_log.send(embed=dishonourable_embed)
                await b_log.send(embed=blacklist_embed)
                logger.info(f"Logged deserted member, {cleaned_nickname}. Roblox ID: {roblox_id}")
            else:
                alt_channel = bot.get_channel(1165368316970405917)
                if alt_channel:
                    await alt_channel.send(f"⚠️ Failed to log deserter discharge for {cleaned_nickname}")
                    logger.error(f"Failed to log deserted member {cleaned_nickname} - main channel not found")
        except Exception as e:
            logger.error(f"Error logging deserter discharge: {str(e)}")
        
        # 🗑️ THIRD: Only AFTER sending embeds, remove from database
        try:
            await bot.db.discharge_user(str(member.id), cleaned_nickname, guild)
            logger.info(f"Removed {cleaned_nickname} ({member.id}) from database")
        except Exception as e:
            logger.error(f"Error removing deserter {cleaned_nickname} ({member.id}) from DB: {e}")


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
            
            # Close any existing session before starting
            if hasattr(bot, "shared_session") and bot.shared_session and not bot.shared_session.closed:
                await bot.shared_session.close()
                bot.shared_session = None
            
            await bot.start(TOKEN)
            
            # Reset on successful run
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
            
            # Clean up session before retry
            if hasattr(bot, "shared_session") and bot.shared_session and not bot.shared_session.closed:
                await bot.shared_session.close()
                bot.shared_session = None
            
            if getattr(e, "status", None) == 429:
                retry_after = float(e.response.headers.get('Retry-After', 30) if getattr(e, "response", None) else 30)
                logger.warning(f"Rate limited during login. Waiting {retry_after}s.")
                await asyncio.sleep(retry_after)
                
                # Don't continue the loop - break out and restart fresh
                logger.info("Closing bot and restarting due to rate limit...")
                try:
                    await bot.close()
                except:
                    pass
                continue
                
            logger.error(f"HTTPException in run loop: {e}")
            
        except Exception as e:
            consecutive_failures += 1
            
            # Clean up session
            if hasattr(bot, "shared_session") and bot.shared_session and not bot.shared_session.closed:
                await bot.shared_session.close()
                bot.shared_session = None
            
            logger.error(f"Unexpected error in Discord client: {type(e).__name__}: {e}")

        # Exponential backoff with jitter
        sleep_time = min(max_backoff, backoff * (2 ** consecutive_failures))
        sleep_time *= random.uniform(0.8, 1.2)
        
        logger.info(f"Discord client stopped; restarting in {sleep_time:.1f}s")
        
        # Ensure bot is closed before restart
        try:
            if not bot.is_closed():
                await bot.close()
        except:
            pass
            
        await asyncio.sleep(sleep_time)
        backoff = sleep_time

if __name__ == '__main__':  
    try:
        asyncio.run(run_bot_forever())
    except Exception as e:
        logger.critical(f"Fatal error running bot: {e}", exc_info=True)
        raise




































































































































































