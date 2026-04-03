import os
import asyncio
import discord
import logging
import functools

from typing import Tuple
from dotenv import load_dotenv
from config import Config
from utils import roblox
from utils.helpers import clean_nickname
from utils.roblox import RobloxAPI
from supabase import create_client, Client

logger  = logging.getLogger(__name__)
load_dotenv()

class DatabaseHandler:
    # Function: Initialises class object variables
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            logger.warning("Supabase credentials not provided; DatabaseHandler will be inert.")
            self.supabase = None
        else:
            # create sync client once (cheap); all sync calls are run in threads
            self.supabase = create_client(url, key)

    # Function: Wraps a synchronous function so it runs in a background thread without blocking the async event loop
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
            return False, 0

    # Generic helper to run arbitrary table queries when needed from command handlers
    # Function: Accepts any sync supabase function from outside the class and runs it in a thread - allows flexible one-off queires without adding new methods

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

    async def get_welcome_message_history(self, message_type: str, limit: int = 5):
        """Get historical versions of welcome messages"""
        def _work():
            result = self.supabase.table('welcome_messages') \
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
        self,
        discord_id: str,
        username: str,
        guild: discord.Guild,
        roblox_id: int | None = None
    ) -> bool:
        """
        Create or update a user in the 'users' table with Roblox ID.
        Returns: True if successful, False if failed
        """
        try:
            # Clean the username
            cleaned_username = clean_nickname(username)
            
            
            def _work():
                sup = self.supabase
                res = sup.table('users').select('*').eq('user_id', discord_id).execute()
                
                if getattr(res, 'data', None) and len(res.data) > 0:
                    update_data = {
                        "username": cleaned_username,
                        "roblox_id": roblox_id
                    }
                    return sup.table('users').update(update_data).eq('user_id', discord_id).execute()
                else:
                    new_user = {
                        "user_id": discord_id,
                        "username": cleaned_username,
                        "roblox_id": roblox_id,
                        "xp": 0  
                    }
                    return sup.table('users').insert(new_user).execute()
            
            await self.run_query(_work)
            logger.info(f"Created/updated user {cleaned_username} ({discord_id}), Roblox ID: {roblox_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create/update user {discord_id} in database: {e}")
            return False


