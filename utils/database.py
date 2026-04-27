import os
import asyncio
import discord
import logging

from typing import Tuple
from dotenv import load_dotenv
from config import Config
from utils.helpers import clean_nickname
from supabase import acreate_client, AsyncClient

logger  = logging.getLogger(__name__)
load_dotenv()

class DatabaseHandler:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not self.url or not self.key:
            logger.warning("Supabase credentials not provided; DatabaseHandler will be inert.")
        self.supabase = None
        self._welcome_cache_lock = asyncio.Lock()
        self._user_cache_lock = asyncio.Lock()
        self._hrs_cache_lock = asyncio.Lock()
        self._lrs_cache_lock = asyncio.Lock()
        self.welcome_cache = {}
        self.welcome_initialised = False
        self._user_cache = {}
        self._hrs_cache = {}
        self._lrs_cache = {}
        self.users_table = "users"
        self.s_users_table = "users_store"
        self.hrs_table = "HRs"
        self.lrs_table = "LRs"
        self.crs_table = "criminal_records"
        self.u_roles_table = "user_roles"
        self.wm_table = "welcome_messages"


    async def initialise(self):
        if not self.supabase and self.url and self.key:
            self.supabase: AsyncClient = await acreate_client(self.url, self.key)

        await self._sync_all_users()
        await self._sync_all_hrs()
        await self._sync_all_lrs()
   
    async def _sync_all_users(self):
        """Fetch the entire table to keep cache and leaderboard 100% accurate"""
        try:
            res = await self.supabase.table(self.users_table).select("*").execute()
            
            async with self._user_cache_lock:
                self._user_cache = {str(user['user_id']): user for user in res.data}
                self._user_last_sync = asyncio.get_event_loop().time()
            
            logger.info(f"Synchronized {len(self._user_cache)} users into memory.")
        except Exception as e:
            logger.error(f"Sync failed: {e}")

    async def _sync_all_hrs(self):
        """Fetch the entire HR table to keep cache and leaderboard 100% accurate"""
        try:
            res = await self.supabase.table(self.hrs_table).select("*").execute()
            
            async with self._hrs_cache_lock:
                self._hrs_cache = {str(user['user_id']): user for user in res.data}
            
            logger.info(f"Synchronized {len(self._hrs_cache)} hrs into memory.")
        except Exception as e:
            logger.error(f"Sync failed: {e}")

    async def _sync_all_lrs(self):
        """Fetch the LR entire table to keep cache and leaderboard 100% accurate"""
        try:
            res = await self.supabase.table(self.lrs_table).select("*").execute()
            
            async with self._lrs_cache_lock:
                self._lrs_cache = {str(user['user_id']): user for user in res.data}
            
            logger.info(f"Synchronized {len(self._lrs_cache)} lrs into memory.")
        except Exception as e:
            logger.error(f"Sync failed: {e}")


    async def welcome_initialise(self):
        """Load all welcome messages from database on startup"""
        if self.welcome_initialised:
            return
            
        async with self._welcome_cache_lock:
            try:
                for msg_type in ['hr_welcome', 'rmp_welcome']:
                    data = await self._load_from_database(msg_type)
                    if data:
                        self.welcome_cache[msg_type] = data
                
                logger.info(f"Loaded {len(self.welcome_cache)} welcome messages into cache")
                self.welcome_initialised = True
                
            except Exception as e:
                logger.error(f"Failed to initialize welcome message cache: {e}")   
 
    async def send_change_log(self):
        if not self.supabase:
            return False

        try:
            res = await self.supabase.table("send_change_log").select("*").eq("id", 1).maybe_single().execute() 
            
            send = res.data if res and res.data else False
            
            return send

        except Exception as e:
            logger.exception("send_change_log failed for: %s", e)
            return False
   
    async def get_user(self, user_id: str) -> dict:
        """Retrieves full user data from cache, falling back to Database if missing."""
        user_id_str = str(user_id)

        if user_id_str in self._user_cache:
            return self._user_cache[user_id_str]

        if not self.supabase:
            return {}

        try:
            res = await self.supabase.table(self.users_table).select("*").eq("user_id", user_id_str).maybe_single().execute() 
            
            user_data = res.data if res and res.data else {}
            
            if user_data:
                async with self._user_cache_lock:
                    self._user_cache[user_id_str] = user_data
            
            return user_data

        except Exception as e:
            logger.exception("get_user failed for %s: %s", user_id_str, e)
            return {}

    async def get_stored_user(self, user_id: str) -> dict:
        user_id_str = str(user_id)

        if not self.supabase:
            return {}

        try:
            res = await self.supabase.table(self.s_users_table).select("*").eq("user_id", user_id_str).maybe_single().execute()
            user_data = res.data if res and res.data else {}

            if user_data:
                async with self._user_cache_lock:
                    self._user_cache[user_id_str] = user_data
            
            return user_data

        except Exception as e:
            logger.exception("get_stored_user failed: %s", e)
            return {}

    async def get_hr_info(self, user_id: str) -> dict:
        user_id_str = str(user_id)

        if user_id_str in self._hrs_cache:
            return self._hrs_cache[user_id_str]

        if not self.supabase:
            return {}

        try:
            res = await self.supabase.table(self.hrs_table).select("*").eq("user_id", user_id_str).maybe_single().execute() 
            
            user_data = res.data if res and res.data else {}
            
            if user_data:
                async with self._hrs_cache_lock:
                    self._hrs_cache[user_id_str] = user_data
            
            return user_data
        except Exception as e:
            logger.exception("get_hr_info failed: %s", e)
            return {}

    async def get_lr_info(self, user_id: str):
        user_id_str = str(user_id)

        if user_id_str in self._lrs_cache:
            return self._lrs_cache[user_id_str]

        if not self.supabase:
            return {}

        try:
            res = await self.supabase.table(self.lrs_table).select("*").eq("user_id", user_id_str).maybe_single().execute() 
            
            user_data = res.data if res and res.data else {}
            
            if user_data:
                async with self._lrs_cache_lock:
                    self._lrs_cache[user_id_str] = user_data
            
            return user_data
        except Exception as e:
            logger.exception("get_lr_info failed: %s", e)
            return {}

    async def get_user_xp(self, user_id: str) -> int:
        user_id_str = str(user_id)

        if user_id_str in self._user_cache:
            return self._user_cache[user_id_str].get('xp', 0)

        if not self.supabase:
            return 0

        try:
            res = await self.supabase.table(self.users_table).select("*").eq("user_id", user_id_str).maybe_single().execute()
            user_data = res.data if res and res.data else {}

            if user_data:
                async with self._user_cache_lock:
                    self._user_cache[user_id_str] = user_data
                return user_data.get('xp', 0)
            
            return 0

        except Exception:
            return 0

    async def get_roblox_id(self, user_id: str) -> str:
        user_id_str = str(user_id)

        if user_id_str in self._user_cache:
            return str(self._user_cache[user_id_str].get('roblox_id', ""))

        if not self.supabase:
            return ""

        try:
            res = await self.supabase.table(self.users_table).select("*").eq("user_id", user_id_str).maybe_single().execute()
            user_data = res.data if res and res.data else {}

            if user_data:
                async with self._user_cache_lock:
                    self._user_cache[user_id_str] = user_data
                return str(user_data.get('roblox_id', ""))
            
            return ""

        except Exception as e:
            logger.exception("get_roblox_id failed: %s", e)
            return ""


    async def get_all_users_sorted_by_xp(self) -> list:
        if not self._user_cache:
            if not self.supabase:
                return []
            await self._sync_all_users()

        async with self._user_cache_lock:
            all_users = list(self._user_cache.values())

        sorted_users = sorted(
            all_users, 
            key=lambda x: x.get('xp', 0), 
            reverse=True
        )

        return [(str(user['user_id']), user.get('xp', 0)) for user in sorted_users]

    async def add_xp(self, user_id: str, username: str, xp: int) -> Tuple[bool, int]:
        user_id_str = str(user_id)
        current_xp = await self.get_user_xp(user_id_str)
        new_total = current_xp + xp
        cleaned_name = clean_nickname(username)

        try:
            await self.supabase.table(self.users_table).upsert({
                "user_id": user_id_str,
                "xp": new_total,
                "username": cleaned_name
            }).execute()

            async with self._user_cache_lock:
                if user_id_str in self._user_cache:
                    self._user_cache[user_id_str]['xp'] = new_total
                    self._user_cache[user_id_str]['username'] = cleaned_name
                else:
                    self._user_cache[user_id_str] = {
                        "user_id": user_id_str,
                        "xp": new_total,
                        "username": cleaned_name
                    }
            
            return True, new_total

        except Exception as e:
            logger.exception("add_xp failed: %s", e)
            return False, 0


    async def remove_xp(self, user_id: str, xp: int) -> Tuple[bool, int]:
        user_id_str = str(user_id)
        current_xp = await self.get_user_xp(user_id_str)
        new_total = max(0, current_xp - xp)

        if not self.supabase:
            return False, 0

        try:
            res =await self.supabase.table(self.users_table).update({"xp": new_total}).eq("user_id", user_id_str).execute()
            
            if not res.data:
                return False, 0

            async with self._user_cache_lock:
                if user_id_str in self._user_cache:
                    self._user_cache[user_id_str]['xp'] = new_total
                else:
                    self._user_cache[user_id_str] = {
                        "user_id": user_id_str,
                        "xp": new_total
                    }
            
            return True, new_total

        except Exception as e:
            logger.exception("remove_xp failed: %s", e)
            return False, 0


    async def get_leaderboard(self, category: str, limit: int = 200):
        sorted_users = []

        if category == "XP":
            now = asyncio.get_event_loop().time()
            if now - self._user_last_sync > 300:
                await self._sync_all_users()
            async with self._user_cache_lock:
                all_users = list(self._user_cache.values())

            sorted_users = sorted(
                all_users,
                key=lambda x: x.get('xp', 0),
                reverse=True
            )
            sorted_users = [
                {**u, 'score': u.get('xp', 0)}
                for u in sorted_users
            ]
        elif category == "HR":
            async with self._hrs_cache_lock:
                hr_users = list(self._hrs_cache.values())

            sorted_users = sorted(
                hr_users,
                key=lambda x: (x.get('tryouts', 0) + x.get('events', 0) + x.get('phases', 0) + x.get('inspections', 0) + x.get('joint_events', 0)),
                reverse=True
            )
            sorted_users = [
                {**u, 'score': u.get('tryouts', 0) + u.get('events', 0) + u.get('phases', 0) + u.get('inspections', 0) + u.get('joint_events', 0)}
                for u in sorted_users
            ]
        elif category == "LR Events":
            async with self._lrs_cache_lock:
                lr_users = list(self._lrs_cache.values())

            sorted_users = sorted(
                lr_users,
                key=lambda x: x.get('events_attended', 0),
                reverse=True
            )
            sorted_users = [
                {**u, 'score': u.get('events_attended', 0)}
                for u in sorted_users
            ]
        elif category == "LR Activity":
            async with self._lrs_cache_lock:
                lr_users = list(self._lrs_cache.values())

            sorted_users = sorted(
                lr_users,
                key=lambda x: (x.get('activity', 0) + x.get('time_guarded', 0)),
                reverse=True
            )
            sorted_users = [
                {**u, 'score': u.get('activity', 0) + u.get('time_guarded', 0)}
                for u in sorted_users
            ]
        elif category == "Departments":
            async with self._hrs_cache_lock:
                hr_users = list(self._hrs_cache.values())

            sorted_users = sorted(
                hr_users,
                key=lambda x: x.get('courses', 0),
                reverse=True
            )
            sorted_users = [
                {**u, 'score': u.get('courses', 0)}
                for u in sorted_users
            ]
        else:
            raise ValueError(f"Unknown leaderboard category: '{category}'")

        return sorted_users[:limit] if sorted_users else None

    async def increment_points(self, column: str, table: str, member: discord.Member, points, replace: bool = False):
        try:
            user_id_str = str(member.id)
            cleaned_nickname = clean_nickname(member.display_name)
            
            target_cache = self._hrs_cache if table == "HRs" else self._lrs_cache
            user_data = target_cache.get(user_id_str, {})

            current_points = user_data.get(column) or 0
            new_points = points if replace else (current_points + points)

            await self.create_or_update_user_in_db(discord_id=user_id_str, username=cleaned_nickname)

            await self.supabase.table(table).upsert({
                "user_id": str(member.id),
                "username": cleaned_nickname,
                column: new_points
            }).execute()
        
            logger.info("Updated %s | %s points for %s (%s): %s ➝ %s", table, column, cleaned_nickname, user_id_str, current_points, new_points)

            return current_points, new_points
            
        except Exception as e:
            logger.exception("Failed to increment points in %s | %s for %s: %s", table, column, member.id, e)
        return None, None

    async def increment_points_handler(self, column: str, table: str, member: discord.Member, points, replace: bool = False):
        user_id_str = str(member.id)
        if table == "HRs":
            async with self._hrs_cache_lock: 
                _, points = await self.increment_points(column, table, member, points, replace)
                if points is not None:
                    if user_id_str not in self._hrs_cache:
                        self._hrs_cache[user_id_str] = {}

                    self._hrs_cache[user_id_str][column] = points
                    return True
        else: 
            async with self._lrs_cache_lock:
                _, points = await self.increment_points(column, table, member, points, replace)
                if points is not None:
                    if user_id_str not in self._lrs_cache:
                        self._lrs_cache[user_id_str] = {}

                    self._lrs_cache[user_id_str][column] = points
                    return True

        return False


    async def discharge_user(self, user_id: str, username: str, guild: discord.Guild) -> bool:
        """Removing and Archiving user data to user_store"""
        user_id_str = str(user_id)
        
        user_data = await self.get_user(user_id_str)
        if not user_data:
            return False

        color = discord.Color.green()
        title = "Removed from Database"
        icon_url = Config.CHECK_URL
        description = f"**{username}** (`{user_id}`) has been successfully removed from the database and stored in backup for retrieval."
        success = False

        try:
            store_data = user_data.copy()
            store_data.pop("rank", None)
            store_data.pop("division", None)

            await self.supabase.table(self.s_users_table).upsert(store_data).execute()

            res = await self.supabase.table(self.users_table).delete().eq("user_id", user_id_str).execute()
            if not res.data:
                return False

            async with self._user_cache_lock:
                self._user_cache.pop(user_id_str, None)

            logger.info("Successfully discharged and archived %s", username)
            success = True
        except Exception as e:
            logger.exception("discharge_user failed for %s: %s", username, e)
            color = discord.Color.red()
            title = "Database Removal Failed"
            icon_url = Config.CANCEL_URL 
            description = f"An error occurred while removing **{username}** from the database."

        log_channel = guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
        if log_channel:
            try:
                embed = discord.Embed(description=description, color=color).set_footer(text="Will be deleted in a month.")
                embed.set_author(name=title, icon_url=icon_url)
                await log_channel.send(embed=embed)
            except Exception as log_error:
                logger.error("Failed to send log embed: %s", log_error)
        
        return success


    async def delete_stored_user(self, user_id: str) -> bool:
        if not self.supabase:
            return False
        
        try:
            res = await self.supabase.table(self.s_users_table).select("*").eq("user_id", str(user_id)).maybe_single().execute()
            if not res:
                return False
            
            await self.supabase.table(self.s_users_table).delete().eq("user_id", str(user_id)).execute()
            return True
        except Exception as e:
            logger.exception("delete_user failed: %s", e)
            return False

    async def add_to_hr(self, user_id: str, username: str) -> bool:
        if not self.supabase:
            return False

        user_id_str = str(user_id)
        cleaned_name = clean_nickname(username)
        await self.create_or_update_user_in_db(user_id, cleaned_name)

        update_data = {
            "user_id": user_id_str,
            "username": cleaned_name,
        }

        try:
            await self.supabase.table(self.hrs_table).upsert(update_data).execute()
            
            async with self._hrs_cache_lock:
                if user_id_str not in self._hrs_cache:
                    self._hrs_cache[user_id_str] = {}
                
                self._hrs_cache[user_id_str].update(update_data)

            logger.info("Added/Updated %s (%s) in %s table and cache.", cleaned_name, user_id_str, self.hrs_table)
            return True

        except Exception as e:
            logger.exception("Failed to add %s to %s: %s", user_id_str, self.hrs_table, e)
            return False

    async def add_to_lr(self, user_id: str, username: str) -> bool:
        if not self.supabase:
            return False

        user_id_str = str(user_id)
        cleaned_name = clean_nickname(username)
        await self.create_or_update_user_in_db(user_id, cleaned_name)

        update_data = {
            "user_id": user_id_str,
            "username": cleaned_name,
        }

        try:
            await self.supabase.table(self.lrs_table).upsert(update_data).execute()
            
            async with self._lrs_cache_lock:
                if user_id_str not in self._lrs_cache:
                    self._lrs_cache[user_id_str] = {}
                
                self._lrs_cache[user_id_str].update(update_data)

            logger.info("Added/Updated %s (%s) in %s table and cache.", cleaned_name, user_id_str, self.lrs_table)
            return True

        except Exception as e:
            logger.exception("Failed to add %s to %s: %s", user_id_str, self.lrs_table, e)
            return False

    async def remove_from_hr(self, user_id: str) -> bool:
        """Remove a user from the LRs table and the local LR cache."""
        if not self.supabase:
            return False

        user_id_str = str(user_id)

        try:
            res = await self.supabase.table(self.hrs_table).delete().eq("user_id", user_id_str).execute()
            if not res.data:
                return False

            async with self._hrs_cache_lock:
                self._hrs_cache.pop(user_id_str, None)

            logger.info("Removed %s from %s database and cache.", user_id_str, self.hrs_table)
            return True

        except Exception as e:
            logger.exception("Failed to remove %s from %s: %s", user_id_str, self.hrs_table, e)
            return False

    async def remove_from_lr(self, user_id: str) -> bool:
        """Remove a user from the LRs table and the local LR cache."""
        if not self.supabase:
            return False

        user_id_str = str(user_id)

        try:
            res = await self.supabase.table(self.lrs_table).delete().eq("user_id", user_id_str).execute()
            if not res.data:
                return False
            async with self._lrs_cache_lock:
                self._lrs_cache.pop(user_id_str, None)

            logger.info("Removed %s from %s database and cache.", user_id_str, self.lrs_table)
            return True

        except Exception as e:
            logger.exception("Failed to remove %s from %s: %s", user_id_str, self.lrs_table, e)
            return False

    async def save_user_roles(self, user_id: str, username: str, role_ids: list[int]):
        """Save a user's tracked roles into the 'user_roles' table."""
        if not self.supabase:
            logger.warning("Supabase not configured; save_user_roles aborted.")
            return False

        try:
            await self.supabase.table(self.u_roles_table).upsert({
                "user_id": str(user_id),
                "username": clean_nickname(username),
                "roles": role_ids
            }).execute()
            logger.info("Done save_user_roles for %s", user_id)
            return True
        except Exception as e:
            logger.exception(f"save_user_roles failed for %s: %s", user_id, e)
            return False

    async def get_user_roles(self, user_id: str) -> list:
        """Retrieve saved roles for a user."""
        if not self.supabase:
            logger.warning("Supabase not configured; get_user_roles aborted.")
            return []

        try:
            res = await self.supabase.table(self.u_roles_table).select("roles").eq("user_id", str(user_id)).execute()
            logger.info("Retrieved saved roles for %s", user_id)
            return res.data[0].get("roles", []) if res and res.data else []
        except Exception as e:
            logger.exception("get_user_roles failed for %s: %s", user_id, e)
            return []
    
    async def get_criminal_record(self, user_id: str|None=None, username: str|None=None):
        if not self.supabase:
            return None

        filters = []
        if user_id:
            filters.append(f"user_id.eq.{user_id}")
        if username:
            filters.append(f"username.eq.{username}")
        
        if not filters:
            return []

        try:
            res = (
                await self.supabase
                .table(self.crs_table)
                .select("*") 
                .or_(",".join(filters))
                .execute()
            )
            
            return res.data 
            
        except Exception as e:
            logger.exception("Failed to fetch records: %s", e)
            return None


    async def add_criminal_record(self, message_id: str, user_id: str, username: str, link: str):
        try:
            await self.supabase.table(self.crs_table).insert({
                "message_id": str(message_id),
                "user_id": str(user_id),
                "username": str(username),
                "record": str(link)
            }).execute()
            return True
        except Exception as e:
            logger.exception("add_criminal_record failed: %s", e)
            return False

    async def remove_criminal_record(self, message_id: str|None = None, username: str|None = None, rbx_id: str|None = None):
        try:
            if message_id:
                res = await self.supabase.table(self.crs_table).delete().eq("message_id", str(message_id)).execute()
            elif username:
                res = await self.supabase.table(self.crs_table).delete().eq("username", str(username)).execute()
            else:
                res = await self.supabase.table(self.crs_table).delete().eq("user_id", str(rbx_id)).execute()
            
            if not res.data:
                return False

            return True
        except Exception as e:
            logger.exception("remove_criminal_record failed: %s", e)
            return False

    async def _load_from_database(self, message_type: str):
        """Load a specific message type from database"""
        try:
            result = await self.supabase.table(self.wm_table) \
                .select('*') \
                .eq('message_type', message_type) \
                .eq('is_active', True) \
                .limit(1) \
                .execute()
            
            if result and result.data and len(result.data) > 0:
                return {
                    'id': result.data[0]['id'],
                    'message_type': result.data[0]['message_type'],
                    'embeds': result.data[0]['embeds'],
                    'last_updated': result.data[0]['last_updated'],
                    'updated_by': result.data[0]['updated_by'],
                    'version': result.data[0].get('version', 1)
                }
            return None
        except Exception as e:
            logger.error(f"Error loading {message_type} from database: {e}")
            return None
    
    async def update_rank_div(self, user_id, username, rank, division):
        user_id_str = str(user_id)
        update_data = {
                "user_id": user_id_str,
                "username": username,
                "rank": rank,
                "division": division
            }

        try:
            res = await self.supabase.table(self.users_table).upsert(update_data, on_conflict="user_id").execute()
            if res.data:
                async with self._user_cache_lock:
                    if user_id_str not in self._user_cache:
                        self._user_cache[user_id_str] = {}
                    
                    self._user_cache[user_id_str].update(update_data)

                return True
            
            return False
        except Exception as e:
            logger.error(f"update_rank_div failed for %s: %s", user_id_str, e)
            return False

    async def get_welcome_message(self, message_type: str, refresh: bool = False):
        """
        Get a welcome message from cache.
        If refresh=True or not in cache, load from database.
        """
        async with self._welcome_cache_lock:
            if refresh or message_type not in self.welcome_cache:
                data = await self._load_from_database(message_type)
                if data:
                    self.welcome_cache[message_type] = data
            
            return self.welcome_cache.get(message_type)

    async def update_welcome_message(self, message_type: str, embeds_data: list, updated_by: str):
        """
        Update welcome message with multiple embeds.
        """
        async with self._welcome_cache_lock:
            try:
                #Deactive old message
                await self.supabase.table(self.wm_table) \
                    .update({'is_active': False}) \
                    .eq('message_type', message_type) \
                    .eq('is_active', True) \
                    .execute()
                
                
                #Get next version number
                result = await self.supabase.table(self.wm_table) \
                    .select('version') \
                    .eq('message_type', message_type) \
                    .order('version', desc=True) \
                    .limit(1) \
                    .execute()
                
                next_version = 1

                if result and result.data and len(result.data) > 0:
                        next_version = result.data[0]['version'] + 1
                                
                # Insert new active message
                new_message = {
                    'message_type': message_type,
                    'version': next_version,
                    'embeds': embeds_data,
                    'updated_by': updated_by,
                    'is_active': True
                }
                
                result = await self.supabase.table(self.wm_table) \
                    .insert(new_message) \
                    .execute()
                    
                new_message = result.data[0] if result.data else None
                
                if new_message:
                    # Update cache
                    self.welcome_cache[message_type] = {
                        'id': new_message['id'],
                        'message_type': new_message['message_type'],
                        'embeds': new_message['embeds'],
                        'last_updated': new_message['last_updated'],
                        'updated_by': new_message['updated_by'],
                        'version': new_message['version']
                    }
                    
                    logger.info(f"Updated {message_type} welcome message with {len(embeds_data)} embeds")
                    return True, self.welcome_cache[message_type]
                else:
                    logger.error(f"Failed to insert new {message_type} message")
                    return False, None
                    
            except Exception as e:
                logger.error(f"Error updating welcome message {message_type}: {e}")
                return False, None


    async def get_welcome_message_history(self, message_type: str, limit: int = 5):
        """Get historical versions of welcome messages"""
        try:
            res = await self.supabase.table(self.wm_table) \
                .select('*') \
                .eq('message_type', message_type) \
                .order('last_updated', desc=True) \
                .limit(limit) \
                .execute()
            return res.data
        except Exception as e:
            logger.exception("Error getting history for %s: %s", message_type, e)
            return []

    async def create_or_update_user_in_db(
        self,
        discord_id: str,
        username: str,
        guild: discord.Guild | None = None,
        roblox_id: int | None = None,
        xp: int | None = 0
    ) -> bool:
        user_id_str = str(discord_id)
        cleaned_username = clean_nickname(username)

        try:
            res = await self.supabase.table(self.users_table).select('*').eq('user_id', user_id_str).maybe_single().execute()
            
            if res and res.data:
                update_data = {
                    "username": cleaned_username,
                    "roblox_id": roblox_id if roblox_id is not None else res.data.get('roblox_id')
                }
                await self.supabase.table(self.users_table).update(update_data).eq('user_id', user_id_str).execute()
                
                async with self._user_cache_lock:
                    if user_id_str in self._user_cache:
                        self._user_cache[user_id_str].update(update_data)
                    else:
                        full_data = res.data.copy()
                        full_data.update(update_data)
                        self._user_cache[user_id_str] = full_data
            else:
                new_user = {
                    "user_id": user_id_str,
                    "username": cleaned_username,
                    "roblox_id": roblox_id,
                    "xp": xp if xp is not None else 0,
                    "rank": "Unknown",
                    "division": "Unknown"
                }
                await self.supabase.table(self.users_table).insert(new_user).execute()
                logger.info("Created New User: %s (%s), Roblox ID: %s", cleaned_username, user_id_str, roblox_id)

                async with self._user_cache_lock:
                    self._user_cache[user_id_str] = new_user
        
            return True
        except Exception as e:
            logger.exception("Failed to create/update user %s in database: %s", user_id_str, e)
            return False
    
    async def reset_hrs(self) -> bool:
        if not self.supabase:
            return False

        try:            
            res = await self.supabase.table(self.hrs_table).update({
                'tryouts': 0,
                'events': 0,
                'phases': 0,
                'courses': 0,
                'inspections': 0,
                'joint_events': 0
            }).neq('user_id', '0').execute()
            
            if not res.data:
                return False

            async with self._hrs_cache_lock:
                for user_id in self._hrs_cache:
                    self._hrs_cache[user_id].update({
                        'tryouts': 0,
                        'events': 0,
                        'phases': 0,
                        'courses': 0,
                        'inspections': 0,
                        'joint_events': 0
                    })

            logger.info("Successfully reset all HR statistics in database and cache.")
            return True
        except Exception as e:
            logger.exception("reset_hrs failed: %s", e)
            return False

    async def reset_lrs(self) -> bool:
        if not self.supabase:
            return False

        try:    
            res = await self.supabase.table(self.lrs_table).update({
                'activity': 0,
                'time_guarded': 0,
                'events_attended': 0
            }).neq('user_id', '0').execute()
            
            if not res.data:
                return False

            async with self._lrs_cache_lock:
                for user_id in self._lrs_cache:
                    self._lrs_cache[user_id].update({
                        'activity': 0,
                        'time_guarded': 0,
                        'events_attended': 0
                    })
            

            logger.info("Successfully reset all LR statistics in database and cache.")
            return True
        except Exception as e:
            logger.exception("reset_lrs failed: %s", e)
            return False

