import aiohttp
import asyncio
import logging

from config import Config
from utils.helpers import clean_nickname
from typing import Optional

logger = logging.getLogger(__name__)

class RobloxAPI:
    def __init__(self, bot):
        self.bot = bot

    async def get_user_id(self, username: str) -> Optional[int]:
        """Get Roblox user ID from username using shared session"""
        try:
            async with self.bot.rate_limiter:  
                async with self.bot.shared_session.get(
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

    async def get_group_rank(self, user_id: int, group_id: int = Config.RMP) -> Optional[str]:
        """Get user's rank in specific group using shared session"""
        try:
            async with self.bot.rate_limiter:  # Use global rate limiter
                async with self.bot.shared_session.get(
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

    async def get_roblox_rank(self, discord_name: str, group_id: int = 4972920) -> str:
        """Get Roblox rank from Discord display name with better error handling"""
        try:
            # Clean the nickname and extract potential Roblox username
            cleaned_name = clean_nickname(discord_name)
            
            if not cleaned_name or cleaned_name.lower() == "unknown":
                logger.warning(f"Invalid Discord name for Roblox lookup: {discord_name}")
                return "Invalid Name"
            
            # Try to find Roblox user ID
            user_id = await self.get_user_id(cleaned_name)
            if not user_id:
                logger.warning(f"Roblox user not found for Discord name: {cleaned_name}")
                return "Not Found"
            
            # Get group rank with retry logic for rate limits
            rank = await self.get_group_rank(user_id, group_id)
            if not rank:
                logger.info(f"User {cleaned_name} (ID: {user_id}) has no specific rank in group")
                return "Member"  # Default to "Member" if no specific rank found
            
            return rank
            
        except Exception as e:
            logger.error(f"Failed to get Roblox rank for {discord_name}: {e}")
            return "Error"


    async def get_roblox_id_from_username(self, username: str) -> Optional[int]:
        """
        Get Roblox ID from username using the Roblox API.
        Returns: Roblox ID (int) or None if not found
        """
        
        try:
            url = "https://users.roblox.com/v1/usernames/users"
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': Config.USER_AGENT
            }
            
            payload = {
                "usernames": [username],
                "excludeBannedUsers": False
            }
            
            async with self.bot.shared_session.post(url, json=payload, headers=headers, timeout=15) as response:
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


