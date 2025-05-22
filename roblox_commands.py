import discord
import time
import random
from discord.ext import commands
from discord import app_commands
import urllib.parse
import aiohttp
import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional
from aiohttp.resolver import AsyncResolver

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)

logger = logging.getLogger(__name__)
logger.info("Main logger configured")

# Configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
TIMEOUT = aiohttp.ClientTimeout(total=10)
BGROUP_IDS = {32578828, 6447250, 4973512, 14286518, 32014700, 15229694, 15224554, 14557406, 14609194, 5029915}
BRITISH_ARMY_GROUP_ID = 4972535
REQUIREMENTS = {'age': 90, 'friends': 7, 'groups': 5, 'badges': 120}
MAX_CONCURRENT_REQUESTS = 5
REQUEST_RETRIES = 3
REQUEST_RETRY_DELAY = 1.0

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json"
}

request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def widen_text(text: str) -> str:
    return text.upper().translate(
        str.maketrans(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789! ',
            'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９！　'
        )
    )

def create_progress_bar(percentage: float, meets_req: bool) -> str:
    filled = min(10, round(percentage / 10))
    return (("🟩" if meets_req else "🟥") * filled) + ("⬜" * (10 - filled))

async def create_session():
    return aiohttp.ClientSession(
        timeout=TIMEOUT,
        headers=DEFAULT_HEADERS,
        connector=aiohttp.TCPConnector(
            resolver=AsyncResolver(),
            force_close=True,
            enable_cleanup_closed=True
        )
    )

async def fetch_group_rank(session: aiohttp.ClientSession, user_id: int) -> str:
    url = f"https://groups.roblox.com/v2/users/{user_id}/groups/roles"
    try:
        data = await fetch_with_retry(session, url)
        if data and 'data' in data:
            for group in data['data']:
                if group.get('group', {}).get('id') == BRITISH_ARMY_GROUP_ID:
                    return group.get('role', {}).get('name', 'Guest')
        return 'Not in Group'
    except Exception as e:
        logger.warning(f"Failed to fetch group rank for user {user_id}: {str(e)}")
        return 'Unknown'

async def _fetch_url(session, url):
    async with request_semaphore:
        try:
            async with session.get(url, headers=DEFAULT_HEADERS) as response:
                if response.status == 404:
                    return None  # User doesn't exist or endpoint not found
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                logger.warning(f"Roblox API rejected request to {url} - may require authentication")
            elif e.status == 429:
                logger.warning(f"Rate limited on {url}")
                await asyncio.sleep(10)  # Longer wait for rate limits
            raise
        except Exception as e:
            logger.error(f"Request to {url} failed: {str(e)}")
            raise

async def fetch_with_retry(session: aiohttp.ClientSession, url: str) -> Any:
    last_error = None
    for attempt in range(REQUEST_RETRIES):
        try:
            return await _fetch_url(session, url)
        except aiohttp.ClientConnectorError as e:
            if "DNS" in str(e):
                logger.warning(f"DNS resolution failed for {url}, attempt {attempt+1}")
                await asyncio.sleep(2 ** attempt)
                last_error = e
                continue
            raise
        except Exception as e:
            last_error = e
            await asyncio.sleep(REQUEST_RETRY_DELAY * (attempt + 1))
            continue
    
    logger.error(f"Max retries exceeded for {url}: {str(last_error)}")
    raise last_error if last_error else Exception(f"Failed to fetch {url}")

async def fetch_badge_count(session: aiohttp.ClientSession, user_id: int) -> int:
    endpoints = [
        (f"https://badges.roblox.com/v1/users/{user_id}/badges", "data"),
        (f"https://accountinformation.roblox.com/v1/users/{user_id}/roblox-badges", "robloxBadges"),
    ]
    
    last_valid_count = 0
    
    for url, data_key in endpoints:
        try:
            data = await fetch_with_retry(session, url)
            if data:
                if data_key and data_key in data and isinstance(data[data_key], list):
                    count = len(data[data_key])
                    if count > last_valid_count:
                        last_valid_count = count
                elif isinstance(data, list):
                    count = len(data)
                    if count > last_valid_count:
                        last_valid_count = count
        except Exception as e:
            logger.debug(f"Badge endpoint {url} failed: {str(e)}")
            continue

    if last_valid_count == 0:
        logger.warning(f"All badge endpoints failed for user {user_id}")
        return -1  # Special value indicating failure
    
    return last_valid_count

def create_sc_command(bot: commands.Bot):
    @bot.tree.command(name="sc", description="Security check a Roblox user")
    @app_commands.describe(user_id="The Roblox user ID to check")
    @app_commands.checks.cooldown(rate=1, per=10.0)
    async def sc(interaction: discord.Interaction, user_id: int):
        try:
            # Create fresh session for this command
            async with await create_session() as session:
                try:
                    await interaction.response.defer(thinking=True)
                except discord.errors.NotFound:
                    logger.warning("Interaction timed out before response")
                    return

                # Fetch all data in parallel
                try:
                    profile, groups, friends, avatar, badges, rank = await asyncio.gather(
                        fetch_with_retry(session, f"https://users.roblox.com/v1/users/{user_id}"),
                        fetch_with_retry(session, f"https://groups.roblox.com/v2/users/{user_id}/groups/roles"),
                        fetch_with_retry(session, f"https://friends.roblox.com/v1/users/{user_id}/friends/count"),
                        fetch_with_retry(session, f"https://thumbnails.roblox.com/v1/users/avatar?userIds={user_id}&size=150x150&format=Png"),
                        fetch_badge_count(session, user_id),
                        fetch_group_rank(session, user_id),
                        return_exceptions=True
                    )
                except asyncio.TimeoutError:
                    return await interaction.followup.send(
                        "⌛ Command timed out while fetching Roblox data",
                        ephemeral=True
                    )

                # Process results - special handling for user ID 1 (Roblox)
                if user_id == 1:
                    profile = {'name': 'Roblox', 'created': '2006-01-01T00:00:00Z'}  # Fake profile for testing
                    created_at = datetime.fromisoformat(profile['created'].replace('Z', '+00:00'))
                    age_days = (datetime.now(timezone.utc) - created_at).days
                    friends_count = 0
                    groups_count = 0
                    badge_count = 0
                    british_army_rank = 'Not in Group'
                    warning = ""
                elif profile is None or isinstance(profile, Exception) or not profile.get('name'):
                    embed = discord.Embed(
                        title="🔴 User Not Found",
                        description="The specified Roblox user could not be found.",
                        color=discord.Color.red()
                    )
                    return await interaction.followup.send(embed=embed)
                else:
                    username = profile.get('name', 'Unknown')
                    created_at = datetime.fromisoformat(profile['created'].replace('Z', '+00:00')) if profile.get('created') else None
                    age_days = (datetime.now(timezone.utc) - created_at).days if created_at else 0

                    friends_count = friends.get('count', 0) if not isinstance(friends, Exception) else 0
                    groups_count = len(groups.get('data', [])) if not isinstance(groups, Exception) else 0
                    badge_count = badges if not isinstance(badges, Exception) and badges != -1 else 0
                    warning = "⚠️ Could not verify badges" if isinstance(badges, Exception) or badges == -1 else ""
                    british_army_rank = rank if not isinstance(rank, Exception) else 'Unknown'

                metrics = {
                    'age': {'value': age_days, 'percentage': min(100, (age_days / REQUIREMENTS['age']) * 100), 'meets_req': age_days >= REQUIREMENTS['age']},
                    'friends': {'value': friends_count, 'percentage': min(100, (friends_count / REQUIREMENTS['friends']) * 100), 'meets_req': friends_count >= REQUIREMENTS['friends']},
                    'groups': {'value': groups_count, 'percentage': min(100, (groups_count / REQUIREMENTS['groups']) * 100), 'meets_req': groups_count >= REQUIREMENTS['groups']},
                    'badges': {'value': badge_count, 'percentage': min(100, (badge_count / max(1, REQUIREMENTS['badges'])) * 100), 'meets_req': badge_count >= REQUIREMENTS['badges']}
                }

                banned_groups = []
                if not isinstance(groups, Exception) and groups and isinstance(groups, dict):
                    banned_groups = [
                        f"• {group['group']['name']}" 
                        for group in groups.get('data', []) 
                        if group and group.get('group', {}).get('id') in BGROUP_IDS
                    ]

                embed = discord.Embed(
                    title=f"{widen_text(profile.get('name', 'Unknown'))}",
                    url=f"https://www.roblox.com/users/{user_id}/profile",
                    color=discord.Color.red() if banned_groups else discord.Color.green(),
                    description=f"📅 **Created Account:** <t:{int(created_at.timestamp())}:R>\n🎖️ **British Army Rank:** {british_army_rank}",
                    timestamp=datetime.utcnow()
                )

                if not isinstance(avatar, Exception) and avatar and avatar.get('data'):
                    embed.set_thumbnail(url=avatar['data'][0]['imageUrl'])

                emoji_map = {"age": "📅", "friends": "👥", "groups": "🏘️", "badges": "🎖️"}
                for name, metric in metrics.items():
                    emoji = emoji_map.get(name.lower(), "")
                    status_icon = "✅" if metric['meets_req'] else "❌"
                    progress_bar = create_progress_bar(metric['percentage'], metric['meets_req'])

                    field_value = f"{progress_bar} {metric['value']}/{REQUIREMENTS[name]}\nProgress: {round(metric['percentage'])}%"
                    if name == 'badges' and warning:
                        field_value += f"\n{warning}"

                    embed.add_field(name=f"{emoji} {name.capitalize()} {status_icon}", value=field_value, inline=True)

                if banned_groups:
                    embed.add_field(name="🚨 Banned/Main Groups Detected", value="\n".join(banned_groups), inline=False)
                else:
                    embed.add_field(name="✅", value="User is not in any banned groups or main regiments.", inline=True)

                embed.set_author(
                    name=f"Roblox User Check • {profile.get('name', 'Unknown')}",
                    icon_url=interaction.user.avatar.url if interaction.user.avatar else None
                )
                embed.set_footer(text=f"Requested by {interaction.user.display_name} | Roblox User ID: {user_id}")

                await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"[SC COMMAND ERROR]: {str(e)}", exc_info=True)
            try:
                await interaction.followup.send(
                    "⚠️ An error occurred while checking this user. Roblox's APIs may be experiencing issues.",
                    ephemeral=True
                )
            except:
                pass  # Ignore followup errors if interaction is already dead

    @sc.error
    async def sc_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CommandOnCooldown):
            await interaction.response.send_message(
                f"⌛ Command on cooldown. Try again in {error.retry_after:.1f}s",
                ephemeral=True
            )
        else:
            logger.error("Unhandled error in sc command:", exc_info=error)
            try:
                await interaction.response.send_message(
                    "⚠️ An unexpected error occurred.",
                    ephemeral=True
                )
            except:
                pass  # Ignore if interaction is already dead
    
    return sc
