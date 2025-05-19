import discord  
import time
from discord.ext import commands
from discord import app_commands
from rate_limiter import RateLimiter
from decorators import has_allowed_role
import aiohttp
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Configuration 
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
BGROUP_IDS = {32578828, 6447250, 4973512, 14286518, 32014700, 15229694, 15224554, 14557406, 14609194, 5029915}
BRITISH_ARMY_GROUP_ID = 4972535
TIMEOUT = aiohttp.ClientTimeout(total=10)
REQUIREMENTS = {'age': 90, 'friends': 7, 'groups': 5, 'badges': 120}
CACHE = {}
CACHE_TTL = 300  # 5 minutes cache
REQUEST_RETRIES = 3
REQUEST_RETRY_DELAY = 1.0  # Base delay in seconds

# Helper functions 
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

async def fetch_with_retry(session: aiohttp.ClientSession, url: str, max_retries: int = REQUEST_RETRIES) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 429:
                    retry_after = float(response.headers.get('Retry-After', REQUEST_RETRY_DELAY))
                    await asyncio.sleep(retry_after)
                    continue
                if response.status == 200:
                    return await response.json()
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                logger.error(f"[API ERROR] {url}: {str(e)}")
                raise
            wait_time = (2 ** attempt) * REQUEST_RETRY_DELAY  # Exponential backoff
            await asyncio.sleep(wait_time)
    return None

async def fetch_with_cache(session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, Any]]:
    cache_key = f"req_{hash(url)}"
    if cache_key in CACHE and (time.time() - CACHE[cache_key]['timestamp']) < CACHE_TTL:
        return CACHE[cache_key]['data']
    try:
        data = await fetch_with_retry(session, url)
        if data:
            CACHE[cache_key] = {'data': data, 'timestamp': time.time()}
            return data
    except Exception as e:
        logger.error(f"[CACHE ERROR] {url}: {str(e)}")
    return None

async def fetch_group_rank(session: aiohttp.ClientSession, user_id: int) -> str:
    url = f"https://groups.roblox.com/v2/users/{user_id}/groups/roles"
    data = await fetch_with_cache(session, url)
    if data and 'data' in data:
        for group in data['data']:
            if group.get('group', {}).get('id') == BRITISH_ARMY_GROUP_ID:
                return group.get('role', {}).get('name', 'Guest')
    return 'Not in Group'

async def fetch_badge_count(session: aiohttp.ClientSession, user_id: int) -> int:
    try:
        badge_count = 0
        cursor = ""
        while True:
            url = f"https://badges.roblox.com/v1/users/{user_id}/badges?limit=100&sortOrder=Asc"
            if cursor:
                url += f"&cursor={cursor}"
            data = await fetch_with_cache(session, url)
            if not data or "data" not in data:
                break
            badge_count += len(data["data"])
            if not data.get("nextPageCursor"):
                break
            cursor = data["nextPageCursor"]
        if badge_count > 0:
            return badge_count
            
        inventory_url = f"https://inventory.roblox.com/v1/users/{user_id}/items/Collectible/1?limit=100"
        inventory_data = await fetch_with_cache(session, inventory_url)
        if inventory_data and "data" in inventory_data:
            return len(inventory_data["data"])
            
        legacy_url = f"https://api.roblox.com/users/{user_id}/badges"
        legacy_data = await fetch_with_cache(session, legacy_url)
        if legacy_data and isinstance(legacy_data, list):
            return len(legacy_data)
        return 0
    except Exception as e:
        logger.error(f"[BADGE COUNT ERROR] {e}")
        return 0

async def safe_followup(interaction: discord.Interaction, *args, **kwargs):
    """Wrapper for followup.send with rate limit handling"""
    try:
        await interaction.client.rate_limiter.wait_if_needed('followup_messages')
        return await interaction.followup.send(*args, **kwargs)
    except discord.errors.HTTPException as e:
        if e.status == 429:
            retry_after = float(e.response.headers.get('Retry-After', REQUEST_RETRY_DELAY))
            await asyncio.sleep(retry_after)
            return await interaction.followup.send(*args, **kwargs)
        raise

def create_sc_command(bot: commands.Bot):
    @bot.tree.command(name="sc", description="Security check a Roblox user")
    @app_commands.describe(user_id="The Roblox user ID to check")
    @has_allowed_role()
    async def sc(interaction: discord.Interaction, user_id: int):
        try:
            # Immediate deferral
            await interaction.response.defer()
            
            # Rate limiting for command execution
            await bot.rate_limiter.wait_if_needed(bucket="sc_command")
            
            if user_id <= 0:
                await safe_followup(
                    interaction,
                    "❌ Invalid Roblox User ID. Please provide a positive number.",
                    ephemeral=True
                )
                return
                
            async with aiohttp.ClientSession(
                headers={"User-Agent": USER_AGENT}, 
                timeout=TIMEOUT,
                connector=aiohttp.TCPConnector(limit_per_host=5)  # Limit concurrent connections
            ) as session:
                urls = {
                    'profile': f"https://users.roblox.com/v1/users/{user_id}",
                    'groups': f"https://groups.roblox.com/v2/users/{user_id}/groups/roles",
                    'friends': f"https://friends.roblox.com/v1/users/{user_id}/friends/count",
                    'avatar': f"https://thumbnails.roblox.com/v1/users/avatar?userIds={user_id}&size=150x150&format=Png"
                }
                
                tasks = {
                    'profile': fetch_with_cache(session, urls['profile']),
                    'groups': fetch_with_cache(session, urls['groups']),
                    'friends': fetch_with_cache(session, urls['friends']),
                    'avatar': fetch_with_cache(session, urls['avatar']),
                    'badges': fetch_badge_count(session, user_id),
                    'rank': fetch_group_rank(session, user_id)
                }
                
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                data = dict(zip(tasks.keys(), results))
                
                if not data['profile'] or isinstance(data['profile'], Exception):
                    embed = discord.Embed(
                        title="🔴 User Not Found",
                        description="The specified Roblox user could not be found.",
                        color=discord.Color.red()
                    )
                    await safe_followup(interaction, embed=embed)
                    return
                
                username = data['profile'].get('name', 'Unknown')
                created_at = datetime.fromisoformat(data['profile']['created'].replace('Z', '+00:00')) if data['profile'].get('created') else None
                age_days = (datetime.now(timezone.utc) - created_at).days if created_at else 0
                
                friends_count = data['friends'].get('count', 0) if data['friends'] and not isinstance(data['friends'], Exception) else 0
                groups_count = len(data['groups'].get('data', [])) if data['groups'] and not isinstance(data['groups'], Exception) else 0
                if isinstance(data['badges'], Exception):
                    badge_count = 0
                    warning = "⚠️ Badges may be private or user has none"
                else:
                    badge_count = data['badges']
                    warning = "⚠️ Badges may be private or user has none" if badge_count == 0 else ""
                british_army_rank = data['rank'] if not isinstance(data['rank'], Exception) else 'Unknown'
                
                metrics = {
                    'age': {
                        'value': age_days,
                        'percentage': min(100, (age_days / REQUIREMENTS['age']) * 100),
                        'meets_req': age_days >= REQUIREMENTS['age']
                    },
                    'friends': {
                        'value': friends_count,
                        'percentage': min(100, (friends_count / REQUIREMENTS['friends']) * 100),
                        'meets_req': friends_count >= REQUIREMENTS['friends']
                    },
                    'groups': {
                        'value': groups_count,
                        'percentage': min(100, (groups_count / REQUIREMENTS['groups']) * 100),
                        'meets_req': groups_count >= REQUIREMENTS['groups']
                    },
                    'badges': {
                        'value': badge_count,
                        'percentage': min(100, (badge_count / max(1, REQUIREMENTS['badges'])) * 100),
                        'meets_req': badge_count >= REQUIREMENTS['badges']
                    }
                }
                
                banned_groups = []
                if data['groups'] and not isinstance(data['groups'], Exception):
                    banned_groups = [
                        f"• {group['group']['name']}" 
                        for group in data['groups'].get('data', []) 
                        if group and group['group']['id'] in BGROUP_IDS
                    ]
                
                emoji_map = {
                    "age": "📅",
                    "friends": "👥",
                    "groups": "🏘️",
                    "badges": "🎖️"
                }

                embed = discord.Embed(
                    title=f"{widen_text(username)}",
                    url=f"https://www.roblox.com/users/{user_id}/profile",
                    color=discord.Color.red() if banned_groups else discord.Color.green(),
                    description=(
                        f"📅 **Created Account:** (<t:{int(created_at.timestamp())}:R>)\n"
                        f"🎖️ **British Army Rank:** {british_army_rank}"
                    ),
                    timestamp=datetime.utcnow()
                )

                if data['avatar'] and not isinstance(data['avatar'], Exception) and data['avatar'].get('data'):
                    embed.set_thumbnail(url=data['avatar']['data'][0]['imageUrl'])

                for name, metric in metrics.items():
                    emoji = emoji_map.get(name.lower(), "")
                    status_icon = "✅" if metric['meets_req'] else "❌"
                    progress_bar = create_progress_bar(metric['percentage'], metric['meets_req'])

                    field_value = f"{progress_bar} {metric['value']}/{REQUIREMENTS[name]}\nProgress: {round(metric['percentage'])}%"
                    if name == 'badges' and warning:
                        field_value += f"\n{warning}"
                    
                    embed.add_field(
                        name=f"{emoji} {name.capitalize()} {status_icon}",
                        value=field_value,
                        inline=True
                    )

                if banned_groups:
                    embed.add_field(
                        name="🚨 Banned/Main Groups Detected",
                        value="\n".join(banned_groups),
                        inline=False
                    )
                else:
                    embed.add_field(
                        name="✅",
                        value="User is not in any banned groups or main regiments.",
                        inline=True
                    )
                            
                embed.set_author(
                    name=f"Roblox User Check • {username}",
                    icon_url=interaction.user.avatar.url if interaction.user.avatar else None
                )
                embed.set_footer(text=f"Requested by {interaction.user.display_name} | Roblox User ID: {user_id}")
                await safe_followup(interaction, embed=embed)

        except app_commands.CommandOnCooldown as e:
            await safe_followup(
                interaction,
                f"⌛ Command on cooldown. Try again in {e.retry_after:.1f}s",
                ephemeral=True
            )
            
        except Exception as e:
            logger.error(f"[SC COMMAND ERROR]: {str(e)}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "⚠️ An error occurred while processing your request.",
                    ephemeral=True
                )
            else:
                await safe_followup(
                    interaction,
                    "⚠️ An error occurred while processing your request.",
                    ephemeral=True
                )
    
    return sc
