import discord
import time
import random
from discord.ext import commands
from discord import app_commands
import urllib.parse
import aiohttp
import asyncio
from datetime import datetime, timezone
import socket
import aiodns
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
TIMEOUT = aiohttp.ClientTimeout(total=10)
BGROUP_IDS = {32578828, 6447250, 4973512, 14286518, 32014700, 15229694, 15224554, 14557406, 14609194, 5029915}
BRITISH_ARMY_GROUP_ID = 4972535
REQUIREMENTS = {'age': 90, 'friends': 7, 'groups': 5, 'badges': 120}
CACHE = {}
CACHE_TTL = 300  # 5 minutes cache
BAD_REQUEST_CACHE_TTL = 60  # 1 minute for failed requests
REQUEST_RETRIES = 3
REQUEST_RETRY_DELAY = 1.0
REQUEST_TIMEOUT = 10
MAX_PAGES = 3

# Global concurrency limiter
MAX_CONCURRENT_REQUESTS = 5
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

ROBLOX_API_IPS = {
    'api.roblox.com': '172.67.209.252',
    'www.roblox.com': '172.67.209.252',
    'groups.roblox.com': '172.67.209.252',
    'friends.roblox.com': '172.67.209.252',
    'thumbnails.roblox.com': '172.67.209.252',
    'accountinformation.roblox.com': '172.67.209.252',
    'badges.roblox.com': '172.67.209.252'
}

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json"
}

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

async def fetch_group_rank(session: aiohttp.ClientSession, user_id: int) -> str:
    url = f"https://groups.roblox.com/v2/users/{user_id}/groups/roles"
    data = await fetch_with_cache(session, url)
    if data and 'data' in data:
        for group in data['data']:
            if group.get('group', {}).get('id') == BRITISH_ARMY_GROUP_ID:
                return group.get('role', {}).get('name', 'Guest')
    return 'Not in Group'

async def fetch_with_retry(session: aiohttp.ClientSession, url: str, max_retries: int = 3):
    last_error = None
    for attempt in range(max_retries):
        try:
            try:
                return await _fetch_url(session, url)
            except Exception as e:
                if any(err in str(e).lower() for err in ["dns", "name resolution"]):
                    parsed = urllib.parse.urlparse(url)
                    if parsed.hostname in ROBLOX_API_IPS:
                        ip_url = url.replace(
                            f"{parsed.scheme}://{parsed.hostname}",
                            f"{parsed.scheme}://{ROBLOX_API_IPS[parsed.hostname]}"
                        )
                        headers = {'Host': parsed.hostname}
                        headers.update(DEFAULT_HEADERS)
                        return await _fetch_url(session, ip_url, headers=headers)
                raise
        except Exception as e:
            last_error = e
            wait_time = (2 ** attempt) * (0.5 + random.random())
            await asyncio.sleep(wait_time)
    
    logger.error(f"Max retries exceeded for {url}: {str(last_error)}")
    return None

async def _fetch_url(session, url, headers=None):
    async with request_semaphore:
        try:
            final_headers = DEFAULT_HEADERS.copy()
            if headers:
                final_headers.update(headers)
                
            async with session.get(
                url, 
                headers=final_headers,
                raise_for_status=True
            ) as response:
                return await response.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                logger.warning(f"Roblox API rejected request to {url} - may require authentication")
            raise
        except Exception as e:
            logger.error(f"Request to {url} failed: {str(e)}")
            raise

async def fetch_with_cache(session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, Any]]:
    cache_key = f"req_{hash(url)}"
    if cache_key in CACHE:
        cache_data = CACHE[cache_key]
        if time.time() - cache_data['timestamp'] < (BAD_REQUEST_CACHE_TTL if cache_data.get('error') else CACHE_TTL):
            if cache_data.get('error'):
                raise Exception("Cached error")
            return cache_data['data']
        del CACHE[cache_key]

    try:
        data = await fetch_with_retry(session, url)
        if data:
            CACHE[cache_key] = {'data': data, 'timestamp': time.time()}
            return data
    except Exception as e:
        logger.error(f"Cache failed for {url}: {str(e)}")
        CACHE[cache_key] = {'timestamp': time.time(), 'error': True}
        raise

async def fetch_badge_count(session: aiohttp.ClientSession, user_id: int) -> int:
    endpoints = [
        # Primary endpoint - badges API
        lambda: f"https://badges.roblox.com/v1/users/{user_id}/badges?limit=100",
        # Fallback endpoint - account info API
        lambda: f"https://accountinformation.roblox.com/v1/users/{user_id}/roblox-badges",
        # Legacy endpoint
        lambda: f"https://api.roblox.com/users/{user_id}/badges"
    ]
    
    for endpoint in endpoints:
        url = endpoint()
        try:
            data = await fetch_with_cache(session, url)
            if data:
                if isinstance(data, list):
                    return len(data)
                if isinstance(data.get("data"), list):
                    return len(data["data"])
                if isinstance(data.get("robloxBadges"), list):  # Handle account info format
                    return len(data["robloxBadges"])
        except Exception as e:
            logger.warning(f"Badge endpoint {url} failed: {str(e)}")
            continue

    logger.error("All badge endpoints failed")
    return 0

def create_sc_command(bot: commands.Bot):
    @bot.tree.command(name="sc", description="Security check a Roblox user")
    @app_commands.describe(user_id="The Roblox user ID to check")
    @app_commands.checks.cooldown(rate=1, per=10.0)
    async def sc(interaction: discord.Interaction, user_id: int):
        try:
            
            session = bot.shared_session if hasattr(bot, 'shared_session') else None
            if not session:
                return await interaction.response.send_message(
                    "⚠️ Bot is not ready yet. Please try again in a moment.",
                    ephemeral=True
                )
                
            #Deferral
            await interaction.response.defer(thinking=True)

            urls = {
                'profile': f"https://users.roblox.com/v1/users/{user_id}",
                'groups': f"https://groups.roblox.com/v2/users/{user_id}/groups/roles",
                'friends': f"https://friends.roblox.com/v1/users/{user_id}/friends/count",
                'avatar': f"https://thumbnails.roblox.com/v1/users/avatar?userIds={user_id}&size=150x150&format=Png"
            }

            try:
                tasks = {
                    'profile': fetch_with_cache(session, urls['profile']),
                    'groups': fetch_with_cache(session, urls['groups']),
                    'friends': fetch_with_cache(session, urls['friends']),
                    'avatar': fetch_with_cache(session, urls['avatar']),
                    'badges': fetch_badge_count(session, user_id),
                    'rank': fetch_group_rank(session, user_id)
                }

                results = await asyncio.wait_for(
                    asyncio.gather(*tasks.values(), return_exceptions=True),
                    timeout=25.0
                )
                data = dict(zip(tasks.keys(), results))
            except asyncio.TimeoutError:
                return await interaction.followup.send(
                    "⌛ Command timed out while fetching Roblox data. Please try again.",
                    ephemeral=True
                )

            if not data['profile'] or isinstance(data['profile'], Exception):
                embed = discord.Embed(
                    title="🔴 User Not Found",
                    description="The specified Roblox user could not be found.",
                    color=discord.Color.red()
                )
                return await interaction.followup.send(embed=embed)

            username = data['profile'].get('name', 'Unknown')
            created_at = datetime.fromisoformat(data['profile']['created'].replace('Z', '+00:00')) if data['profile'].get('created') else None
            age_days = (datetime.now(timezone.utc) - created_at).days if created_at else 0

            friends_count = data['friends'].get('count', 0) if data['friends'] and not isinstance(data['friends'], Exception) else 0
            groups_count = len(data['groups'].get('data', [])) if data['groups'] and not isinstance(data['groups'], Exception) else 0
            badge_count = data['badges'] if not isinstance(data['badges'], Exception) else 0
            warning = "⚠️ Could not verify badges (API limits)" if isinstance(data['badges'], Exception) else (
                "⚠️ Badges may be private or user has none" if badge_count == 0 else "")
            british_army_rank = data['rank'] if not isinstance(data['rank'], Exception) else 'Unknown'

            metrics = {
                'age': {'value': age_days, 'percentage': min(100, (age_days / REQUIREMENTS['age']) * 100), 'meets_req': age_days >= REQUIREMENTS['age']},
                'friends': {'value': friends_count, 'percentage': min(100, (friends_count / REQUIREMENTS['friends']) * 100), 'meets_req': friends_count >= REQUIREMENTS['friends']},
                'groups': {'value': groups_count, 'percentage': min(100, (groups_count / REQUIREMENTS['groups']) * 100), 'meets_req': groups_count >= REQUIREMENTS['groups']},
                'badges': {'value': badge_count, 'percentage': min(100, (badge_count / max(1, REQUIREMENTS['badges'])) * 100), 'meets_req': badge_count >= REQUIREMENTS['badges']}
            }

            banned_groups = []
            if data['groups'] and not isinstance(data['groups'], Exception):
                banned_groups = [
                    f"• {group['group']['name']}" 
                    for group in data['groups'].get('data', []) 
                    if group and group['group']['id'] in BGROUP_IDS
                ]

            embed = discord.Embed(
                title=f"{widen_text(username)}",
                url=f"https://www.roblox.com/users/{user_id}/profile",
                color=discord.Color.red() if banned_groups else discord.Color.green(),
                description=f"📅 **Created Account:** <t:{int(created_at.timestamp())}:R>\n🎖️ **British Army Rank:** {british_army_rank}",
                timestamp=datetime.utcnow()
            )

            if data['avatar'] and not isinstance(data['avatar'], Exception) and data['avatar'].get('data'):
                embed.set_thumbnail(url=data['avatar']['data'][0]['imageUrl'])

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
                name=f"Roblox User Check • {username}",
                icon_url=interaction.user.avatar.url if interaction.user.avatar else None
            )
            embed.set_footer(text=f"Requested by {interaction.user.display_name} | Roblox User ID: {user_id}")

            await interaction.followup.send(embed=embed)
    

        except Exception as e:
            logger.error(f"[SC COMMAND ERROR]: {str(e)}", exc_info=True)
            try:
                await interaction.followup.send(
                    "⚠️ An error occurred while processing your request.",
                    ephemeral=True
                )
            except:
                pass
    @sc.error
    async def sc_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CommandOnCooldown):
            # Send the cooldown message as a new response
            if interaction.response.is_done():
                await interaction.followup.send(
                    f"⌛ Command on cooldown. Try again in {error.retry_after:.1f}s",
                    ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    f"⌛ Command on cooldown. Try again in {error.retry_after:.1f}s",
                    ephemeral=True
                )
        else:
            logger.error("Unhandled error in sc command:", exc_info=error)
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "⚠️ An unexpected error occurred.",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        "⚠️ An unexpected error occurred.",
                        ephemeral=True
                    )
            except:
                pass
    
    return sc
