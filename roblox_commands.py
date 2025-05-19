import discord  
import time
import random
from discord.ext import commands
from discord import app_commands
from rate_limiter import RateLimiter
from decorators import has_allowed_role
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
CACHE_TTL = 300
REQUEST_RETRIES = 3
REQUEST_RETRY_DELAY = 1.0
REQUEST_TIMEOUT = 10  # seconds
MAX_PAGES = 3  # For paginated endpoints
DNS_RETRIES = 2

# Global concurrency limiter
MAX_CONCURRENT_REQUESTS = 5
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def check_dns_connectivity():
    """Improved DNS resolution with multiple fallback methods"""
    domains = ['api.roblox.com', 'www.roblox.com']
    try:
        # Method 1: Try aiodns first
        try:
            resolver = aiodns.DNSResolver(loop=asyncio.get_running_loop())
            await asyncio.gather(*[resolver.query(domain, 'A') for domain in domains])
            return True
        except Exception:
            # Method 2: Fallback to socket (sync in thread)
            def sync_resolve():
                try:
                    for domain in domains:
                        socket.gethostbyname(domain)
                    return True
                except socket.gaierror:
                    return False
            
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, sync_resolve)
    except Exception as e:
        logger.error(f"[DNS ERROR] All resolution methods failed: {str(e)}")
        return False

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

async def fetch_with_retry(session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    last_error = None
    for attempt in range(max_retries):
        try:
            # Verify DNS connectivity first
            if not await check_dns_connectivity():
                raise Exception("DNS resolution failed")
                
            async with request_semaphore:
                async with session.get(url, headers={"User-Agent": USER_AGENT}) as response:
                    if response.status == 429:
                        retry_after = float(response.headers.get('Retry-After', 1.0)) * (1 + random.random())
                        logger.warning(f"Rate limited on {url}, retrying in {retry_after:.2f}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    if response.status != 200:
                        logger.warning(f"Non-200 status {response.status} for {url}")
                        return None
                        
                    return await response.json()
                    
        except Exception as e:
            last_error = e
            wait_time = (2 ** attempt) * (0.5 + random.random())
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            await asyncio.sleep(wait_time)
    
    logger.error(f"Max retries exceeded for {url}: {str(last_error)}")
    return None

async def fetch_with_cache(session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, Any]]:
    cache_key = f"req_{hash(url)}"
    if cache_key in CACHE:
        if time.time() - CACHE[cache_key]['timestamp'] < CACHE_TTL:
            return CACHE[cache_key]['data']
        del CACHE[cache_key]

    try:
        data = await fetch_with_retry(session, url)
        if data:
            CACHE[cache_key] = {'data': data, 'timestamp': time.time()}
            return data
    except Exception as e:
        logger.error(f"Cache failed for {url}: {str(e)}")
    return None

async def fetch_badge_count(session: aiohttp.ClientSession, user_id: int) -> int:
    endpoints = [
        lambda: f"https://badges.roblox.com/v1/users/{user_id}/badges?limit=100",
        lambda: f"https://inventory.roblox.com/v1/users/{user_id}/items/Collectible/1?limit=100",
        lambda: f"https://accountinformation.roblox.com/v1/users/{user_id}/roblox-badges",
        lambda: f"https://api.roblox.com/users/{user_id}/badges"
    ]

    for endpoint in endpoints:
        url = endpoint()
        try:
            if "badges?" in url:
                badge_count = 0
                cursor = ""
                for _ in range(3):
                    current_url = f"{url}&cursor={cursor}" if cursor else url
                    data = await fetch_with_cache(session, current_url)
                    if not data or not data.get("data"):
                        break
                    badge_count += len(data["data"])
                    cursor = data.get("nextPageCursor")
                    if not cursor:
                        break
                if badge_count > 0:
                    return badge_count
            else:
                data = await fetch_with_cache(session, url)
                if data:
                    if isinstance(data, list):
                        return len(data)
                    if isinstance(data.get("data"), list):
                        return len(data["data"])
        except Exception as e:
            logger.warning(f"Badge endpoint {url} failed: {str(e)}")
            continue

    logger.error("All badge endpoints failed")
    return 0  

async def safe_followup(interaction: discord.Interaction, *args, **kwargs):
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
    @app_commands.checks.cooldown(rate=1, per=10.0)
    @has_allowed_role()
    async def sc(interaction: discord.Interaction, user_id: int):
        try:
            # Defer immediately to prevent timeout
            await interaction.response.defer(thinking=True)
            
            # Validate user ID
            if user_id <= 0:
                return await interaction.followup.send(
                    "❌ Invalid Roblox User ID. Please provide a positive number.",
                    ephemeral=True
                )

            # Get shared session
            session = bot.shared_session if hasattr(bot, 'shared_session') else None
            if not session:
                return await interaction.followup.send(
                    "⚠️ Bot is not ready yet. Please try again in a moment.",
                    ephemeral=True
                )

            # API endpoints
            urls = {
                'profile': f"https://users.roblox.com/v1/users/{user_id}",
                'groups': f"https://groups.roblox.com/v2/users/{user_id}/groups/roles",
                'friends': f"https://friends.roblox.com/v1/users/{user_id}/friends/count",
                'avatar': f"https://thumbnails.roblox.com/v1/users/avatar?userIds={user_id}&size=150x150&format=Png"
            }

            # Fetch all data with timeout
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

            # Handle user not found
            if not data['profile'] or isinstance(data['profile'], Exception):
                embed = discord.Embed(
                    title="🔴 User Not Found",
                    description="The specified Roblox user could not be found.",
                    color=discord.Color.red()
                )
                return await interaction.followup.send(embed=embed)

            # Process data
            username = data['profile'].get('name', 'Unknown')
            created_at = datetime.fromisoformat(data['profile']['created'].replace('Z', '+00:00')) if data['profile'].get('created') else None
            age_days = (datetime.now(timezone.utc) - created_at).days if created_at else 0

            friends_count = data['friends'].get('count', 0) if data['friends'] and not isinstance(data['friends'], Exception) else 0
            groups_count = len(data['groups'].get('data', [])) if data['groups'] and not isinstance(data['groups'], Exception) else 0
            badge_count = data['badges'] if not isinstance(data['badges'], Exception) else 0
            warning = "⚠️ Badges may be private or user has none" if badge_count == 0 else ""
            british_army_rank = data['rank'] if not isinstance(data['rank'], Exception) else 'Unknown'

            # Prepare metrics
            metrics = {
                'age': {'value': age_days, 'percentage': min(100, (age_days / REQUIREMENTS['age']) * 100), 'meets_req': age_days >= REQUIREMENTS['age']},
                'friends': {'value': friends_count, 'percentage': min(100, (friends_count / REQUIREMENTS['friends']) * 100), 'meets_req': friends_count >= REQUIREMENTS['friends']},
                'groups': {'value': groups_count, 'percentage': min(100, (groups_count / REQUIREMENTS['groups']) * 100), 'meets_req': groups_count >= REQUIREMENTS['groups']},
                'badges': {'value': badge_count, 'percentage': min(100, (badge_count / max(1, REQUIREMENTS['badges'])) * 100), 'meets_req': badge_count >= REQUIREMENTS['badges']}
            }

            # Check banned groups
            banned_groups = []
            if data['groups'] and not isinstance(data['groups'], Exception):
                banned_groups = [
                    f"• {group['group']['name']}" 
                    for group in data['groups'].get('data', []) 
                    if group and group['group']['id'] in BGROUP_IDS
                ]

            # Create embed
            embed = discord.Embed(
                title=f"{widen_text(username)}",
                url=f"https://www.roblox.com/users/{user_id}/profile",
                color=discord.Color.red() if banned_groups else discord.Color.green(),
                description=f"📅 **Created Account:** <t:{int(created_at.timestamp())}:R>\n🎖️ **British Army Rank:** {british_army_rank}",
                timestamp=datetime.utcnow()
            )

            # Add thumbnail if available
            if data['avatar'] and not isinstance(data['avatar'], Exception) and data['avatar'].get('data'):
                embed.set_thumbnail(url=data['avatar']['data'][0]['imageUrl'])

            # Add metrics fields
            emoji_map = {"age": "📅", "friends": "👥", "groups": "🏘️", "badges": "🎖️"}
            for name, metric in metrics.items():
                emoji = emoji_map.get(name.lower(), "")
                status_icon = "✅" if metric['meets_req'] else "❌"
                progress_bar = create_progress_bar(metric['percentage'], metric['meets_req'])

                field_value = f"{progress_bar} {metric['value']}/{REQUIREMENTS[name]}\nProgress: {round(metric['percentage'])}%"
                if name == 'badges' and warning:
                    field_value += f"\n{warning}"

                embed.add_field(name=f"{emoji} {name.capitalize()} {status_icon}", value=field_value, inline=True)

            # Add banned groups if found
            if banned_groups:
                embed.add_field(name="🚨 Banned/Main Groups Detected", value="\n".join(banned_groups), inline=False)
            else:
                embed.add_field(name="✅", value="User is not in any banned groups or main regiments.", inline=True)

            # Set author and footer
            embed.set_author(
                name=f"Roblox User Check • {username}",
                icon_url=interaction.user.avatar.url if interaction.user.avatar else None
            )
            embed.set_footer(text=f"Requested by {interaction.user.display_name} | Roblox User ID: {user_id}")

            # Send final response
            await interaction.followup.send(embed=embed)

        except app_commands.CommandOnCooldown as e:
            await interaction.followup.send(
                f"⌛ Command on cooldown. Try again in {e.retry_after:.1f}s",
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"[SC COMMAND ERROR]: {str(e)}", exc_info=True)
            try:
                await interaction.followup.send(
                    "⚠️ An error occurred while processing your request.",
                    ephemeral=True
                )
            except:
                pass  # If all response methods fail

    return sc
