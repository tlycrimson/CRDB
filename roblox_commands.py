import time
from decorators import has_allowed_role
import discord
import aiohttp 
from datetime import datetime
from datetime import timezone
from discord.ext import commands
import asyncio
from typing import Optional, Dict, Any

# Configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
BGROUP_IDS = {32578828, 4219097, 6447250, 4973512, 14286518, 32014700, 15229694, 15224554, 14557406, 14609194, 5029915}  # Using set for faster lookups
BRITISH_ARMY_GROUP_ID = 4972535
TIMEOUT = aiohttp.ClientTimeout(total=10)
REQUIREMENTS = {'age': 90, 'friends': 8, 'groups': 10, 'badges': 150}
CACHE = {}  # Simple in-memory cache
CACHE_TTL = 300  # 5 minutes cache duration

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

async def fetch_with_cache(session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, Any]]:
    """Fetch data with simple caching mechanism"""
    cache_key = f"req_{hash(url)}"
    if cache_key in CACHE and (time.time() - CACHE[cache_key]['timestamp']) < CACHE_TTL:
        return CACHE[cache_key]['data']
    
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                CACHE[cache_key] = {'data': data, 'timestamp': time.time()}
                return data
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"[API ERROR] {url}: {str(e)}")
    return None

async def fetch_group_rank(session: aiohttp.ClientSession, user_id: int) -> str:
    """Fetch user's rank in British Army group with optimized query"""
    url = f"https://groups.roblox.com/v2/users/{user_id}/groups/roles"
    data = await fetch_with_cache(session, url)
    if data and 'data' in data:
        for group in data['data']:
            if group.get('group', {}).get('id') == BRITISH_ARMY_GROUP_ID:
                return group.get('role', {}).get('name', 'Guest')
    return 'Not in Group'

async def fetch_badge_count(session: aiohttp.ClientSession, user_id: int) -> int:
    """Reliable badge count fetcher with multiple verification methods"""
    try:
        # Primary method - Modern badges API with pagination
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
            
        # Fallback method - Inventory API for collectibles
        inventory_url = f"https://inventory.roblox.com/v1/users/{user_id}/items/Collectible/1?limit=100"
        inventory_data = await fetch_with_cache(session, inventory_url)
        
        if inventory_data and "data" in inventory_data:
            return len(inventory_data["data"])
            
        # Final fallback - Legacy API
        legacy_url = f"https://api.roblox.com/users/{user_id}/badges"
        legacy_data = await fetch_with_cache(session, legacy_url)
        
        if legacy_data and isinstance(legacy_data, list):
            return len(legacy_data)
            
        return 0
        
    except Exception as e:
        print(f"[BADGE COUNT ERROR] {e}")
        return 0

@has_allowed_role()
@commands.command(name="sc")
async def sc(ctx: commands.Context, user_id: int):
    """Optimized Roblox user checker with British Army rank"""
    try:
        async with ctx.typing(), aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT) as session:
            # Prepare all API endpoints
            urls = {
                'profile': f"https://users.roblox.com/v1/users/{user_id}",
                'groups': f"https://groups.roblox.com/v2/users/{user_id}/groups/roles",
                'friends': f"https://friends.roblox.com/v1/users/{user_id}/friends/count",
                'avatar': f"https://thumbnails.roblox.com/v1/users/avatar?userIds={user_id}&size=150x150&format=Png"
            }
            
            # Fetch all data concurrently
            tasks = {
                'profile': fetch_with_cache(session, urls['profile']),
                'groups': fetch_with_cache(session, urls['groups']),
                'friends': fetch_with_cache(session, urls['friends']),
                'avatar': fetch_with_cache(session, urlFs['avatar']),
                'badges': fetch_badge_count(session, user_id),
                'rank': fetch_group_rank(session, user_id)
            }
            
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            data = dict(zip(tasks.keys(), results))
            
            # Process profile data
            if not data['profile'] or isinstance(data['profile'], Exception):
                return await ctx.send("🔴 User not found")
            
            username = data['profile'].get('name', 'Unknown')
            created_at = datetime.fromisoformat(data['profile']['created'].replace('Z', '+00:00')) if data['profile'].get('created') else None
            age_days = (datetime.now(timezone.utc) - created_at).days if created_at else 0
            
            # Process other metrics
            friends_count = data['friends'].get('count', 0) if data['friends'] and not isinstance(data['friends'], Exception) else 0
            groups_count = len(data['groups'].get('data', [])) if data['groups'] and not isinstance(data['groups'], Exception) else 0
            badge_count = data['badges'] if not isinstance(data['badges'], Exception) else 0
            british_army_rank = data['rank'] if not isinstance(data['rank'], Exception) else 'Unknown'
            
            # Calculate metrics
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
            
            # Check banned groups
            banned_groups = []
            if data['groups'] and not isinstance(data['groups'], Exception):
                banned_groups = [
                    f"• {group['group']['name']}" 
                    for group in data['groups'].get('data', []) 
                    if group and group['group']['id'] in BGROUP_IDS
                ]
            
                        # Build improved embed
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

                warning = ""
                if name == 'badges':
                    warning = "\n⚠️ Badges may be private or user has none"
                elif metric['percentage'] >= 100 and metric['value'] < REQUIREMENTS['badges']:
                    warning = "\n ⚠️ Requirement may have changed"
                if REQUIREMENTS['badges'] <= 0:
                    warning = "\n ⚠️ Badge requirement not set"

                embed.add_field(
                    name=f"{emoji} {name.capitalize()} {status_icon}",
                    value=(
                        f"{progress_bar} {metric['value']}/{REQUIREMENTS[name]}\n"
                        f"Progress: {round(metric['percentage'])}%{warning}"
                    ),
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
                    value=" User is not in any banned groups or main regiments.",
                    inline=True
                )
            

            embed.set_author(
                name=f"Roblox User Check • {username}",
                icon_url=ctx.author.avatar.url if ctx.author.avatar else discord.Embed.Empty
            )
            embed.set_footer(text=f"Requested by {ctx.author.display_name} | Roblox User ID: {user_id}")
            await ctx.send(embed=embed)
            
    except Exception as e:
        print(f"[COMMAND ERROR] {e}")
        await ctx.send(embed=discord.Embed(
            title="⚠️ Error",
            description="Failed to process request. The user may have private settings.",
            color=discord.Color.red()
        ))

def setup(bot):
    bot.add_command(sc)
