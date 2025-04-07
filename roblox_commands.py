import discord
import requests
import json
from datetime import datetime
from discord.ext import commands
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio

# Configuration - Updated for Render compatibility
USER_AGENT = "RMPBot/2.0 (+https://github.com/tlycrimson/RMP-Discord-Bot)"
GROUP_IDS = [32578828, 4219097]  # Your Roblox group IDs
TIMEOUT = 15  # Increased timeout for Render's network

# Official Roblox API endpoints (prioritized)
ROBLOX_ENDPOINTS = [
    "https://users.roblox.com",  # Primary
    "https://api.roblox.com",    # Legacy
    "https://groups.roblox.com"  # Fallback
]

def create_session():
    """Optimized session for Render's environment"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[408, 429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_maxsize=20,  # Better for concurrent requests
        pool_block=True
    )
    session.mount('https://', adapter)
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Connection": "keep-alive"
    })
    return session

async def fetch_roblox_data(session, endpoint, user_id, path=""):
    """Unified data fetcher with proper error handling"""
    url = f"{endpoint}/v1/users/{user_id}{path}"
    try:
        response = await asyncio.to_thread(
            session.get, 
            url, 
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[API ERROR] {endpoint}{path}: {str(e)}")
        return None

@commands.command(name="sc")
async def sc(ctx, user_id: int):
    """Optimized Roblox user lookup command"""
    try:
        async with ctx.typing():
            session = create_session()
            
            # Try all endpoints for basic user data
            user_data = None
            for endpoint in ROBLOX_ENDPOINTS:
                user_data = await fetch_roblox_data(session, endpoint, user_id)
                if user_data:
                    break
            
            if not user_data:
                return await ctx.send("🔴 Roblox API unavailable. Try again later.")

            # Process core data
            username = user_data.get("name", user_data.get("username", "Unknown"))
            created = user_data.get("created", user_data.get("joinDate"))
            created_at = datetime.strptime(created, "%Y-%m-%dT%H:%M:%S.%fZ") if created else None
            account_age = (datetime.utcnow() - created_at).days // 365 if created_at else "Unknown"

            # Parallel fetching for badges and groups
            tasks = []
            for endpoint in ROBLOX_ENDPOINTS[:2]:  # Only check first two endpoints
                tasks.append(fetch_roblox_data(session, endpoint, user_id, "/badges"))
                for group_id in GROUP_IDS:
                    tasks.append(fetch_roblox_data(session, endpoint, user_id, f"/groups/{group_id}"))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process badges
            badges = [r for r in results if isinstance(r, list)]
            total_badges = len(badges[0]) if badges else 0
            free_badges = sum(1 for b in (badges[0] if badges else []) if b.get("isFree", False))

            # Process groups
            groups = []
            for i, group_id in enumerate(GROUP_IDS, start=2):  # Skip badge results
                group_data = results[i] if i < len(results) else None
                if group_data and group_data.get("role"):
                    groups.append(str(group_id))

            # Build optimized embed
            embed = discord.Embed(
                title=f"{username}",
                color=0xb44e4e,
                url=f"https://www.roblox.com/users/{user_id}/profile"
            )
            embed.set_thumbnail(url=f"https://www.roblox.com/headshot-thumbnail/image?userId={user_id}")
            embed.add_field(name="📅 Account Age", value=f"{account_age} years" if isinstance(account_age, int) else account_age)
            embed.add_field(name="🎖️ Badges", value=f"{total_badges} (Free: {free_badges})")
            embed.add_field(name="👥 Banned Groups", value=", ".join(groups) if groups else "None", inline=False)
            embed.set_footer(text=f"User ID: {user_id} | Requested by {ctx.author.display_name}")

            await ctx.send(embed=embed)

    except Exception as e:
        print(f"[COMMAND ERROR] SC: {type(e).__name__}: {str(e)}")
        await ctx.send("⚠️ Service temporarily unavailable. Please try again in a minute.")

def setup(bot):
    bot.add_command(sc)
