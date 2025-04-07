import discord
import socket
import requests
import json
from datetime import datetime
from discord.ext import commands
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
USER_AGENT = "RMPBot/1.0 (+https://github.com/tlycrimson/RMP-Discord-Bot)"
OFFICIAL_API_URL = "https://api.roblox.com/users"
GROUP_IDS = [32578828, 4219097]

def create_session():
    """Configure requests session with retries"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

async def fetch_roblox_data(session, url):
    """Universal fetch with error handling"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[API ERROR] Failed to fetch {url}: {str(e)}")
        return None

@commands.command(name="sc")
async def sc(ctx, user_id: int):
    """Fetch and display Roblox user info"""
    try:
        async with ctx.typing():
            session = create_session()
            base_url = OFFICIAL_API_URL
            
            # 1. Fetch user data
            user_url = f"{base_url}/{user_id}"
            user_data = await fetch_roblox_data(session, user_url)
            if not user_data:
                return await ctx.send("❌ Failed to fetch user data. Roblox API may be down.")

            if "errorMessage" in user_data:
                return await ctx.send(f"❌ Roblox error: {user_data['errorMessage']}")

            # 2. Process data
            username = user_data["username"]
            created_at = datetime.strptime(user_data["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
            account_age = (datetime.utcnow() - created_at).days // 365

            # 3. Fetch badges
            badges_url = f"{base_url}/{user_id}/badges"
            badges_data = await fetch_roblox_data(session, badges_url) or []
            total_badges = len(badges_data)
            free_badges = [b for b in badges_data if b.get("isFree")]

            # 4. Check groups
            groups = []
            for group_id in GROUP_IDS:
                group_url = f"{base_url}/{user_id}/groups/{group_id}"
                group_data = await fetch_roblox_data(session, group_url)
                if group_data and group_data.get("success"):
                    groups.append(str(group_id))

            # 5. Build embed
            embed = discord.Embed(
                title=f"Roblox User: {username}",
                color=discord.Color.blue()
            )
            embed.add_field(name="Account Age", value=f"{account_age} years", inline=False)
            embed.add_field(name="Total Badges", value=total_badges, inline=False)
            embed.add_field(name="Free Badges", value=len(free_badges), inline=False)
            embed.add_field(name="Groups", value=", ".join(groups) if groups else "None", inline=False)
            
            await ctx.send(embed=embed)

    except Exception as e:
        print(f"[CRITICAL ERROR] {type(e).__name__}: {str(e)}")
        await ctx.send("❌ Service temporarily unavailable. Please try again later.")

def setup(bot):
    bot.add_command(sc)
