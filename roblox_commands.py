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
GROUP_IDS = [32578828, 4219097]

# Known Roblox API IPs (update these periodically)
ROBLOX_IPS = [
    "172.253.118.95",  # Primary
    "142.250.190.46",  # Secondary
    "api.roblox.com"   # Official domain as last resort
]

def create_session():
    """Configure requests session with retries"""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,  # Fewer retries since we're trying multiple IPs
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.verify = False  # Required for IP direct access
    return session

async def try_fetch(session, user_id, ip):
    """Attempt to fetch data using a specific IP/domain"""
    base_url = f"https://{ip}" if ip not in ["api.roblox.com"] else f"https://{ip}"
    headers = {
        "Host": "api.roblox.com",
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    
    try:
        # Try user endpoint
        user_url = f"{base_url}/users/{user_id}"
        response = session.get(user_url, headers=headers, timeout=10)
        response.raise_for_status()
        user_data = response.json()
        
        # Try badges endpoint to confirm full API access
        badges_url = f"{base_url}/users/{user_id}/badges"
        session.get(badges_url, headers=headers, timeout=5)
        
        return user_data
    except Exception as e:
        print(f"[FAILOVER] Failed with {ip}: {str(e)}")
        return None

@commands.command(name="sc")
async def sc(ctx, user_id: int):
    """Fetch and display Roblox user info"""
    try:
        async with ctx.typing():
            session = create_session()
            user_data = None
            
            # Try all available IPs/domains
            for ip in ROBLOX_IPS:
                user_data = await try_fetch(session, user_id, ip)
                if user_data:
                    break
            
            if not user_data:
                return await ctx.send("❌ Roblox API is currently unreachable. Please try again later.")

            # Process data
            username = user_data["username"]
            created_at = datetime.strptime(user_data["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
            account_age = (datetime.utcnow() - created_at).days // 365

            # Fetch badges
            badges_url = f"https://{ROBLOX_IPS[0]}/users/{user_id}/badges"
            badges_data = (await try_fetch(session, user_id, ROBLOX_IPS[0])) or []
            total_badges = len(badges_data) if isinstance(badges_data, list) else 0
            free_badges = [b for b in badges_data if isinstance(badges_data, list) and b.get("isFree")]

            # Check groups
            groups = []
            for group_id in GROUP_IDS:
                group_url = f"https://{ROBLOX_IPS[0]}/users/{user_id}/groups/{group_id}"
                group_data = session.get(group_url, headers={"Host": "api.roblox.com"}, timeout=5).json()
                if group_data and group_data.get("success"):
                    groups.append(str(group_id))

            # Build embed
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
        print(f"[FATAL ERROR] {type(e).__name__}: {str(e)}")
        await ctx.send("❌ Service error. Contact bot administrator.")

def setup(bot):
    bot.add_command(sc)
