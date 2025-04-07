import discord
import socket
import requests
from datetime import datetime
from discord.ext import commands
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Get fresh IP with DNS fallback
def get_roblox_ip():
    try:
        return socket.gethostbyname('api.roblox.com')
    except socket.gaierror:
        return "172.253.118.95"  # Fallback IP (update if needed)

ROBLOX_IP = get_roblox_ip()
ROBLOX_API_URL = f"https://{ROBLOX_IP}/users"  # No trailing slash!

# Configure session with retries and SSL bypass
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('https://', HTTPAdapter(max_retries=retry_strategy))
session.verify = False  # Required for IP direct access

HEADERS = {
    "Host": "api.roblox.com",
    "User-Agent": "RMPBot/1.0"
}

@commands.command(name="sc")
async def sc(ctx, user_id: int):
    try:
        async with ctx.typing():
            # Fetch main user data
            response = session.get(
                f"{ROBLOX_API_URL}/{user_id}",
                headers=HEADERS,
                timeout=10
            )
            user_data = response.json()

            if user_data.get("errorMessage"):
                await ctx.send(f"❌ Roblox API error: {user_data['errorMessage']}")
                return

            # Process data
            created_at = datetime.strptime(user_data["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
            account_age = (datetime.utcnow() - created_at).days // 365

            # Fetch badges
            badges_url = f"https://{ROBLOX_IP}/users/{user_id}/badges"
            badges_data = session.get(badges_url, headers=HEADERS, timeout=10).json()
            total_badges = len(badges_data)
            free_badges = [badge for badge in badges_data if badge.get("isFree")]

            # Check groups
            group_ids = [32578828, 4219097]
            groups = []
            for group_id in group_ids:
                group_url = f"https://{ROBLOX_IP}/users/{user_id}/groups/{group_id}"
                group_data = session.get(group_url, headers=HEADERS, timeout=10).json()
                if group_data.get("success"):
                    groups.append(group_id)

            # Build embed
            embed = discord.Embed(title=f"Roblox User: {user_data['username']}")
            embed.add_field(name="Account Age", value=f"{account_age} years", inline=False)
            embed.add_field(name="Total Badges", value=total_badges, inline=False)
            embed.add_field(name="Free Badges", value=len(free_badges), inline=False)
            embed.add_field(name="Groups", value=", ".join(map(str, groups)) if groups else "None", inline=False)
            
            await ctx.send(embed=embed)

    except requests.exceptions.Timeout:
        await ctx.send("⏳ Roblox API timed out. Try again later.")
    except requests.exceptions.RequestException as e:
        await ctx.send(f"❌ Network error: {str(e)}")
    except Exception as e:
        await ctx.send(f"❌ Unexpected error: {str(e)}")
