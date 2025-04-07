import discord
import requests
from datetime import datetime
from discord.ext import commands
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROBLOX_API_URL = "https://api.roblox.com/users/"

# Configure retries for Heroku's flaky network
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

@commands.command(name="sc")
async def sc(ctx, user_id: int):
    try:
        async with ctx.typing():  # Shows "bot is typing"
            # Fetch user data with retries
            user_url = f"{ROBLOX_API_URL}{user_id}"
            user_data = session.get(user_url, timeout=10).json()

            if user_data.get("errorMessage"):
                await ctx.send(f"❌ Roblox API error: {user_data['errorMessage']}")
                return

            # Calculate account age
            created_at = datetime.strptime(user_data["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
            account_age = (datetime.utcnow() - created_at).days // 365

            # Fetch badges
            badges_url = f"https://api.roblox.com/users/{user_id}/badges"
            badges_data = session.get(badges_url, timeout=10).json()
            total_badges = len(badges_data)
            free_badges = [badge for badge in badges_data if badge.get("isFree")]

            # Check group membership
            group_ids = [32578828, 4219097]
            groups = []
            for group_id in group_ids:
                group_data = session.get(
                    f"https://api.roblox.com/users/{user_id}/groups/{group_id}",
                    timeout=10
                ).json()
                if group_data.get("success"):
                    groups.append(group_id)

            # Build embed
            embed = discord.Embed(title=f"Roblox User: {user_data['username']}")
            embed.add_field(name="Account Age", value=f"{account_age} years", inline=False)
            embed.add_field(name="Total Badges", value=len(badges_data), inline=False)
            embed.add_field(name="Free Badges", value=len(free_badges), inline=False)
            embed.add_field(name="Groups", value=", ".join(map(str, groups)) if groups else "None", inline=False)
            
            await ctx.send(embed=embed)

    except requests.exceptions.Timeout:
        await ctx.send("⏳ Roblox API timed out. Try again later.")
    except Exception as e:
        await ctx.send(f"❌ Unexpected error: {str(e)}")
