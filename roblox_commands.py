# roblox_commands.py
import discord
import requests
from datetime import datetime
from discord.ext import commands

# Updated API URLs
USER_INFO_URL = "https://users.roblox.com/v1/users/"
BADGES_URL = "https://badges.roblox.com/v1/users/{}/badges"
GROUPS_URL = "https://groups.roblox.com/v2/users/{}/groups/roles"

@commands.command()
async def sc(ctx, user_id: int):
    try:
        # Get user info
        user_resp = requests.get(f"{USER_INFO_URL}{user_id}")
        if user_resp.status_code != 200:
            await ctx.send("⚠️ Failed to fetch user info.")
            return
        user_data = user_resp.json()

        username = user_data["name"]
        created_at = datetime.strptime(user_data["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
        account_age = (datetime.utcnow() - created_at).days // 365

        # Get badges info
        badge_resp = requests.get(BADGES_URL.format(user_id))
        badges_data = badge_resp.json() if badge_resp.status_code == 200 else []
        total_badges = len(badges_data)
        free_badges = [b for b in badges_data if b.get("awarderType") == "GamePass"]  # Hypothetical condition
        total_free_badges = len(free_badges)

        # Get group info
        group_resp = requests.get(GROUPS_URL.format(user_id))
        groups_data = group_resp.json().get("data", []) if group_resp.status_code == 200 else []
        target_groups = [32578828, 4219097]
        in_groups = [str(g["group"]["id"]) for g in groups_data if g["group"]["id"] in target_groups]

        # Build the embed
        embed = discord.Embed(title=f"Roblox User Info: {username}", color=discord.Color.blue())
        embed.add_field(name="Account Age", value=f"{account_age} years", inline=False)
        embed.add_field(name="Total Badges", value=f"{total_badges}", inline=False)
        embed.add_field(name="Free Badges", value=f"{total_free_badges}", inline=False)
        embed.add_field(name="Groups", value=", ".join(in_groups) if in_groups else "Not in specified groups", inline=False)

        await ctx.send(embed=embed)

    except requests.exceptions.RequestException as e:
        await ctx.send(f"🚫 Network error: {e}")
    except Exception as e:
        await ctx.send(f"❌ Unexpected error: {e}")

def setup(bot):
    bot.add_command(sc)
