# roblox_commands.py
import discord
import requests
from datetime import datetime
from discord.ext import commands

ROBLOX_API_URL = "https://api.roblox.com/users/"

@commands.command(name="sc")
async def sc(ctx, user_id: int):
    try:
        await ctx.trigger_typing()  # Show "bot is typing"
        user_data = requests.get(f"{ROBLOX_API_URL}{user_id}", timeout=10).json()
        ...
    except requests.exceptions.Timeout:
        await ctx.send("⏳ API timed out. Try again later.")

        if user_data.get("errorMessage"):
            await ctx.send(f"Error: {user_data['errorMessage']}")
            return

        username = user_data["username"]
        account_creation = user_data["created"]
        friends_count = user_data["friendsCount"]
        
        created_at = datetime.strptime(account_creation, "%Y-%m-%dT%H:%M:%S.%fZ")
        account_age = (datetime.utcnow() - created_at).days // 365
        
        badges_url = f"https://api.roblox.com/users/{user_id}/badges"
        badges_data = requests.get(badges_url).json()
        total_badges = len(badges_data)
        
        free_badges = [badge for badge in badges_data if badge["isFree"]]
        total_free_badges = len(free_badges)

        group_ids = [32578828, 4219097]
        groups = []
        for group_id in group_ids:
            group_url = f"https://api.roblox.com/users/{user_id}/groups/{group_id}"
            group_data = requests.get(group_url).json()
            if group_data.get("success"):
                groups.append(group_id)
        
        embed = discord.Embed(title=f"Roblox User Info: {username}")
        embed.add_field(name="Friends", value=f"{friends_count}", inline=False)
        embed.add_field(name="Account Age", value=f"{account_age} years", inline=False)
        embed.add_field(name="Total Badges", value=f"{total_badges}", inline=False)
        embed.add_field(name="Free Badges", value=f"{total_free_badges}", inline=False)
        embed.add_field(name="Groups", value=", ".join(map(str, groups)) if groups else "Not in specified groups", inline=False)
        
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

def setup(bot):
    bot.add_command(commands.Command(sc, name="sc"))  # Explicitly register the command
