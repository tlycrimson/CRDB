# roblox_commands.py
import discord
import requests
from datetime import datetime
from discord.ext import commands  # <- Import commands

# The Roblox user info API
ROBLOX_API_URL = "https://api.roblox.com/users/"

# Command to check Roblox user stats
@commands.command()  # <- This is necessary for defining a bot command
async def roblox_user(ctx, user_id: int):
    # Fetch the user's data from Roblox API
    user_url = f"{ROBLOX_API_URL}{user_id}"
    user_data = requests.get(user_url).json()

    if user_data.get("errorMessage"):
        await ctx.send(f"Error: {user_data['errorMessage']}")
        return

    username = user_data["username"]
    account_creation = user_data["created"]
    friends_count = user_data["friendsCount"]
    
    # Get account age in years (simplified)
    created_at = datetime.strptime(account_creation, "%Y-%m-%dT%H:%M:%S.%fZ")
    account_age = (datetime.utcnow() - created_at).days // 365
    
    # Fetch the user's badges
    badges_url = f"https://api.roblox.com/users/{user_id}/badges"
    badges_data = requests.get(badges_url).json()
    total_badges = len(badges_data)
    
    # You can implement logic to filter free badges if needed.
    free_badges = [badge for badge in badges_data if badge["isFree"]]
    total_free_badges = len(free_badges)

    # Check if the user is in the specific groups
    group_ids = [32578828, 4219097]
    groups = []
    for group_id in group_ids:
        group_url = f"https://api.roblox.com/users/{user_id}/groups/{group_id}"
        group_data = requests.get(group_url).json()
        if group_data.get("success"):
            groups.append(group_id)
    
    # Embed the information in a neat format
    embed = discord.Embed(title=f"Roblox User Info: {username}")
    embed.add_field(name="Friends", value=f"{friends_count}", inline=False)
    embed.add_field(name="Account Age", value=f"{account_age} years", inline=False)
    embed.add_field(name="Total Badges", value=f"{total_badges}", inline=False)
    embed.add_field(name="Free Badges", value=f"{total_free_badges}", inline=False)
    embed.add_field(name="Groups", value=", ".join(map(str, groups)) if groups else "Not in specified groups", inline=False)
    
    # Send the embed
    await ctx.send(embed=embed)
