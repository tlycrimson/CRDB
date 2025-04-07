# main.py
import os
import discord
import requests
from datetime import datetime
from discord.ext import commands
from roblox_commands import roblox_user

#TOKEN
TOKEN = os.getenv('DISCORD_TOKEN')

#
# INTENTS
#
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = commands.Bot(command_prefix='rb!', intents=intents)

#
# VERIFICATION THAT BOT IS ONLINE
#
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

#
# SECURITY CHECK
#
# The roblox_user function is now imported from roblox_commands.py, no need to define it here again.

#
# Ping Pong command
#
@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')

#
# ACTIVELY CHECKS FOR DESERTERS
#
# IDs for monitoring and notification
ROLE_ID_TO_MONITOR = 722006506014507040  # The role to monitor
NOTIFY_ROLE_ID = 1335394269535666216     # The role to @mention
NOTIFY_CHANNEL_ID = 722002957738180620  # The channel to send the notification

#
# DESERTER CHECKER
#
@bot.event
async def on_member_remove(member):
    guild = member.guild
    role = guild.get_role(ROLE_ID_TO_MONITOR)

    if role and role in member.roles:
        notify_role = guild.get_role(NOTIFY_ROLE_ID)
        channel = guild.get_channel(NOTIFY_CHANNEL_ID)

        if channel and notify_role:
            notifyembed = discord.Embed(
                title="🚨 Possible Deserter!",
                description=f"{member.mention} with the {role.mention} role has left the server.",
                color=discord.Color.red()
            )

            await channel.send(
                content=f"{notify_role.mention}",
                embed=notifyembed)

            
bot.run(TOKEN)
