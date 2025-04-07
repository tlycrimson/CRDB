import os
import discord
from discord.ext import commands

from flask import Flask
import threading

# Health check server
server = Flask(__name__)
@server.route('/')
def home():
    return "Bot is running", 200

def run_web():
    server.run(host="0.0.0.0", port=8080)

# Start web server in a thread
threading.Thread(target=run_web).start()

TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Load commands directly (no extension system for simplicity)
from roblox_commands import sc
bot.add_command(sc)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')

# ACTIVELY CHECKS FOR DESERTERS
ROLE_ID_TO_MONITOR = 722006506014507040  # The role to monitor
NOTIFY_ROLE_ID = 1335394269535666216     # The role to @mention
NOTIFY_CHANNEL_ID = 722002957738180620  # The channel to send the notification

# DESERTER CHECKER
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

# Run the bot with the token
bot.run(TOKEN)
