# bot.py
import os
import discord
from discord.ext import commands

TOKEN = os.getenv('DISCORD_TOKEN')  # From Heroku config
intents = discord.Intents.default()
intents.message_content = True  # Needed for basic command response
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')

bot.run(TOKEN)
