import os
import discord
from discord.ext import commands
from flask import Flask
from waitress import serve  # Production-ready WSGI server

# Initialize Flask app for health checks
app = Flask(__name__)

@app.route('/health')
def health():
    return "OK", 200

# Discord bot setup
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Load commands
from roblox_commands import sc
bot.add_command(sc)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')
    await bot.change_presence(activity=discord.Game(name="Monitoring Deserters"))

@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')

# Deserter checker (unchanged)
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

def run_flask():
    """Run production WSGI server"""
    serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

if __name__ == '__main__':
    # Start Flask in production mode
    import threading
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Start Discord bot
    bot.run(TOKEN)
