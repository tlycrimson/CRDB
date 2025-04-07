import os
import discord
from discord.ext import commands
from flask import Flask
from waitress import serve
from threading import Thread

app = Flask(__name__)

@app.route('/health')
def health():
    return "OK", 200

TOKEN = os.getenv('DISCORD_TOKEN')

class MyBot(commands.Bot):
    async def on_ready(self):
        print(f'Logged in as {self.user}!')
        await self.change_presence(activity=discord.Game(name="Monitoring Deserters"))

    async def setup_hook(self):
        from roblox_commands import sc
        self.add_command(sc)

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = MyBot(command_prefix='!', intents=intents)

@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')

# Deserter checker
@bot.event
async def on_member_remove(member):
    guild = member.guild
    role = guild.get_role(722006506014507040)  # The role to monitor
    notify_role = guild.get_role(1335394269535666216)  # The role to @mention
    channel = guild.get_channel(722002957738180620)  # Notification channel

    if role and role in member.roles:
        notifyembed = discord.Embed(
            title="🚨 Possible Deserter!",
            description=f"{member.mention} with the {role.mention} role has left the server.",
            color=discord.Color.red()
        )
        await channel.send(content=f"{notify_role.mention}", embed=notifyembed)

def run():
    """Run both services"""
    # Start Flask server in a separate thread
    Thread(target=lambda: serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 8080))).start()
    
    # Start Discord bot
    bot.run(TOKEN)

if __name__ == '__main__':
    run()
