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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._commands_registered = False

    async def on_ready(self):
        if not self._commands_registered:
            from roblox_commands import sc
            self.add_command(sc)
            self._commands_registered = True
        print(f'Logged in as {self.user}!')
        await self.change_presence(activity=discord.Game(name="Monitoring Deserters"))

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
    role = guild.get_role(722006506014507040)
    notify_role = guild.get_role(1335394269535666216)
    channel = guild.get_channel(722002957738180620)

    if role and role in member.roles:
        notifyembed = discord.Embed(
            title="🚨 Possible Deserter!",
            description=f"{member.mention} with the {role.mention} role has left the server.",
            color=discord.Color.red()
        )
        await channel.send(content=f"{notify_role.mention}", embed=notifyembed)

def run():
    """Run both services"""
    # Verified correct parentheses below (count them - 6 opening, 6 closing)
    Thread(target=lambda: serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))).start()
    
    # Start Discord bot
    bot.run(TOKEN)

if __name__ == '__main__':
    run()
