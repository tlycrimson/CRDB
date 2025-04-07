import os
import discord
from discord.ext import commands
from flask import Flask
from waitress import serve

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

# Initialize bot with proper intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = MyBot(command_prefix='!', intents=intents)

@bot.command()
async def ping(ctx):
    """Fixed ping command (will only respond once)"""
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
            await channel.send(content=f"{notify_role.mention}", embed=notifyembed)

def run():
    """Run both services without conflict"""
    from threading import Thread
    Thread(target=lambda: serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 8080))).start()
    bot.run(TOKEN)

if __name__ == '__main__':
    run()
