# main.py
import os
import discord
from discord.ext import commands

#TOKEN
TOKEN = os.getenv('DISCORD_TOKEN')

#
# INTENTS
#
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = commands.Bot(command_prefix='rb!ping', intents=intents)

#
# VERIFICATION THAT BOT IS ONLINE
#
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')



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

@bot.event
async def on_member_remove(member):
    guild = member.guild
    role = guild.get_role(ROLE_ID_TO_MONITOR)

    if role and role in member.roles:
        notify_role = guild.get_role(NOTIFY_ROLE_ID)
        channel = guild.get_channel(NOTIFY_CHANNEL_ID)

        if channel and notify_role:
            embed = discord.Embed(
                title="🚨 Member Left!",
                description=f"`{member}` with the role **{role.name}** has left the server.",
                color=discord.Color.red()
            )
            embed.add_field(name="Notify", value=notify_role.mention, inline=False)
            embed.set_footer(text="Stay alert!")

            await channel.send(embed=embed)

            
bot.run(TOKEN)
