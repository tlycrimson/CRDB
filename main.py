import os
from decorators import has_allowed_role, min_rank_required
import discord
from discord.ext import commands
from dotenv import load_dotenv
from flask import Flask

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Config
ALLOWED_ROLE_ID = 1165368311840784508
DESERTER_ROLE_ID = 1165368311727521795
DESERTER_ALERT_CHANNEL_ID = 1197664960294170724
TRACKED_REACTIONS = {"✅", "❌"}

class ReactionLogger:
    def __init__(self, bot):
        self.bot = bot
        self.monitor_channel_ids = set()
        self.log_channel_id = None

    async def setup(self, ctx, log_channel: discord.TextChannel, *monitor_channels: discord.TextChannel):
        if len(monitor_channels) > 6:
            return await ctx.send("⚠️ You can monitor up to 6 channels.")

        self.monitor_channel_ids = {ch.id for ch in monitor_channels}
        self.log_channel_id = log_channel.id

        valid_mentions = [ch.mention for ch in monitor_channels if ch]
        if not valid_mentions or not self.bot.get_channel(self.log_channel_id):
            return await ctx.send("❌ Invalid channel(s) provided.")

        await ctx.send(
            f"✅ Setup complete!\n"
            f"Monitoring {len(valid_mentions)} channels: {', '.join(valid_mentions)}\n"
            f"Logging to: {log_channel.mention}\n"
            f"Reactions tracked: ✅ and ❌"
        )

    async def add_monitor_channels(self, ctx, *channels: discord.TextChannel):
        new_ids = {ch.id for ch in channels} - self.monitor_channel_ids
        if len(self.monitor_channel_ids | new_ids) > 6:
            return await ctx.send("⚠️ Max 6 channels can be monitored.")

        self.monitor_channel_ids.update(new_ids)
        await ctx.send(f"✅ Added {len(new_ids)} channel(s). Now monitoring {len(self.monitor_channel_ids)} total.")

    async def remove_monitor_channels(self, ctx, *channels: discord.TextChannel):
        removed = sum(1 for ch in channels if ch.id in self.monitor_channel_ids and self.monitor_channel_ids.remove(ch.id))
        await ctx.send(f"✅ Removed {removed} channel(s). Now monitoring {len(self.monitor_channel_ids)} total.")

    async def list_monitored_channels(self, ctx):
        if not self.monitor_channel_ids:
            return await ctx.send("No channels are currently being monitored.")

        mentions = [self.bot.get_channel(cid).mention for cid in self.monitor_channel_ids if self.bot.get_channel(cid)]
        embed = discord.Embed(
            title="📋 Monitored Channels",
            description="\n".join(mentions) or "None found",
            color=discord.Color.blue()
        )
        embed.set_footer(text=f"Total: {len(mentions)} channel(s)")
        await ctx.send(embed=embed)

    async def log_reaction(self, payload):
        if payload.channel_id not in self.monitor_channel_ids or str(payload.emoji) not in TRACKED_REACTIONS:
            return

        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return

        channel = guild.get_channel(payload.channel_id)
        user = guild.get_member(payload.user_id)
        log_channel = guild.get_channel(self.log_channel_id)

        if not (channel and user and log_channel):
            return

        try:
            message = await channel.fetch_message(payload.message_id)
        except discord.DiscordException:
            return

        content = message.content
        content = (content[:100] + "...") if len(content) > 100 else content

        embed = discord.Embed(
            title="📝 Reaction Logged",
            description=f"{user.mention} reacted with {payload.emoji}",
            color=discord.Color.blue()
        )
        embed.add_field(name="Channel", value=channel.mention, inline=True)
        embed.add_field(name="Author", value=message.author.mention, inline=True)
        embed.add_field(name="Message", value=content, inline=False)
        embed.add_field(name="Jump to", value=f"[Click here]({message.jump_url})", inline=False)
        embed.set_footer(text=f"User ID: {user.id} • Message ID: {message.id}")

        await log_channel.send(embed=embed)


class Bot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = intents.members = intents.guilds = intents.reactions = True
        super().__init__(command_prefix="!", intents=intents)
        self.reaction_logger = ReactionLogger(self)

    async def setup_hook(self):
        from roblox_commands import setup
        setup(self)

        @self.event
        async def on_raw_reaction_add(payload):
            await self.reaction_logger.log_reaction(payload)

        @self.event
        async def on_raw_reaction_remove(payload):
            await self.reaction_logger.log_reaction(payload)

    async def check_allowed_role(self, ctx):
        return any(role.id == ALLOWED_ROLE_ID for role in getattr(ctx.author, "roles", []))


bot = Bot()

# --- Commands ---
@bot.command()
@has_allowed_role()
@min_rank_required("High Command")
async def SRL(ctx, log_channel: discord.TextChannel, *monitor_channels: discord.TextChannel):
    await bot.reaction_logger.setup(ctx, log_channel, *monitor_channels)

@bot.command()
@has_allowed_role()
@min_rank_required("High Command")
async def ADC(ctx, *channels: discord.TextChannel):
    await bot.reaction_logger.add_monitor_channels(ctx, *channels)

@bot.command()
@has_allowed_role()
@min_rank_required("High Command")
async def RMC(ctx, *channels: discord.TextChannel):
    await bot.reaction_logger.remove_monitor_channels(ctx, *channels)

@bot.command()
@has_allowed_role()
@min_rank_required("High Command")
async def list_monitored_channels(ctx):
    await bot.reaction_logger.list_monitored_channels(ctx)

@bot.command()
@has_allowed_role()
@min_rank_required("High Command")
async def ping(ctx):
    await ctx.send("🏓 Pong!")

# --- Command List ---
@bot.command()
@has_allowed_role()
async def commands(ctx):
    """List all available commands with their descriptions"""
    embed = discord.Embed(
        title="📜 Available Commands",
        description="Here are all the commands you can use:",
        color=discord.Color.blue()
    )
    
    # Reaction Logger Commands
    embed.add_field(
        name="🔍 Reaction Monitoring",
        value=(
            "`!SRL <log_channel> <monitor_channel1> ...` - Setup reaction logger (max 6 channels)\n"
            "`!ADC <channel1> <channel2> ...` - Add channels to monitor\n"
            "`!RMC <channel1> <channel2> ...` - Remove monitored channels\n"
            "`!list_monitored_channels` - List currently monitored channels\n"
        ),
        inline=False
    )
    
    # Utility Commands
    embed.add_field(
        name="🛠️ Utility",
        value=(
            "`!ping` - Check bot responsiveness\n"
            "`!commands` - Show this help message\n"
        ),
        inline=False
    )
    
    # Roblox Commands (assuming you have the SC command)
    embed.add_field(
        name="🎮 Roblox Tools",
        value=(
            "`!sc <user_id>` - Security Check Roblox user info\n"
        ),
        inline=False
    )
    
    # Permissions Note
    embed.add_field(
        name="⚠️ Permissions",
        value="Most commands require High Command role or special permissions",
        inline=False
    )
    
    await ctx.send(embed=embed)

# --- Events ---
@bot.event
async def on_command_error(ctx, error):
    if not isinstance(error, commands.CheckFailure):
        await ctx.send(f"⚠️ An error occurred: `{error}`")

@bot.event
async def on_member_remove(member):
    guild = member.guild
    deserter_role = guild.get_role(DESERTER_ROLE_ID)
    notify_role = guild.get_role(ALLOWED_ROLE_ID)
    alert_channel = guild.get_channel(DESERTER_ALERT_CHANNEL_ID)

    if deserter_role in member.roles:
        embed = discord.Embed(
            title="🚨 Deserter Alert",
            description=f"{member.mention} with role {deserter_role.mention} left the server!",
            color=discord.Color.red()
        )
        await alert_channel.send(notify_role.mention, embed=embed)

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is running", 200

if __name__ == '__main__':
    import threading
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080)).start()
    bot.run(TOKEN)
