import os
import time
import asyncio
import threading
import discord
from decorators import min_rank_required
from rate_limiter import RateLimiter
from discord import app_commands
from config import Config
from discord.ext import commands
from dotenv import load_dotenv
from flask import Flask
from typing import Optional, Set, Dict, List, Tuple
from roblox_commands import create_sc_command

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")


# --- Utility Classes ---

class ReactionLogger:
    """Handles reaction monitoring and logging"""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.monitor_channel_ids: Set[int] = set()
        self.log_channel_id: Optional[int] = None
        self.rate_limiter = RateLimiter(calls_per_minute=45)

    async def _create_embed(self, title: str, description: str, 
                          color: discord.Color = discord.Color.blue(), 
                          ephemeral: bool = False) -> Dict:
        """Helper to create consistent embeds"""
        embed = discord.Embed(
            title=title,
            description=description,
            color=color
        )
        embed.set_footer(text=f"Executed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return {"embed": embed, "ephemeral": ephemeral}

    async def _process_channels(self, interaction: discord.Interaction, 
                              input_str: str) -> Tuple[List[discord.TextChannel], List[str]]:
        """Process channel input string into valid channels and invalid names"""
        valid_channels = []
        invalid_names = []
        
        for name in (n.strip() for n in input_str.split(',') if n.strip()):
            if name.startswith('<#') and name.endswith('>'):
                channel_id = int(name[2:-1])
                if channel := interaction.guild.get_channel(channel_id):
                    valid_channels.append(channel)
                    continue
            
            if channel := discord.utils.get(interaction.guild.text_channels, name=name):
                valid_channels.append(channel)
            else:
                invalid_names.append(name)
                
        return valid_channels, invalid_names

    async def setup(self, interaction: discord.Interaction, 
                   log_channel: discord.TextChannel, 
                   monitor_channels: str):
        """Initialize reaction monitoring system"""
        channels, invalid = await self._process_channels(interaction, monitor_channels)
        
        if len(channels) > Config.MAX_MONITORED_CHANNELS:
            response = await self._create_embed(
                "⚠️ Channel Limit Exceeded",
                f"You can monitor up to {Config.MAX_MONITORED_CHANNELS} channels.",
                discord.Color.red(),
                True
            )
            await interaction.response.send_message(**response)
            return

        self.monitor_channel_ids = {ch.id for ch in channels}
        self.log_channel_id = log_channel.id
        
        response = await self._create_embed(
            "✅ Setup Complete" if not invalid else "⚠️ Partial Setup",
            f"Now monitoring {len(channels)} channels" + 
            (f"\nCouldn't find: {', '.join(invalid)}" if invalid else ""),
            discord.Color.green() if not invalid else discord.Color.orange()
        )
        await interaction.response.send_message(**response)

    async def add_channels(self, interaction: discord.Interaction, channels: str):
        """Add channels to monitor"""
        new_channels, invalid = await self._process_channels(interaction, channels)
        
        if len(self.monitor_channel_ids) + len(new_channels) > Config.MAX_MONITORED_CHANNELS:
            response = await self._create_embed(
                "⚠️ Channel Limit Exceeded",
                f"Cannot add {len(new_channels)} channels. Max is {Config.MAX_MONITORED_CHANNELS}.",
                discord.Color.red(),
                True
            )
            await interaction.response.send_message(**response)
            return

        self.monitor_channel_ids.update(ch.id for ch in new_channels)
        
        response = await self._create_embed(
            "✅ Channels Added" if not invalid else "⚠️ Partial Success",
            f"Added {len(new_channels)} channels to monitoring" + 
            (f"\nCouldn't find: {', '.join(invalid)}" if invalid else ""),
            discord.Color.green() if not invalid else discord.Color.orange()
        )
        await interaction.response.send_message(**response)

    async def remove_channels(self, interaction: discord.Interaction, channels: str):
        """Remove channels from monitoring"""
        remove_channels, invalid = await self._process_channels(interaction, channels)
        removed = []
        
        for ch in remove_channels:
            if ch.id in self.monitor_channel_ids:
                self.monitor_channel_ids.remove(ch.id)
                removed.append(ch.name)
        
        response = await self._create_embed(
            "✅ Channels Removed" if removed else "⚠️ No Channels Removed",
            (f"Stopped monitoring: {', '.join(removed)}" if removed else "No matching channels were being monitored") +
            (f"\nCouldn't find: {', '.join(invalid)}" if invalid else ""),
            discord.Color.green() if removed else discord.Color.orange()
        )
        await interaction.response.send_message(**response)

    async def list_channels(self, interaction: discord.Interaction):
        """List currently monitored channels"""
        if not self.monitor_channel_ids:
            response = await self._create_embed(
                "ℹ️ No Channels Monitored",
                "Currently not monitoring any channels.",
                discord.Color.blue(),
                True
            )
            await interaction.response.send_message(**response)
            return

        channel_names = []
        guild = interaction.guild
        
        for channel_id in self.monitor_channel_ids:
            if channel := guild.get_channel(channel_id):
                channel_names.append(f"• {channel.mention}")
        
        response = await self._create_embed(
            "📋 Monitored Channels",
            "\n".join(channel_names) if channel_names else "No channels found",
            discord.Color.blue(),
            True
        )
        await interaction.response.send_message(**response)

    async def log_reaction(self, payload: discord.RawReactionActionEvent):
        """Log reactions from monitored channels (only for users with monitoring role)"""
        # Check if reaction is in monitored channel and is a tracked emoji
        if (payload.channel_id not in self.monitor_channel_ids or 
        str(payload.emoji) not in Config.TRACKED_REACTIONS):
            return

        await self.rate_limiter.wait_if_needed()
            
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return

        # Get the member who reacted
        member = guild.get_member(payload.user_id)
        if not member:
            return

        # Check if member has the required role
        monitor_role = guild.get_role(Config.MONITOR_ROLE_ID)
        if not monitor_role or monitor_role not in member.roles:
            return

        channel = guild.get_channel(payload.channel_id)
        log_channel = guild.get_channel(self.log_channel_id)

        if not all((channel, member, log_channel)):
            return

        try:
            message = await channel.fetch_message(payload.message_id)
            content = (message.content[:100] + "...") if len(message.content) > 100 else message.content
                
            embed = discord.Embed(
                title="📝 Reaction Logged",
                description=f"{member.mention} (with {monitor_role.name} role) reacted with {payload.emoji}",
                color=discord.Color.blue()
                    )
            
            embed.add_field(name="Channel", value=channel.mention)
            embed.add_field(name="Author", value=message.author.mention)
            embed.add_field(name="Message", value=content, inline=False)
            embed.add_field(name="Jump to", value=f"[Click here]({message.jump_url})", inline=False)
                
            await log_channel.send(embed=embed)
        except discord.NotFound:
            return
        except Exception as e:
                print(f"[REACTION LOG ERROR] {type(e).__name__}: {str(e)}")

# --- Bot Initialization ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True
intents.reactions = True

bot = commands.Bot(intents=intents, command_prefix="!")
bot.rate_limiter = RateLimiter(calls_per_minute=45) 
bot.reaction_logger = ReactionLogger(bot)

# --- Decorators ---
def has_allowed_role():
    async def predicate(interaction: discord.Interaction):
        if interaction.guild is None:
            return False
            
        member = interaction.guild.get_member(interaction.user.id)
        if not member:
            return False
            
        if member.guild_permissions.administrator:
            return True
            
        allowed_role = interaction.guild.get_role(Config.ALLOWED_ROLE_ID)
        if allowed_role and allowed_role in member.roles:
            return True
            
        await interaction.response.send_message(
            "You don't have permission to use this command.",
            ephemeral=True
        )
        return False
    return app_commands.check(predicate)

# --- Slash Commands ---
@bot.tree.command(name="commands", description="List all available commands")
@has_allowed_role()
async def command_list(interaction: discord.Interaction):
    """Display help menu with all commands"""
    embed = discord.Embed(
        title="📜 Available Commands",
        color=discord.Color.blue()
    )
    
    categories = {
        "🔍 Reaction Monitoring": [
            "/reaction-setup - Setup reaction logger",
            "/reaction-add - Add channels to monitor",
            "/reaction-remove - Remove monitored channels",
            "/reaction-list - List monitored channels"
        ],
        "🛠️ Utility": [
            "/ping - Check bot responsiveness",
            "/commands - Show this help message"
        ],
        "🎮 Roblox Tools": [
            "/sc - Security Check Roblox user"
        ]
    }
    
    for name, value in categories.items():
        embed.add_field(name=name, value="\n".join(value), inline=False)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="ping", description="Check bot latency")
async def ping(interaction: discord.Interaction):
    """Check bot responsiveness"""
    latency = round(bot.latency * 1000)
    await interaction.response.send_message(
        f"🏓 Pong! Latency: {latency}ms",
        ephemeral=True
    )

@bot.tree.command(name="reaction-setup", description="Setup reaction monitoring")
@min_rank_required(Config.MONITOR_ROLE_ID)
async def reaction_setup(
    interaction: discord.Interaction,
    log_channel: discord.TextChannel,
    monitor_channels: str
):
    await bot.reaction_logger.setup(interaction, log_channel, monitor_channels)

@bot.tree.command(name="reaction-add", description="Add channels to monitor")
@min_rank_required(Config.MONITOR_ROLE_ID)
async def reaction_add(
    interaction: discord.Interaction,
    channels: str
):
    await bot.reaction_logger.add_channels(interaction, channels)

@bot.tree.command(name="reaction-remove", description="Remove channels from monitoring")
@min_rank_required(Config.MONITOR_ROLE_ID)
async def reaction_remove(
    interaction: discord.Interaction,
    channels: str
):
    await bot.reaction_logger.remove_channels(interaction, channels)

@bot.tree.command(name="reaction-list", description="List monitored channels")
@min_rank_required(Config.MONITOR_ROLE_ID)
async def reaction_list(interaction: discord.Interaction):
    await bot.reaction_logger.list_channels(interaction)

# --- Event Handlers ---
@bot.event
async def on_ready():
    print(f"[✅] logged in as {bot.user}")
    
    # Make sure the bot has a rate limiter
    if not hasattr(bot, 'rate_limiter'):
        bot.rate_limiter = RateLimiter(calls_per_minute=45)

    # Create and register SC command
    create_sc_command(bot)
    
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} commands")
        print("Commands:", [cmd.name for cmd in bot.tree.get_commands()])
    except Exception as e:
        print(f"Command sync error: {e}")

@bot.event
async def on_member_remove(member: discord.Member):
    """Handle members leaving with deserter role"""
    guild = member.guild
    if not (deserter_role := guild.get_role(Config.DESERTER_ROLE_ID)):
        return
        
    if deserter_role not in member.roles:
        return
        
    if not (alert_channel := guild.get_channel(Config.DESERTER_ALERT_CHANNEL_ID)):
        return
        
    embed = discord.Embed(
        title="🚨 Deserter Alert",
        description=f"{member.mention} with role {deserter_role.mention} left!",
        color=discord.Color.red()
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    
    await alert_channel.send(
        content=f"<@&{Config.ALLOWED_ROLE_ID}>",
        embed=embed
    )

@bot.event 
async def on_member_update(before: discord.Member, after: discord.Member):
    """Send welcome message when RMP role is added"""
    if not (rmp_role := after.guild.get_role(Config.RMP_ROLE_ID)):
        return
        
    if rmp_role in before.roles or rmp_role not in after.roles:
        return
        
    embed = discord.Embed(
        title="Welcome to the Royal Military Police",
         description="**1.** Make sure to read all of the rules found in <#1165368313925353580>\n\n"
                    "**2.** You can NOT enforce the MSL (Manual of Service Law).\n\n"
                    "**3.** You can't use your L85 unless you are doing it for Self-Militia. (Self-defence)\n\n"
                    "**4.** Make sure to follow the Chain Of Command. Inspector > Chief Inspector > Superintendent > Major > Lieutenant Colonel > Colonel > Commander > Provost Marshal\n\n"
                    "**5.** For phases, you may wait for one to be hosted in <#1207367013698240584> or request the phase you need in <#1270700562433839135>.\n\n"
                    "**6.** All the information about the Defence School of Policing and Guarding is found in both <#1237062439720452157> and <#1207366893631967262>\n\n"
                    "**7.** Choose your timezone here https://discord.com/channels/1165368311085809717/1165368313925353578\n\n"
                    "**8.** You will be ranked Private but if you ever decide to leave RMP you will get your original rank back.\n\n"
                    "**Besides that, good luck with your phases!**",
        color=discord.Color.red()
        )
    
    try:
        await after.send(embed=embed)
    except discord.Forbidden:
        if welcome_channel := after.guild.get_channel(722002957738180620):
            await welcome_channel.send(f"{after.mention}", embed=embed)

@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    """Handle reaction events"""
    await bot.reaction_logger.log_reaction(payload)

# --- Flask Setup ---
app = Flask(__name__)
keep_alive = True

@app.route('/')
def home():
    return "Bot is running", 200

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global keep_alive
    keep_alive = False
    return "Shutting down...", 200

def run_flask():
    """Run Flask in a background thread"""
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    try:
        bot.run(TOKEN)
    finally:
        keep_alive = False
