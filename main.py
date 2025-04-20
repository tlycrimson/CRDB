import os
import re
import time
import asyncio
import threading
import aiohttp
import discord
import logging
import random
from decorators import min_rank_required, has_allowed_role
from rate_limiter import RateLimiter
from discord import app_commands
from config import Config
from discord.ext import commands
from dotenv import load_dotenv
from flask import Flask
from typing import Optional, Set, Dict, List, Tuple
from roblox_commands import create_sc_command
from datetime import datetime

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GOOGLE_SCRIPT_URL = os.getenv("GOOGLE_SCRIPT_URL")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global rate limiter configuration
GLOBAL_RATE_LIMIT = 25  # requests per minute
COMMAND_COOLDOWN = 5    # seconds between command uses per user

# --- Enhanced Rate Limiter Class ---
class EnhancedRateLimiter:
    """Improved rate limiter with jitter and bucket support"""
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.last_call_time = 0
        self.buckets = {}
        self.locks = {}  # Add locks for thread safety
        
    async def wait_if_needed(self, bucket: str = "global"):
        """Wait if needed to avoid rate limits"""
        # Initialize bucket if not exists
        if bucket not in self.buckets:
            self.buckets[bucket] = {'last_call': 0, 'count': 0}
            self.locks[bucket] = asyncio.Lock()
            
        async with self.locks[bucket]:
            now = time.time()
            bucket_data = self.buckets[bucket]
            
            # Calculate time since last call
            elapsed = now - bucket_data['last_call']
            
            # Add jitter and minimum delay
            required_delay = max(1.0, (60 / self.calls_per_minute) - elapsed)
            
            if required_delay > 0:
                logger.debug(f"Rate limit wait: {required_delay:.2f}s for bucket {bucket}")
                await asyncio.sleep(required_delay)
                
            bucket_data['last_call'] = time.time()
            bucket_data['count'] += 1

# --- API Request Helper with Retry Logic ---
class DiscordAPI:
    """Helper class for Discord API requests with retry logic"""
    @staticmethod
    async def execute_with_retry(coro, max_retries=3, initial_delay=1.0):
        """Execute a Discord API call with automatic retry on rate limits"""
        for attempt in range(max_retries):
            try:
                return await coro
            except discord.errors.HTTPException as e:
                if e.status == 429:
                    retry_after = float(e.response.headers.get('Retry-After', initial_delay * (attempt + 1)))
                    logger.warning(f"Rate limited. Attempt {attempt + 1}/{max_retries}. Waiting {retry_after:.2f}s")
                    await asyncio.sleep(retry_after)
                    continue
                raise
            except Exception as e:
                logger.error(f"API Error: {type(e).__name__}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(initial_delay * (attempt + 1))
        
        raise Exception(f"Failed after {max_retries} attempts")

# --- Utility Classes ---
class ReactionLogger:
    """Handles reaction monitoring and logging with improved rate limiting"""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.monitor_channel_ids = set(Config.DEFAULT_MONITOR_CHANNELS)
        self.log_channel_id = Config.DEFAULT_LOG_CHANNEL
        self.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)
        
    async def on_ready_setup(self):
        """Setup monitoring when bot starts"""
        await DiscordAPI.execute_with_retry(self._on_ready_setup_impl())
        
    async def _on_ready_setup_impl(self):
        guild = self.bot.guilds[0]  # For the first guild the bot is in
        
        # Verify channels exist
        valid_channels = set()
        for channel_id in self.monitor_channel_ids:
            if channel := guild.get_channel(channel_id):
                valid_channels.add(channel.id)
        
        self.monitor_channel_ids = valid_channels
        
        # Verify log channel exists
        if not guild.get_channel(self.log_channel_id):
            logger.warning(f"Default log channel {self.log_channel_id} not found!")
            self.log_channel_id = None

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

    async def log_reaction(self, payload: discord.RawReactionActionEvent):
        """Log reactions from monitored channels with rate limiting"""
        try:
            await self.rate_limiter.wait_if_needed(bucket="reaction_log")
            await self._log_reaction_impl(payload)
        except Exception as e:
            logger.error(f"Failed to log reaction: {type(e).__name__}: {str(e)}")

    async def _log_reaction_impl(self, payload: discord.RawReactionActionEvent):
        """Actual reaction logging implementation"""
        logger.info(f"\n--- REACTION DETECTED ---\n"
                   f"Channel: {payload.channel_id}\n"
                   f"User: {payload.user_id}\n"
                   f"Emoji: {payload.emoji}\n")
        
        if payload.channel_id in Config.IGNORED_CHANNELS and str(payload.emoji) in Config.IGNORED_EMOJI:
            return

        if (payload.channel_id not in self.monitor_channel_ids or 
            str(payload.emoji) not in Config.TRACKED_REACTIONS):
            return
            
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return

        member = guild.get_member(payload.user_id)
        if not member:
            return

        monitor_role = guild.get_role(Config.MONITOR_ROLE_ID)
        if not monitor_role or monitor_role not in member.roles:
            return

        channel = guild.get_channel(payload.channel_id)
        log_channel = guild.get_channel(self.log_channel_id)

        if not all((channel, member, log_channel)):
            return

        try:
            message = await DiscordAPI.execute_with_retry(
                channel.fetch_message(payload.message_id)
            )
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
                
            await DiscordAPI.execute_with_retry(
                log_channel.send(embed=embed)
            )
            
            logger.info(f"Attempting to update points for: {member.display_name}")
            update_success = await self.bot.sheets.update_points(member)
            logger.info(f"Update {'succeeded' if update_success else 'failed'}")
            
        except discord.NotFound:
            return
        except Exception as e:
            logger.error(f"Reaction log error: {type(e).__name__}: {str(e)}")


# --- SheetDB Logger with rate limiting ---
class SheetDBLogger:
    def __init__(self):
        self.script_url = os.getenv("GOOGLE_SCRIPT_URL")
        if not self.script_url:
            logger.error("❌ Google Script URL not configured in environment variables")
            self.ready = False
        else:
            self.ready = True
            logger.info("✅ SheetDB Logger configured with Google Apps Script")
            logger.debug(f"Script URL: {self.script_url.split('?')[0]}...")

    async def update_points(self, member: discord.Member):
        if not self.ready:
            logger.error("🛑 SheetDB Logger not properly initialized")
            return False

        username = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
        logger.info(f"🔄 Attempting to update points for: {username}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.script_url,
                    json={"username": username},
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    response_text = (await response.text()).strip()
                    
                    success_conditions = [
                        response.status == 200,
                        "Error" not in response_text,
                        "Unauthorized" not in response_text,
                        len(response_text) > 0
                    ]
                    
                    if all(success_conditions):
                        logger.info(f"✅ Successfully updated points for {username}")
                        logger.debug(f"Response: {response_text}")
                        return True
                    else:
                        logger.error(f"❌ Failed to update points - Status: {response.status}")
                        logger.error(f"Response: {response_text}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.error("⏰ Timeout while connecting to Google Script")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"🌐 Network error: {type(e).__name__}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"⚠️ Unexpected error: {type(e).__name__}: {str(e)}")
            return False

# --- Bot Initialization ---
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.guilds = True
intents.reactions = True

bot = commands.Bot(
    intents=intents,
    command_prefix="!",
    activity=discord.Activity(type=discord.ActivityType.watching, name="out for RMP"),
    # Add these to handle rate limits better
    max_messages=None,
    heartbeat_timeout=60.0
)
bot.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)
bot.reaction_logger = ReactionLogger(bot)
bot.api = DiscordAPI()  # Add our API helper

# --- Command Error Handler ---
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"⏳ This command is on cooldown. Try again in {error.retry_after:.1f} seconds.", ephemeral=True)
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("🔒 You don't have permission to use this command.", ephemeral=True)
    elif isinstance(error, discord.errors.HTTPException) and error.status == 429:
        retry_after = error.response.headers.get('Retry-After', 5)
        await ctx.send(f"⚠️ Too many requests. Please wait {retry_after} seconds before trying again.", ephemeral=True)
    else:
        logger.error(f"Command error: {type(error).__name__}: {str(error)}")
        await ctx.send("❌ An error occurred while processing your command.", ephemeral=True)


# Initialize in on_ready()
@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    logger.info(f"Connected to {len(bot.guilds)} guild(s)")
    
    # Initialize SheetDB Logger
    bot.sheets = SheetDBLogger()
    if not bot.sheets.ready:
        logger.warning("SheetDB Logger not initialized properly")
    else:
        logger.info("SheetDB Logger initialized successfully")
        
    # Initialize reaction logger
    await bot.reaction_logger.on_ready_setup()

    # Register commands only if not already registered
    if not any(cmd.name == 'sc' for cmd in bot.tree.get_commands()):
        from roblox_commands import create_sc_command
        create_sc_command(bot)
        logger.info("Registered /sc command")
    
    # Sync commands with retry
    try:
        synced = await bot.api.execute_with_retry(
            bot.tree.sync(),
            max_retries=5,
            initial_delay=2.0
        )
        logger.info(f"Synced {len(synced)} commands")
    except Exception as e:
        logger.error(f"Command sync error: {e}")

# --- New Debug Commands ---
@bot.tree.command(name="force-update", description="Manually test sheet updates")
@has_allowed_role()
async def force_update(interaction: discord.Interaction, username: str):
    """Manually test sheet updates"""
    await interaction.response.defer(ephemeral=True)
    
    class FakeMember:
        def __init__(self, name):
            self.display_name = name
            self.name = name
    
    logger.info(f"Manual update test for username: {username}")
    success = await bot.sheets.update_points(FakeMember(username))
    
    if success:
        await interaction.followup.send(
            f"✅ Successfully updated points for {username}",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"❌ Failed to update points for {username} - check logs",
            ephemeral=True
        )

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
            "/commands - Show this help message",
            "/sheetdb-test - Test SheetDB connection"
        ],
        "🎮 Roblox Tools": [
            "/sc - Security Check Roblox user"
        ]
    }
    
    for name, value in categories.items():
        embed.add_field(name=name, value="\n".join(value), inline=False)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="ping", description="Check bot latency")
@has_allowed_role()
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

@bot.tree.command(name="sheetdb-test", description="Test SheetDB connection")
async def sheetdb_test(interaction: discord.Interaction):
    """Test the SheetDB integration"""
    await interaction.response.defer(ephemeral=True)
    
    if not hasattr(bot, 'sheets') or not bot.sheets.ready:
        await interaction.followup.send("❌ SheetDB Logger not initialized", ephemeral=True)
        return
    
    test_member = interaction.user
    success = await bot.sheets.update_points(test_member)
    
    if success:
        await interaction.followup.send(
            "✅ SheetDB update test successful",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            "❌ SheetDB update test failed - check logs",
            ephemeral=True
        )

# --- Event Handlers ---
@bot.event
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandInvokeError):
        if isinstance(error.original, discord.errors.HTTPException) and error.original.status == 429:
            retry_after = error.original.response.headers.get('Retry-After', 5)
            await interaction.followup.send(
                f"⚠️ Too many requests. Please wait {retry_after} seconds before trying again.",
                ephemeral=True
            )
            return
 
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
        description=f"{member.mention} with role the {deserter_role.mention} left the server!",
        color=discord.Color.red()
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    
    await alert_channel.send(
        content=f"<@&{Config.HIGH_COMMAND_ROLE_ID}>",
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
    except discord.HTTPException as e:
        print(f"Failed to send welcome message: {e}")

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

async def run_bot():
    while True:
        try:
            await bot.start(TOKEN)
        except discord.errors.HTTPException as e:
            if e.status == 429:
                retry_after = e.response.headers.get('Retry-After', 30)
                logger.warning(f"Rate limited during login. Waiting {retry_after} seconds...")
                await asyncio.sleep(float(retry_after))
                continue
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
        else:
            break

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    asyncio.run(run_bot())
