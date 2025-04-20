import os
import re
import time
import asyncio
import random
import threading
import aiohttp
import discord
import logging
from decorators import min_rank_required, has_allowed_role
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
COMMAND_COOLDOWN = 10   # Increased from 5 to 10 seconds between command uses per user

# --- Enhanced Rate Limiter Class ---
class EnhancedRateLimiter:
    """Improved rate limiter with jitter and bucket support"""
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.last_call_time = 0
        self.buckets = {}  # For per-command rate limiting
        
    async def wait_if_needed(self, bucket: str = "global"):
        """Wait if needed to avoid rate limits, with jitter and bucket support"""
        now = time.time()
        
        # Initialize bucket if not exists
        if bucket not in self.buckets:
            self.buckets[bucket] = {'last_call': 0, 'count': 0}
            
        bucket_data = self.buckets[bucket]
        
        # Calculate time since last call
        elapsed = now - bucket_data['last_call']
        
        # Add jitter to avoid synchronized requests
        jitter = random.uniform(0.8, 1.2)
        required_delay = max(0, (60 / self.calls_per_minute) * jitter - elapsed)
        
        if required_delay > 0:
            logger.debug(f"Rate limit wait: {required_delay:.2f}s for bucket {bucket}")
            await asyncio.sleep(required_delay)
            
        bucket_data['last_call'] = time.time()
        bucket_data['count'] += 1
        
        # Track global rate too
        self.last_call_time = time.time()

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

# --- Updated Reaction Logger with Rate Limiting ---
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

    # ... [rest of your ReactionLogger methods with similar rate limiting improvements] ...

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

# --- Updated SheetDB Logger ---
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
    activity=discord.Activity(type=discord.ActivityType.watching, name="for reactions")
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

# --- Updated Command Implementations ---
@bot.tree.command(name="sc", description="Security Check Roblox user")
@app_commands.checks.cooldown(1, COMMAND_COOLDOWN, key=lambda i: (i.guild_id, i.user.id))
@has_allowed_role()
async def sc(interaction: discord.Interaction, username: str):
    """Security check command with rate limiting"""
    try:
        # Defer the response first
        try:
            await bot.api.execute_with_retry(
                interaction.response.defer(thinking=True)
            )
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            raise

        # Process the command (replace with your actual SC logic)
        result = f"Security check results for {username} would appear here"
        
        # Send follow-up with retry
        await bot.api.execute_with_retry(
            interaction.followup.send(result)
        )
        
    except Exception as e:
        logger.error(f"SC command failed: {type(e).__name__}: {str(e)}")
        if not interaction.response.is_done():
            await interaction.response.send_message(
                "⚠️ An error occurred while processing your request",
                ephemeral=True
            )

# ... [rest of your existing commands with similar rate limiting improvements] ...

# --- Updated on_ready ---
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
    
    # Register commands
    from roblox_commands import create_sc_command
    create_sc_command(bot)
    
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

# ... [rest of your existing Flask setup and event handlers] ...

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Run bot with restart logic
    while True:
        try:
            asyncio.run(run_bot())
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Bot crashed: {type(e).__name__}: {str(e)}")
            logger.info("Restarting in 10 seconds...")
            time.sleep(10)
