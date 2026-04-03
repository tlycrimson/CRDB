import os
import asyncio
import aiohttp
import discord
import logging
import random
from discord.ext import commands
from dotenv import load_dotenv

from utils.runners import run_bot_forever
from config import Config
from utils.database import DatabaseHandler
from utils.network import EnhancedRateLimiter, GlobalRateLimiter, DiscordAPI
from utils.rank_tracker import RankTracker

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise ValueError("DISCORD_TOKEN not found in .env file")


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# --- Bot Class Definition ---
class CRDB(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.members = True  
        intents.message_content = True       
        intents.reactions = True
        
        logger.info("Bot initializing with intents: members=%s, message_content=%s, reactions=%s, guilds=%s",
                    intents.members, intents.message_content, intents.reactions, intents.guilds)
        
        super().__init__(
            command_prefix="!.",
            intents=intents,
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="out for RMP"
            ),
            max_messages=5000,
            heartbeat_timeout=60.0,
            guild_ready_timeout=2.0,
            member_cache_flush_time=None,
            chunk_guilds_at_startup=True,
            status=discord.Status.online
        )
        
        self.db = None
        self.rate_limiter = None
        self.global_rate_limiter = None
        self.rank_tracker = None
        self.shared_session = None
        self.reaction_logger = None  
        
    async def setup_hook(self):
        logger.info("Running setup_hook...")
        
        self.db = DatabaseHandler()
        logger.info("DatabaseHandler initialised")
        
        self.rate_limiter = EnhancedRateLimiter(calls_per_minute=Config.GLOBAL_RATE_LIMIT)
        self.global_rate_limiter = GlobalRateLimiter()
        logger.info("Rate limiters initialised")
        
        DiscordAPI.initialize(self.rate_limiter)
        logger.info("DiscordAPI with rate limits initalised.")
        
        connector = aiohttp.TCPConnector(
            limit=15, 
            limit_per_host=4, 
            enable_cleanup_closed=True
        )
        self.shared_session = aiohttp.ClientSession(
            headers={"User-Agent": Config.USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=12, connect=5, sock_connect=3, sock_read=6),
            connector=connector,
            trust_env=True,
        )
        logger.info("Shared HTTP session initialised")
        
        self.rank_tracker = RankTracker(self)
        logger.info("RankTracker initialised")
        
        cogs_to_load = [
            "cogs.xp",
            "cogs.welcome", 
            "cogs.moderation",
            "cogs.reactions",
            "cogs.admin",
            "cogs.utility",
            "cogs.sc",  
        ]
        
        for cog in cogs_to_load:
            try:
                await self.load_extension(cog)
                logger.info(f"Loaded cog: {cog}")
            except Exception as e:
                logger.error(f"Failed to load cog {cog}: {e}")
        
        self.reaction_logger = self.get_cog("ReactionLoggerCog")
        if self.reaction_logger:
            logger.info("Retrieved and stored ReactionLoggerCog reference.")
            
        logger.info("Setup hook completed!")


# --- Create bot instance ---
bot = CRDB()


# --- Event Handlers ---  
@bot.event
async def on_ready():
    logger.info("=" * 50)
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    logger.info(f"Connected to {len(bot.guilds)} guild(s)")
    
    loaded_cogs = list(bot.cogs.keys())
    logger.info(f"Loaded cogs: {', '.join(loaded_cogs)}")
    
    try:
        await asyncio.wait_for(bot.tree.sync(), timeout=15.0)
        logger.info("Commands synced successfully")
    except asyncio.TimeoutError:
        logger.warning("Command sync timed out (continuing)")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")
    
    if bot.reaction_logger and hasattr(bot.reaction_logger, 'start_cleanup_task'):
        try:
            await bot.reaction_logger.start_cleanup_task()
            logger.info("Started reaction logger cleanup task")
        except Exception as e:
            logger.error(f"Failed to start cleanup task: {e}")
    
    logger.info("=" * 50)


@bot.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord")
    
    # Close shared session
    if bot.shared_session and not bot.shared_session.closed:
        await bot.shared_session.close()
        bot.shared_session = None
        logger.info("Closed shared aiohttp session")


@bot.event
async def on_resumed():
    """Called when bot resumes connection"""
    logger.info("Bot successfully resumed")
    
    # Recreate HTTP session if needed
    if not bot.shared_session or bot.shared_session.closed:
        connector = aiohttp.TCPConnector(limit=15, limit_per_host=4, enable_cleanup_closed=True)
        bot.shared_session = aiohttp.ClientSession(
            headers={"User-Agent": Config.USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=12, connect=5, sock_connect=3, sock_read=6),
            connector=connector,
            trust_env=True,
        )
        logger.info("Recreated shared aiohttp session after resume")

# --- Event Handlers ---

# --- Global Command Error Handler ---
@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
    """Global error handler for slash commands"""
    try:
        if isinstance(error, discord.app_commands.CommandOnCooldown):
            await interaction.response.send_message(
                f"⏳ Command on cooldown. Try again in {error.retry_after:.1f}s",
                ephemeral=True
            )
        elif isinstance(error, discord.app_commands.MissingPermissions):
            await interaction.response.send_message(
                "🔒 You don't have permission to use this command.",
                ephemeral=True
            )
        elif isinstance(error, discord.app_commands.CheckFailure):
            # This handles our custom permission checks
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "⛔ You don't have permission to use this command.",
                    ephemeral=True
                )
        else:
            logger.error(f"Unhandled command error: {type(error).__name__}: {error}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "❌ An error occurred while processing your command.",
                    ephemeral=True
                )
            else:
                await interaction.followup.send(
                    "❌ An error occurred while processing your command.",
                    ephemeral=True
                )
    except Exception as e:
        logger.error(f"Error in error handler: {e}")


# --- Entry Point ---
if __name__ == '__main__':
    try:
        asyncio.run(run_bot_forever(bot, TOKEN))
    except KeyboardInterrupt:
        logger.info("Bot shut down by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
