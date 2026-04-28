import os
import asyncio
import aiohttp
import discord
import logging
from discord.ext import commands
from dotenv import load_dotenv

from utils.roblox import RobloxClient
from utils.permissions import PermissionsCache
from utils.views import restore_approval_views
from utils.runners import run_bot_forever
from config import Config
from utils.database import DatabaseHandler
from utils.network import EnhancedRateLimiter, GlobalRateLimiter, DiscordAPI
from utils.rank_tracker import RankTracker

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise ValueError("DISCORD TOKEN NOT FOUND!")


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)
# Silence Requests
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)

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
            command_prefix="!",
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
        self.roblox = None
        self.permissions = None

    async def setup_hook(self):
        logger.info("Running setup_hook...")
        
        self.db = DatabaseHandler()
        await self.db.initialise()
        logger.info("DatabaseHandler initialised")
        
        self.rate_limiter = EnhancedRateLimiter(calls_per_minute=Config.GLOBAL_RATE_LIMIT)
        self.global_rate_limiter = GlobalRateLimiter()
        logger.info("Rate limiters initialised")
        
        DiscordAPI.initialize(self.rate_limiter)
        logger.info("DiscordAPI with rate limits initalised.")
        
        self.roblox = RobloxClient()
        await self.roblox.start()
        logger.info("Roblox Client initialised.")

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

        self.permissions = PermissionsCache(self)
        await self.permissions.initialise()
        logger.info("Permissions intialised")
        
        cogs_to_load = [
            "cogs.xp",
            "cogs.welcome", 
            "cogs.moderation",
            "cogs.reactions",
            "cogs.admin",
            "cogs.utility",
            "cogs.sc",  
            "cogs.messages"
        ]
        
        for cog in cogs_to_load:
            try:
                await self.load_extension(cog)
                logger.info(f"Loaded cog: {cog}")
            except Exception as e:
                logger.error(f"Failed to load cog {cog}: {e}")
        
            
        logger.info("Setup hook completed!")

    async def close(self):
        logger.info("Bot shutting down, closing connections...")

        if self.roblox:
            await self.roblox.close()
            logger.info("RobloxClient session closed")

        if self.shared_session and not self.shared_session.closed:
            await self.shared_session.close()
            logger.info("Shared aiohttp session closed")

        await super().close()

# --- Create bot instance ---
bot = CRDB()


# --- Event Handlers ---  
@bot.event
async def on_ready():
    logger.info("=" * 50)
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    logger.info(f"Connected to {len(bot.guilds)} guild(s)")

    try:
        await restore_approval_views(bot)
    except Exception as e:
        logger.exception("restore_approval_views failed: %s", e)
   
    try:
        await asyncio.wait_for(bot.tree.sync(), timeout=15.0)
        logger.info("Commands synced successfully")
    except asyncio.TimeoutError:
        logger.warning("Command sync timed out (continuing)")
    except Exception as e:
        logger.exception(f"Failed to sync commands: %s", e)
   
    
    if bot.db:
        try:
            await bot.db.welcome_initialise()
        except Exception as e:
            logger.exception("welcome_initialise failed: %s", e)

    if bot.reaction_logger and hasattr(bot.reaction_logger, 'start_cleanup_task'):
        try:
            await bot.reaction_logger.start_cleanup_task()
            logger.info("Started reaction logger cleanup task")
        except Exception as e:
            logger.exception(f"Failed to start cleanup task: %s", e)

    BANNER_PATH = "banner.gif"
    if os.path.exists(BANNER_PATH):
        try:
            with open(BANNER_PATH, "rb") as f:
                banner_bytes = f.read()
            await bot.user.edit(banner=banner_bytes)
            logger.info("Bot Banner updated successfully!")
        except discord.HTTPException as e:
            logger.info("Failed to set banner: {e}")
    else:
        logger.info("Banner not found at {BANNER_PATH}")
 
    logger.info("=" * 50)


# --- Event Handlers ---
@bot.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord")
    

@bot.event
async def on_resumed():
    """Called when bot resumes connection"""
    logger.info("Bot successfully resumed")
    

# --- Global Command Error Handler ---
@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    """Global error handler for slash commands"""
    try:
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(
                f"```⏳ Command on cooldown. Try again in {error.retry_after:.1f}s```",
                ephemeral=True
            )
        elif isinstance(error, commands.MissingPermissions):
            if ctx.interaction:
                await ctx.send(
                    "```⛔ You don't have permission to use this command.```",
                    ephemeral=True
                )
        elif isinstance(error, commands.CheckFailure):
            if ctx.interaction:
                await ctx.send(
                    "```⛔ You don't have permission to use this command.```",
                    ephemeral=True
                )
        elif isinstance(error, commands.CommandInvokeError):
                if isinstance(error.original, discord.Forbidden):
                    try:
                        await ctx.send("``❌ I don't have the necessary permissions (like 'Embed Links') in this channel. Try using the `/` version!```")
                    except discord.Forbidden:
                        try:
                            await ctx.author.send(f"```❌ I couldn't respond in {ctx.channel.mention} because I'm missing permissions there.```")
                        except discord.Forbidden:
                            pass
        elif isinstance(error, (commands.BadLiteralArgument, commands.BadArgument, commands.MissingRequiredArgument)):
                    command = ctx.command
                    command_structure = command.usage if command.usage else command.signature
                    
                    error_msg = (
                        f"```❌ Invalid Command Usage```\n"
                        f"**Structure:**\n```{ctx.clean_prefix}{ctx.invoked_with} {command_structure}```\n"
                    )

                    if isinstance(error, commands.BadLiteralArgument):
                        options = ", ".join(f"`{l}`" for l in error.literals)
                        error_msg += f"**Valid Choices:** {options}"
                    
                    await ctx.send(error_msg, ephemeral=True)
        else:
            logger.error(f"Unhandled command error: {type(error).__name__}: {error}", exc_info=True)
            await ctx.send(
                "```❌ An error occurred while processing your command.```",
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
