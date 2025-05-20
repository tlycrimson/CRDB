import os
import re
import time
import asyncio
import threading
import aiohttp
import discord
import logging
import random
import aiodns
import socket
from typing import Optional, Set, Dict, List, Tuple, Any, Literal
from decorators import min_rank_required, has_allowed_role
from rate_limiter import RateLimiter
from discord import app_commands
from config import Config
from discord.ext import commands
from dotenv import load_dotenv
from flask import Flask
from roblox_commands import create_sc_command
from datetime import datetime, timezone

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GOOGLE_SCRIPT_URL = os.getenv("GOOGLE_SCRIPT_URL")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
TIMEOUT = aiohttp.ClientTimeout(total=10)

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

# Global rate limiter configuration
GLOBAL_RATE_LIMIT = 15  # requests per minute
COMMAND_COOLDOWN = 10    # seconds between command uses per user

async def check_dns_connectivity() -> bool:
    """DNS resolution with multiple fallback methods"""
    domains = ['api.roblox.com', 'www.roblox.com']
    dns_servers = [
        '1.1.1.1',  # Cloudflare
        '8.8.8.8',  # Google
        '208.67.222.222'  # OpenDNS
    ]
    
    # Try public DNS servers first
    for server in dns_servers:
        try:
            resolver = aiodns.DNSResolver()
            resolver.nameservers = [server]
            
            for domain in domains:
                try:
                    result = await resolver.query(domain, 'A')
                    logger.info(f"DNS {server} resolved {domain} to {result}")
                    return True
                except aiodns.error.DNSError as e:
                    logger.warning(f"DNS {server} failed for {domain}: {str(e)}")
                    continue
        except Exception as e:
            logger.warning(f"DNS server {server} failed: {str(e)}")
            continue
    
    # Fallback to system DNS
    try:
        resolver = aiodns.DNSResolver()
        for domain in domains:
            try:
                result = await resolver.query(domain, 'A')
                logger.info(f"System DNS resolved {domain} to {result}")
                return True
            except aiodns.error.DNSError as e:
                logger.warning(f"System DNS query failed for {domain}: {str(e)}")
                continue
    except Exception as e:
        logger.warning(f"System DNS resolver failed: {str(e)}")

    # Final fallback to socket (sync)
    def sync_resolve():
        try:
            for domain in domains:
                try:
                    ip = socket.gethostbyname(domain)
                    logger.info(f"Socket resolved {domain} to {ip}")
                    return True
                except socket.gaierror as e:
                    logger.warning(f"Socket resolution failed for {domain}: {str(e)}")
                    continue
            return False
        except Exception as e:
            logger.error(f"Socket resolution error: {str(e)}")
            return False
    
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, sync_resolve)

class EnhancedRateLimiter:
    """Improved rate limiter with jitter and bucket support"""
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.buckets = {}
        self.locks = {}
        
    async def wait_if_needed(self, bucket: str = "global"):
        """Wait if needed to avoid rate limits"""
        if bucket not in self.buckets:
            self.buckets[bucket] = {'last_call': 0, 'count': 0}
            self.locks[bucket] = asyncio.Lock()
            
        async with self.locks[bucket]:
            now = time.time()
            bucket_data = self.buckets[bucket]
            
            elapsed = now - bucket_data['last_call']
            required_delay = max(1.0, (60 / self.calls_per_minute) - elapsed)
            
            if required_delay > 0:
                logger.debug(f"Rate limit wait: {required_delay:.2f}s for bucket {bucket}")
                await asyncio.sleep(required_delay)
                
            bucket_data['last_call'] = time.time()
            bucket_data['count'] += 1

    async def setup(self, interaction: discord.Interaction, log_channel: discord.TextChannel, monitor_channels: str):
        """Setup reaction monitoring"""
        await interaction.response.defer(ephemeral=True)
        try:
            channel_ids = [int(cid.strip()) for cid in monitor_channels.split(',')]
            self.monitor_channel_ids = set(channel_ids)
            self.log_channel_id = log_channel.id
            await interaction.followup.send("✅ Reaction monitoring setup complete", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Setup failed: {str(e)}", ephemeral=True)

    async def add_channels(self, interaction: discord.Interaction, channels: str):
        """Add channels to monitor"""
        await interaction.response.defer(ephemeral=True)
        try:
            channel_ids = [int(cid.strip()) for cid in channels.split(',')]
            self.monitor_channel_ids.update(channel_ids)
            await interaction.followup.send(f"✅ Added {len(channel_ids)} channels to monitoring", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to add channels: {str(e)}", ephemeral=True)

    async def remove_channels(self, interaction: discord.Interaction, channels: str):
        """Remove channels from monitoring"""
        await interaction.response.defer(ephemeral=True)
        try:
            channel_ids = [int(cid.strip()) for cid in channels.split(',')]
            self.monitor_channel_ids.difference_update(channel_ids)
            await interaction.followup.send(f"✅ Removed {len(channel_ids)} channels from monitoring", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to remove channels: {str(e)}", ephemeral=True)

    async def list_channels(self, interaction: discord.Interaction):
        """List monitored channels"""
        await interaction.response.defer(ephemeral=True)
        if not self.monitor_channel_ids:
            await interaction.followup.send("❌ No channels being monitored", ephemeral=True)
            return
            
        channel_list = "\n".join(f"• <#{cid}>" for cid in self.monitor_channel_ids)
        embed = discord.Embed(
            title="Monitored Channels",
            description=channel_list,
            color=discord.Color.blue()
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

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

class ReactionLogger:
    """Handles reaction monitoring and logging"""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.monitor_channel_ids = set(Config.DEFAULT_MONITOR_CHANNELS)
        self.log_channel_id = Config.DEFAULT_LOG_CHANNEL
        self.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)
        
    async def on_ready_setup(self):
        """Setup monitoring when bot starts"""
        await DiscordAPI.execute_with_retry(self._on_ready_setup_impl())
        
    async def _on_ready_setup_impl(self):
        guild = self.bot.guilds[0]
        valid_channels = set()
        for channel_id in self.monitor_channel_ids:
            if channel := guild.get_channel(channel_id):
                valid_channels.add(channel.id)
        
        self.monitor_channel_ids = valid_channels
        
        if not guild.get_channel(self.log_channel_id):
            logger.warning(f"Default log channel {self.log_channel_id} not found!")
            self.log_channel_id = None

    async def log_reaction(self, payload: discord.RawReactionActionEvent):
        """Log reactions from monitored channels"""
        try:
            await self.rate_limiter.wait_if_needed(bucket="reaction_log")
            await self._log_reaction_impl(payload)
        except Exception as e:
            logger.error(f"Failed to log reaction: {type(e).__name__}: {str(e)}")

    async def _log_reaction_impl(self, payload: discord.RawReactionActionEvent):
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return

        member = guild.get_member(payload.user_id)
        if not member:
            return
            
        monitor_role = guild.get_role(Config.MONITOR_ROLE_ID)
        if not monitor_role or monitor_role not in member.roles:
            return
        
        if payload.channel_id in Config.IGNORED_CHANNELS and str(payload.emoji) in Config.IGNORED_EMOJI:
            return

        if (payload.channel_id not in self.monitor_channel_ids or 
            str(payload.emoji) not in Config.TRACKED_REACTIONS):
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
                title="🧑‍💻 LD Activity Logged",
                description=f"{member.mention} reacted with {payload.emoji}",
                color=discord.Color.purple()
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

class SheetDBLogger:
    def __init__(self):
        self.script_url = os.getenv("GOOGLE_SCRIPT_URL")
        if not self.script_url:
            logger.error("❌ Google Script URL not configured in environment variables")
            self.ready = False
        else:
            self.ready = True
            logger.info("✅ SheetDB Logger configured with Google Apps Script")

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
                        return True
                    else:
                        logger.error(f"❌ Failed to update points - Status: {response.status}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.error("⏰ Timeout while connecting to Google Script")
            return False
        except Exception as e:
            logger.error(f"⚠️ Unexpected error: {type(e).__name__}: {str(e)}")
            return False

class MessageTracker:
    """Tracks messages sent by users with a specific role in specified channels"""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.monitor_channel_ids = set(Config.MESSAGE_TRACKER_CHANNELS)
        self.log_channel_id = Config.MESSAGE_TRACKER_LOG_CHANNEL
        self.tracked_role_id = Config.MESSAGE_TRACKER_ROLE_ID
        self.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)
        
    async def on_ready_setup(self):
        """Setup monitoring when bot starts"""
        guild = self.bot.guilds[0]
        valid_channels = set()
        for channel_id in self.monitor_channel_ids:
            if channel := guild.get_channel(channel_id):
                valid_channels.add(channel.id)
        
        self.monitor_channel_ids = valid_channels
        
        if not guild.get_channel(self.log_channel_id):
            logger.warning(f"Message tracker log channel {self.log_channel_id} not found!")
            self.log_channel_id = None

        if not guild.get_role(self.tracked_role_id):
            logger.warning(f"Message tracker role {self.tracked_role_id} not found!")

    async def log_message(self, message: discord.Message):
        """Log messages from monitored channels by users with tracked role"""
        try:
            await self.rate_limiter.wait_if_needed(bucket="message_log")
            await self._log_message_impl(message)
        except Exception as e:
            logger.error(f"Failed to log message: {type(e).__name__}: {str(e)}")

    async def _log_message_impl(self, message: discord.Message):
        if message.author.bot:
            return
            
        if message.channel.id not in self.monitor_channel_ids:
            return
            
        if not isinstance(message.author, discord.Member):
            return
            
        tracked_role = message.guild.get_role(self.tracked_role_id)
        if not tracked_role or tracked_role not in message.author.roles:
            return
            
        log_channel = message.guild.get_channel(self.log_channel_id)
        if not log_channel:
            return
            
        content = message.content
        if len(content) > 1000:
            content = content[:1000] + "... [truncated]"
            
        embed = discord.Embed(
            title="🎓 ED Activity Logged",
            description=f"{message.author.mention} has marked an exam or logged a course!",
            color=discord.Color.pink(),
            timestamp=message.created_at
        )
        
        embed.add_field(name="Channel", value=message.channel.mention)
        embed.add_field(name="Message ID", value=message.id)
        embed.add_field(name="Content", value=content, inline=False)
        embed.add_field(name="Jump to", value=f"[Click here]({message.jump_url})", inline=False)
        
        if message.attachments:
            attachment = message.attachments[0]
            if attachment.url.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'webp')):
                embed.set_image(url=attachment.url)
            else:
                embed.add_field(name="Attachment", value=f"[{attachment.filename}]({attachment.url})", inline=False)
        
        await log_channel.send(embed=embed)

    async def add_channels(self, interaction: discord.Interaction, channels: str):
        """Add channels to monitor"""
        await interaction.response.defer(ephemeral=True)
        try:
            channel_ids = [int(cid.strip()) for cid in channels.split(',')]
            self.monitor_channel_ids.update(channel_ids)
            await interaction.followup.send(f"✅ Added {len(channel_ids)} channels to monitoring", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to add channels: {str(e)}", ephemeral=True)

    async def remove_channels(self, interaction: discord.Interaction, channels: str):
        """Remove channels from monitoring"""
        await interaction.response.defer(ephemeral=True)
        try:
            channel_ids = [int(cid.strip()) for cid in channels.split(',')]
            self.monitor_channel_ids.difference_update(channel_ids)
            await interaction.followup.send(f"✅ Removed {len(channel_ids)} channels from monitoring", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to remove channels: {str(e)}", ephemeral=True)

    async def list_channels(self, interaction: discord.Interaction):
        """List monitored channels"""
        await interaction.response.defer(ephemeral=True)
        if not self.monitor_channel_ids:
            await interaction.followup.send("❌ No channels being monitored", ephemeral=True)
            return
            
        channel_list = "\n".join(f"• <#{cid}>" for cid in self.monitor_channel_ids)
        embed = discord.Embed(
            title="Monitored Message Channels",
            description=channel_list,
            color=discord.Color.blue()
        )
        await interaction.followup.send(embed=embed, ephemeral=True)
        
class ConfirmView(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.value = None

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = True
        await interaction.response.defer()
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = False
        await interaction.response.defer()
        self.stop()

# --- Bot Initialization ---
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.guilds = True
intents.reactions = True

bot = commands.Bot(
    intents=intents,
    command_prefix="!.",
    activity=discord.Activity(type=discord.ActivityType.watching, name="out for RMP"),
    max_messages=None,
    heartbeat_timeout=60.0
)
bot.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)
bot.reaction_logger = ReactionLogger(bot)
bot.message_tracker = MessageTracker(bot)
bot.api = DiscordAPI()

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

# Creating Sc command
create_sc_command(bot)

@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    logger.info(f"Connected to {len(bot.guilds)} guild(s)")

    bot.sheets = SheetDBLogger()
    if not bot.sheets.ready:
        logger.warning("SheetDB Logger not initialized properly")
    else:
        logger.info("SheetDB Logger initialized successfully")
        
    await bot.reaction_logger.on_ready_setup()
    await bot.message_tracker.on_ready_setup()
    
    bot.shared_session = aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        connector=aiohttp.TCPConnector(
            force_close=True,
            enable_cleanup_closed=True,
            resolver=aiohttp.AsyncResolver() if await check_dns_connectivity() else None
        )
    )

    try:
        synced = await bot.api.execute_with_retry(
            bot.tree.sync(),
            max_retries=5,
            initial_delay=2.0
        )
        logger.info(f"Synced {len(synced)} commands")
    except Exception as e:
        logger.error(f"Command sync error: {e}")

@bot.event
async def on_disconnect():
    if hasattr(bot, 'shared_session') and bot.shared_session:
        await bot.shared_session.close()
        logger.info("Closed shared HTTP session")
        
# --- Commands --- 
# Discharge Command
@bot.tree.command(name="discharge", description="Notify members of honorable/dishonorable discharge")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def discharge(
    interaction: discord.Interaction,
    members: str,  # Comma-separated user mentions/IDs
    reason: str,
    discharge_type: Literal["Honourable", "Dishonorable"] = "Honourable",
    evidence: Optional[discord.Attachment] = None
):
    view = ConfirmView()
    await interaction.response.send_message("Confirm discharge?", view=view, ephemeral=True)
    await view.wait()
    if not view.value:
        return

    try:
        # Reason character limit check
        if len(reason) > 1000:
            await interaction.followup.send(
                "❌ Reason must be under 1000 characters",
                ephemeral=True
            )
            return

        member_ids = []
        for mention in members.split(','):
            mention = mention.strip()
            try:
                if mention.startswith('<@') and mention.endswith('>'):
                    member_id = int(mention[2:-1].replace('!', ''))  # Handle nicknames
                else:
                    member_id = int(mention)
                member_ids.append(member_id)
            except ValueError:
                logger.warning(f"Invalid member identifier: {mention}")

        await interaction.followup.defer(ephemeral=True)
        await bot.rate_limiter.wait_if_needed(bucket="discharge")

        discharged_members = []
        for member_id in member_ids:
            if member := interaction.guild.get_member(member_id):
                discharged_members.append(member)
            else:
                logger.warning(f"Member {member_id} not found in guild")

        if not discharged_members:
            await interaction.followup.send("❌ No valid members found.", ephemeral=True)
            return

        # Embed creation
        color = discord.Color.green() if discharge_type == "Honourable" else discord.Color.red()
        embed = discord.Embed(
            title=f"{discharge_type} Discharge Notification",
            color=color,
            timestamp=datetime.now(timezone.utc)
        )
        embed.add_field(name="Reason", value=reason, inline=False)

        if evidence:
            embed.add_field(name="Evidence", value=f"[Attachment Link]({evidence.url})", inline=False)
            mime, _ = mimetypes.guess_type(evidence.filename)
            if mime and mime.startswith("image/"):
                embed.set_image(url=evidence.url)

        embed.set_footer(text=f"Discharged by {interaction.user.display_name}")

        success_count = 0
        failed_members = []

        for member in discharged_members:
            try:
                try:
                    await member.send(embed=embed)
                except discord.Forbidden:
                    if channel := interaction.guild.get_channel(1165368316970405917):
                        await channel.send(f"{member.mention}", embed=embed)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to notify {member.display_name}: {str(e)}")
                failed_members.append(member.mention)

        result_embed = discord.Embed(
            title="Discharge Summary",
            color=color
        )
        result_embed.add_field(name="Action", value=f"{discharge_type} Discharge", inline=False)
        result_embed.add_field(
            name="Results",
            value=f"✅ Successfully notified: {success_count}\n❌ Failed: {len(failed_members)}",
            inline=False
        )
        if failed_members:
            result_embed.add_field(name="Failed Members", value=", ".join(failed_members), inline=False)

        await interaction.followup.send(embed=result_embed, ephemeral=True)

        # Log to D_LOG_CHANNEL_ID
        if d_log := interaction.guild.get_channel(Config.D_LOG_CHANNEL_ID):
            log_embed = discord.Embed(
                title=f"{discharge_type} Discharge Log",
                color=color,
                timestamp=datetime.now(timezone.utc)
            )
            log_embed.add_field(
                name="Type",
                value=f"🔰 {discharge_type} Discharge" if discharge_type == "Honourable" else f"⚠️ {discharge_type} Discharge",
                inline=False
            )
            log_embed.add_field(name="Reason", value=f"```{reason}```", inline=False)
            log_embed.add_field(
                name="Affected Members",
                value="\n".join(m.mention for m in discharged_members) or "None",
                inline=False
            )
            if evidence:
                log_embed.add_field(name="Evidence", value=f"[View Attachment]({evidence.url})", inline=True)

            log_embed.add_field(name="Processed By", value=interaction.user.mention, inline=True)
            log_embed.add_field(name="Channel", value=interaction.channel.mention, inline=True)

            await d_log.send(embed=log_embed)

    except Exception as e:
        logger.error(f"Discharge command failed: {e}")
        await interaction.followup.send("❌ An error occurred while processing the discharge.", ephemeral=True)
    

#LD Reaction Monitor Set-up Command           
@bot.tree.command(name="message-tracker-setup", description="Setup message monitoring")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)  
async def message_tracker_setup(
    interaction: discord.Interaction,
    log_channel: discord.TextChannel,
    monitor_channels: str,
    role: discord.Role
):
    """Setup message tracking"""
    await interaction.response.defer(ephemeral=True)
    try:
        channel_ids = [int(cid.strip()) for cid in monitor_channels.split(',')]
        bot.message_tracker.monitor_channel_ids = set(channel_ids)
        bot.message_tracker.log_channel_id = log_channel.id
        bot.message_tracker.tracked_role_id = role.id
        await interaction.followup.send("✅ Message tracking setup complete", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"❌ Setup failed: {str(e)}", ephemeral=True)

@bot.tree.command(name="message-tracker-add", description="Add channels to message monitoring")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)  # Or whatever role you want
async def message_tracker_add(
    interaction: discord.Interaction,
    channels: str
):
    """Add channels to message tracking"""
    await bot.message_tracker.add_channels(interaction, channels)

@bot.tree.command(name="message-tracker-remove", description="Remove channels from message monitoring")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID) 
async def message_tracker_remove(
    interaction: discord.Interaction,
    channels: str
):
    """Remove channels from message tracking"""
    await bot.message_tracker.remove_channels(interaction, channels)

@bot.tree.command(name="message-tracker-list", description="List currently monitored channels")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)  
async def message_tracker_list(interaction: discord.Interaction):
    """List channels being tracked for messages"""
    await bot.message_tracker.list_channels(interaction)


@bot.tree.command(name="force-update", description="Manually test sheet updates")
@has_allowed_role()
async def force_update(interaction: discord.Interaction, username: str):
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

@bot.tree.command(name="commands", description="List all available commands")
@has_allowed_role()
async def command_list(interaction: discord.Interaction):
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
        "💬 Message Tracking": [
            "/message-tracker-setup - Setup message tracking",
            "/message-tracker-add - Add channels to monitor",
            "/message-tracker-remove - Remove monitored channels",
            "/message-tracker-list - List monitored channels"
        ],
        "🛠️ Utility": [
            "/ping - Check bot responsiveness",
            "/commands - Show this help message",
            "/sheetdb-test - Test SheetDB connection",
            "/sc - Security Check Roblox user",
            "/discharge - Sends discharge notification to user and logs in discharge logs."
        ]
    }
    
    for name, value in categories.items():
        embed.add_field(name=name, value="\n".join(value), inline=False)
    
    await interaction.response.send_message(embed=embed)
    

@bot.tree.command(name="ping", description="Check bot latency")
@has_allowed_role()
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)
    await interaction.response.send_message(
        f"🏓 Pong! Latency: {latency}ms",
        ephemeral=True
    )

@bot.tree.command(name="reaction-setup", description="Setup reaction monitoring")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def reaction_setup(
    interaction: discord.Interaction,
    log_channel: discord.TextChannel,
    monitor_channels: str
):
    await bot.reaction_logger.rate_limiter.setup(interaction, log_channel, monitor_channels)

@bot.tree.command(name="reaction-add", description="Add channels to monitor")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def reaction_add(
    interaction: discord.Interaction,
    channels: str
):
    await bot.reaction_logger.rate_limiter.add_channels(interaction, channels)

@bot.tree.command(name="reaction-remove", description="Remove channels from monitoring")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def reaction_remove(
    interaction: discord.Interaction,
    channels: str
):
    await bot.reaction_logger.rate_limiter.remove_channels(interaction, channels)

@bot.tree.command(name="reaction-list", description="List monitored channels")
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def reaction_list(interaction: discord.Interaction):
    await bot.reaction_logger.rate_limiter.list_channels(interaction)

@bot.tree.command(name="sheetdb-test", description="Test SheetDB connection")
async def sheetdb_test(interaction: discord.Interaction):
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

# HR Welcome Message
async def send_hr_welcome(member: discord.Member):
    if not (welcome_channel := member.guild.get_channel(Config.DESERTER_ALERT_CHANNEL_ID)):
        logger.warning("HR welcome channel not found!")
        return

    embed = discord.Embed(
        title="🎉 Welcome to the HR Team!",
        description=(
            f"{member.mention}\n\n"
            "**Please note the following:**\n"
            "• Request for document access in [HR Documents](https://discord.com/channels/1165368311085809717/1165368317532438646)\n"
            "• You are exempted from quota this week only - you start next week ([Quota Info](https://discord.com/channels/1165368311085809717/1206998095552978974))\n"
            "• Uncomplete quota = strike\n"
            "• One failed tryout allowed if your try quota portion ≥2\n"
            "• Ask for help anytime - we're friendly!\n"
            "• Are you Captain+ in BA? Apply for departments: [Applications](https://discord.com/channels/1165368311085809717/1165368316970405916)"
        ),
        color=discord.Color.gold(),  
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.set_footer(text="We're excited to have you on the team!")
    
    try:
        await welcome_channel.send(content=f"{member.mention}", embed=embed)
        logger.info(f"Sent HR welcome to {member.display_name}")
    except discord.Forbidden:
        logger.error(f"Missing permissions to send HR welcome for {member.id}")
    except Exception as e:
        logger.error(f"Failed to send HR welcome: {str(e)}")

# RMP Welcome Message
async def send_rmp_welcome(member: discord.Member):
    # First embed (welcome message)
    embed1 = discord.Embed(
        title="Welcome to the Royal Military Police",
        description="**1.** Make sure to read all of the rules found in <#1165368313925353580>\n\n"
                   "**2.** You **MUST** read the RMP main guide and MSL before starting your duties.\n\n"
                   "**3.** You can't use your L85 unless you are doing it for Self-Militia. (Self-defence)\n\n"
                   "**4.** Make sure to follow the Chain Of Command. Inspector > Chief Inspector > Superintendent > Major > Lieutenant Colonel > Colonel > Commander > Provost Marshal\n\n"
                   "**5.** For phases, you may wait for one to be hosted in <#1207367013698240584> or request the phase you need in <#1270700562433839135>.\n\n"
                   "**6.** All the information about the Defence School of Policing and Guarding is found in both <#1237062439720452157> and <#1207366893631967262>\n\n"
                   "**7.** Choose your timezone here https://discord.com/channels/1165368311085809717/1165368313925353578\n\n"
                   "**8.** Please read the following announcements if you're unsure on some stuff: [TC Mini Guide ](https://discord.com/channels/1165368311085809717/1237062439720452157/1363987177394143354), [PD and Stage Rules](https://discord.com/channels/1165368311085809717/1165368314462212158/1365042797052432384) and [Uniform policy](https://discord.com/channels/1165368311085809717/1165368314462212158/1366080537638342729).\n\n"
                   "**Besides that, good luck with your phases!**",
        color=discord.Color.red()
    )

    # Second embed (detailed rules)
    embed2 = discord.Embed(
        title="Trainee Constable Brochure",
        color=discord.Color.blue()
    )
    
    embed2.add_field(
        name="__TOP 5 RULES__",
        value="> **1**. You **MUST** read the RMP main guide and MSL before starting your duties.\n"
              "> **2**. You **CANNOT** enforce the MSL. Only the Parade Deck (PD) rules **AFTER** you pass your phase 1.\n"
              "> **3**. You **CANNOT** use your bike on the PD or the pavements.\n"
              "> **4**. You **MUST** use good spelling and grammar to the best of your ability.\n"
              "> **5**. You **MUST** remain mature and respectful at all times.",
        inline=False
    )
    
    embed2.add_field(
        name="__Who's allowed on the PD at all times?__",
        value="↠ Royal Army Medical Corps,\n"
              "↠ Royal Military Police,\n" 
              "↠ Intelligence Corps.",
        inline=False
    )
    
    embed2.add_field(
        name="__Who's allowed on the PD when carrying out duties?__",
        value="↠ United Kingdom Special Forces,\n"
              "↠ Grenadier Guards,\n"
              "↠ Foreign Relations,\n"
              "↠ Royal Logistic Corps,\n"
              "↠ Adjutant General's Corps,\n"
              "↠ High Ranks, RSM, CSM and ASM hosting,\n"
              "↠ Regimental personnel watching one of their regiment's events inside Pad area.",
        inline=False
    )
    
    embed2.add_field(
        name="__How do I enforce PD rules on people not allowed on it?__",
        value="1. Give them their first warning to get off the PD, \"W1, off the PD!\"\n"
              "2. Wait 3-5 seconds for them to listen if they don't give them their second warning, \"W2, off the PD!\"\n"
              "3. Wait 3-5 seconds for them to listen if they don't kill them.",
        inline=False
    )
    
    embed2.add_field(
        name="__Who's allowed ON the actual Stage at all times?__",
        value="↠ Major General and above,\n"
              "↠ Royal Family (they have a purple name tag),\n"
              "↠ Home Command Regiments who have been given permission to guard a Major General or Royal Family member,\n"
              "↠ Those who have been given permission by a Lieutenant General.",
        inline=False
    )
    
    embed2.add_field(
        name="__Who's allowed to past the red line in-front of the Stage?__",
        value="↠ Major General and above,\n"
              "↠ Royal Family,\n"
              "↠ Those who have been given permission by a Lieutenant General,\n"
              "↠ COMBATIVE Home Command Regiments:\n"
              "- Royal Military Police,\n"
              "- United Kingdom Forces,\n"
              "- Household Division.\n\n"
              "**Kill those not allowed who touch or past the red line.**",
        inline=False
    )
    
    embed2.add_field(
        name="\u200b",  # Empty field name for spacing
        value="**LASTLY, IF YOU'RE UNSURE ON SOMETHING ASK SOMEONE ABOVE YOU BEFORE TAKING ACTION!**",
        inline=False
    )

    try:
        await member.send(embeds=[embed1, embed2])
    except discord.Forbidden:
        if welcome_channel := member.guild.get_channel(722002957738180620):
            await welcome_channel.send(f"{member.mention}", embeds=[embed1, embed2])
            logger.info(f"Sending welcome message to {member.display_name} ({member.id})")
    except discord.HTTPException as e:
        logger.error(f"Failed to send welcome message: {e}")

@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    # Check for HR role
    if hr_role := after.guild.get_role(Config.HR_ROLE_ID):
        if hr_role not in before.roles and hr_role in after.roles:
            await send_hr_welcome(after)
    
    # Check for RMP role
    if rmp_role := after.guild.get_role(Config.RMP_ROLE_ID):
        if rmp_role not in before.roles and rmp_role in after.roles:
            await send_rmp_welcome(after)

@bot.event
async def on_member_remove(member: discord.Member):
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
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    await bot.reaction_logger.log_reaction(payload)

@bot.event
async def on_message(message: discord.Message):
    # Process commands first
    await bot.process_commands(message)
    
    # Then track messages
    await bot.message_tracker.log_message(message)

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
