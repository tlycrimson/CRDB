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
import mimetypes
from typing import Optional, Set, Dict, List, Tuple, Any, Literal
from decorators import min_rank_required, has_allowed_role
from rate_limiter import RateLimiter
from discord import app_commands
from config import Config
from discord.ext import commands
from discord.utils import escape_markdown
from dotenv import load_dotenv
from flask import Flask
from roblox_commands import create_sc_command
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GOOGLE_SCRIPT_URL = os.getenv("GOOGLE_SCRIPT_URL")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
TIMEOUT = aiohttp.ClientTimeout(total=10)

# XP Limit Configuration
MAX_XP_PER_ACTION =  20  # Maximum XP that can be given/taken in a single action
MAX_EVENT_XP_PER_USER = 20 # Maximum XP per user in event distributions
MAX_EVENT_TOTAL_XP = 5000  # Maximum total XP for entire event distribution

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


# Username Cleaner
def clean_nickname(nickname: str) -> str:
    """Remove tags like [INS] from nicknames and clean whitespace"""
    if not nickname:  # Handle None or empty string
        return "Unknown"
    cleaned = re.sub(r'\[.*?\]', '', nickname).strip()
    return cleaned or nickname  # Fallback to original if empty after cleaning


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


# --- CLASSES ---
class EnhancedRateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.buckets = {}
        self.locks = {}
        self.global_lock = asyncio.Lock()
        self.last_global_reset = time.time()
        self.global_count = 0
        # Add bucket cleanup tracking
        self.last_bucket_cleanup = time.time()
        
    async def _cleanup_old_buckets(self):
        """Periodically clean up unused buckets"""
        async with self.global_lock:
            now = time.time()
            if now - self.last_bucket_cleanup > 3600:  # Cleanup every hour
                stale_buckets = [
                    bucket for bucket, data in self.buckets.items()
                    if now - data['last_call'] > 86400  # 24 hours inactive
                ]
                for bucket in stale_buckets:
                    del self.buckets[bucket]
                    if bucket in self.locks:
                        del self.locks[bucket]
                self.last_bucket_cleanup = now

    async def wait_if_needed(self, bucket: str = "global"):
        """Improved rate limiting with global and local limits"""
        now = time.time()
        
        # Periodic bucket cleanup
        if random.random() < 0.01:  # 1% chance to trigger cleanup
            await self._cleanup_old_buckets()

        # Global rate limiting
        async with self.global_lock:
            # Reset global counter every second
            if now - self.last_global_reset >= 1.0:
                self.last_global_reset = now
                self.global_count = 0
                
            # Allow bursts but stay under 50/sec limit
            if self.global_count >= 48:  # Increased from 45 to 48
                wait_time = 1.0 - (now - self.last_global_reset)
                await asyncio.sleep(wait_time)
                self.last_global_reset = time.time()
                self.global_count = 0
                
            self.global_count += 1
            
        # Local bucket rate limiting
        if bucket not in self.buckets:
            async with self.global_lock:  # Prevent race condition on bucket creation
                if bucket not in self.buckets:  # Double-check
                    self.buckets[bucket] = {
                        'last_call': 0,
                        'count': 0,
                        'window_start': now
                    }
                    self.locks[bucket] = asyncio.Lock()
            
        async with self.locks[bucket]:
            bucket_data = self.buckets[bucket]
            
            # Reset count if we're in a new minute window
            if now - bucket_data['window_start'] > 60:
                bucket_data['count'] = 0
                bucket_data['window_start'] = now
                
            # Calculate required delay
            elapsed = now - bucket_data['last_call']
            required_delay = max(0.02, (60 / self.calls_per_minute) - elapsed)
            
            if required_delay > 0:
                await asyncio.sleep(required_delay)
                
            # Update bucket data
            bucket_data['last_call'] = time.time()
            bucket_data['count'] += 1

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

#supabase set-up
class DatabaseHandler:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(self.url, self.key)
    
    async def add_xp(self, user_id: str, username: str, xp: int):
        try:
            current_xp = await self.get_user_xp(user_id)
            new_xp = current_xp + xp
            
            data, _ = self.supabase.table('users').upsert({
                "user_id": str(user_id),
                "username": clean_nickname(username),
                "xp": new_xp,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }).execute()
            
            logger.info(f"XP added: {user_id} gained {xp} XP (was {current_xp}, now {new_xp})")
            return True, new_xp  # Return both success and new XP
        except Exception as e:
            logger.error(f"Failed to add XP for {user_id}: {str(e)}")
            return False, current_xp
            
    async def take_xp(self, user_id: str, username: str, xp: int):
        try:
            current_xp = await self.get_user_xp(user_id)
            new_xp = max(0, current_xp - xp)  # Prevent negative XP
            
            data, _ = self.supabase.table('users').upsert({
                "user_id": str(user_id),
                "username": clean_nickname(username),
                "xp": new_xp,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }).execute()
            
            logger.info(f"XP removed: {user_id} lost {xp} XP (was {current_xp}, now {new_xp})")
            return True, new_xp  # Now returning both success status and new XP
        except Exception as e:
            logger.error(f"Failed to remove XP from {user_id}: {str(e)}")
            return False, current_xp  # Return current XP on failure
    
    async def log_event(self, user_id: str, event_type: str):
        try:
            week_num = datetime.now().isocalendar()[1]
            data, _ = self.supabase.table('events').insert({
                "user_id": str(user_id),
                "event_type": event_type,
                "week_number": week_num
            }).execute()
            logger.info(f"Event logged: {user_id} - {event_type} (Week {week_num})")
            return True
        except Exception as e:
            logger.error(f"Failed to log event for {user_id}: {str(e)}")
            return False
    
    async def get_user_xp(self, user_id: str):
        try:
            data = self.supabase.table('users').select("xp").eq("user_id", str(user_id)).execute()
            xp = data.data[0].get('xp', 0) if data.data else 0
            logger.debug(f"Retrieved XP for {user_id}: {xp}")  #
            return xp
        except Exception as e:
            logger.error(f"Failed to get XP for {user_id}: {str(e)}")
            return 0
    
    async def clear_weekly_events(self):
        try:
            current_week = datetime.now().isocalendar()[1]
            self.supabase.table('events').delete().lt("week_number", current_week).execute()
            logger.info(f"Cleared events older than week {current_week}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear weekly events: {str(e)}")
            return False

# Logs XP changes in logging channel
async def log_xp_to_discord(
    admin: discord.User,
    user: discord.User,
    xp_change: int,
    new_total: int,
    reason: str
):
    """Log XP changes to a Discord channel instead of Supabase."""
    log_channel = bot.get_channel(Config.DEFAULT_LOG_CHANNEL)  # Replace with your channel ID
    if not log_channel:
        logger.error("XP log channel not found!")
        return False

    embed = discord.Embed(
        title="📊 XP Change Logged",
        color=discord.Color.green() if xp_change > 0 else discord.Color.red(),
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(name="Staff", value=f"{admin.mention} (`{admin.id}`)", inline=True)
    embed.add_field(name="User", value=f"{user.mention} (`{user.id}`)", inline=True)
    embed.add_field(name="XP Change", value=f"`{xp_change:+}`", inline=True)
    embed.add_field(name="New Total", value=f"`{new_total}`", inline=True)
    embed.add_field(name="Reason", value=f"```{reason}```", inline=False)
    
    try:
        await log_channel.send(embed=embed)
        return True
    except Exception as e:
        logger.error(f"Failed to log XP to Discord: {str(e)}")
        return False


# Reaction Logger for LD
class ReactionLogger:
    """Handles reaction monitoring and logging with hard-wired configuration"""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.monitor_channel_ids = set(Config.DEFAULT_MONITOR_CHANNELS)
        self.log_channel_id = Config.DEFAULT_LOG_CHANNEL
        self.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)
        self.event_channel_ids = [Config.W_EVENT_LOG_CHANNEL_ID, Config.EVENT_LOG_CHANNEL_ID]
        self.phase_log_channel_id = Config.PHASE_LOG_CHANNEL_ID
        self.tryout_log_channel_id = Config.TRYOUT_LOG_CHANNEL_ID
        self.course_log_channel_id = Config.COURSE_LOG_CHANNEL_ID
        self.activity_log_channel_id = Config.ACTIVITY_LOG_CHANNEL_ID
    
            
    async def on_ready_setup(self):
        """Verify configured channels when bot starts"""
        guild = self.bot.guilds[0]
        valid_channels = set()
        for channel_id in self.monitor_channel_ids:
            if channel := guild.get_channel(channel_id):
                valid_channels.add(channel.id)
        
        self.monitor_channel_ids = valid_channels
        
        if not guild.get_channel(self.log_channel_id):
            logger.warning(f"Default log channel {self.log_channel_id} not found!")
            self.log_channel_id = None
            
    async def _log_reaction_impl(self, payload: discord.RawReactionActionEvent):
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
    
        if (payload.channel_id not in self.monitor_channel_ids or 
            str(payload.emoji) not in Config.TRACKED_REACTIONS):
            return
    
        if (payload.channel_id in Config.IGNORED_CHANNELS and 
            str(payload.emoji) in Config.IGNORED_EMOJI):
            return
    
        member = guild.get_member(payload.user_id)
        if not member:
            return
            
        if Config.LD_ROLE_ID:
            monitor_role = guild.get_role(Config.LD_ROLE_ID)
            if not monitor_role or monitor_role not in member.roles:
                return
    
        channel = guild.get_channel(payload.channel_id)
        log_channel = guild.get_channel(self.log_channel_id)
    
        if not all((channel, member, log_channel)):
            return
    
        try:
            # Add small delay before processing
            await asyncio.sleep(0.5)
            
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
                
            await log_channel.send(embed=embed)
            
            logger.info(f"Attempting to update points for: {member.display_name}")
            update_success = await self.bot.sheets.update_points(member)
            
            
        except discord.NotFound:
            return
        except Exception as e:
            logger.error(f"Reaction log error: {type(e).__name__}: {str(e)}")
            error_embed = discord.Embed(
                title="❌ Error",
                description=f"Failed to log reaction: {str(e)}",
                color=discord.Color.red()
            )
            await log_channel.send(embed=error_embed)
   
    async def _log_event_reaction_impl(self, payload: discord.RawReactionActionEvent):
        """Handle event logging without confirmation"""
        if payload.channel_id not in self.event_channel_ids or str(payload.emoji) != "✅":
            return
            
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
     
        member = guild.get_member(payload.user_id)
        if not member:
            return
            
        ld_role = guild.get_role(Config.LD_ROLE_ID)
        if not ld_role or ld_role not in member.roles:
            return
            
        channel = guild.get_channel(payload.channel_id)
        log_channel = guild.get_channel(self.log_channel_id)
        if not channel or not log_channel:
            return
            
        try:
            # Add small delay before processing
            await asyncio.sleep(0.5)
            
            message = await channel.fetch_message(payload.message_id)
            
            # Extract host
            host_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
            if host_mention:
                host_id = int(host_mention.group(1))
            else:
                logger.info(f"Host not mentioned. Attempting to use author id instead.")
                host_id = message.author.id
            
            host_member = guild.get_member(host_id)
            if not host_member:
                return
                
            cleaned_host_name = clean_nickname(host_member.display_name)
            
            
            # Determine event type and name
            event_content = message.content.lower()
            hr_update_field = "events"  # default
            
            # Check for special event types first (preserves existing inspection/joint logic)
            if payload.channel_id == Config.W_EVENT_LOG_CHANNEL_ID:
                if re.search(r'\bjoint\b', message.content, re.IGNORECASE): 
                    hr_update_field = "joint_events"
                elif re.search(r'\b(inspection|pi|Inspection)\b', message.content, re.IGNORECASE):
                    hr_update_field = "inspections"
            
            # Extract event name (looking for "Event: [name]")
            event_name_match = re.search(r'Event:\s*(.*?)(?:\n|$)', message.content, re.IGNORECASE)
            event_name = event_name_match.group(1).strip() if event_name_match else hr_update_field.replace("_", " ").title()
            
            # Record host in HRs table with dynamic event type
            try:
                await self._update_hr_record(
                    user_id=host_id,
                    username=cleaned_host_name,
                    **{hr_update_field: 1}
                )
                logger.info(f"✅ Logged host {cleaned_host_name} to HR table")
            except Exception as e:
                logger.error(f"❌ Failed to log host {host_id}: {str(e)}")
                
            # Extract attendees/passed
            attendees_section = re.search(r'(?:Attendees:|Passed:)\s*((?:<@!?\d+>\s*)+)', message.content, re.IGNORECASE)  
            attendee_mentions = re.findall(r'<@!?(\d+)>', attendees_section.group(1))
            
            if not attendee_mentions:
                logger.info("No attendees found in message. Skipping LR logging.")
            else:
                attendee_mentions = re.findall(r'<@!?(\d+)>', attendees_section.group(1))
    
            # Get HR role
            hr_role = guild.get_role(Config.HR_ROLE_ID) if Config.HR_ROLE_ID else None
            
            # Identify HR attendees
            hr_attendees = []
            if hr_role:
                hr_attendees = [
                    attendee_id for attendee_id in attendee_mentions
                    if (attendee := guild.get_member(int(attendee_id))) and hr_role in attendee.roles
                ]
                
            # Record non-HR attendees in LRs table
            success_count = 0
            for attendee_id in attendee_mentions:
                try:
                    attendee_member = guild.get_member(int(attendee_id))
                    if not attendee_member:
                        continue
                        
                    if hr_role and hr_role in attendee_member.roles:
                        continue
                        
                    await self._update_lr_record(
                        user_id=attendee_id,
                        username=clean_nickname(attendee_member.display_name),
                        events_attended=1
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to record attendee {attendee_id}: {str(e)}")
                    continue
                    
            # Send completion embed to DEFAULT_LOG_CHANNEL
            done_embed = discord.Embed(
                title="✅ Event Logged Successfully",
                color=discord.Color.green()
            )
            done_embed.add_field(name="Host", value=f"{host_member.mention}", inline=True)
            done_embed.add_field(name="Attendees Recorded", value=str(success_count), inline=True)
            
            # Only add HR attendees field if there are any
            if hr_attendees:
                done_embed.add_field(
                    name="HR Attendees Excluded", 
                    value=str(len(hr_attendees)), 
                    inline=False
                )
                
            done_embed.add_field(name="Logged By", value=member.mention, inline=False)
            done_embed.add_field(name="Event Type", value=event_name, inline=True)  # Using extracted event name
            done_embed.add_field(name="Message", value=f"[Jump to Event]({message.jump_url})", inline=False)
            
            await log_channel.send(embed=done_embed)
    
        except Exception as e:
            logger.error(f"Error processing event reaction: {str(e)}")
            error_embed = discord.Embed(
                title="❌ Event Log Error",
                description=str(e),
                color=discord.Color.red()
            )
            await log_channel.send(embed=error_embed)


    async def _log_training_reaction_impl(self, payload: discord.RawReactionActionEvent):
        """Handle training logs (phases, tryouts, courses)"""
        channel_id = payload.channel_id
        column_to_update = None
        
        if channel_id == self.phase_log_channel_id:
            column_to_update = "phases"
        elif channel_id == self.tryout_log_channel_id:
            column_to_update = "tryouts"
        elif channel_id == self.course_log_channel_id:
            column_to_update = "courses"
        else:
            return
            
        if str(payload.emoji) != "✅":
            return
            
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
            
        member = guild.get_member(payload.user_id)
        if not member:
            return
            
        ld_role = guild.get_role(Config.LD_ROLE_ID)
        if not ld_role or ld_role not in member.roles:
            return

        try:
            # Add small delay before processing
            await asyncio.sleep(0.5)
            
            message = await guild.get_channel(channel_id).fetch_message(payload.message_id)
            
            # Extract user from message
            user_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
            if user_mention:
                user_id = int(user_mention.group(1))
            else:
                logger.info(f"User not mentioned. Attempting to use author id instead.")
                user_id = message.author.id
                
            user_member = guild.get_member(user_id)
            if not user_member:
                return
                
            # Update HRs record
            await self._update_hr_record(
                user_id=user_id,
                username=clean_nickname(user_member.display_name),
                **{column_to_update: 1}
            )

         
        
            # Log the update
            titles = {
                self.phase_log_channel_id: "📊 Phase Logged",
                self.tryout_log_channel_id: "📊 Tryout Logged",
                self.course_log_channel_id: "📊 Course Logged"
            }
            
            title = titles.get(channel_id, "📊 Training Logged")
            log_channel = guild.get_channel(self.log_channel_id)
            
            if log_channel:
                embed = discord.Embed(title=title, color=discord.Color.blue())
                embed.add_field(name="Host", value=user_member.mention)
                embed.add_field(name="Logged By", value=member.mention)
                await log_channel.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error processing {column_to_update} reaction: {str(e)}")
            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                error_embed = discord.Embed(
                    title="❌ Training Log Error",
                    description=str(e),
                    color=discord.Color.red()
                )
                await log_channel.send(embed=error_embed)


    async def _log_activity_reaction_impl(self, payload: discord.RawReactionActionEvent):
        """Handle activity logs (time guarded and activity)"""
        if payload.channel_id != self.activity_log_channel_id or str(payload.emoji) != "✅":
            return
            
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
            
        member = guild.get_member(payload.user_id)
        if not member:
            return
            
        ld_role = guild.get_role(Config.LD_ROLE_ID)
        if not ld_role or ld_role not in member.roles:
            return
            
        try:
            # Add small delay before processing
            await asyncio.sleep(0.5)
            
            message = await guild.get_channel(payload.channel_id).fetch_message(payload.message_id)
            
            # Extract user
            user_mention = re.search(r'<@!?(\d+)>', message.content)
            if user_mention:
                user_id = int(user_mention.group(1))
            else:
                logger.info(f"User not mentioned. Attempting to use author id instead.")
                user_id = message.author.id
                
            user_member = guild.get_member(user_id)
            if not user_member:
                return
                
            # Check for Time: or Guarded: in message
            time_match = re.search(r'Time:\s*(\d+)', message.content)
            guarded_present = "Guarded:" in message.content
            
            updates = {}
            if time_match:
                if guarded_present:
                    updates['time_guarded'] = int(time_match.group(1))
                else:
                    updates['activity'] = int(time_match.group(1))

                
            if updates:
                await self._update_lr_record(
                    user_id=user_id,
                    username=clean_nickname(user_member.display_name),
                    **updates
                )
                
                # Log to default channel
                log_channel = guild.get_channel(self.log_channel_id)
                if log_channel:
                    embed = discord.Embed(
                        title="⏱ Activity Logged",
                        color=discord.Color.green()
                    )
                    embed.add_field(name="Member", value=user_member.mention)
                    if 'activity' in updates:
                        embed.add_field(name="Activity Time", value=f"{updates['activity']} mins")
                    if 'time_guarded' in updates:
                        embed.add_field(name="Guarded Time", value=f"{updates['time_guarded']} mins")
                    embed.add_field(name="Logged By", value=member.mention)
                    embed.add_field(name="Message", value=f"[Jump to Log]({message.jump_url})")
                    
                    await log_channel.send(embed=embed)
                    
        except Exception as e:
            logger.error(f"Error processing activity reaction: {str(e)}")
            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                error_embed = discord.Embed(
                    title="❌ Activity Log Error",
                    description=str(e),
                    color=discord.Color.red()
                )
                await log_channel.send(embed=error_embed)
                
    async def _update_hr_record(self, user_id: int, username: str, **increments):
        """Helper to update HRs table with incremental values"""
        try:
            existing = self.bot.db.supabase.table('HRs') \
                .select('*') \
                .eq('user_id', str(user_id)) \
                .execute()
                
            if existing.data:
                update_data = {'username': username}
                for col, val in increments.items():
                    current = existing.data[0].get(col, 0)
                    update_data[col] = current + val
                    
                self.bot.db.supabase.table('HRs') \
                    .update(update_data) \
                    .eq('user_id', str(user_id)) \
                    .execute()
            else:
                new_data = {
                    'user_id': str(user_id),
                    'username': username,
                    'events': 0,
                    'tryouts': 0,
                    'phases': 0,
                    'courses': 0,
                    'inspections': 0,
                    'joint_events': 0
                }
                for col, val in increments.items():
                    new_data[col] = val
                    
                self.bot.db.supabase.table('HRs').insert(new_data).execute()
                
        except Exception as e:
            logger.error(f"Failed to update HR record for {user_id}: {str(e)}")
            raise

    async def _update_lr_record(self, user_id: int, username: str, **increments):
        """Helper to update LRs table with incremental values"""
        try:
            existing = self.bot.db.supabase.table('LRs') \
                .select('*') \
                .eq('user_id', str(user_id)) \
                .execute()
                
            if existing.data:
                update_data = {'username': username}
                for col, val in increments.items():
                    current = existing.data[0].get(col, 0)
                    update_data[col] = current + val
                    
                self.bot.db.supabase.table('LRs') \
                    .update(update_data) \
                    .eq('user_id', str(user_id)) \
                    .execute()
            else:
                new_data = {
                    'user_id': str(user_id),
                    'username': username,
                    'events_attended': 0,
                    'time_guarded': 0,
                    'activity': 0
                }
                for col, val in increments.items():
                    new_data[col] = val
                    
                self.bot.db.supabase.table('LRs').insert(new_data).execute()
                
        except Exception as e:
            logger.error(f"Failed to update LR record for {user_id}: {str(e)}")
            raise

    async def log_reaction(self, payload: discord.RawReactionActionEvent):
            """Main reaction handler that routes to specific loggers"""
            try:
                await self.rate_limiter.wait_if_needed(bucket="reaction_log")
                await self._log_reaction_impl(payload)
                await self._log_event_reaction_impl(payload)
                await self._log_training_reaction_impl(payload)
                await self._log_activity_reaction_impl(payload)
            except Exception as e:
                logger.error(f"Failed to log reaction: {type(e).__name__}: {str(e)}")
                guild = self.bot.get_guild(payload.guild_id)
                if guild:
                    log_channel = guild.get_channel(self.log_channel_id)
                    if log_channel:
                        error_embed = discord.Embed(
                            title="❌ Reaction Logging Error",
                            description=str(e),
                            color=discord.Color.red()
                        )
                        await log_channel.send(embed=error_embed)

class ConfirmView(discord.ui.View):
    def __init__(self, *, timeout: float = 30.0):
        super().__init__(timeout=timeout)
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

    async def on_timeout(self):
        """Handle when the view times out"""
        self.value = False
        self.stop()


class SheetDBLogger:
    def __init__(self):
        self.script_url = os.getenv("GOOGLE_SCRIPT_URL")
        if not self.script_url:
            logger.error("❌ Google Script URL not configured in environment variables")
            self.ready = False
        else:
            self.ready = True
            logger.info(f"✅ SheetDB Logger configured with Google Apps Script at {self.script_url}")

        # Defined API keys for both trackers
        self.tracker_keys = {
            "LD": "LD_KEY",  # For reaction tracking
            "ED": "ED_KEY"  # For message tracking
        }

    async def update_points(self, member: discord.Member, is_message_tracker: bool = False):
        """Update points for a member, specifying tracker type"""
        if not self.ready:
            logger.error("🛑 SheetDB Logger not properly initialized")
            return False

        username = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
        tracker_type = "ED" if is_message_tracker else "LD"
        api_key = self.tracker_keys[tracker_type]
        
        logger.info(f"🔄 Attempting to update {username}'s points in {tracker_type} tracker")
        logger.debug(f"🔍 Member: {member.id}, Display Name: {member.display_name}, Cleaned Username: {username}")

        payload = {
            "username": username,
            "key": api_key,
            "tracker": tracker_type
        }

        logger.info(f"📤 Sending payload to Google Script: {payload}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.script_url,  
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    response_text = (await response.text()).strip()
                    
                    # Log raw response for debugging
                    logger.debug(f"📥 Raw response from Google Script: {response_text}")
                    logger.debug(f"🔢 Response status: {response.status}")

                    # Success conditions
                    success = (
                        response.status == 200 and 
                        "Error" not in response_text and
                        "Unauthorized" not in response_text and
                        ("Success" in response_text or 
                         "Updated" in response_text or 
                         "Added" in response_text)
                    )

                    if success:
                        logger.info(f"✅ Updated {username}'s points in {tracker_type} tracker")
                        logger.debug(f"Response: {response_text}")
                        return True
                    else:
                        logger.error(f"❌ Failed to update {tracker_type} points - Status: {response.status}, Response: {response_text}")
                        return False

        except asyncio.TimeoutError:
            logger.error("⏰ Timeout while connecting to Google Script")
            return False
        except Exception as e:
            logger.error(f"⚠️ Unexpected error: {type(e).__name__}: {str(e)}", exc_info=True)
            return False
            


class MessageTracker:
    """Tracks messages sent by users with a specific role in specified channels"""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.monitor_channel_ids = set(Config.MESSAGE_TRACKER_CHANNELS)
        self.log_channel_id = Config.MESSAGE_TRACKER_LOG_CHANNEL
        self.tracked_role_id = Config.MESSAGE_TRACKER_ROLE_ID
        self.rate_limiter = EnhancedRateLimiter(calls_per_minute=GLOBAL_RATE_LIMIT)
        self.bot.add_listener(self.log_message, 'on_message')
            
    async def on_ready_setup(self):
        """Setup monitoring when bot starts"""
        guild = self.bot.guilds[0]
        valid_channels = set()
        for channel_id in self.monitor_channel_ids:
            if channel := guild.get_channel(channel_id):
                valid_channels.add(channel.id)
                logger.info(f"👁️ Monitoring channel: #{channel.name} ({channel.id})")
            else:
                logger.warning(f"⚠️ Channel ID {channel_id} not found in guild!")
        
        self.monitor_channel_ids = valid_channels
        
        if log_channel := guild.get_channel(self.log_channel_id):
            logger.info(f"📝 Log channel set to ({log_channel.id})")
        else:
            logger.warning(f"⚠️ Message tracker log channel {self.log_channel_id} not found!")
            self.log_channel_id = None
    
        if tracked_role := guild.get_role(self.tracked_role_id):
            logger.info(f"🎯 Tracking role set to {tracked_role.id}")
        else:
            logger.warning(f"⚠️ Message tracker role {self.tracked_role_id} not found!")
            

    async def _safe_interaction_response(self, interaction: discord.Interaction):
        """Safely handle interaction responses with proper error handling"""
        try:
            if not interaction.response.is_done():
                await interaction.response.defer(ephemeral=True)
            return True
        except discord.errors.NotFound:
            logger.error("Interaction expired before response could be sent")
            return False
        except Exception as e:
            logger.error(f"Failed to defer interaction: {str(e)}", exc_info=True)
            return False
            
    async def add_channels(self, interaction: discord.Interaction, channels: str):
        """Add channels to monitor"""
        if not await self._safe_interaction_response(interaction):
            return
            
        try:
            channel_ids = [int(cid.strip()) for cid in channels.split(',')]
            self.monitor_channel_ids.update(channel_ids)
            await interaction.followup.send(f"✅ Added {len(channel_ids)} channels to monitoring", ephemeral=True)
        except ValueError:
            await interaction.followup.send("❌ Invalid channel ID format. Please provide comma-separated channel IDs.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in add_channels: {str(e)}", exc_info=True)
            await interaction.followup.send(f"❌ Failed to add channels: {str(e)}", ephemeral=True)

    async def remove_channels(self, interaction: discord.Interaction, channels: str):
        """Remove channels from monitoring"""
        if not await self._safe_interaction_response(interaction):
            return
            
        try:
            channel_ids = [int(cid.strip()) for cid in channels.split(',')]
            self.monitor_channel_ids.difference_update(channel_ids)
            await interaction.followup.send(f"✅ Removed {len(channel_ids)} channels from monitoring", ephemeral=True)
        except ValueError:
            await interaction.followup.send("❌ Invalid channel ID format. Please provide comma-separated channel IDs.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in remove_channels: {str(e)}", exc_info=True)
            await interaction.followup.send(f"❌ Failed to remove channels: {str(e)}", ephemeral=True)

    async def list_channels(self, interaction: discord.Interaction):
        """List monitored channels"""
        if not await self._safe_interaction_response(interaction):
            return
            
        try:
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
        except Exception as e:
            logger.error(f"Error in list_channels: {str(e)}", exc_info=True)
            try:
                await interaction.followup.send("❌ Failed to list channels due to an error", ephemeral=True)
            except:
                pass
        

    async def log_message(self, message: discord.Message):
        """Log messages from monitored channels by users with tracked role"""
        try:
            await self.rate_limiter.wait_if_needed(bucket="message_log")
            await self._log_message_impl(message)
        except Exception as e:
            logger.error(f"❌ Failed to log message: {type(e).__name__}: {str(e)}", exc_info=True)

    async def _log_message_impl(self, message: discord.Message):
        if message.author.bot:
            logger.debug("🤖 Ignoring bot message")
            return
            
        if message.channel.id not in self.monitor_channel_ids:
            logger.debug(f"🚫 Ignoring message from non-monitored channel #{message.channel.name}")
            return
            
        if not isinstance(message.author, discord.Member):
            logger.warning("⚠️ Message author is not a Member object")
            return
            
        tracked_role = message.guild.get_role(self.tracked_role_id)
        if not tracked_role:
            logger.error(f"❌ Tracked role {self.tracked_role_id} not found in guild!")
            return
            
        if tracked_role not in message.author.roles:
            logger.debug(f"👤 User {message.author.display_name} doesn't have the tracked role")
            return
            
        log_channel = message.guild.get_channel(self.log_channel_id)
        if not log_channel:
            logger.error("❌ Log channel not available")
            return
            
        # Log the message details
        logger.info(f"✉️ Processing message from {message.author.display_name} in #{message.channel.id}")
        
        content = message.content
        if len(content) > 300:
            content = content[:300] + "... [truncated]"
            
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
        
        logger.info(f"🔢 Attempting to update message tracker points for: {message.author.display_name}")
        update_success = await self.bot.sheets.update_points(message.author, is_message_tracker=True)
        logger.info(f"📊 Message tracker update {'✅ succeeded' if update_success else '❌ failed'}")

    
    

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
bot.db = DatabaseHandler()

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
# /addxp Command
@bot.tree.command(name="add-xp", description="Add XP to a user")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HR_ROLE_ID)
async def add_xp(interaction: discord.Interaction, user: discord.User, xp: int):
    # Validate XP amount
    if xp <= 0:
        await interaction.response.send_message(
            "❌ XP amount must be positive.",
            ephemeral=True
        )
        return
    if xp > MAX_XP_PER_ACTION:
        await interaction.response.send_message(
            f"❌ Cannot give more than {MAX_XP_PER_ACTION} XP at once.",
            ephemeral=True
        )
        return

    cleaned_name = clean_nickname(user.display_name)
    current_xp = await bot.db.get_user_xp(user.id)
    
    # Additional safety check
    if current_xp > 100000:  # Extreme value check
        await interaction.response.send_message(
            "❌ User has unusually high XP. Contact admin.",
            ephemeral=True
        )
        return
    
    success, new_total = await bot.db.add_xp(user.id, cleaned_name, xp)
    
    if success:
        await interaction.response.send_message(
            f"✅ Added {xp} XP to {cleaned_name}. New total: {new_total} XP"
        )
        # Log the XP change
        await log_xp_to_discord(interaction.user, user, xp, new_total, "Manual Addition")
         
    else:
        await interaction.response.send_message(
            "❌ Failed to add XP. Notify admin.",
            ephemeral=True
        )


# /take-xp Command
@bot.tree.command(name="take-xp", description="Takes XP from user")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HR_ROLE_ID)
async def take_xp(interaction: discord.Interaction, user: discord.User, xp: int):
    if xp <= 0:
        await interaction.response.send_message(
            "❌ XP amount must be positive. Use /addxp to give XP.",
            ephemeral=True
        )
        return
    if xp > MAX_XP_PER_ACTION:
        await interaction.response.send_message(
            f"❌ Cannot remove more than {MAX_XP_PER_ACTION} XP at once.",
            ephemeral=True
        )
        return

    cleaned_name = clean_nickname(user.display_name)
    current_xp = await bot.db.get_user_xp(user.id)
    
    if xp > current_xp:
        await interaction.response.send_message(
            f"❌ User only has {current_xp} XP. Cannot take {xp}.",
            ephemeral=True
        )
        return
    
    success, new_total = await bot.db.take_xp(user.id, cleaned_name, xp)
    
    if success:
        message = f"✅ Removed {xp} XP from {cleaned_name}. New total: {new_total} XP"
        if new_total == 0:
            message += "\n⚠️ User's XP has reached 0"
        await interaction.response.send_message(message)
        # Log the XP change
        await log_xp_to_discord(interaction.user, user, -xp, new_total, "Manual Removal")
        
    else:
        await interaction.response.send_message(
            "❌ Failed to take XP. Notify admin.",
            ephemeral=True
        )


# /xp Command
@bot.tree.command(name="xp", description="Check yours or someone else's XP")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.RMP_ROLE_ID) 
async def xp_command(interaction: discord.Interaction, user: Optional[discord.User] = None):
    """Check a user's XP and leaderboard position"""
    try:
        await interaction.response.defer()
        
        # Rate limiting
        await bot.rate_limiter.wait_if_needed(bucket=f"xp_cmd_{interaction.user.id}")
        
        target_user = user or interaction.user
        cleaned_name = clean_nickname(target_user.display_name)
        
        # Get XP and leaderboard data
        xp = await bot.db.get_user_xp(target_user.id)
        result = await asyncio.to_thread(
            lambda: bot.db.supabase.table('users')
            .select("user_id", "xp")
            .order("xp", desc=True)
            .execute()
        )
        
        # Process leaderboard position
        position = next(
            (idx for idx, entry in enumerate(result.data, 1) 
             if str(entry['user_id']) == str(target_user.id)),
            None
        )
        
        # Build embed
        embed = discord.Embed(
            title=f"📊 XP Profile: {cleaned_name}",
            color=discord.Color.green()
        ).set_thumbnail(url=target_user.display_avatar.url)
        
        embed.add_field(name="Current XP", value=f"```{xp}```", inline=True)
        
        if position:
            embed.add_field(name="Leaderboard Position", value=f"```#{position}```", inline=True)
            if len(result.data) > 10:
                percentile = (position / len(result.data)) * 100
                embed.add_field(
                    name="Percentile", 
                    value=f"```Top {100 - percentile:.1f}%```", 
                    inline=False
                )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"XP command error: {str(e)}")
        await handle_command_error(interaction, e)

async def handle_command_error(interaction: discord.Interaction, error: Exception):
    """Centralized error handling for commands"""
    try:
        if isinstance(error, discord.NotFound):
            await interaction.followup.send(
                "⚠️ Operation timed out. Please try again.",
                ephemeral=True
            )
        else:
            await interaction.followup.send(
                "❌ An error occurred. Please try again later.",
                ephemeral=True
            )
    except:
        if interaction.channel:
            await interaction.channel.send(
                f"{interaction.user.mention} ❌ Command failed. Please try again.",
                delete_after=10
            )

# Leadebaord Command
@bot.tree.command(name="leaderboard", description="View the top 15 users by XP")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.RMP_ROLE_ID)  
async def leaderboard(interaction: discord.Interaction):
    try:
        # Rate limiting for Supabase
        await bot.rate_limiter.wait_if_needed(bucket="supabase_query")
        
        # Fetch top 15 users from Supabase
        result = bot.db.supabase.table('users') \
            .select("user_id", "username", "xp") \
            .order("xp", desc=True) \
            .limit(15) \
            .execute()
        
        data = result.data
        
        if not data:
            await interaction.response.send_message("❌ No leaderboard data available.", ephemeral=True)
            return
        
        leaderboard_lines = []
        
        for idx, entry in enumerate(data, start=1):
            try:
                user_id = int(entry['user_id'])
                user = interaction.guild.get_member(user_id) or await bot.fetch_user(user_id)
                display_name = clean_nickname(user.display_name) if user else entry.get('username', f"Unknown ({user_id})")
                xp = entry['xp']
                leaderboard_lines.append(f"**#{idx}** - {display_name}: `{xp} XP`")
            except Exception as e:
                logger.error(f"Error processing leaderboard entry {idx}: {str(e)}")
                continue
        
        embed = discord.Embed(
            title="🏆 XP Leaderboard (Top 15)",
            description="\n".join(leaderboard_lines) or "No data available",
            color=discord.Color.gold()
        )
        
        embed.set_footer(text=f"Requested by {interaction.user.display_name}")
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in /leaderboard command: {str(e)}")
        await interaction.response.send_message(
            "❌ Failed to fetch leaderboard data. Please try again later.",
            ephemeral=True
        )


# Give Event XP Command
@bot.tree.command(name="give-event-xp", description="Give XP to attendees mentioned in an event log message")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HR_ROLE_ID)
async def give_event_xp(
    interaction: discord.Interaction,
    message_link: str,
    xp_amount: int,
    attendees_section: Literal["Attendees:", "Passed:"] = "Attendees:"
):
    # Rate Limit
    await bot.rate_limiter.wait_if_needed(bucket="global_xp_update")
    # Validate XP amount first
    if xp_amount <= 0:
        await interaction.response.send_message(
            "❌ XP amount must be positive.",
            ephemeral=True
        )
        return
    if xp_amount > MAX_EVENT_XP_PER_USER:
        await interaction.response.send_message(
            f"❌ Cannot give more than {MAX_EVENT_XP_PER_USER} XP per user in events.",
            ephemeral=True
        )
        return

    # Defer the response immediately to prevent timeout
    await interaction.response.defer()
    initial_message = await interaction.followup.send("⏳ Attempting to give XP...", wait=True)
    
    try:
        # Add timeout for the entire operation
        async with asyncio.timeout(60):  # 60 second timeout for entire operation
            # Rate limiting check with timeout
            try:
                await asyncio.wait_for(
                    bot.rate_limiter.wait_if_needed(bucket=f"give_xp_{interaction.user.id}"),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                await initial_message.edit(content="⌛ Rate limit check timed out. Please try again.")
                return

            # Parse and validate message link
            if not message_link.startswith('https://discord.com/channels/'):
                await initial_message.edit(content="❌ Invalid message link format")
                return
                
            try:
                parts = message_link.split('/')
                guild_id = int(parts[4])
                channel_id = int(parts[5])
                message_id = int(parts[6])
            except (IndexError, ValueError):
                await initial_message.edit(content="❌ Invalid message link format")
                return
            
            if guild_id != interaction.guild.id:
                await initial_message.edit(content="❌ Message must be from this server")
                return
                
            # Fetch the message with timeout
            try:
                channel = interaction.guild.get_channel(channel_id)
                if not channel:
                    await initial_message.edit(content="❌ Channel not found")
                    return
                    
                try:
                    message = await asyncio.wait_for(
                        channel.fetch_message(message_id),
                        timeout=10.0
                    )
                except discord.NotFound:
                    await initial_message.edit(content="❌ Message not found")
                    return
                except discord.Forbidden:
                    await initial_message.edit(content="❌ No permission to read that channel")
                    return
            except asyncio.TimeoutError:
                await initial_message.edit(content="⌛ Timed out fetching message")
                return
                
            # Process attendees section
            content = message.content
            section_index = content.find(attendees_section)
            if section_index == -1:
                await initial_message.edit(content=f"❌ Could not find '{attendees_section}' in the message")
                return
                
            mentions_section = content[section_index + len(attendees_section):]
            mentions = re.findall(r'<@!?(\d+)>', mentions_section)
            
            if not mentions:
                await initial_message.edit(content=f"❌ No user mentions found after '{attendees_section}'")
                return
                
            # Process users with progress updates
            unique_mentions = list(set(mentions))
            total_potential_xp = xp_amount * len(unique_mentions)
            
            if total_potential_xp > MAX_EVENT_TOTAL_XP:
                await initial_message.edit(
                    content=f"❌ Event would give {total_potential_xp} XP total (max is {MAX_EVENT_TOTAL_XP}). Reduce XP or attendees."
                )
                return
                
            await initial_message.edit(content=f"🎯 Processing XP for {len(unique_mentions)} users...")
            
            success_count = 0
            failed_users = []
            processed_users = 0
            
            for i, user_id in enumerate(unique_mentions, 1):
                try:
                    # Update progress every 5 users
                    if i % 5 == 0 or i == len(unique_mentions):
                        await initial_message.edit(
                            content=f"⏳ Processing {i}/{len(unique_mentions)} users ({success_count} successful)..."
                        )
                    
                    # Rate limit between users
                    if i > 1:
                        await asyncio.sleep(0.75)  # Slightly longer delay
                        
                    member = interaction.guild.get_member(int(user_id))
                    if not member:
                        failed_users.append(f"Unknown user ({user_id})")
                        continue
                        
                    try:
                        current_xp = await asyncio.wait_for(
                            bot.db.get_user_xp(member.id),
                            timeout=5.0
                        )
                        if current_xp + xp_amount > 100000:
                            failed_users.append(f"{clean_nickname(member.display_name)} (would exceed max XP)")
                            continue
                            
                        success, new_total = await asyncio.wait_for(
                            bot.db.add_xp(member.id, member.display_name, xp_amount),
                            timeout=5.0
                        )
                        
                        if success:
                            success_count += 1
                            await interaction.followup.send(
                                f"✨ **{clean_nickname(interaction.user.display_name)}** gave {xp_amount} XP to {member.mention} (New total: {new_total} XP)",
                                silent=True
                            )
                            await log_xp_to_discord(interaction.user, member, xp_amount, new_total, f"Event: {message.jump_url}")
        
                        else:
                            failed_users.append(clean_nickname(member.display_name))
                            
                    except asyncio.TimeoutError:
                        failed_users.append(f"{clean_nickname(member.display_name)} (timeout)")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing user {user_id}: {str(e)}")
                    failed_users.append(f"User {user_id} (error)")
                    continue
                    
            # Final summary
            result_message = [
                f"✅ **XP Distribution Complete**",
                f"**Given by:** {interaction.user.mention}",
                f"**XP per user:** {xp_amount}",
                f"**Successful distributions:** {success_count}",
                f"**Total XP given:** {xp_amount * success_count}"
            ]
            
            if failed_users:
                result_message.append(f"\n**Failed distributions:** {len(failed_users)}")
                for chunk in [failed_users[i:i + 10] for i in range(0, len(failed_users), 10)]:
                    await interaction.followup.send("• " + "\n• ".join(chunk), ephemeral=True)
            
            await interaction.followup.send("\n".join(result_message))
            
    except asyncio.TimeoutError:
        await initial_message.edit(content="⌛ Command timed out. Some XP may have been awarded.")
    except Exception as e:
        logger.error(f"Error in give_event_xp: {str(e)}", exc_info=True)
        await initial_message.edit(content="❌ An unexpected error occurred. Please check logs.")


# Discharge Command
@bot.tree.command(name="discharge", description="Notify members of honourable/dishonourable discharge and log it")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def discharge(
    interaction: discord.Interaction,
    members: str,  # Comma-separated user mentions/IDs
    reason: str,
    discharge_type: Literal["Honourable", "Dishonourable"] = "Honourable",
    evidence: Optional[discord.Attachment] = None
):
    view = ConfirmView()
    await interaction.response.send_message("Confirm discharge?", view=view, ephemeral=True)
    await view.wait()
    if not view.value:
        return

    try:
        # Input Sanitization 
        reason = escape_markdown(reason) 
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

        processing_msg = await interaction.followup.send(
            "⚙️ Processing discharge...",
            ephemeral=True,
            wait=True
        )

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
                    if channel := interaction.guild.get_channel(1219410104240050236):
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

        await processing_msg.edit(
            content=None,
            embed=result_embed
        )

        # Log to D_LOG_CHANNEL_ID
        if d_log := interaction.guild.get_channel(Config.D_LOG_CHANNEL_ID):
            log_embed = discord.Embed(
                title=f"Discharge Log",
                color=color,
                timestamp=datetime.now(timezone.utc)
            )
            log_embed.add_field(
                name="Type",
                value=f"🔰 {discharge_type} Discharge" if discharge_type == "Honourable" else f"🚨 {discharge_type} Discharge",
                inline=False
            )
            log_embed.add_field(name="Reason", value=f"```{reason}```", inline=False)
            
            # Format member mentions with their cleaned nicknames
            member_entries = []
            for member in discharged_members:
                # Clean the nickname by removing any tags like [INS]
                cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
                member_entries.append(f"{member.mention} | {cleaned_nickname}")
            
            log_embed.add_field(
                name="Discharged Members",
                value="\n".join(member_entries) or "None",
                inline=False
            )
            
            if evidence:
                log_embed.add_field(name="Evidence", value=f"[View Attachment]({evidence.url})", inline=True)

            log_embed.add_field(name="Discharged By", value=interaction.user.mention, inline=True)
            
            await d_log.send(embed=log_embed)

    except Exception as e:
        logger.error(f"Discharge command failed: {e}")
        await interaction.followup.send("❌ An error occurred while processing the discharge.", ephemeral=True)
    

#LD Reaction Monitor Set-up Command           
@bot.tree.command(name="message-tracker-setup", description="Setup message monitoring")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
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
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)  # Or whatever role you want
async def message_tracker_add(
    interaction: discord.Interaction,
    channels: str
):
    """Add channels to message tracking"""
    await bot.message_tracker.add_channels(interaction, channels)

@bot.tree.command(name="message-tracker-remove", description="Remove channels from message monitoring")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID) 
async def message_tracker_remove(
    interaction: discord.Interaction,
    channels: str
):
    """Remove channels from message tracking"""
    await bot.message_tracker.remove_channels(interaction, channels)

@bot.tree.command(name="message-tracker-list", description="List currently monitored channels")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)  
async def message_tracker_list(interaction: discord.Interaction):
    """List channels being tracked for messages"""
    await bot.message_tracker.list_channels(interaction)


@bot.tree.command(name="force-update", description="Manually test sheet updates")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)  
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
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.RMP_ROLE_ID)  
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
            "/discharge - Sends discharge notification to user and logs in discharge logs",
            "/force-update - Manually test sheets update"
        ],
         "⭐ XP": [
            "/add-xp - Gives xp to user",
            "/take-xp - Takes xp from user",
            "/give-event-xp - Gives xp to attendees/passers in event logs",
            "/xp - Checks amount of xp user has",
            "/leaderboard - View the top 15 users by XP"
        ]
    }
    
    for name, value in categories.items():
        embed.add_field(name=name, value="\n".join(value), inline=False)
    
    await interaction.response.send_message(embed=embed)
    

@bot.tree.command(name="ping", description="Check bot latency")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@has_allowed_role()
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)
    await interaction.response.send_message(
        f"🏓 Pong! Latency: {latency}ms",
        ephemeral=True
    )

@bot.tree.command(name="reaction-setup", description="Setup reaction monitoring")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def reaction_setup(
    interaction: discord.Interaction,
    log_channel: discord.TextChannel,
    monitor_channels: str
):
    await bot.reaction_logger.setup(interaction, log_channel, monitor_channels)

@bot.tree.command(name="reaction-add", description="Add channels to monitor")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def reaction_add(
    interaction: discord.Interaction,
    channels: str
):
    await bot.reaction_logger.add_channels(interaction, channels)

@bot.tree.command(name="reaction-remove", description="Remove channels from monitoring")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def reaction_remove(
    interaction: discord.Interaction,
    channels: str
):
    await bot.reaction_logger.remove_channels(interaction, channels)

@bot.tree.command(name="reaction-list", description="List monitored channels")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
@min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
async def reaction_list(interaction: discord.Interaction):
    await bot.reaction_logger.list_channels(interaction)

@bot.tree.command(name="sheetdb-test", description="Test SheetDB connection")
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
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
        if isinstance(error.original, discord.errors.HTTPException):
            if error.original.status == 429:
                retry_after = float(error.original.response.headers.get('Retry-After', 5))
                logger.warning(f"Rate limited - waiting {retry_after} seconds")
                
                # Edit original response if possible
                try:
                    await interaction.edit_original_response(
                        content=f"⚠️ Too many requests. Waiting {retry_after:.1f} seconds..."
                    )
                except:
                    pass
                    
                await asyncio.sleep(retry_after)
                
                # Retry the command
                try:
                    await interaction.followup.send(
                        "Retrying command after rate limit...",
                        ephemeral=True
                    )
                    await bot.tree.call(interaction)
                except Exception as retry_error:
                    logger.error(f"Retry failed: {retry_error}")
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
            "• Request for document access in [HR Documents](https://discord.com/channels/1165368311085809717/1165368317532438646).\n"
            "• You are exempted from quota this week only - you start next week ([Quota Info](https://discord.com/channels/1165368311085809717/1206998095552978974)).\n"
            "• Uncomplete quota = strike.\n"
            "• One failed tryout allowed if your try quota portion ≥2.\n"
            "• Ask for help anytime - we're friendly!\n"
            "• Are you Captain+ in BA? Apply for departments: [Applications](https://discord.com/channels/1165368311085809717/1165368316970405916)."
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
        title=" 👮| Welcome to the Royal Military Police",
        description="Congratulations on passing your security check, you're officially a TRAINING member of the police force. Please be sure to read the information found below.\n\n"
                   "> ** 1.** Make sure to read all of the rules found in <#1165368313925353580> and in the brochure found below.\n\n"
                   "> **2.** You **MUST** read the RMP main guide and MSL before starting your duties.\n\n"
                   "> **3.** You can't use your L85 unless you are doing it for Self-Militia or enforcing the PD rules. (Self-defence)\n\n"
                   "> **4.** Make sure to follow the Chain Of Command. 2nd Lieutenant > Lieutenant > Captain > Major > Lieutenant Colonel > Colonel > Brigadier > Major General\n\n"
                   "> **5.** For phases, you may wait for one to be hosted in <#1207367013698240584> or request the phase you need in <#1270700562433839135>.\n\n"
                   "> **6.** All the information about the Defence School of Policing and Guarding is found in both <#1237062439720452157> and <#1207366893631967262>\n\n"
                   "> **7.** Choose your timezone here https://discord.com/channels/1165368311085809717/1165368313925353578\n\n"
                   "**Besides that, good luck with your phases!**",
        color=discord.Color.from_str("#330000") 
    )

    # Second embed (detailed rules)
    embed2 = discord.Embed(
        title="Trainee Constable Brochure",
        color=discord.Color.from_str("#660000")
    )
    
    embed2.add_field(
        name="**TOP 5 RULES**",
        value="> **1**. You **MUST** read the RMP main guide and MSL before starting your duties.\n"
              "> **2**. You **CANNOT** enforce the MSL. Only the Parade Deck (PD) rules **AFTER** you pass your phase 1.\n"
              "> **3**. You **CANNOT** use your bike on the PD or the pavements.\n"
              "> **4**. You **MUST** use good spelling and grammar to the best of your ability.\n"
              "> **5**. You **MUST** remain mature and respectful at all times.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED ON THE PD AT ALL TIMES?__",
        value="> ↠ Royal Army Medical Corps,\n"
              "> ↠ Royal Military Police,\n" 
              "> ↠ Intelligence Corps.\n"
              "> ↠ Royal Family.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED ON THE PD WHEN CARRYING OUT THEIR DUTIES?**",
        value="> ↠ United Kingdom Special Forces,\n"
              "> ↠ Grenadier Guards,\n"
              "> ↠ Foreign Relations,\n"
              "> ↠ Royal Logistic Corps,\n"
              "> ↠ Adjutant General's Corps,\n"
              "> ↠ High Ranks, RSM, CSM and ASM hosting,\n"
              "> ↠ Regimental personnel watching one of their regiment's events inside Pad area.",
        inline=False
    )
    
    embed2.add_field(
        name="**HOW DO I ENFORCE PD RULES ON PEOPLE NOT ALLOWED ON IT?**",
        value="> 1. Give them their first warning to get off the PD, \"W1, off the PD!\"\n"
              "> 2. Wait 3-5 seconds for them to listen; if they don't, give them their second warning, \"W2, off the PD!\"\n"
              "> 3. Wait 3-5 seconds for them to listen; if they don't kill them.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED __ON__ THE ACTUAL STAGE AT ALL TIMES**",
        value="> ↠ Major General and above,\n"
              "> ↠ Royal Family (they should have a purple name tag),\n"
              "> ↠ Those who have been given permission by a Lieutenant General.",
        inline=False
    )
    
    embed2.add_field(
        name="**WHO'S ALLOWED TO PASS THE RED LINE IN-FRONT OF THE STAGE?**",
        value="> ↠ Major General and above,\n"
              "> ↠ Royal Family,\n"
              "> ↠ Those who have been given permission by a Lieutenant General,\n"
              "> ↠ COMBATIVE Home Command Regiments:\n"
              "> - Royal Military Police,\n"
              "> - United Kingdom Forces,\n"
              "> - Household Division.\n"
              "> **Kill those not allowed who touch or past the red line.**",
        inline=False
    )
    
    embed2.add_field(
        name="\u200b",  
        value="**LASTLY, IF YOU'RE UNSURE ABOUT SOMETHING, ASK SOMEONE USING THE CHAIN OF COMMAND BEFORE TAKING ACTION!**",
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

#Deserter Monitor
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
        description=f"{member.mention} just deserted! Please log this in BA.",
        color=discord.Color.red()
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    
    await alert_channel.send(
        content=f"<@&{Config.HIGH_COMMAND_ROLE_ID}>",
        embed=embed
    )

    # For the deserter checker discharge log
    dishonourable_embed = discord.Embed(
        title="Discharge Log",
        color=discord.Color.red(),
        timestamp=datetime.now(timezone.utc)
    )
    
    dishonourable_embed.add_field(
        name="Type",
        value="🚨 Dishonourable Discharge",
        inline=False
    )
    dishonourable_embed.add_field(
        name="Reason", 
        value="```Desertion.```",
        inline=False
    )
    
    # Add member information (assuming you have a member object)
    cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
    dishonourable_embed.add_field(
        name="Discharged Members",
        value=f"{member.mention} | {cleaned_nickname}",
        inline=False
    )
    
    dishonourable_embed.add_field(
        name="Discharged By", 
        value=f"None - Unauthorised",
        inline=True
    )
    
    
    # Set footer
    dishonourable_embed.set_footer(text="Desertion Monitor System")

    # DESERTER | BLacklist logs
    current_date = datetime.now(timezone.utc)
    ending_date = current_date + timedelta(days=30)  # Adding approximately one month
    
    blacklist_embed = discord.Embed(
        title="⛔ Blacklist",
        color=discord.Color.red(),
        timestamp=current_date
    )
    
    blacklist_embed.add_field(
        name="Issuer:",
        value="MP Assistant",
        inline=False
    )
    blacklist_embed.add_field(
        name="Name:", 
        value=f"{cleaned_nickname}",
        inline=False
    )
    
    blacklist_embed.add_field(
        name="Starting date", 
        value=f"<t:{int(current_date.timestamp())}:D>",
        inline=False
    )
    
    blacklist_embed.add_field(
        name="Ending date", 
        value=f"<t:{int(ending_date.timestamp())}:D>",
        inline=False
    )

    blacklist_embed.add_field(
        name="Reason:", 
        value="Desertion.",
        inline=False
    )
    
    
    # Set footer
    blacklist_embed.set_footer(text="Desertion Monitor System")

    try:
        d_log = bot.get_channel(Config.D_LOG_CHANNEL_ID)
        b_log = bot.get_channel(Config.B_LOG_CHANNEL_ID)
        
        if d_log:
            await d_log.send(embed=dishonourable_embed)
            await b_log.send(embed=blacklist_embed)
            logger.info(f"Logged deserted member, {cleaned_nickname}.")
        else:
            # If main channel fails, send simple message to alternative channel
            alt_channel = bot.get_channel(1165368316970405917)
            if alt_channel:
                await alt_channel.send(f"⚠️ Failed to log deserter discharge for {cleaned_nickname} in main channel")
                logger.error(f"Failed to log deserted member {cleaned_nickname} - main channel not found")
    except Exception as e:
        logger.error(f"Error logging deserter discharge: {str(e)}")
    

    
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
