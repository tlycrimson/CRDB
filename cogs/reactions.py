import re
import asyncio
import discord
import logging
from config import Config
from discord.ext import commands, tasks
from utils import embedBuilder
from utils.helpers import clean_nickname
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)

SECURITY_CHECK_PATTERN = re.compile(r"Security\s*Check(?:/s|\(s\)|s)?:\s*(\d+)", re.IGNORECASE)
HOST_PATTERN = re.compile(r'host:\s*<@!?(\d+)>', re.IGNORECASE)
CO_HOST_PATTERN = re.compile(r'co-host:\s*(.*?)(?=\n(?:host|attendees|passed|failed|ping|proof):|$)', re.IGNORECASE)
MENTION_PATTERN = re.compile(r'<@!?(\d+)>')
EVENT_NAME_PATTERN = re.compile(r'Event:\s*(.*?)(?:\n|$)', re.IGNORECASE)
ATTENDEES_PATTERN = re.compile(r'(?:Attendees:|Passed:)\s*((?:<@!?\d+>[\s,]*)+)', re.IGNORECASE)
TIME_PATTERN = re.compile(r'Time:\s*(\d+)')

@dataclass
class ReactionHandler:
    handler: Callable
    channels: set[int] | None = None

class TransactionTracker:
    """Tracks DB updates and sent messages to roll them back if an error occurs."""
    def __init__(self, cog):
        self.cog = cog
        self.db_rollbacks: List[Tuple[Callable, tuple]] = []
        self.sent_messages: List[discord.Message] = []

    async def update_hr(self, member: discord.Member, updates: dict) -> bool:
        success = await self.cog._update_hr_record(member, updates)
        if success:
            inverse = {k: -v for k, v in updates.items()}
            self.db_rollbacks.append((self.cog._update_hr_record, (member, inverse)))
        return success

    async def update_lr(self, member: discord.Member, updates: dict) -> bool:
        success = await self.cog._update_lr_record(member, updates)
        if success:
            inverse = {k: -v for k, v in updates.items()}
            self.db_rollbacks.append((self.cog._update_lr_record, (member, inverse)))
        return success

    async def add_xp(self, user_id: str, display_name: str, xp: int):
        success, new_total = await self.cog.bot.db.add_xp(user_id, display_name, xp)
        if success:
            self.db_rollbacks.append((self.cog.bot.db.add_xp, (user_id, display_name, -xp)))
        return success, new_total

    def track_message(self, message: discord.Message):
        self.sent_messages.append(message)

    async def rollback(self):
        """Deletes sent messages and reverts DB changes in reverse order."""
        logger.info("Error occurred in reacton handler - starting rollback")
        for msg in self.sent_messages:
            try:
                await msg.delete()
            except discord.NotFound:
                pass
            except Exception as e:
                logger.error(f"Failed to delete message during rollback: {e}")
                
        for func, args in reversed(self.db_rollbacks):
            try:
                await func(*args)
            except Exception as e:
                logger.error(f"Failed to rollback DB action: {e}")

        logger.info("Rollback finished")

class ReactionLoggerCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.monitor_channel_ids = set(Config.DEFAULT_MONITOR_CHANNELS)
        self.log_channel_id = Config.DEFAULT_LOG_CHANNEL
        self.event_channel_ids = [Config.W_EVENT_LOG_CHANNEL_ID, Config.EVENT_LOG_CHANNEL_ID]
        self.phase_log_channel_id = Config.PHASE_LOG_CHANNEL_ID
        self.tryout_log_channel_id = Config.TRYOUT_LOG_CHANNEL_ID
        self.course_log_channel_id = Config.COURSE_LOG_CHANNEL_ID
        self.activity_log_channel_id = Config.ACTIVITY_LOG_CHANNEL_ID
        self.tc_supervision_log_channel_id = Config.TC_SUPERVISION_CHANNEL_ID
        self.sc_log_channel_id = Config.SC_LOGS_CHANNEL_ID
        self.exam_monitor_channel_ids = Config.EXAM_AND_INDUCTION_MONITOR_CHANNELS
        
        self.REACTION_HANDLERS = [
            ReactionHandler("_log_event_reaction_impl", channels=set(self.event_channel_ids)),
            ReactionHandler("_log_training_reaction_impl", channels={self.phase_log_channel_id, self.tryout_log_channel_id, self.course_log_channel_id, self.tc_supervision_log_channel_id}),
            ReactionHandler("_log_activity_reaction_impl", channels={self.activity_log_channel_id}),
            ReactionHandler("_log_security_check_log_reaction_impl", channels={self.sc_log_channel_id}),
            ReactionHandler("_log_la_and_examiner_impl", channels=set(self.exam_monitor_channel_ids)),
            ReactionHandler("_log_dbl_reaction_impl", channels=set(self.monitor_channel_ids))
        ]
        self.cleanup_loop.start()

    async def cog_unload(self):
        self.cleanup_loop.cancel() 

    @tasks.loop(hours=24)
    async def cleanup_loop(self):
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            await self.bot.db.supabase.table('processed_reactions')\
                .delete()\
                .lt('processed_at', cutoff.isoformat())\
                .execute()
            logger.info("Cleaned up old processed reactions")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")    
        
    @cleanup_loop.before_loop
    async def before_cleanup(self):
        await self.bot.wait_until_ready()

    async def configure(self, interaction=None, log_channel=None, monitor_channels=None):
        if interaction and log_channel and monitor_channels:
            channel_ids = [int(cid.strip()) for cid in monitor_channels.split(',')]
            self.monitor_channel_ids = set(channel_ids)
            await interaction.followup.send("Reaction tracking setup complete", ephemeral=True)
            
    async def on_ready_setup(self):
        if not self.bot.guilds:
            logger.error("Bot is not in any guilds!")
            return

        guild = self.bot.guilds[0]
        valid_channels = set()
        for channel_id in self.monitor_channel_ids:
            if channel := guild.get_channel(channel_id):
                valid_channels.add(channel.id)
        
        self.monitor_channel_ids = valid_channels
        self.log_channel = guild.get_channel(self.log_channel_id)

        if not self.log_channel:
            logger.warning(f"Default log channel {self.log_channel_id} not found for guild: {guild.name}!")
        else:
            logger.info(f"Default Log channel configured for {guild.name}.")
    
    @commands.Cog.listener()
    async def on_ready(self):
        await self.on_ready_setup()

    async def is_reaction_processed(self, message_id: int, member: discord.Member) -> bool:
        try:
            res = await self.bot.db.supabase.table('processed_reactions')\
                .select('id')\
                .eq('message_id', str(message_id))\
                .eq('user_id', str(member.id))\
                .execute()
            return bool(res.data)
        except Exception as e:
            logger.error(f"Error checking processed reaction: {e}")
            return False

    async def mark_reaction_processed(self, message_id: int, user_id: int):
        try:
            await self.bot.db.supabase.table('processed_reactions')\
                .upsert(
                    {
                        'message_id': str(message_id),
                        'user_id': str(user_id)
                    },
                    on_conflict='message_id,user_id'
                ).execute()
        except Exception as e:
            logger.error(f"Error marking reaction processed: {e}")

    async def _handle_reaction_error(self, e: Exception, member: discord.Member, handler: str = "") -> None:
        logger.error(f"Failed in {handler}: {type(e).__name__}: {e}", exc_info=True)
        if self.log_channel:
            error_embed = discord.Embed(
                description="An error occurred while logging this reaction. Operational changes were rolled back.",
                color=discord.Color.red(),
            )
            error_embed.set_author(name="Reaction Logging Error", icon_url=Config.CANCEL_URL)
            await self.log_channel.send(content=member.mention, embed=error_embed)

    async def health_check(self):
        try:
            add_listeners = [
                l for l in self.bot.extra_events.get('on_raw_reaction_add', [])
                if getattr(l, '__self__', None) is self
            ]
            status = {
                "add_listeners": len(add_listeners),
                "monitor_channels": len(self.monitor_channel_ids),
                "log_channel_id": self.log_channel_id,
            }
            logger.info(f"🔧 ReactionLogger Health Check: {status}")
            return status
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'error': str(e)}

    async def _log_la_and_examiner_impl(self, payload: discord.RawReactionActionEvent, guild: discord.Guild, member: discord.Member) -> bool:
        emoji = str(payload.emoji)
        if emoji == "☑️" or emoji not in Config.TRACKED_REACTIONS:
            return False

        hicom_role = guild.get_role(Config.HIGH_COMMAND_ROLE_ID)
        inductor_role = guild.get_role(Config.LA_ROLE_ID)
        
        if not ((hicom_role and hicom_role in member.roles) or (inductor_role and inductor_role in member.roles)):
            return False
                
        channel = guild.get_channel(payload.channel_id)
        if not channel:
            return False
    
        tx = TransactionTracker(self)
        try:
            message = await channel.fetch_message(payload.message_id)
            
            if payload.channel_id == Config.LA_INDUCTION_CHANNEL_ID: 
                await tx.update_hr(member, {"courses": Config.POINTS_PER_ACTIVITY})
                embed = embedBuilder.build_inductor_record(message.author, member, message, Config.POINTS_PER_ACTIVITY, emoji)
            else:
                points = Config.POINTS_PER_ACTIVITY * 2
                await tx.update_hr(message.author, {"courses": points})
                embed = embedBuilder.build_examiner_record(message.author, member, message, points)
                
            msg = await self.log_channel.send(embed=embed)
            tx.track_message(msg)
            return True
        except Exception as e:
            await tx.rollback()
            raise e
    
    async def _log_security_check_log_reaction_impl(self, payload: discord.RawReactionActionEvent, guild: discord.Guild, member: discord.Member) -> bool:
        emoji = str(payload.emoji)
        if emoji == "☑️" or emoji not in Config.TRACKED_REACTIONS:
            return False

        hicom_role = guild.get_role(Config.HIGH_COMMAND_ROLE_ID)
        if not hicom_role or hicom_role not in member.roles:
            return False
    
        channel = guild.get_channel(payload.channel_id)
        if not channel:
            return False
    
        message = await channel.fetch_message(payload.message_id)
    
        match = SECURITY_CHECK_PATTERN.search(message.content)
        if not match:
            error_embed = discord.Embed(
                description="Could not log that reaction as no security check count was found.",
                color=discord.Color.red(),
            )
            error_embed.set_author(name="Reaction Logging Error", icon_url=Config.CANCEL_URL)
            await self.log_channel.send(content=member.mention, embed=error_embed)
            return False
        
        security_checks = int(match.group(1))
        points = Config.POINTS_PER_ACTIVITY * security_checks
        
        tx = TransactionTracker(self)
        try:
            await tx.update_hr(message.author, {"courses": points})
            embed = embedBuilder.build_sc_check_log(member, message.author, security_checks, points, message)
            msg = await self.log_channel.send(embed=embed)
            tx.track_message(msg)
            return True
        except Exception as e:
            await tx.rollback()
            raise e

    async def _log_dbl_reaction_impl(self, payload: discord.RawReactionActionEvent, guild: discord.Guild, member: discord.Member) -> bool:
        if str(payload.emoji) not in Config.TRACKED_REACTIONS:
            return False
    
        if Config.DB_LOGGER_ROLE_ID:
            monitor_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
            if not monitor_role or monitor_role not in member.roles:
                return False
    
        channel = guild.get_channel(payload.channel_id)
        if not all((channel, member, self.log_channel)):
            return False
        
        tx = TransactionTracker(self)
        try:
            await asyncio.sleep(0.5)
            message = await channel.fetch_message(payload.message_id)
            await tx.update_hr(member, {"courses": Config.POINTS_PER_ACTIVITY})
            embed = embedBuilder.build_db_logger_record(member, message, Config.POINTS_PER_ACTIVITY, payload.emoji)
            msg = await self.log_channel.send(embed=embed)
            tx.track_message(msg)
            return True
        except discord.NotFound:
            return False
        except Exception as e:
            await tx.rollback()
            raise e
   
    async def _log_event_reaction_impl(self, payload: discord.RawReactionActionEvent, guild: discord.Guild, member: discord.Member) -> bool:
        if payload.channel_id not in self.event_channel_ids or str(payload.emoji) != "✅":
            return False

        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return False

        channel = guild.get_channel(payload.channel_id)
        if not channel or not self.log_channel:
            return False

        tx = TransactionTracker(self)
        try:
            await asyncio.sleep(0.5)
            message = await channel.fetch_message(payload.message_id)

            exempt_role_ids = {Config.HR_ROLE_ID, Config.HQ_ROLE_ID, Config.HIGH_COMMAND_ROLE_ID}
            exempt_roles = {guild.get_role(rid) for rid in exempt_role_ids} - {None}

            host_mention = HOST_PATTERN.search(message.content)
            host_id = int(host_mention.group(1)) if host_mention else message.author.id
            host_member = guild.get_member(host_id) or await guild.fetch_member(host_id)
            
            co_hosts = {}
            co_host_mentions = CO_HOST_PATTERN.search(message.content)
            
            if co_host_mentions:
                co_host_ids = MENTION_PATTERN.findall(co_host_mentions.group(1))
                for mid in set(co_host_ids):
                    co_host_id = int(mid)  
                    if co_host_id == host_id:
                        continue
                    co_host = guild.get_member(co_host_id) or await guild.fetch_member(co_host_id)
                    if co_host:
                        cleaned_name = f"{clean_nickname(co_host.display_name)} (`{co_host_id}`)"
                        co_hosts[cleaned_name] = co_host

            cleaned_host_name = clean_nickname(host_member.display_name)
            hr_update_field = "events"
            
            if payload.channel_id == Config.W_EVENT_LOG_CHANNEL_ID:
                content = message.content
                if re.search(r'\bjoint\b', content, re.IGNORECASE):
                    hr_update_field = "joint_events"
                elif re.search(r'\b(inspection|pi)\b', content, re.IGNORECASE):
                    hr_update_field = "inspections"

            event_name_match = EVENT_NAME_PATTERN.search(message.content)
            event_name = event_name_match.group(1).strip() if event_name_match else hr_update_field.replace("_", " ").title()

            await tx.update_hr(host_member, {hr_update_field: 1})
            _, new_total = await tx.add_xp(str(host_member.id), host_member.display_name, 1)
           
            co_host_names = "\n".join(co_hosts.keys()) if co_hosts else None
            for co_host in co_hosts.values():
                await tx.update_hr(co_host, {hr_update_field: 0.5})

            attendees_section = ATTENDEES_PATTERN.search(message.content)
            if not attendees_section:
                return False

            attendee_ids = list({int(uid) for uid in MENTION_PATTERN.findall(attendees_section.group(1))})

            async def fetch_member(uid: int) -> discord.Member | None:
                return guild.get_member(uid) or await guild.fetch_member(uid)

            fetched = await asyncio.gather(*[fetch_member(uid) for uid in attendee_ids], return_exceptions=True)
            attendee_members = [m for m in fetched if isinstance(m, discord.Member)]

            successful_attendees = []
            for attendee in attendee_members:
                name_str = f"{clean_nickname(attendee.display_name)} | {attendee.id}"
                if exempt_roles & set(attendee.roles):  
                    successful_attendees.append(name_str)
                    continue
                
                success = await tx.update_lr(attendee, {"events_attended": 1})
                successful_attendees.append(name_str if success else f"{name_str} (failed to update points)")
            
            await tx.update_hr(member, {"courses": Config.POINTS_PER_ACTIVITY})
            
            db_embed = embedBuilder.build_db_logger_record(member, message, Config.POINTS_PER_ACTIVITY, payload.emoji)
            log_embed = embedBuilder.build_event_log(member, message, host_member, co_host_names, event_name, "\n".join(successful_attendees))
            
            # SEND ALWAYS RETURNS A SINGLE MESSAGE OBJECT EVEN FOR MULTIPLE EMBEDS
            msg = await self.log_channel.send(embeds=[db_embed, log_embed])
            tx.track_message(msg)
            
            await asyncio.sleep(0.3)
            admin_cog = self.bot.get_cog("XPCog")
            if admin_cog:
                await admin_cog.log_xp_to_discord(member, [f"{cleaned_host_name} | {host_member.id}: {new_total-1} ↠ {new_total}"], 1, f"[Hosting]({message.jump_url})", message.id)
            return True
        except Exception as e:
            await tx.rollback()
            raise e

    async def _log_training_reaction_impl(self, payload: discord.RawReactionActionEvent, guild: discord.Guild, member: discord.Member) -> bool:
        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return False

        mapping = {
            self.phase_log_channel_id: "phases",
            self.tryout_log_channel_id: "tryouts",
            self.course_log_channel_id: "phases",
            self.tc_supervision_log_channel_id: "phases",
        }

        column_to_update = mapping.get(payload.channel_id)
        if not column_to_update or str(payload.emoji) != "✅":
            return False

        tx = TransactionTracker(self)
        try:
            await asyncio.sleep(0.5)
            channel = guild.get_channel(payload.channel_id)
            if not channel:
                return False

            message = await channel.fetch_message(payload.message_id)
            user_mention = HOST_PATTERN.search(message.content)
            user_id = int(user_mention.group(1)) if user_mention else message.author.id
            host_member = guild.get_member(user_id) or await guild.fetch_member(user_id)

            await tx.update_hr(host_member, {column_to_update: 1})

            display_mapping = {
                self.phase_log_channel_id: "Phase",
                self.tryout_log_channel_id: "Tryout",
                self.course_log_channel_id: "Course",
                self.tc_supervision_log_channel_id: "TC Supervision",
            } 
            
            title = display_mapping.get(payload.channel_id)
            log_embed = embedBuilder.build_event_log(member, message, host_member, event_type=title)

            await tx.update_hr(member, {"courses": Config.POINTS_PER_ACTIVITY})
            db_embed = embedBuilder.build_db_logger_record(member, message, Config.POINTS_PER_ACTIVITY, payload.emoji)

            # Fixed multi-embed return object handling
            msg = await self.log_channel.send(embeds=[db_embed, log_embed])
            tx.track_message(msg)
            
            return True
        except Exception as e:
            await tx.rollback()
            raise e

    async def _log_activity_reaction_impl(self, payload: discord.RawReactionActionEvent, guild: discord.Guild, member: discord.Member) -> bool:
        if str(payload.emoji) != "✅":
            return False

        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return False

        channel = guild.get_channel(payload.channel_id)
        if not channel:
            return False

        tx = TransactionTracker(self)
        try:
            await asyncio.sleep(0.5)
            message = await channel.fetch_message(payload.message_id)

            exempt_role_ids = {Config.HR_ROLE_ID, Config.HQ_ROLE_ID, Config.HIGH_COMMAND_ROLE_ID}
            exempt_roles = {guild.get_role(rid) for rid in exempt_role_ids} - {None}

            user_mention = MENTION_PATTERN.search(message.content)
            user_id = int(user_mention.group(1)) if user_mention else message.author.id
            user_member = guild.get_member(user_id) or await guild.fetch_member(user_id)

            if exempt_roles & set(user_member.roles):
                embed = discord.Embed(description="Cannot log that. User is a high rank.", color=discord.Color.red())
                await self.log_channel.send(content=member.mention, embed=embed)  
                return False

            updates = {}
            is_time_guarded = False
            time_match = TIME_PATTERN.search(message.content)
            if time_match:
                minutes = int(time_match.group(1))
                if "Guarded:" in message.content:
                    updates["time_guarded"] = minutes
                    is_time_guarded = True
                else:
                    updates["activity"] = minutes

            if not updates:
                return False

            await tx.update_lr(user_member, updates)

            total_minutes = updates.get("activity", 0) + updates.get("time_guarded", 0)
            xp_to_award = total_minutes // 30

            if xp_to_award > 0:
                await tx.add_xp(str(user_member.id), user_member.display_name, xp_to_award)

            log_embed = embedBuilder.build_activity_log(member, message, user_member, total_minutes, is_time_guarded, xp_to_award)  
            await tx.update_hr(member, {"courses": Config.POINTS_PER_ACTIVITY})
            db_embed = embedBuilder.build_db_logger_record(member, message, Config.POINTS_PER_ACTIVITY, payload.emoji)

            # Fixed multi-embed return object handling
            msg = await self.log_channel.send(embeds=[db_embed, log_embed])
            tx.track_message(msg)
            
            return True
        except Exception as e:
            await tx.rollback()
            raise e

    async def _update_hr_record(self, member: discord.Member, updates: dict) -> bool:
        try:
            success = True
            for column, points in updates.items():
                result = await self.bot.db.increment_points_handler(column=column, table=self.bot.db.hrs_table, member=member, points=points)
                if not result:
                    success = False
            return success
        except Exception:
            logger.exception("ReactionLogger._update_hr_record failed")
            return False
    
    async def _update_lr_record(self, member: discord.Member, updates: dict) -> bool:
        try:
            success = True
            for column, points in updates.items():
                result = await self.bot.db.increment_points_handler(column=column, table=self.bot.db.lrs_table, member=member, points=points)
                if not result:
                    success = False
            return success
        except Exception:
            logger.exception("ReactionLogger._update_lr_record failed")
            return False

    async def update_hr(self, member: discord.Member, updates: dict):
        try:
            await self._update_hr_record(member, updates)
            logger.info(f"HR record updated for {member.display_name}: {updates}")
        except Exception as e:
            logger.error(f"Failed to update HR record for {member.display_name}: {e}")

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        await self.bot.rate_limiter.wait_if_needed(bucket="reaction_log")

        if payload.member and payload.member.bot:
            return

        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
    
        member = guild.get_member(payload.user_id)
        if not member:
            try:
                member = await guild.fetch_member(payload.user_id)
            except discord.NotFound:
                return

        if await self.is_reaction_processed(payload.message_id, member):
            error_embed = discord.Embed(
                    description="That log has already been processed recently. If you wish to still log try using the force-log command.",
                    color=discord.Color.red(),
            )
            error_embed.set_author(name="Duplicate Log Prevention", icon_url=Config.SHIELD_WARNING_ICON)
            msg = await self.log_channel.send(content=member.mention, embed=error_embed)
            return
        
        try:
            for h in self.REACTION_HANDLERS:
                if h.channels is None or payload.channel_id in h.channels:
                    try:
                        success = await getattr(self, h.handler)(payload, guild, member)
                        if success:
                            logger.info(f"Processed Reaction event | msg={payload.message_id} channel={payload.channel_id} user={payload.user_id} | reaction={payload.emoji}")
                            await self.mark_reaction_processed(payload.message_id, payload.user_id)
                            break
                    except Exception as e:
                        await self._handle_reaction_error(e, member, h.handler)
        except Exception as e:
            logger.error(f"ReactionLogger.on_raw_reaction_add failed: {e}", exc_info=True)

async def setup(bot):
    await bot.add_cog(ReactionLoggerCog(bot))
