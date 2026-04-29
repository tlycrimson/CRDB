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
from typing import Callable

logger = logging.getLogger(__name__)

@dataclass
class ReactionHandler:
    handler: Callable
    channels: set[int] | None = None


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
                ReactionHandler("_log_dbl_reaction_impl", channels=set(self.monitor_channel_ids)),
                ReactionHandler("_log_event_reaction_impl", channels=set(self.event_channel_ids)),
                ReactionHandler("_log_training_reaction_impl", channels={self.phase_log_channel_id, self.tryout_log_channel_id, self.course_log_channel_id, self.tc_supervision_log_channel_id}),
                ReactionHandler("_log_activity_reaction_impl", channels={self.activity_log_channel_id}),
                ReactionHandler("_log_security_check_log_reaction_impl", channels={self.sc_log_channel_id}),
                ReactionHandler("_log_la_and_examiner_impl", channels=set(self.exam_monitor_channel_ids))
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
        """Setup reaction logger with optional parameters"""
        if interaction and log_channel and monitor_channels:
            channel_ids = [int(cid.strip()) for cid in monitor_channels.split(',')]
            self.monitor_channel_ids = set(channel_ids)
            await interaction.followup.send("Reaction tracking setup complete", ephemeral=True)
            
      
    async def on_ready_setup(self):
        """Verify configured channels when bot starts"""
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
        """Check if reaction was already processed"""
        try:
            res = await self.bot.db.supabase.table('processed_reactions')\
                .select('id')\
                .eq('message_id', str(message_id))\
                .eq('user_id', str(member.id))\
                .execute()
            
            if res.data: 
                return True
        except Exception as e:
            logger.error(f"Error checking processed reaction: {e}")

        return False

    async def mark_reaction_processed(self, message_id: int, user_id: int):
        """Mark reaction as processed"""
        try:
            await self.bot.db.supabase.table('processed_reactions')\
                .upsert(
                        {
                            'message_id': str(message_id),
                            'user_id': str(user_id)
                        },
                        on_conflict='message_id,user_id'
                )\
                .execute()
        except Exception as e:
            logger.error(f"Error marking reaction processed: {e}")

    async def _handle_reaction_error(self, e: Exception, member: discord.Member, handler: str = "") -> None:
        logger.error(f"Failed in {handler}: {type(e).__name__}: {e}", exc_info=True)
        if self.log_channel:
            error_embed = discord.Embed(
                description="An error occurred while logging this reaction.",
                color=discord.Color.red(),
            )
            error_embed.set_author(name="Reaction Logging Error", icon_url=Config.CANCEL_URL)
            await self.log_channel.send(content=member.mention, embed=error_embed)

    async def health_check(self):
        """Check the health of the reaction logger"""
        try:
            # Check if listeners are registered
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
        
        if not (
            (hicom_role and hicom_role in member.roles) or
            (inductor_role and inductor_role in member.roles)
        ):
            return False
                
        channel = guild.get_channel(payload.channel_id)
        if not channel:
            return False
    
        message = await channel.fetch_message(payload.message_id)
        author = message.author
        logger = member 

        if payload.channel_id == Config.LA_INDUCTION_CHANNEL_ID: 
            await self._update_hr_record(author, {"courses": Config.POINTS_PER_ACTIVITY})

            embed = embedBuilder.build_inductor_record(author, logger, message, Config.POINTS_PER_ACTIVITY, emoji)
            await self.log_channel.send(embed=embed)

        else:
            points = Config.POINTS_PER_ACTIVITY*2
            await self._update_hr_record(author, {"courses": points})

            embed = embedBuilder.build_examiner_record(author, logger, message, points)
            await self.log_channel.send(embed=embed)

        return True
    
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
        log_author = message.author
    
        match = re.search(
            r"Security\s*Check(?:/s|\(s\)|s)?:\s*(\d+)",
            message.content,
            re.IGNORECASE
        )
        
        if not match:
            logger.warning(
                f"No Security Check count found in message {payload.message_id}"
            )
            error_embed = discord.Embed(
                description="Could not log that reaction as no security check count was found.",
                color=discord.Color.red(),
            )
            error_embed.set_author(name="Reaction Logging Error", icon_url=Config.CANCEL_URL)
            await self.log_channel.send(content=member.mention, embed=error_embed)
            return False
        
        security_checks = int(match.group(1))
    
        points = Config.POINTS_PER_ACTIVITY * security_checks
        await self._update_hr_record(log_author, {"courses": points})

        embed = embedBuilder.build_sc_check_log(member, log_author, security_checks, points, message)
        await self.log_channel.send(embed=embed)

        logger.info(
            f"Security Check log Approved: ApprovedBy={member.id}, Logger={log_author.id}"
        )

        return True

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
        
        try:
            await asyncio.sleep(0.5)
            message = await channel.fetch_message(payload.message_id)

            await self._update_hr_record(member, {"courses": Config.POINTS_PER_ACTIVITY})
            
            embed = embedBuilder.build_db_logger_record(member, message, Config.POINTS_PER_ACTIVITY, payload.emoji)
            await self.log_channel.send(embed=embed)
            return True
        except discord.NotFound:
            return False
        except Exception as e:
            logger.error(f"Reaction log error: {type(e).__name__}: {str(e)}")
            await self._handle_reaction_error(e, member, "_log_dbl_reaction_impl")
            return False 
   
    async def _log_event_reaction_impl(self, payload: discord.RawReactionActionEvent, guild: discord.Guild, member: discord.Member)-> bool:
        """Handle event logging without confirmation"""
        if payload.channel_id not in self.event_channel_ids or str(payload.emoji) != "✅":
            return False

        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return False

        channel = guild.get_channel(payload.channel_id)
        if not channel or not self.log_channel:
            return False

        try:
            await asyncio.sleep(0.5)  # Prevent bursts
            message = await channel.fetch_message(payload.message_id)

            # --- Pre-fetch all roles at once ---
            exempt_role_ids = {Config.HR_ROLE_ID, Config.HQ_ROLE_ID, Config.HIGH_COMMAND_ROLE_ID}
            exempt_roles = {guild.get_role(rid) for rid in exempt_role_ids} - {None}

            # --- Extract host ---
            host_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
            host_id = int(host_mention.group(1)) if host_mention else message.author.id
            host_member = guild.get_member(host_id) or await guild.fetch_member(host_id)
            cleaned_host_name = clean_nickname(host_member.display_name)

            # --- Determine event type ---
            hr_update_field = "events"
            if payload.channel_id == Config.W_EVENT_LOG_CHANNEL_ID:
                content = message.content
                if re.search(r'\bjoint\b', content, re.IGNORECASE):
                    hr_update_field = "joint_events"
                elif re.search(r'\b(inspection|pi)\b', content, re.IGNORECASE):  # IGNORECASE covers capitalisation
                    hr_update_field = "inspections"

            event_name_match = re.search(r'Event:\s*(.*?)(?:\n|$)', message.content, re.IGNORECASE)
            event_name = event_name_match.group(1).strip() if event_name_match else hr_update_field.replace("_", " ").title()

            # --- Update host HR record ---
            await self._update_hr_record(host_member, {hr_update_field: 1})

            # --- Parse attendees ---
            attendees_section = re.search(
                r'(?:Attendees:|Passed:)\s*((?:<@!?\d+>[\s,]*)+)',
                message.content,
                re.IGNORECASE
            )
            if not attendees_section:
                return

            attendee_ids = list({int(uid) for uid in re.findall(r'<@!?(\d+)>', attendees_section.group(1))})

            # --- Fetch all attendee members concurrently ---
            async def fetch_member(uid: int) -> discord.Member | None:
                return guild.get_member(uid) or await guild.fetch_member(uid)

            fetched = await asyncio.gather(*[fetch_member(uid) for uid in attendee_ids], return_exceptions=True)
            attendee_members = [m for m in fetched if isinstance(m, discord.Member)]

            # --- Process attendees ---
            hr_excluded_count, successful_attendees = 0, []

            update_tasks = []
            attendees_to_update = []

            for attendee in attendee_members:
                if exempt_roles & set(attendee.roles):  
                    hr_excluded_count += 1
                    continue
                
                update_tasks.append(self._update_lr_record(attendee, {"events_attended": 1}))
                attendees_to_update.append(attendee)

            results = await asyncio.gather(*update_tasks)

            for attendee, success in zip(attendees_to_update, results):
                name_str = f"{clean_nickname(attendee.display_name)} | {attendee.id}"
                
                if success:
                    successful_attendees.append(name_str)
                else:
                    successful_attendees.append(f"{name_str} (failed to update points)")
                    
            # --- Send log embed ---
            embed = embedBuilder.build_event_log(member, message, host_member, event_name, "\n".join(successful_attendees), hr_excluded_count)
            await self.log_channel.send(embed=embed)

            logger.info(
                f"Event logged successfully: "
                f"Host={host_member} ({host_member.id}), "
                f"Logged_By={member} ({member.id}), "
                f"MessageID={message.id}"
            )
            return True
        except Exception as e:
            await self._handle_reaction_error(e, member, "_log_event_reaction_impl")
            return False

    async def _log_training_reaction_impl(self, payload: discord.RawReactionActionEvent, guild: discord.Guild, member: discord.Member) -> bool:
        """Handle training logs (phases, tryouts, courses)"""
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

        try:
            await asyncio.sleep(0.5)
            
            channel =  guild.get_channel(payload.channel_id)
            if not channel:
                return False

            message = await channel.fetch_message(payload.message_id)

            user_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
            user_id = int(user_mention.group(1)) if user_mention else message.author.id
            host_member = guild.get_member(user_id) or await guild.fetch_member(user_id)

            await self._update_hr_record(host_member, {column_to_update: 1})

            mapping = {
                self.phase_log_channel_id: "Phase",
                self.tryout_log_channel_id: "Tryout",
                self.course_log_channel_id: "Course",
                self.tc_supervision_log_channel_id: "TC Supervision",
            } 
            
            title = mapping.get(payload.channel_id)
            embed = embedBuilder.build_event_log(member, message, host_member, title)
            await self.log_channel.send(embed=embed)

            logger.info(
                    f"Training Event Succesfully logged: "
                    f"Host={host_member} ({host_member.id}), "
                    f"Logged_By={member} ({member.id}), "
            )
            
            return True
        except Exception as e:
            await self._handle_reaction_error(e, member, "_log_training_reaction_impl")
            return False


    async def _log_activity_reaction_impl(self, payload: discord.RawReactionActionEvent, guild: discord.Guild, member: discord.Member)-> bool:
        """Handle activity logs (time guarded and activity)"""
        if str(payload.emoji) != "✅":
            return False

        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return False

        channel = guild.get_channel(payload.channel_id)
        if not channel:
            return False

        exempt_role_ids = {Config.HR_ROLE_ID, Config.HQ_ROLE_ID, Config.HIGH_COMMAND_ROLE_ID}
        exempt_roles = {guild.get_role(rid) for rid in exempt_role_ids} - {None}

        try:
            await asyncio.sleep(0.5)
            message = await channel.fetch_message(payload.message_id)

            user_mention = re.search(r'<@!?(\d+)>', message.content)
            user_id = int(user_mention.group(1)) if user_mention else message.author.id
            user_member = guild.get_member(user_id) or await guild.fetch_member(user_id)

            if exempt_roles & set(user_member.roles):
                embed = discord.Embed(
                    description="Cannot log that. User is a high rank and or has the role.",
                    color=discord.Color.red()
                ).set_author(name="Log Prevention", icon_url=Config.SHIELD_WARNING_ICON)
                await self.log_channel.send(content=member.mention, embed=embed)  
                return

            updates = {}
            is_time_guarded = False
            time_match = re.search(r'Time:\s*(\d+)', message.content)
            if time_match:
                minutes = int(time_match.group(1))
                if "Guarded:" in message.content:
                    updates["time_guarded"] = minutes
                    is_time_guarded = True
                else:
                    updates["activity"] = minutes

            if not updates:
                return

            await self._update_lr_record(user_member, updates)

            total_minutes = updates.get("activity", 0) + updates.get("time_guarded", 0)
            xp_to_award = total_minutes // 30

            if xp_to_award > 0:
                success, _ = await self.bot.db.add_xp(
                    str(user_member.id),
                    user_member.display_name,
                    xp_to_award
                )
                if success:
                    logger.info(f"Gave {xp_to_award} XP to {user_member.display_name} ({user_member.id}) for {total_minutes} mins activity")

            embed = embedBuilder.build_activity_log(member, message, user_member, total_minutes, is_time_guarded, xp_to_award)  
            await self.log_channel.send(embed=embed)
            
            return True
        except Exception as e:
            await self._handle_reaction_error(e, member, "_log_activity_reaction_impl")
            return False


    # Won't Combine these Functions In Case I require new logic for LRs and HRs
    async def _update_hr_record(self, member: discord.Member, updates: dict) -> bool:
        try:
            column, points = next(iter(updates.items()))
            return await self.bot.db.increment_points_handler(column=column, table=self.bot.db.hrs_table, member=member, points=points)
        except Exception:
            logger.exception("ReactionLogger._update_hr_record failed")
            return False
    
    
    async def _update_lr_record(self, member: discord.Member, updates: dict) -> bool:
        try:
            column, points = next(iter(updates.items()))
            return await self.bot.db.increment_points_handler(column=column, table=self.bot.db.lrs_table, member=member, points=points)

        except Exception:
            logger.exception("ReactionLogger._update_lr_record failed")
            return False

    async def update_hr(self, member: discord.Member, updates: dict):
        """Public method to update HR record from other classes"""
        try:
            await self._update_hr_record(member, updates)
            logger.info(f"HR record updated for {member.display_name}: {updates}")
        except Exception as e:
            logger.error(f"Failed to update HR record for {member.display_name}: {e}")

                        
      # --- Event listener ---
    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Triggered when a reaction is added anywhere the bot can see."""

        await self.bot.rate_limiter.wait_if_needed(bucket="reaction_log")

        if payload.member.bot:
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
            logger.error(
                f"Duplicate reaction ignored | "
                f"msg={payload.message_id}, channel={payload.message_id} user={member.id}"
            )
            error_embed = discord.Embed(
                    description="That log has already been processed recently. If you wish to still log try /force-log.",
                    color=discord.Color.red(),
            )
            error_embed.set_author(
                    name="Duplicate Log Prevention",
                     icon_url= Config.SHIELD_WARNING_ICON
            )
            return await self.log_channel.send(content=member.mention, embed=error_embed)
        
    
        try:
            for h in self.REACTION_HANDLERS:
                if h.channels is None or payload.channel_id in h.channels:
                    try:
                        success = await getattr(self, h.handler)(payload, guild, member)
                        if success:
                            logger.info(f"Processed Reaction event | msg={payload.message_id} channel={payload.channel_id} user={payload.user_id} reaction={payload.emoji}")
                            await self.mark_reaction_processed(payload.message_id, payload.user_id)
                    except Exception as e:
                        await self._handle_reaction_error(e, member, h.handler)

        except Exception as e:
            logger.error(f"ReactionLogger.on_raw_reaction_add failed: {e}", exc_info=True)

async def setup(bot):
    await bot.add_cog(ReactionLoggerCog(bot))
