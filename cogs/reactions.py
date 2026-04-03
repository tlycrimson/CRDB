import re
import asyncio
import discord
import logging
from config import Config
from discord.ext import commands
from utils.helpers import clean_nickname
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

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
        
    # Clean up old processed reactions in db
    async def start_cleanup_task(self):
        """Clean up old database entries periodically"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(86400)  # Cleanup daily
                try:
                    # Clean entries older than 30 days
                    def _cleanup():
                        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                        self.bot.db.supabase.table('processed_messages')\
                            .delete()\
                            .lt('processed_at', cutoff.isoformat())\
                            .execute()
                        self.bot.db.supabase.table('processed_reactions')\
                            .delete()\
                            .lt('processed_at', cutoff.isoformat())\
                            .execute()
                    await self.bot.db.run_query(_cleanup)
                    logger.info("Cleaned up old processed messages/reactions")
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())


    async def configure(self, interaction=None, log_channel=None, monitor_channels=None):
        """Setup reaction logger with optional parameters"""
        if interaction and log_channel and monitor_channels:
            # Setup from command
            channel_ids = [int(cid.strip()) for cid in monitor_channels.split(',')]
            self.monitor_channel_ids = set(channel_ids)
            self.log_channel_id = log_channel.id
            await interaction.followup.send("✅ Reaction tracking setup complete", ephemeral=True)
            
        # Supabase connection check
        try:
            await self.bot.db.run_query(lambda: self.bot.db.supabase.table("LD").select("count").limit(1).execute())
            logger.info("✅ Supabase connection validated")
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")

    
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
        
        log_channel = guild.get_channel(self.log_channel_id)
        if not log_channel:
            logger.warning(f"⚠️ Default log channel {self.log_channel_id} not found! Trying other.")
            log_channel = guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
        else:
            logger.info("Default Log channel configured.")
        
        if not log_channel:
            logger.error("❌ No valid log channel found for ReactionLogger!")
            
    async def is_reaction_processed(self, message_id: int, user_id: int) -> bool:
        """Check if reaction was already processed"""
        try:
            def _check():
                res = self.bot.db.supabase.table('processed_reactions')\
                    .select('id')\
                    .eq('message_id', message_id)\
                    .eq('user_id', user_id)\
                    .execute()
                return len(res.data) > 0
            return await self.bot.db.run_query(_check)
        except Exception as e:
            logger.error(f"Error checking processed reaction: {e}")
            return False

    async def mark_reaction_processed(self, message_id: int, user_id: int):
        """Mark reaction as processed"""
        try:
            def _insert():
                self.bot.db.supabase.table('processed_reactions')\
                    .insert({
                        'message_id': message_id,
                        'user_id': user_id
                    })\
                    .execute()
            await self.bot.db.run_query(_insert)
        except Exception as e:
            logger.error(f"Error marking reaction processed: {e}")

        
    async def log_reaction(self, payload: discord.RawReactionActionEvent):
        """Main reaction handler that routes to specific loggers with DB + memory duplicate checks."""
        
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
    
        member = guild.get_member(payload.user_id)
        if not member:
            try:
                member = await guild.fetch_member(payload.user_id)
            except discord.NotFound:
                return
    
        # === DUPLICATE GUARD (silent) ===
        if await self.is_reaction_processed(payload.message_id, payload.user_id):
            logger.debug(
                f"Duplicate reaction ignored | "
                f"msg={payload.message_id} user={payload.user_id}"
            )
            return
            
        
        logger.info(f"🔍 Reaction detected: {payload.emoji} in channel {payload.channel_id} by user {payload.user_id}")

        # === For Background Checkers ===
        if payload.channel_id == Config.BGC_LOGS_CHANNEL:
            emoji = str(payload.emoji)
            if emoji == "☑️" or emoji not in Config.TRACKED_REACTIONS:
                return
        
            hicom_role = guild.get_role(Config.HIGH_COMMAND_ROLE_ID)
            if not hicom_role or hicom_role not in member.roles:
                return
        
            channel = guild.get_channel(payload.channel_id)
            if not channel:
                return
        
            message = await channel.fetch_message(payload.message_id)
            log_author = message.author
        
        
            match = re.search(
                r"Security\s*Check(?:\(s\)|s)?:\s*(\d+)",
                message.content,
                re.IGNORECASE
            )
            
            if not match:
                logger.warning(
                    f"No Security Check count found in message {payload.message_id}"
                )
                return
            
            security_checks = int(match.group(1))
        
            log_channel = guild.get_channel(self.log_channel_id)
            if not log_channel:
                return
            
            points = Config.POINTS_PER_ACTIVITY * security_checks
            embed = discord.Embed(
                title="🪪 Security Check Log Approved",
                color=discord.Color.blue()
            )
            embed.add_field(name="Approved by", value=f"{member.mention} ({member.id})", inline=False)
            embed.add_field(name="Logger", value=f"{log_author.mention}", inline=False)
            embed.add_field(name="Amount of Checks", value=security_checks, inline=False)
            embed.add_field(name="Points Awarded", value=points, inline=True)
            embed.add_field(name="Log ID", value=f"`{payload.message_id}`", inline=False)
        
            await log_channel.send(embed=embed)
            await self._update_hr_record(log_author, {"courses": points})
        
            logger.info(
                f"🪪 Security Check log Approved: ApprovedBy={member.id}, Logger={log_author.id}"
            )
            # Prevents it from going to other log pipelines
            await self.mark_reaction_processed(payload.message_id, payload.user_id)
            return
        


        # === For Exam graders ===
        if payload.channel_id in Config.EXAM_MONITOR_CHANNELS:
            emoji = str(payload.emoji)
            if emoji == "☑️" or emoji not in Config.TRACKED_REACTIONS:
                return
        
            hicom_role = guild.get_role(Config.HIGH_COMMAND_ROLE_ID)
            inductor_role = guild.get_role(Config.LA_ROLE_ID)
            
            if not (
                (hicom_role and hicom_role in member.roles) or
                (inductor_role and inductor_role in member.roles)
            ):
                return
                    
            channel = guild.get_channel(payload.channel_id)
            if not channel:
                return
        
            message = await channel.fetch_message(payload.message_id)
            examiner = message.author
        
            log_channel = guild.get_channel(self.log_channel_id)
            if not log_channel:
                return
            
            if payload.channel_id == Config.LA_INDUCTION_CHANNEL_ID: 
                embed = discord.Embed(
                    title="📝 Inductor Activity Logged",
                    color=discord.Color.pink()
                )
                
                embed.add_field(name="Inductor", value=f"{member.mention}", inline=False)
                embed.add_field(name="Message Request", value=f"<#{payload.channel_id}>", inline=True)
                embed.add_field(name="Status", value=f"{emoji}", inline=True)
                embed.add_field(name="Points Awarded", value=Config.POINTS_PER_ACTIVITY, inline=True)
                await log_channel.send(embed=embed)
                logger.info(
                f"📝 Inductor Activity logged: Inductor={member.id}, Channel={payload.channel_id}"
                ) 
                await self._update_hr_record(member, {"courses": Config.POINTS_PER_ACTIVITY})
            else:
                points = Config.POINTS_PER_ACTIVITY*2
                embed = discord.Embed(
                    title="📝 Examiner Activity Logged",
                    color=discord.Color.pink()
                )
                embed.add_field(name="Examiner", value=f"{examiner.mention} ({examiner.id})", inline=False)
                embed.add_field(name="Exam Type", value=f"<#{payload.channel_id}>", inline=True)
                embed.add_field(name="Exam Message", value=f"`{payload.message_id}`", inline=False)
                embed.add_field(name="Points Awarded", value=points, inline=True)
                await log_channel.send(embed=embed)
                logger.info(
                f"📝 Examiner logged: Examiner={examiner.id}, Channel={payload.channel_id}"
                )
                await self._update_hr_record(examiner, {"courses": points})
        
            await self.mark_reaction_processed(payload.message_id, payload.user_id)      
            return
        
        # === Route to handlers ===
        try:
            await self.bot.rate_limiter.wait_if_needed(bucket="reaction_log")
            await self._log_reaction_impl(payload)
            await self._log_event_reaction_impl(payload, member)
            await self._log_training_reaction_impl(payload, member)
            await self._log_activity_reaction_impl(payload, member)
            await self.mark_reaction_processed(payload.message_id, payload.user_id)
        except Exception as e:
            logger.error(f"Failed to log reaction: {type(e).__name__}: {e}")
            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                error_embed = discord.Embed(
                    title="❌ Reaction Logging Error",
                    description="An error occured while logging this reaction.",
                    color=discord.Color.red(),
                )
                await log_channel.send(content=member.mention, embed=error_embed)

     
    async def _log_reaction_impl(self, payload: discord.RawReactionActionEvent):
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
    
        if (payload.channel_id not in self.monitor_channel_ids or 
            str(payload.emoji) not in Config.TRACKED_REACTIONS):
            return
    
        if str(payload.emoji) in Config.IGNORED_EMOJI:
            return
    
        member = guild.get_member(payload.user_id)
        if not member:
            try:
                member = await guild.fetch_member(payload.user_id)
            except discord.NotFound:
                logger.info("Member not found for reaction event; skipping")
                return
            except Exception:
                logger.exception("Failed to fetch member for reaction", exc_info=True)
                return
    
        if Config.DB_LOGGER_ROLE_ID:
            monitor_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
            if not monitor_role or monitor_role not in member.roles:
                return
    
        channel = guild.get_channel(payload.channel_id)
        log_channel = guild.get_channel(self.log_channel_id)
    
        if not all((channel, member, log_channel)):
            return
        
        try:
            await asyncio.sleep(0.5)
            message = await channel.fetch_message(payload.message_id)

            content = (message.content[:100] + "...") if len(message.content) > 100 else message.content

            embed = discord.Embed(
                title="🧑‍💻 DB Logger Activity Recorded",
                description=f"{member.mention} reacted with {payload.emoji}",
                color=discord.Color.purple()
            )
            embed.add_field(name="Channel", value=channel.mention)
            embed.add_field(name="Author", value=message.author.mention)
            embed.add_field(name="Message", value=content, inline=False)
            embed.add_field(name="Points Awarded", value=Config.POINTS_PER_ACTIVITY, inline=False)
            embed.add_field(name="Jump to", value=f"[Click here]({message.jump_url})", inline=False)

            try:
                await asyncio.wait_for(
                    log_channel.send(embed=embed),
                    timeout=5
                )
            except asyncio.TimeoutError: 
                logger.error("log_channel.send() timed out")
                return

            logger.info(f"Attempting to update points for: {member.display_name}")
            await self._update_hr_record(member, {"courses": Config.POINTS_PER_ACTIVITY})
            logger.info(f"✅ Added {Config.POINTS_PER_ACTIVITY} points to {member.display_name} for activity.")


        except discord.NotFound:
            return
        except Exception as e:
            logger.error(f"Reaction log error: {type(e).__name__}: {str(e)}")
            if log_channel:
                error_embed = discord.Embed(
                    title="❌ Error",
                    description=f"Failed to log reaction: {str(e)}",
                    color=discord.Color.red()
                )
                await log_channel.send(embed=error_embed)


   
    async def _log_event_reaction_impl(self, payload: discord.RawReactionActionEvent, member: discord.Member):
        """Handle event logging without confirmation"""
        if payload.channel_id not in self.event_channel_ids or str(payload.emoji) != "✅":
            return

        guild = member.guild
        if not guild:
            return

        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return

        channel = guild.get_channel(payload.channel_id)
        log_channel = guild.get_channel(self.log_channel_id)
        if not channel or not log_channel:
            return

        try:
            await asyncio.sleep(0.5)  # Prevent bursts
            message = await channel.fetch_message(payload.message_id)

            # Extract host
            host_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
            host_id = int(host_mention.group(1)) if host_mention else message.author.id
            host_member = guild.get_member(host_id) or await guild.fetch_member(host_id)

            cleaned_host_name = clean_nickname(host_member.display_name)

            # Determine event type
            hr_update_field = "events"
            if payload.channel_id == Config.W_EVENT_LOG_CHANNEL_ID:
                if re.search(r'\bjoint\b', message.content, re.IGNORECASE):
                    hr_update_field = "joint_events"
                elif re.search(r'\b(inspection|pi|Inspection)\b', message.content, re.IGNORECASE):
                    hr_update_field = "inspections"

            event_name_match = re.search(r'Event:\s*(.*?)(?:\n|$)', message.content, re.IGNORECASE)
            event_name = event_name_match.group(1).strip() if event_name_match else hr_update_field.replace("_", " ").title()

            # Update HR table
            await self._update_hr_record(host_member, {hr_update_field: 1})
            logger.info(f"✅ Logged host {cleaned_host_name} to HR table")

            # Process attendees
            attendees_section = re.search(r'(?:Attendees:|Passed:)\s*((?:<@!?\d+>\s*)+)', message.content, re.IGNORECASE)
            if not attendees_section:
                return
            attendee_mentions = re.findall(r'<@!?(\d+)>', attendees_section.group(1))

            hr_role = guild.get_role(Config.HR_ROLE_ID)
            hr_attendees, success_count = [], 0

            for attendee_id in attendee_mentions:
                attendee_member = guild.get_member(int(attendee_id)) or await guild.fetch_member(int(attendee_id))
                if not attendee_member:
                    continue
                if hr_role and hr_role in attendee_member.roles:
                    hr_attendees.append(attendee_id)
                    continue
                await self._update_lr_record(attendee_member, {"events_attended": 1})
                success_count += 1

            # Embed
            done_embed = discord.Embed(title="✅ Event Logged Successfully", color=discord.Color.green())
            done_embed.add_field(name="Host", value=host_member.mention, inline=True)
            done_embed.add_field(name="Attendees Recorded", value=str(success_count), inline=True)
            if hr_attendees:
                done_embed.add_field(name="HR Attendees Excluded", value=str(len(hr_attendees)), inline=False)
            done_embed.add_field(name="Logged By", value=member.mention, inline=False)
            done_embed.add_field(name="Event Type", value=event_name, inline=True)
            done_embed.add_field(name="Message", value=f"[Jump to Event]({message.jump_url})", inline=False)

            await log_channel.send(content=member.mention, embed=done_embed)
            logger.info(
                f"✅ Event logged successfully: "
                f"Host={host_member} ({host_member.id}), "
                f"Attendees={success_count}, "
                f"HR_Excluded={len(hr_attendees) if hr_attendees else 0}, "
                f"Logged_By={member} ({member.id}), "
                f"EventType={event_name}, "
                f"MessageID={message.id}"
            )


        except Exception as e:
            logger.error(f"Error processing event reaction: {e}")
            await log_channel.send(embed=discord.Embed(title="❌ Event Log Error", description=str(e), color=discord.Color.red()))



    async def _log_training_reaction_impl(self, payload: discord.RawReactionActionEvent, member: discord.Member):
        """Handle training logs (phases, tryouts, courses)"""
        mapping = {
            self.phase_log_channel_id: "phases",
            self.tryout_log_channel_id: "tryouts",
            self.course_log_channel_id: "phases",
            self.tc_supervision_log_channel_id: "phases",
        }
        column_to_update = mapping.get(payload.channel_id)
        if not column_to_update or str(payload.emoji) != "✅":
            return

        guild = member.guild
        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return

        try:
            await asyncio.sleep(0.5)
            
            channel = await guild.get_channel(payload.channel_id)
            if not channel:
                return

            message = await channel.fetch_message(payload.message_id)

            user_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
            user_id = int(user_mention.group(1)) if user_mention else message.author.id
            user_member = guild.get_member(user_id) or await guild.fetch_member(user_id)

            await self._update_hr_record(user_member, {column_to_update: 1})

            title = {
                "phases": "📊 Phase Logged",
                "tryouts": "📊 Tryout Logged",
                "courses": "📊 Course Logged",
            }.get(column_to_update, "📊 Training Logged")

            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                embed = discord.Embed(title=title, color=discord.Color.blue())
                if payload.channel_id == self.tc_supervision_log_channel_id:
                    embed.add_field(name="Supervisor", value=user_member.mention)
                else:
                    embed.add_field(name="Host", value=user_member.mention)
                embed.add_field(name="Logged By", value=member.mention)
                await log_channel.send(embed=embed)
                logger.info(
                    f"📘 Event log embed sent: "
                    f"Title='{title}', "
                    f"Host={user_member} ({user_member.id}), "
                    f"Logged_By={member} ({member.id}), "
                    f"Channel=#{log_channel.name}"
                )

        except Exception as e:
            logger.error(f"Error processing {column_to_update} reaction: {e}")
            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                await log_channel.send(embed=discord.Embed(title="❌ Log Error", description=str(e), color=discord.Color.red()))


    async def _log_activity_reaction_impl(self, payload: discord.RawReactionActionEvent, member: discord.Member):
        """Handle activity logs (time guarded and activity)"""
        if payload.channel_id != self.activity_log_channel_id or str(payload.emoji) != "✅":
            return

        guild = member.guild
        db_logger_role = guild.get_role(Config.DB_LOGGER_ROLE_ID)
        if not db_logger_role or db_logger_role not in member.roles:
            return

        try:
            await asyncio.sleep(0.5)

            channel = await guild.get_channel(payload.channel_id)
            if not channel:
                return

            message = await channel.fetch_message(payload.message_id)

            user_mention = re.search(r'<@!?(\d+)>', message.content)
            user_id = int(user_mention.group(1)) if user_mention else message.author.id
            user_member = guild.get_member(user_id) or await guild.fetch_member(user_id)

            updates = {}
            time_match = re.search(r'Time:\s*(\d+)', message.content)
            if time_match:
                if "Guarded:" in message.content:
                    updates["time_guarded"] = int(time_match.group(1))
                else:
                    updates["activity"] = int(time_match.group(1))

            if updates:
                #Update LR record first
                await self._update_lr_record(user_member, updates)
                
                # 🟩 XP logic: 1 XP per 30 mins (activity or guarded)
                total_minutes = 0
                if "activity" in updates:
                    total_minutes += updates["activity"]
                if "time_guarded" in updates:
                    total_minutes += updates["time_guarded"]
            
                xp_to_award = total_minutes // 30
                if xp_to_award > 0:
                    success, new_xp = await self.bot.db.add_xp(
                        str(user_member.id),
                        user_member.display_name,
                        xp_to_award
                    )
                    if success:
                        logger.info(f"⭐ Gave {xp_to_award} XP to {user_member.display_name} ({user_member.id}) for {total_minutes} mins activity")

                log_channel = guild.get_channel(self.log_channel_id)
                if log_channel:
                    embed = discord.Embed(title="⏱ Activity Logged", color=discord.Color.green())
                    embed.add_field(name="Member", value=user_member.mention)
                    if "activity" in updates:
                        embed.add_field(name="Activity Time", value=f"{updates['activity']} mins")
                    if "time_guarded" in updates:
                        embed.add_field(name="Guarded Time", value=f"{updates['time_guarded']} mins")
                    if xp_to_award > 0:
                        embed.add_field(name="XP Awarded", value=f"+{xp_to_award} XP")
                    embed.add_field(name="Logged By", value=member.mention)
                    embed.add_field(name="Message", value=f"[Jump to Log]({message.jump_url})")
                    await log_channel.send(content=member.mention, embed=embed)

        except Exception as e:
            logger.error(f"Error processing activity reaction: {e}")
            log_channel = guild.get_channel(self.log_channel_id)
            if log_channel:
                await log_channel.send(embed=discord.Embed(title="❌ Activity Log Error", description=str(e), color=discord.Color.red()))

                
    async def _update_hr_record(self, member: discord.Member, updates: dict):
        u_str = str(member.id)
    
        def _work():
            sup = self.bot.db.supabase
            row = sup.table('HRs').select('*').eq('user_id', u_str).execute()
            
            FLOAT_COLUMNS = {"courses"} 

            if getattr(row, "data", None):
                existing = row.data[0]
                incremented = {}
            
                for key, value in updates.items():
                    if isinstance(value, (int, float)):
                        current = existing.get(key, 0) or 0
            
                        # If this column is meant to store floats
                        if key in FLOAT_COLUMNS:
                            incremented[key] = float(current) + float(value)
                        else:
                            incremented[key] = int(current) + int(value)
                    else:
                        incremented[key] = value
            
                return sup.table("HRs").update({
                    **incremented,
                    "username": clean_nickname(member.display_name)
                }).eq("user_id", u_str).execute()
            
            else:
                payload = {
                    "user_id": u_str,
                    "username": clean_nickname(member.display_name),
                    **updates
                }
                return sup.table("HRs").insert(payload).execute()
    
        try:
            await self.bot.db.run_query(_work)
        except Exception:
            logger.exception("ReactionLogger._update_hr_record failed")
    
    
    async def _update_lr_record(self, member: discord.Member, updates: dict):
        u_str = str(member.id)
    
        def _work():
            sup = self.bot.db.supabase
            row = sup.table('LRs').select('*').eq('user_id', u_str).execute()
    
            if getattr(row, "data", None):
                existing = row.data[0]
                # Increment numerical fields
                incremented = {}
                for key, value in updates.items():
                    if isinstance(value, int):
                        incremented[key] = existing.get(key, 0) + value
                    else:
                        incremented[key] = value
                return sup.table('LRs').update({
                    **incremented,
                    "username": clean_nickname(member.display_name)
                }).eq('user_id', u_str).execute()
            else:
                payload = {
                    'user_id': u_str,
                    "username": clean_nickname(member.display_name),
                    **updates
                }
                return sup.table('LRs').insert(payload).execute()
    
        try:
            await self.bot.db.run_query(_work)
        except Exception:
            logger.exception("ReactionLogger._update_lr_record failed")
            
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
        try:
            await self.log_reaction(payload)
        except Exception as e:
            logger.error(f"ReactionLogger.on_raw_reaction_add failed: {e}", exc_info=True)

async def setup(bot):
    await bot.add_cog(ReactionLoggerCog(bot))
