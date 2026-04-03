import re
import logging 
import discord
from config import Config
from datetime import datetime, timezone
from utils.decorators import has_allowed_role, min_rank_required 
from discord.ext import commands
from discord import app_commands
from utils.views import ConfirmView

logger = logging.getLogger(__name__)

class AdminCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # Edit database command
    @app_commands.command(name="edit-db", description="Edit a specific user's record in the HR or LR table.")
    @has_allowed_role()
    async def edit_db(self, interaction: discord.Interaction, user: discord.User, column: str, value: str):
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        if not guild:
            await interaction.followup.send("❌ This command can only be used in a server.")
            return
            
        member = guild.get_member(user.id)
        if not member:
            try:
                member = await guild.fetch_member(user.id)
            except discord.NotFound:
                await interaction.followup.send(f"❌ {user.mention} not found in this server.")
                return
                
        hr_role = guild.get_role(Config.HR_ROLE_ID)
        is_hr = hr_role and hr_role in member.roles
        table = "HRs" if is_hr else "LRs"
        user_id = str(user.id)

        # Define available columns based on role
        hr_columns = ["tryouts", "events", "phases", "courses", "inspections", "joint_events"]
        lr_columns = ["activity", "time_guarded", "events_attended"]
        
        # Validate column based on role
        available_columns = hr_columns if is_hr else lr_columns
        if column not in available_columns:
            await interaction.followup.send(
                f"❌ Invalid column `{column}` for {table} table. "
                f"Available columns for {'HRs' if is_hr else 'LRs'}: {', '.join(available_columns)}"
            )
            return

        def _work():
            sup = self.bot.db.supabase
            res = sup.table(table).select("*").eq("user_id", user_id).execute()
            return res

        try:
            res = await self.bot.db.run_query(_work)
            if not res.data:
                await interaction.followup.send(f"❌ No record found for {user.mention} in `{table}` table.")
                return
            if len(res.data) > 1:
                await interaction.followup.send(f"❌ Multiple records found for {user.mention} in `{table}` table.")
                return

            old_value = res.data[0].get(column, "N/A")
            try:
                value_converted = int(value)
            except ValueError:
                try:
                    value_converted = float(value)
                except ValueError:
                    value_converted = value

            def _update_work():
                return self.bot.db.supabase.table(table).update({column: value_converted}).eq("user_id", user_id).execute()

            await self.bot.db.run_query(_update_work)
            await interaction.followup.send(
                f"✅ Updated `{column}` for {user.mention} from `{old_value}` to `{value_converted}`."
            )
        except Exception as e:
            logger.exception("edit_db failed: %s", e)
            await interaction.followup.send(f"❌ Failed to update data: `{e}`")


    # Add autocomplete for the column parameter
    @edit_db.autocomplete('column')
    async def edit_db_column_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str
    ):
        user_option = next(
            (opt for opt in interaction.data.get('options', []) if opt['name'] == 'user'),
            None
        )
        if not user_option or not interaction.guild:
            return []

        user_id = int(user_option['value'])
        guild = interaction.guild

        member = guild.get_member(user_id)
        if not member:
            try:
                member = await guild.fetch_member(user_id)
            except discord.NotFound:
                return []

        hr_role = guild.get_role(Config.HR_ROLE_ID)
        is_hr = hr_role and hr_role in member.roles

        # Display name -> DB column
        hr_columns = {
            "Tryouts": "tryouts",
            "Events": "events",
            "Phases": "phases",
            "Logistics": "courses",       
            "Inspections": "inspections",
            "Joint Events": "joint_events",
        }

        lr_columns = {
            "Activity": "activity",
            "Time Guarded": "time_guarded",
            "Events Attended": "events_attended",
        }

        available = hr_columns if is_hr else lr_columns

        return [
            discord.app_commands.Choice(name=display, value=db_value)
            for display, db_value in available.items()
            if current.lower() in display.lower()
        ][:25]


    
    # Reset Database Command
    @app_commands.command(name="reset-db", description="Reset the LR and HR tables.")
    @min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
    async def reset_db(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        guild = interaction.guild
        if not guild:
            await interaction.followup.send(
                "❌ This command can only be used in a server.",
                ephemeral=True
            )
            return

        view = ConfirmView(author=interaction.user)

        await interaction.followup.send(
            "⚠️ **Are you sure you want to reset the database?**\n"
            "This will reset **ALL LR and HR stats**.\n\n"
            "Click **Confirm** to proceed or **Cancel** to abort.",
            view=view,
            ephemeral=True
        )

        # Wait for user input (or timeout)
        await view.wait()

        if view.value is not True:
            await interaction.followup.send(
                "❎ Database reset cancelled.",
                ephemeral=True
            )
            return

        def _reset_work():
            sup = self.bot.db.supabase
            sup.table('HRs').update({
                'tryouts': 0,
                'events': 0,
                'phases': 0,
                'courses': 0,
                'inspections': 0,
                'joint_events': 0
            }).neq('user_id', 0).execute()

            sup.table('LRs').update({
                'activity': 0,
                'time_guarded': 0,
                'events_attended': 0
            }).neq('user_id', 0).execute()

            return True

        try:
            await self.bot.db.run_query(_reset_work)
            await interaction.followup.send(
                "✅ **Database reset successfully!**",
                ephemeral=True
            )
        except Exception as e:
            logger.exception("reset_db failed: %s", e)
            await interaction.followup.send(
                f"❌ Error resetting database:\n```{e}```",
                ephemeral=True
            )




    # Manual Fallback Log Command (covers all ReactionLogger cases)
    @app_commands.command(name="force-log", description="Force log a message by link (reaction logger fallback)")
    @has_allowed_role()
    async def force_log(self, interaction: discord.Interaction, message_link: str):
        """Force log the exact message from a link - acts as fallback for reaction logger"""
        await interaction.response.defer(ephemeral=True)

        try:
            # Parse the link
            match = re.match(r"https://discord\.com/channels/(\d+)/(\d+)/(\d+)", message_link)
            if not match:
                await interaction.followup.send("❌ Invalid message link.", ephemeral=True)
                return

            guild_id, channel_id, message_id = map(int, match.groups())
            
            if guild_id != interaction.guild.id:
                await interaction.followup.send("❌ Message must be from this server.", ephemeral=True)
                return

            channel = interaction.guild.get_channel(channel_id)
            if not channel:
                await interaction.followup.send("❌ Channel not found.", ephemeral=True)
                return

            # Fetch the specific message
            message = await channel.fetch_message(message_id)
            
            logger.info(f"🔧 Force-log: Processing message {message_id} in #{channel.name} by {interaction.user.display_name}")

            processed = False
            results = []

            # 1. EVENT LOGS (event channels)
            if channel_id in self.bot.reaction_logger.event_channel_ids:
                try:
                    # Extract host
                    host_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
                    host_id = int(host_mention.group(1)) if host_mention else message.author.id
                    host_member = interaction.guild.get_member(host_id) or await interaction.guild.fetch_member(host_id)

                    # Determine event type
                    hr_update_field = "events"
                    if channel_id == Config.W_EVENT_LOG_CHANNEL_ID:
                        if re.search(r'\bjoint\b', message.content, re.IGNORECASE):
                            hr_update_field = "joint_events"
                        elif re.search(r'\b(inspection|pi)\b', message.content, re.IGNORECASE):
                            hr_update_field = "inspections"

                    # Update HR table for host
                    await self.bot.reaction_logger._update_hr_record(host_member, {hr_update_field: 1})
                    
                    # Process attendees
                    attendees_section = re.search(r'(?:Attendees:|Passed:)\s*((?:<@!?\d+>\s*)+)', message.content, re.IGNORECASE)
                    success_count = 0
                    hr_attendees = []
                    
                    if attendees_section:
                        attendee_mentions = re.findall(r'<@!?(\d+)>', attendees_section.group(1))
                        hr_role = interaction.guild.get_role(Config.HR_ROLE_ID)
                        
                        for attendee_id in attendee_mentions:
                            attendee_member = interaction.guild.get_member(int(attendee_id))
                            if not attendee_member:
                                continue
                            if hr_role and hr_role in attendee_member.roles:
                                hr_attendees.append(attendee_id)
                                continue
                            await self.bot.reaction_logger._update_lr_record(attendee_member, {"events_attended": 1})
                            success_count += 1

                    
                    results.append(f"✅ Event: {success_count} attendees + host logged")
                    processed = True
                    logger.info(f"✅ Event force-logged: {success_count} attendees")
                    
                except Exception as e:
                    logger.error(f"Event logging failed: {e}")
                    results.append(f"❌ Event: {str(e)[:50]}")

            # 2. TRAINING LOGS (phases, tryouts, courses)
            elif channel_id in [
                Config.PHASE_LOG_CHANNEL_ID,
                Config.TRYOUT_LOG_CHANNEL_ID, 
                Config.COURSE_LOG_CHANNEL_ID,
            ]:
                try:
                    user_mention = re.search(r'host:\s*<@!?(\d+)>', message.content, re.IGNORECASE)
                    user_id = int(user_mention.group(1)) if user_mention else message.author.id
                    user_member = interaction.guild.get_member(user_id) or await interaction.guild.fetch_member(user_id)

                    column_to_update = {
                        Config.PHASE_LOG_CHANNEL_ID: "phases",
                        Config.TRYOUT_LOG_CHANNEL_ID: "tryouts",
                        Config.COURSE_LOG_CHANNEL_ID: "courses",
                    }.get(channel_id)

                    if column_to_update:
                        await self.bot.reaction_logger._update_hr_record(user_member, {column_to_update: 1})
                        
                        results.append(f"✅ {column_to_update.title()} logged")
                        
                        processed = True
                        
                        logger.info(f"✅ {column_to_update} force-logged")
                        
                except Exception as e:
                    logger.error(f"Training logging failed: {e}")
                    results.append(f"❌ Training: {str(e)[:50]}")

        
            # 3. ACTIVITY LOGS
            elif channel_id == Config.ACTIVITY_LOG_CHANNEL_ID:
                try:
                    user_mention = re.search(r'<@!?(\d+)>', message.content)
                    user_id = int(user_mention.group(1)) if user_mention else message.author.id
                    user_member = interaction.guild.get_member(user_id) or await interaction.guild.fetch_member(user_id)
            
                    updates = {}
                    time_match = re.search(r'Time:\s*(\d+)', message.content)
                    if time_match:
                        minutes = int(time_match.group(1))
                        if "Guarded:" in message.content:
                            updates["time_guarded"] = minutes
                        else:
                            updates["activity"] = minutes
            
                    if updates:
                        # Update LR record
                        await self.bot.reaction_logger._update_lr_record(user_member, updates)
            
                        # 🟩 NEW: Award XP (1 XP per 30 mins total)
                        total_minutes = updates.get("activity", 0) + updates.get("time_guarded", 0)
                        xp_to_award = total_minutes // 30
                        xp_text = ""
                        if xp_to_award > 0:
                            success, new_xp = await self.bot.db.add_xp(
                                str(user_member.id),
                                user_member.display_name,
                                xp_to_award
                            )
                            if success:
                                xp_text = f" (+{xp_to_award} XP)"
                                logger.info(f"⭐ Gave {xp_to_award} XP to {user_member.display_name} ({user_member.id}) for {total_minutes} mins (force-log)")
            
            
                        # Finish up and include XP info in results
                        field_name = "time_guarded" if "time_guarded" in updates else "activity"
                        results.append(f"✅ Activity: {updates[field_name]} mins logged{xp_text}")
                        processed = True
                        logger.info(f"✅ Activity force-logged: {updates[field_name]} mins{xp_text}")
            
                except Exception as e:
                    logger.error(f"Activity logging failed: {e}")
                    results.append(f"❌ Activity: {str(e)[:50]}")


            elif channel_id in self.bot.reaction_logger.monitor_channel_ids:
                try:
                    processed = True
                    logger.info("✅ Activity force-logged")
                    
                except Exception as e:
                    logger.error(f"Force activity logging failed: {e}")
                    results.append(f"❌ DB_LOGGER: {str(e)[:50]}")

            # Send results
            if processed:
                result_text = "\n".join(results)
                await interaction.followup.send(
                    f"✅ **Force-log completed**\n"
                    f"**Message:** [Jump to message]({message.jump_url})\n"
                    f"**Channel:** #{channel.name}\n"
                    f"**Results:**\n{result_text}",
                    ephemeral=True
                )
                
                # Also log to the main log channel
                log_channel = interaction.guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
                if log_channel:
                    embed = discord.Embed(
                        title="🔄 Manual Log Entry (Force-log)",
                        color=discord.Color.orange(),
                        timestamp=datetime.now(timezone.utc)
                    )
                    embed.add_field(name="Logged by", value=interaction.user.mention, inline=True)
                    embed.add_field(name="Channel", value=channel.mention, inline=True)
                    embed.add_field(name="Message Type", value=channel.name, inline=True)
                    embed.add_field(name="Results", value=result_text, inline=False)
                    embed.add_field(name="Message", value=f"[Jump to message]({message.jump_url})", inline=False)
                    
                    await log_channel.send(embed=embed)
                    
            else:
                await interaction.followup.send(
                    f"❌ No applicable log types found for this message.\n"
                    f"**Channel:** #{channel.name}\n"
                    f"**Supported channels:** Events, Training, Activity logs, or DB_LOGGER-monitored channels",
                    ephemeral=True
                )

        except Exception as e:
            logger.error(f"force-log failed: {e}", exc_info=True)
            await interaction.followup.send("❌ Failed to force-log message.", ephemeral=True)

    #Save Roles Command
    @app_commands.command(name="save-roles", description="Save a user's tracked roles to the database.")
    @has_allowed_role()
    async def save_roles(self, interaction: discord.Interaction, member: discord.Member):
        # Defer immediately
        await interaction.response.defer(ephemeral=True)

        tracked_ids = Config.TRACKED_ROLE_IDS
        matched_roles = [r for r in member.roles if r.id in tracked_ids]
        role_ids = [r.id for r in matched_roles]

        try:
            success = await self.bot.db.save_user_roles(
                user_id=str(member.id),
                username=member.display_name,
                role_ids=role_ids
            )
        except Exception as e:
            embed = discord.Embed(
                title="❌ Error Saving Roles",
                description=f"Failed to save roles for {member.mention}.\n{e}",
                color=discord.Color.red()
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
            return

        # Build status embed
        embed = discord.Embed(
            title="✅ Roles Saved" if success else "❌ Error Saving Roles",
            description=(
                f"Saved **{len(role_ids)}** tracked roles for {member.mention}.\n"
                f"**Roles:** {', '.join([r.name for r in matched_roles]) or 'None'}"
            ) if success else f"Failed to save roles for {member.mention}.",
            color=discord.Color.green() if success else discord.Color.red()
        )

        # Log to default channel
        log_channel = interaction.guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
        if log_channel:
            await log_channel.send(embed=embed)

        # Send ephemeral follow-up to the user
        await interaction.followup.send(embed=embed, ephemeral=True)

    #Restore Roles Command
    @app_commands.command(name="restore-roles", description="Restore saved roles for a user.")
    async def restore_roles(self, interaction: discord.Interaction, member: discord.Member):
        # Defer the interaction immediately with error handling
        try:
            await interaction.response.defer(ephemeral=True)
            deferred = True
        except discord.NotFound:
            # If deferral fails, the interaction might have timed out
            deferred = False
        except Exception as e:
            print(f"Error deferring interaction: {e}")
            deferred = False

        # Fetch saved roles from Supabase
        try:
            saved_roles = await self.bot.db.get_user_roles(str(member.id))
        except Exception as e:
            error_msg = f"❌ Failed to fetch saved roles: {e}"
            if deferred:
                await interaction.followup.send(error_msg, ephemeral=True)
            else:
                try:
                    await interaction.response.send_message(error_msg, ephemeral=True)
                except discord.InteractionResponded:
                    await interaction.followup.send(error_msg, ephemeral=True)
            return

        if not saved_roles:
            msg = f"⚠️ No saved roles found for {member.mention}."
            if deferred:
                await interaction.followup.send(msg, ephemeral=True)
            else:
                try:
                    await interaction.response.send_message(msg, ephemeral=True)
                except discord.InteractionResponded:
                    await interaction.followup.send(msg, ephemeral=True)
            return

        # Format for Dyno command: ?role <user_id> <role id 1>, <role id 2>, <role id 3>
        roles_string = ", ".join([str(role_id) for role_id in saved_roles])
        dyno_command = f"?role {member.id} {roles_string}"

        # Create embed with Dyno command
        embed = discord.Embed(
            title=f"✅ Roles restored for {member.display_name}",
            description=f"Use the following command in chat to reassign roles:\n`{dyno_command}`",
            color=discord.Color.green()
        )
        
        if deferred:
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            try:
                await interaction.response.send_message(embed=embed, ephemeral=True)
            except discord.InteractionResponded:
                await interaction.followup.send(embed=embed, ephemeral=True)

async def setup(bot):
    await bot.add_cog(AdminCog(bot))
