import asyncio
import logging 
import discord
from config import Config
from typing import Optional, Literal
from utils.decorators import  has_modular_permission 
from discord.ext import commands
from discord import app_commands
from utils.helpers import clean_nickname, MockPayload
from utils.views import ConfirmView

logger = logging.getLogger(__name__)



class ModerationCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    # Edit database command
    @commands.hybrid_command(
            name="edit-user",
            aliases=["eu"],
            usage="<member> <column: tryouts/events/phases/courses(logistics)/inspections/joint_events/activity/time_guarded/events_attended> <value>",
            description="Edit a specific user's record in the HR or LR table."
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("moderation")
    async def edit_db(
            self, 
            ctx: commands.Context, 
            user: discord.User, 
            column: str,
            value: str
    ):
        if ctx.interaction:
            await ctx.interaction.response.defer(ephemeral=False)

        guild = ctx.guild
        user_id = str(user.id)

        member = guild.get_member(user.id) or await guild.fetch_member(user.id)
        if not member:
            return await ctx.send(f"```❌ User not found in this server.```")

        hr_role = guild.get_role(Config.HR_ROLE_ID)
        is_hr = hr_role and hr_role in member.roles

        hr_columns = ["tryouts", "events", "phases", "courses", "inspections", "joint_events"]
        lr_columns = ["activity", "time_guarded", "events_attended"]
        
        table_name = "HRs" if is_hr else "LRs"
        db_table = self.bot.db.hrs_table if is_hr else self.bot.db.lrs_table
        available_columns = hr_columns if is_hr else lr_columns

        if column.lower() not in available_columns:
            return await ctx.send(
                f"```❌ Invalid column '{column}' for {table_name}.\n"
                f"Available: {', '.join(available_columns)}```"
            )

        try:
            if is_hr:
                res = await self.bot.db.get_hr_info(user_id)
            else:
                res = await self.bot.db.get_lr_info(user_id) 

            if not res:
                return await ctx.send(f"```❌ No record found for {user.display_name} in {table_name} table.```")

            old_value = res.get(column, 0)

            try:
                value_converted = float(value) if '.' in value else int(value)
            except ValueError:
                value_converted = value

            update_success = await self.bot.db.increment_points_handler(
                column, db_table, member, value, replace=True
            )

            if not update_success:
                raise Exception("Database update returned False")

            await ctx.send(
                f"```✅ Updated {column} for {clean_nickname(user.display_name)}\n"
                f"Table: {table_name}\n"
                f"Change: {old_value} ➔ {value_converted}```"
            )

        except Exception as e:
            logger.exception(f"edit_db failed: {e}")
            await ctx.send(f"```❌ Failed to update data. Internal error logged.```")


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


       
    # Remove User Command
    @commands.hybrid_command(
            name="remove-user",
            usage="<member>",
            aliases=["remove"],
            description="Remove a user from the database"
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("moderation")
    async def remove_user(self, ctx: commands.Context, user: discord.User):
        
        if ctx.interaction:
            await ctx.interaction.response.defer()
        user_id = str(user.id)
        
        try:
            member = await self.bot.db.get_user(user_id)
            if member:
                username = member['username']
                await self.bot.db.discharge_user(user_id, username, ctx.guild)
                return await ctx.send(f"```✅ Successfully removed {username} from the database.```")
            else:
                return await ctx.send("```❌ That user is not in the database.```")
        except Exception as e:
            logger.error(f"remove_user failed: %s", e)
            await ctx.send("```❌ Failed to remove user from the database.```")
       
        return

    # Manual Fallback Log Command (covers all ReactionLogger cases)
    @commands.hybrid_command(
            name="force-log", 
            aliases=["log", "fl"],
            usage="<link> <reaction>",
            description="Force log a message by link"
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("moderation")
    async def force_log(self, ctx: commands.Context, message_link: str, reaction: str):

        if ctx.interaction:
            await ctx.interaction.response.defer(ephemeral=True)
        
        member = ctx.guild.get_member(ctx.author.id)

        # 1.1 Validation Check - Parsing Link
        try:
            parts = message_link.split('/')
            guild_id = int(parts[4])
            channel_id = int(parts[5])
            message_id = int(parts[6])
        except (IndexError, ValueError):
            return await ctx.send(content="```❌ Invalid message link format.```")
        
        if guild_id != ctx.guild.id:
            return await ctx.send(content="```❌ Message must be from this server.```")
        try:
            channel = ctx.guild.get_channel(channel_id)
            if not channel:
                return await ctx.send(content="```❌ Channel not found.```")
            try:
                message = await asyncio.wait_for(
                    channel.fetch_message(message_id),
                    timeout=10.0
                )
            except discord.NotFound:
                return await ctx.send(content="```❌ Message not found.```")
            except discord.Forbidden:
                return await ctx.send(content="```❌ No permissions to read that channel.```")
        except asyncio.TimeoutError:
            await ctx.send(content="```⌛ Timed out fetching message.```")
            return

        #1.2 Validation Check - Reaction
        if reaction not in Config.TRACKED_REACTIONS:
           return await ctx.send("```❌ That reaction is not being tracked.```") 

        emoji = discord.PartialEmoji.from_str(reaction)
        

        # 2. Get the Cog instance
        rl = self.bot.get_cog("ReactionLoggerCog")
        if not rl:
            logger.error("Could not find ReactionLogger Cog.")
            raise Exception

        #3. Check if log has been processed already
        processed = await rl.is_reaction_processed(message_id, member)
        
        if processed:
            view = ConfirmView(author=ctx.author)

            await ctx.send(
                    f"```This log has already been processed recently.\nAre you sure you want to log this?```", 
                    view=view)

            await view.wait()

            if view.value is not True:
                return await ctx.send(
                    "```❎ Force log cancelled.```",
                )

        # 4. Build the mock payload
        payload = MockPayload(
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            user_id=ctx.author.id,
            emoji=emoji,
            member=ctx.author
        )

        # 5. Route to the matching handler(s)
        try:
            matched = False
            for handler_entry in rl.REACTION_HANDLERS:
                if handler_entry.channels and channel_id in handler_entry.channels:
                    matched = True
                    method = getattr(rl, handler_entry.handler)
                    await method(payload, ctx.guild, member)

            if not matched:
                return await ctx.send(
                    "```❌ The channel and reaction provided are not a tracked combination.```", ephemeral=True
                )

            # 7. Mark as processed only after all handlers ran successfully
            if not processed:
                await rl.mark_reaction_processed(message_id, ctx.author.id)
          
            try:
                channel = ctx.guild.get_channel(channel_id)
                message = await channel.fetch_message(message_id)
                await message.add_reaction(emoji)
            except Exception as e:
                logger.error("Failed to react to message: %s", e)

            await ctx.send(
                "```✅ Forcefully logged: Verified for inputted link and reaction.```",
                ephemeral=True
            )

        except Exception as e:
            logger.error("/force-log failed: %s", e, exc_info=True)
            await ctx.send(
                f"```❌ Failed to log. ```", ephemeral=True
            )


    #Save Roles Command
    @commands.hybrid_command(
            name="save-roles",
            aliases=["sr"],
            usage="<member>",
            description="Save a user's tracked roles to the database."
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("moderation")
    async def save_roles(self, ctx: commands.Context, member: discord.Member):
        if ctx.interaction:
            await ctx.interaction.response.defer()

        tracked_ids = Config.TRACKED_ROLE_IDS
        matched_roles = [r for r in member.roles if r.id in tracked_ids]
        if not matched_roles:
            return await ctx.send(f"```❎ {clean_nickname(member.display_name)} has no tracked roles to save.```")
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
            await ctx.send(embed=embed, ephemeral=True)
            return

        embed = discord.Embed(
            title="✅ Roles Saved" if success else "❌ Error Saving Roles",
            description=(
                f"Saved **{len(role_ids)}** tracked roles for {member.mention}.\n"
                f"**Roles:** {', '.join([r.name for r in matched_roles]) or 'None'}"
            ) if success else f"Failed to save roles for {member.mention}.",
            color=discord.Color.green() if success else discord.Color.red()
        )

        # Log to default channel
        log_channel = ctx.guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
        if log_channel:
            await log_channel.send(embed=embed)

        # Send ephemeral follow-up to the user
        await ctx.send(embed=embed, ephemeral=True)

    #Restore Roles Command
    @commands.hybrid_command(
            name="restore-roles", 
            aliases=["rr"],
            usage="<member>",
            description="Restore saved roles for a user."
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("moderation")
    async def restore_roles(self, ctx: commands.Context, member: discord.Member):
        
        if ctx.interaction:
            await ctx.interaction.response.defer()

        try:
            saved_roles = await self.bot.db.get_user_roles(str(member.id))
        except Exception:
            error_msg = f"```❌ Failed to fetch saved roles.```"
            await ctx.send(error_msg, ephemeral=True)
            return

        if not saved_roles:
            msg = f"```⚠️ No saved roles found for {clean_nickname(member.display_name)}.```"
            await ctx.send(msg, ephemeral=True)
            return

        roles_string = ", ".join([str(role_id) for role_id in saved_roles])
        dyno_command = f"?role {member.id} {roles_string}"

        # Create embed with Dyno command
        embed = discord.Embed(
            description=f"Use the following command in chat to reassign roles:\n`{dyno_command}`",
            color=discord.Color.green()
        ).set_author(name=f"Roles restored for {clean_nickname(member.display_name)}", icon_url= Config.CHECK_URL)        
        await ctx.send(embed=embed)

    #Restore User Command
    @commands.hybrid_command(
            name="restore-user", 
            aliases=["restore"],
            usage="<member>",
            description="Restore user's data."
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("moderation")
    async def restore_user(self, ctx: commands.Context, member: discord.Member):
        
        if ctx.interaction:
            await ctx.interaction.response.defer()
       
        rmp_role = ctx.guild.get_role(Config.RMP_ROLE_ID)
        if rmp_role and rmp_role not in member.roles:
            return await ctx.send("```❌ Cannot restore data if they do not have the RMP role.```")

        try:
            saved_user = await self.bot.db.get_stored_user(str(member.id))
        except Exception:
            error_msg = f"```❌ Failed to fetch saved data.```"
            return await ctx.send(error_msg, ephemeral=True)

        if not saved_user:
            msg = f"```⚠️ No saved data found in backup for {clean_nickname(member.display_name)}.```"
            await ctx.send(msg, ephemeral=True)
            return
        
        username = saved_user['username']
        guild = ctx.guild
        roblox_id = saved_user['roblox_id']
        xp = saved_user['xp']
       
        try:
            await self.bot.db.create_or_update_user_in_db(member.id, username, guild, roblox_id, xp)
            embed = discord.Embed(
                    description=f"User data restored for {clean_nickname(member.display_name)}. Data Restored: discord ID, username and xp. Be aware that if the user has changed username it will revert back to their old username until an action is performed on them.",
                color=discord.Color.green()
            )
            name = "Sucessfully restored data"
            icon_url = Config.CHECK_URL
            embed.set_author(name=name, icon_url=icon_url)
            
            log_channel = guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
            if log_channel:
                try:
                    await log_channel.send(embed=embed)
                except Exception as log_error:
                    logger.error("Failed to send log embed: %s", log_error)
            
            await ctx.send(f"```✅ Sucessfully data restored for {clean_nickname(member.display_name)}.```")
        except Exception as e:
            logger.error("Failed to restore stored user for %s (%s): %s", username, member.id, e)
            await ctx.send("```❌ Failed to restore data.```")
            return

        try:
            await self.bot.db.delete_stored_user(member.id) 
        except Exception as e:
            logger.error("Failed to delete %s (%s) in user_store: %s", username, member.id, e)
            return

    #Manage Case Logs
    @commands.hybrid_command(
            name="manage-case-logs", 
            aliases=["mcl"],
            usage="<rbx username/id> <case-link: (optional)>",
            description="Manage a user's case-log by viewing their record or adding/deleting a record."
    )
    @app_commands.describe(roblox_user="The user you want to manage (id/username)", case_link="The link to the record you want to add/remove (leave blank just to view their criminal record)")
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("moderation")
    async def remove_case_log(self, ctx: commands.Context, roblox_user: str, case_link: Optional[str] = None):

        if ctx.interaction:
            await ctx.interaction.response.defer(ephemeral=False)

        ctx.author = ctx.author
        cleaned_name = clean_nickname(ctx.author.display_name)
        
        if roblox_user.isnumeric():
            user_record = await self.bot.db.get_criminal_record(user_id=roblox_user)
        else:
            user_record = await self.bot.db.get_criminal_record(username=roblox_user)
        
        
        suspect_username = user_record[0]["username"] if user_record else ""

        if not case_link:
            if not user_record:
                await ctx.send("```❌ No recent criminal record found for that user.```")
                return

            limited_records = user_record[:30]

            records = "\n".join(
                f"[Record {i}]({record['record']})"
                for i, record in enumerate(limited_records, start=1)
            )

            if len(user_record) > 30:
                records += f"\n\n*...and {len(user_record) - 30} more records.*"

            record_embed = discord.Embed(
                description=f"Logged cases of {suspect_username} in the past 2 days:\n{records}",
                color=discord.Color.red()
            ).set_author(name=f"{suspect_username}'s Criminal Record", icon_url=Config.CUFFS_ICON)

            record_embed.set_footer(text="Use this command with the 'case_link' argument to add/remove a case.")

            return await ctx.send(embed=record_embed)

        try:
            parts = case_link.split('/')
            guild_id = int(parts[4])
            channel_id = int(parts[5])
            message_id = int(parts[6])
        except (IndexError, ValueError):
            return await ctx.send("```❌ Invalid message link format.```")

        if guild_id != ctx.guild.id:
            return await ctx.send("```❌ Message must be from this server.```")

        try:
            channel = ctx.guild.get_channel(channel_id)
            if not channel or channel_id != Config.CASE_LOGS_CHANNEL_ID:
                return await ctx.send("```❌ Channel not valid or not case-logs.```")

            try:
                message = await asyncio.wait_for(
                    channel.fetch_message(message_id),
                    timeout=10.0
                )
            except discord.NotFound:
                return await ctx.send("```❌ Message not found.```")
            except discord.Forbidden:
                return await ctx.send("```❌ No permissions to read that channel.```")

        except asyncio.TimeoutError:
            return await ctx.send("```⌛ Timed out fetching message.```")

        list_of_ids = [int(record["message_id"]) for record in user_record]
        action = "removed" if message_id in list_of_ids else "added"

        try:
            if action == "removed":
                await self.bot.db.remove_criminal_record(message_id=message_id)
                await ctx.send(f"```✅ Successfully removed the case log on {suspect_username}.```")
            else:
                mlc = self.bot.get_cog("MessageLoggerCog")
                suspect_username = await mlc.case_logs(message, True, roblox_user)
                if suspect_username:
                    await ctx.send(f"```✅ Successfully added the case log on {suspect_username}.```")
                else:
                    return await ctx.send("```❌ An error occurred while trying to add the log to the database. Double check if the provided user aligns with the one in the case-log.```")

            embed = discord.Embed(
                description=f"[Arrest log]({case_link}) on {suspect_username} has been successfully **{action}** by {cleaned_name}.",
                color=discord.Color.green()
            ).set_author(name=f"Case {action.capitalize()}", icon_url=Config.CUFFS_ICON)

            log_channel = ctx.guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
            if log_channel:
                await log_channel.send(embed=embed)

        except Exception as e:
            logger.error("remove_case_log failed: %s", e)
            await ctx.send("```❌ An error occurred while trying to manage the case log.```")

async def setup(bot):
    await bot.add_cog(ModerationCog(bot))
