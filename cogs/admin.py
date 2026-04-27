import asyncio
import logging
import discord
import mimetypes
from discord import app_commands
from discord.ext import commands
from typing import Optional, Literal
from discord.utils import escape_markdown
from datetime import datetime, timedelta, timezone

from config import Config
from utils import embedBuilder
from utils.helpers import clean_nickname, BlacklistData
from utils.views import ConfirmView, ApprovalView, save_pending_blacklist
from utils.decorators import min_rank_required2, has_modular_permission, is_admin_or_dev


logger = logging.getLogger(__name__)

class AdminCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        #self.monitoring_channels_cache = MonitoringChannelsCache(bot)

    #Set Permissions Command
    @commands.hybrid_command(
            name="set-permissions",
            aliases=["sp"], 
            usage="<category: general/xp/mod/admin> <role(optional)>",
            description="View or toggle role permissions for command categories"
    )
    @app_commands.checks.cooldown(1, 5)
    @app_commands.describe(
        category="The command group to manage",
        role="The role to add or remove (leave blank to just view current roles)"
    )
    @is_admin_or_dev()
    async def set_permissions(
        self, 
        ctx: commands.Context, 
        category: Literal["General", "XP Rewards", "Moderation", "Administrative", "general", "xp", "mod", "admin"],
        role: Optional[discord.Role] = None
    ):
        category_low = category.lower() 
        group_mapping = {
            "general": "general",
            "xp rewards": "xp_rewards",
            "xp": "xp_rewards",
            "moderation": "moderation",
            "mod": "moderation",
            "administrative": "administrative",
            "admin": "administrative"
        }

        db_type = group_mapping.get(category_low, "general")
        
        allowed_ids = await self.bot.permissions.get(db_type)
        
        if role is None:
            role_mentions = "\n".join(f"<@&{rid}>" for rid in allowed_ids)
            list_str = role_mentions if role_mentions else "No roles assigned (Admin only)."
            
            embed = discord.Embed(
                description=f"The following roles have access to these commands:\n{list_str}",
                color=discord.Color.blue()
                ).set_author(name=f"Permissions: {category}", icon_url=Config.SCROLL_ICON)
            embed.set_footer(text="Tip: Use this command with the 'role' argument to add/remove access.")

            return await ctx.send(embed=embed, ephemeral=True)
        
        if role.id in allowed_ids:
            allowed_ids.remove(role.id)
            action = "Removed"
        else:
            allowed_ids.append(role.id)
            action = "Added"

        try:
            allowed_ids = list(set(allowed_ids))
            updated = await self.bot.permissions.update(db_type, allowed_ids)
            
            if updated:
                await ctx.send(
                    f"```✅ {action} {role.name} role {'from' if action == 'Removed' else 'to'} the {category} category permissions.```",
                    ephemeral=True
                )
            else:
                raise Exception("Update failed")
        except Exception as e:
            logger.error(f"Error updating permissions: {e}")
            await ctx.send("```❌ Failed to update permissions.```", ephemeral=True)


    # Discharge Command
    @commands.hybrid_command(
            name="discharge",
            usage="<member> <type: h/g/d> <reason> [evidence: attach to message (optional)]",
            description="Notify members of honourable (h)/general (g)/dishonourable (d) discharge and log it"
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("administrative")
    async def discharge(
        self,
        ctx: commands.Context,
        members: str,  # Comma-separated user mentions/IDs
        discharge_type: Literal["Honourable", "General", "Dishonourable", "h", "g", "d"] = "General",
        evidence: Optional[discord.Attachment] = None,
        *, # Puts any left over into last arg
        reason: str = "No reason provided"
    ):
        if ctx.interaction:
            await ctx.interaction.response.defer(ephemeral=True)
        
        type_map = {
                "h": "Honourable",
                "g": "General",
                "d": "Dishonourable",
        }

        charge_type = db_value = type_map.get(discharge_type.lower(), discharge_type.title())

        try:
            reason = escape_markdown(reason) 
            if len(reason) > 1000:
                await ctx.send(
                    "```❌ Reason must be under 1000 characters```",
                    ephemeral=True
                )
                return

            member_ids = []
            for mention in members.split(','):
                mention = mention.strip()
                if mention.startswith('<@') and mention.endswith('>'):
                    member_id = int(mention[2:-1].replace('!', ''))  # Handle nicknames
                else:
                    member_id = int(mention)
                member_ids.append(member_id)

            member_ids = list(set(member_ids))

            if ctx.author.id in member_ids:
                return await ctx.send("```❌ You cannot discharge yourself.```", ephemeral=True)
            
            # Confirmation before action
            view = ConfirmView(author=ctx.author)
            confirmation_msg = (
                        f"Their information will be deleted from the database and could result in permanent loss.\n\n"
                        f"**__Discharge Information__**\n"
                        f"**Type:** {charge_type}\n"
                        f"**Members to be discharged:** {members}\n"
                        f"**Reason:** {reason}\n"
                        f"**Evidence:** {'N/A' if not evidence else ''}\n\n"
                    )

            embed = discord.Embed(description=confirmation_msg, color=discord.Color.orange())
            embed.set_author(
                    name="Are you sure you want to go through this discharge?",
                    icon_url=Config.ALERT_URL
            )
            if evidence:
                embed.set_image(url=evidence.url)

            await ctx.send(embed=embed, view=view, ephemeral=True)
            await view.wait()

            if not view.value:
                await ctx.send("```❎ Discharge cancelled.```", ephemeral=True)
                return False

            processing_msg = await ctx.send(
                "```⚙️ Processing discharge...```",
                ephemeral=True,
            )

            await self.bot.rate_limiter.wait_if_needed(bucket="discharge")

            discharged_members = []
            for member_id in member_ids:
                if member := ctx.guild.get_member(member_id):
                    discharged_members.append(member)

            if not discharged_members:
                await ctx.send("```❌ No valid members found.```", ephemeral=True)
                return
            
            # Embed creation
            color = discord.Color.green() 
            if charge_type == "General":
                color = discord.Color.green() 
            elif charge_type == "Honourable":
                color = discord.Color.gold() 
            else:
                color = discord.Color.red()
                
            embed = discord.Embed(
                title=f"{charge_type} Discharge Notification",
                color=color,
                timestamp=datetime.now(timezone.utc)
                ).set_thumbnail(url=Config.RMP_URL)
            embed.add_field(name="Reason Provided:", value=f"```{reason}```", inline=False)
            
            if charge_type != "Dishonourable":
                embed.description = "**Thank you for your service! Please come back to us when you can.**\n\n"

            if evidence:
                embed.add_field(name="Evidence:", value=f"[Attachment Link]({evidence.url})", inline=False)
                mime, _ = mimetypes.guess_type(evidence.filename)
                if mime and mime.startswith("image/"):
                    embed.set_image(url=evidence.url)

            embed.set_footer(text=f"Discharged by {ctx.author.display_name}")

            success_count = 0
            successful_members = []
            failed_members = []
            
            for member in discharged_members:
                cleaned_nickname = clean_nickname(member.display_name) or member.name
                try:
                    roblox_id = await self.bot.db.get_roblox_id(member.id)

                    member_info = f"{cleaned_nickname} | {member.id} ({member.mention})"
                    if roblox_id:
                        member_info += f" | {roblox_id}"

                    await self.bot.db.discharge_user(str(member.id), cleaned_nickname, ctx.guild)
                    try:
                        await member.send(embed=embed)
                    except discord.Forbidden:
                        if channel := ctx.guild.get_channel(Config.PUBLIC_CHAT_CHANNEL_ID):
                            await channel.send(f"{member.mention}", embed=embed)
                    
                    successful_members.append(member_info)
                    success_count += 1
                except Exception as e:
                    logger.exception(f"Error discharging %s (%s): $s", cleaned_nickname, member.id, e)
                    failed_members.append(member.mention)

            result_embed = discord.Embed(
                title="Discharge Summary",
                color=color
            )

            result_embed.add_field(name="Action:", value=f"{charge_type} Discharge", inline=False)
            result_embed.add_field(
                name="Results:",
                value=f"Successfully notified: {success_count}\nFailed: {len(failed_members)}",
                inline=False
            )

            if failed_members:
                result_embed.add_field(name="Failed Members:", value=", ".join(failed_members), inline=False)

            await processing_msg.edit(
                content=None,
                embed=result_embed
            )

            # Log to D_LOG_CHANNEL_ID
            if d_log := ctx.guild.get_channel(Config.D_LOG_CHANNEL_ID):
                log_embed = embedBuilder.build_discharge_log(
                        charge_type,
                        successful_members,
                        ctx.author,
                        reason=reason,
                        evidence=evidence,
                        color=color
                )

                await d_log.send(embed=log_embed)
                return True
        except Exception as e:
            logger.error(f"Discharge command failed: {e}")
            await ctx.send("```❌ An error occurred while processing the discharge.```", ephemeral=True)
            return False
    
    #Blacklist Command
    @commands.hybrid_command(
            name="blacklist", 
            aliases=["bl"],
            usage="<member> <unit: y/m/d/perm> <amount> <reason> [evidence: attach to message (optional)]",
            description="Blacklist members with specified duration"
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("administrative")
    async def blacklist(
        self,
        ctx: commands.Context,
        members: str,  # Comma-separated user mentions/IDs
        duration_unit: Literal["Permanent", "Years", "Months", "Days", "perm", "y", "m", "d"],
        duration_amount: int = 1,
        evidence: Optional[discord.Attachment] = None,
        *, # Puts any left over into last arg
        reason: str = "No reason provided.",
    ):
        unit_map = {
            "perm": "Permanent", "permanent": "Permanent",
            "y": "Years", "years": "Years",
            "m": "Months", "months": "Months",
            "d": "Days", "days": "Days"
        }

        unit = unit_map.get(duration_unit.lower(), "Permanent")

        view = ConfirmView(author=ctx.author)
        
        member = ctx.guild.get_member(ctx.author.id)
    
        isPMorHigher = await min_rank_required2(Config.PM_ROLE_ID, ctx)
        
                
        reason = escape_markdown(reason) 

        if len(reason) > 1000:
            await ctx.followup.send(
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
        
        member_ids = list(set(member_ids))

        if ctx.author.id in member_ids:
            return await ctx.send("```❌ You cannot blacklist yourself.```", ephemeral=True)

        if isPMorHigher:
            topMsg = "Are you sure you want to go through this blacklist?"
        else:
            topMsg = "Are you sure you want to request this blacklist?"
       
        duration_text = ""

        if unit == "Permanent":
            duration_text = f"Permanent"
        elif unit == "Years":
            duration_text = f"{duration_amount} Year{'s' if duration_amount > 1 else ''}"
        elif unit == "Months":
            duration_text = f"{duration_amount} Month{'s' if duration_amount > 1 else ''}"
        elif unit == "Days":
            duration_text = f"{duration_amount} Day{'s' if duration_amount > 1 else ''}"
            
        confirmation_msg = (
            f"**__Blacklist Information__**\n"
            f"**Members to be blacklisted:** {members}\n"
            f"**Duration:** {duration_text}\n"
            f"**Reason:** {reason}\n"
            f"**Evidence:** {'' if evidence else 'N/A'}\n\n"
        )

        embed = discord.Embed(description=confirmation_msg, color=discord.Color.orange())
        embed.set_author(
                name=f"{topMsg}",
                icon_url=Config.ALERT_URL
        )
        if evidence:
            embed.set_image(url=evidence.url)

        await ctx.send(embed=embed, view=view, ephemeral=True)
        await view.wait()
        
        if not view.value:
            return await ctx.send("```❎ Blacklist cancelled.```", ephemeral=True)
                                    
        ctx_data = BlacklistData(member_ids, reason, unit, duration_amount, evidence, member)

        if isPMorHigher:
           return await self.blacklist_members(ctx_data, ctx)

        request_embed = embedBuilder.build_blacklist_request(members, duration_text, reason, clean_nickname(member.display_name), evidence)     
        
        requestView = ApprovalView(ctx_data, self, self.bot)
        
        channel = ctx.guild.get_channel(Config.B_LOG_CHANNEL_ID)

        msg = await channel.send(
                f"<@&{Config.PM_ROLE_ID}>",
                embed=request_embed, 
                allowed_mentions=discord.AllowedMentions(roles=True), 
                view=requestView)

        await save_pending_blacklist(self.bot, msg.id, channel.id, ctx_data)

        await ctx.send(f"```✅ Blacklist Request has been sent.```", ephemeral=True)
        return

    async def blacklist_members(self, ctx_data, ctx):
        members = ctx_data.members
        reason = ctx_data.reason
        unit = ctx_data.unit
        duration = ctx_data.duration
        evidence = ctx_data.evidence
        issuer = ctx_data.issuer

        
        try:
            processing_msg = await ctx.send(
                            "⚙️ Processing blacklist...",
                            ephemeral=True,
                        )

            blacklisted_members = []
            for member_id in members:
                if member := ctx.guild.get_member(member_id):
                    blacklisted_members.append(member)
                else:
                    logger.warning(f"Member {member_id} not found in guild")

            if not blacklisted_members:
                await ctx.send("❌ No valid members found.", ephemeral=True)
                return

            # Calculate blacklist duration
            if unit == "Permanent":
                blacklist_duration = "Permanent"
            elif unit == "Years":
                blacklist_duration = f"{duration} year{'s' if duration > 1 else ''}"
            elif unit == "Months":
                blacklist_duration = f"{duration} month{'s' if duration > 1 else ''}"
            elif unit == "Days":
                blacklist_duration = f"{duration} day{'s' if duration > 1 else ''}"
            
            # Calculate ending date for blacklist
            ending_date = None
            if unit != "Permanent":
                current_date = datetime.now(timezone.utc)
                if unit == "Years":
                    ending_date = current_date + timedelta(days=duration * 365)
                elif unit == "Months":
                    ending_date = current_date + timedelta(days=duration * 30)  # Approximate
                elif unit == "Days":
                    ending_date = current_date + timedelta(days=duration)

            # Create notification embed for users (Dishonourable discharge + blacklist info)
            embed = discord.Embed(
                title="Blacklist Notification",
                color=discord.Color.red(),
                timestamp=datetime.now(timezone.utc)
            ).set_thumbnail(url=Config.RMP_URL)
            
            embed.add_field(name="Blacklist Duration:", value=blacklist_duration, inline=True)
            
            embed.add_field(name="Starting date:", value=f"<t:{int(datetime.now(timezone.utc).timestamp())}:D>", inline=True)

            if ending_date:
                embed.add_field(
                    name="Ending date:", 
                    value=f"<t:{int(ending_date.timestamp())}:D>",
                    inline=True
                )
        
            embed.add_field(name="Reason:", value=f"```{reason}```", inline=False)

            if evidence:
                embed.add_field(name="Evidence:", value=f"[Attachment Link]({evidence.url})", inline=False)
                mime, _ = mimetypes.guess_type(evidence.filename)
                if mime and mime.startswith("image/"):
                    embed.set_image(url=evidence.url)
        

            if issuer.id == ctx.author.id:
                embed.set_footer(text=f"Blacklisted by {clean_nickname(issuer.display_name)}")
            else:
                embed.set_footer(text=f"Blacklisted by {clean_nickname(issuer.display_name)} with approval of {clean_nickname(ctx.author.display_name)}")

            success_count = 0
            successful_members = []
            failed_members = []

            for member in blacklisted_members:
                cleaned_nickname = clean_nickname(member.display_name) or member.name
                
                try:
                    roblox_id = await self.bot.db.get_roblox_id(member.id)

                    member_info = f"{cleaned_nickname} | {member.id} ({member.mention})"
                    if roblox_id:
                        member_info += f" | {roblox_id}"

                    successful_members.append(member_info)
               
                    try:
                        await member.send(embed=embed)
                    except discord.Forbidden:
                        if channel := ctx.guild.get_channel(Config.PUBLIC_CHAT_CHANNEL_ID):
                            await channel.send(f"{member.mention}", embed=embed)

                    await self.bot.db.discharge_user(str(member.id), cleaned_nickname, ctx.guild)
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {member.display_name}: {str(e)}")
                    failed_members.append(member.mention)

            # Create result embed
            result_embed = discord.Embed(
                title="Blacklist Summary",
                color=discord.Color.red()
            )
            
            result_embed.add_field(name="Action:", value=f"Dishonourable Discharge & Blacklist", inline=False)
            result_embed.add_field(name="Duration:", value=blacklist_duration, inline=False)
            result_embed.add_field(
                name="Results:",
                value=f"Successfully processed: {success_count}\nFailed: {len(failed_members)}",
                inline=False
            )
            
            if failed_members:
                result_embed.add_field(name="Failed Members:", value=", ".join(failed_members), inline=False)

            await processing_msg.edit(
                content=None,
                embed=result_embed
            )

            # ========== LOG TO DISCHARGE CHANNEL ==========
            if d_log := ctx.guild.get_channel(Config.D_LOG_CHANNEL_ID):
                d_embed = embedBuilder.build_discharge_log(
                        "Dishonourable", 
                        successful_members, 
                        issuer, 
                        reason, 
                        "Blacklist", 
                        blacklist_duration, 
                        ending_date, 
                        evidence
                )

                await d_log.send(embed=d_embed)

            # ========== LOG TO BLACKLIST CHANNEL ==========
            if b_log := ctx.guild.get_channel(Config.B_LOG_CHANNEL_ID):
                blacklist_log_embed = embedBuilder.build_blacklist_log(
                        issuer, successful_members, blacklist_duration, reason, ending_date)

                await b_log.send(embed=blacklist_log_embed)

        except Exception as e:
            logger.error(f"Blacklist members function failed: {e}")
            await ctx.send("❌ An error occurred while processing the blacklist.", ephemeral=True)

    # Reset Database Command
    @commands.hybrid_command(
            name="reset-db",
            aliases=["rdb"],
            description="Reset the LR and HR tables."
    )
    @app_commands.checks.cooldown(1, 60.0)
    @has_modular_permission("administrative")
    async def reset_db(self, ctx: commands.Context):
        
        if ctx.interaction:
            await ctx.interaction.response.defer(ephemeral=False)

        view = ConfirmView(author=ctx.author)
        
        embed = discord.Embed(
                description=f"This will reset **ALL LR and HR stats**.\n\nClick **Confirm** to proceed or **Cancel** to abort.",
                color=discord.Color.red()
        )

        embed.set_author(
                name="Are you sure you want to reset the database?",
                icon_url=Config.ALERT_URL
        )

        await ctx.send(embed=embed, view=view)

        await view.wait()

        if view.value is not True:
            return await ctx.send(
                "```❎ Database reset cancelled.```",
            )

        try:
            res_hr, res_lr = await asyncio.gather(
                self.bot.db.reset_hrs(),
                self.bot.db.reset_lrs(),
                return_exceptions=True 
            )

            failed_tables = []
            if not res_hr or isinstance(res_hr, Exception): failed_tables.append("HR")
            if not res_lr or isinstance(res_lr, Exception): failed_tables.append("LR")

            if not failed_tables:
                cleaned_nickname = clean_nickname(ctx.author.display_name)
                embed = discord.Embed(
                    description=f"**{cleaned_nickname}** has just reset the database."
                ).set_author(name="Database Reset", icon_url=Config.RESET_ICON)
                
                return await ctx.send(content="```✅ Database reset successfully!```", embed=embed)

            if len(failed_tables) == 2:
                raise Exception("Both reset functions failed")

            partial_msg = f"✅ {failed_tables[0]} table reset failed, but the other succeeded. Try again."
            return await ctx.send(f"```⚠️ {partial_msg}```")

        except Exception as e:
            logger.exception(f"reset_db failed: {e}")
            return await ctx.send("```❌ Error resetting database.```")

async def setup(bot):
    await bot.add_cog(AdminCog(bot))

