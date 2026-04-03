import re
import mimetypes
import discord
import logging
from config import Config
from datetime import datetime, timedelta, timezone
from utils.decorators import min_rank_required
from discord import app_commands
from discord.utils import escape_markdown
from discord.ext import commands
from utils.views import ConfirmView
from typing import Optional, Literal


logger = logging.getLogger(__name__)

class ModerationCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # Discharge Command
    @app_commands.command(name="discharge", description="Notify members of honourable/general/dishonourable discharge and log it")
    @app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
    @min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
    async def discharge(
        self,
        interaction: discord.Interaction,
        members: str,  # Comma-separated user mentions/IDs
        reason: str,
        discharge_type: Literal["Honourable", "General", "Dishonourable"] = "General",
        evidence: Optional[discord.Attachment] = None
    ):
        view = ConfirmView(author=interaction.user)
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

            await self.bot.rate_limiter.wait_if_needed(bucket="discharge")

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
            color = discord.Color.green() 
            if discharge_type == "General":
                color = discord.Color.green() 
            elif discharge_type == "Honourable":
                color = discord.Color.gold() 
            else:
                color = discord.Color.red()
                
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
                cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
                try:
                    await self.bot.db.discharge_user(str(member.id), cleaned_nickname, interaction.guild)

                except Exception as e:
                    logger.error(f"Error removing {cleaned_nickname} ({member.id}) from DB: {e}")
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
                    value = f"🔰 {discharge_type} Discharge" if discharge_type in ("Honourable", "General") else f"🚨 {discharge_type} Discharge",
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

    @app_commands.command(name="blacklist", description="Blacklist members with specified duration")
    @app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
    @min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
    async def blacklist(
        self,
        interaction: discord.Interaction,
        members: str,  # Comma-separated user mentions/IDs
        reason: str,
        duration_unit: Literal["Permanent", "Years", "Months", "Days"],
        duration_amount: app_commands.Range[int, 1, 100] = 1,
        evidence: Optional[discord.Attachment] = None
    ):
        view = ConfirmView(author=interaction.user)
        
        # Create confirmation message
        duration_text = "Permanent" if duration_unit == "Permanent" else f"{duration_amount} {duration_unit}"
        confirmation_msg = (
            f"⚠️ **Are you sure you want to blacklist member(s)?**\n"
            f"**Duration:** {duration_text}\n"
            f"**Reason:** {reason[:200]}{'...' if len(reason) > 200 else ''}\n\n"
            f"Click **Confirm** to proceed or **Cancel** to abort."
        )
        
        await interaction.response.send_message(confirmation_msg, view=view, ephemeral=True)
        await view.wait()
        
        if not view.value:
            await interaction.followup.send("❎ Blacklist cancelled.", ephemeral=True)
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

            # Parse member IDs
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
                "⚙️ Processing blacklist...",
                ephemeral=True,
                wait=True
            )

            await self.bot.rate_limiter.wait_if_needed(bucket="blacklist")

            # Get member objects
            blacklisted_members = []
            for member_id in member_ids:
                if member := interaction.guild.get_member(member_id):
                    blacklisted_members.append(member)
                else:
                    logger.warning(f"Member {member_id} not found in guild")

            if not blacklisted_members:
                await interaction.followup.send("❌ No valid members found.", ephemeral=True)
                return

            # Calculate blacklist duration
            if duration_unit == "Permanent":
                blacklist_duration = "Permanent"
            elif duration_unit == "Years":
                blacklist_duration = f"{duration_amount} year{'s' if duration_amount > 1 else ''}"
            elif duration_unit == "Months":
                blacklist_duration = f"{duration_amount} month{'s' if duration_amount > 1 else ''}"
            elif duration_unit == "Days":
                blacklist_duration = f"{duration_amount} day{'s' if duration_amount > 1 else ''}"
            
            # Calculate ending date for blacklist
            ending_date = None
            if duration_unit != "Permanent":
                current_date = datetime.now(timezone.utc)
                if duration_unit == "Years":
                    ending_date = current_date + timedelta(days=duration_amount * 365)
                elif duration_unit == "Months":
                    ending_date = current_date + timedelta(days=duration_amount * 30)  # Approximate
                elif duration_unit == "Days":
                    ending_date = current_date + timedelta(days=duration_amount)

            # Create notification embed for users (Dishonourable discharge + blacklist info)
            embed = discord.Embed(
                title="Dishonourable Discharge & Blacklist Notification",
                color=discord.Color.red(),
                timestamp=datetime.now(timezone.utc)
            )
            
            embed.add_field(name="Reason", value=reason, inline=False)
            embed.add_field(name="Blacklist Duration", value=blacklist_duration, inline=False)
            
            if ending_date:
                embed.add_field(
                    name="Blacklist Ends", 
                    value=f"<t:{int(ending_date.timestamp())}:D> (<t:{int(ending_date.timestamp())}:R>)",
                    inline=False
                )
            
            if evidence:
                embed.add_field(name="Evidence", value=f"[Attachment Link]({evidence.url})", inline=False)
                mime, _ = mimetypes.guess_type(evidence.filename)
                if mime and mime.startswith("image/"):
                    embed.set_image(url=evidence.url)

            embed.set_footer(text=f"Blacklisted by {interaction.user.display_name}")

            success_count = 0
            failed_members = []

            # Process each member
            for member in blacklisted_members:
                cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
                
                try:
                    # Remove from database (dishonourable discharge)
                    await self.bot.db.discharge_user(str(member.id), cleaned_nickname, interaction.guild)
                    
                    # Try to notify the user
                    try:
                        await member.send(embed=embed)
                    except discord.Forbidden:
                        # If DMs are closed, try public channel
                        if channel := interaction.guild.get_channel(1219410104240050236):
                            await channel.send(f"{member.mention}", embed=embed)
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {member.display_name}: {str(e)}")
                    failed_members.append(member.mention)

            # Create result embed
            result_embed = discord.Embed(
                title="Blacklist Summary",
                color=discord.Color.red()
            )
            
            result_embed.add_field(name="Action", value=f"Dishonourable Discharge & Blacklist", inline=False)
            result_embed.add_field(name="Duration", value=blacklist_duration, inline=False)
            result_embed.add_field(
                name="Results",
                value=f"✅ Successfully processed: {success_count}\n❌ Failed: {len(failed_members)}",
                inline=False
            )
            
            if failed_members:
                result_embed.add_field(name="Failed Members", value=", ".join(failed_members), inline=False)

            await processing_msg.edit(
                content=None,
                embed=result_embed
            )

            # ========== LOG TO DISCHARGE CHANNEL (AS DISHONOURABLE) ==========
            if d_log := interaction.guild.get_channel(Config.D_LOG_CHANNEL_ID):
                log_embed = discord.Embed(
                    title="Discharge Log",
                    color=discord.Color.red(),
                    timestamp=datetime.now(timezone.utc)
                )
                
                # Log as dishonourable discharge (same format as /discharge command)
                log_embed.add_field(name="Type", value="🚨 Dishonourable Discharge", inline=False)
                log_embed.add_field(name="Sub-Type", value="⛔ With Blacklist", inline=True)
                log_embed.add_field(name="Blacklist Duration", value=blacklist_duration, inline=True)
                
                if ending_date:
                    log_embed.add_field(
                        name="Blacklist Ends", 
                        value=f"<t:{int(ending_date.timestamp())}:D>",
                        inline=True
                    )
                
                log_embed.add_field(name="Reason", value=f"```{reason}```", inline=False)
                
                # Format member mentions with their cleaned nicknames (same format as /discharge)
                member_entries = []
                for member in blacklisted_members:
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

            # ========== LOG TO BLACKLIST CHANNEL ==========
            if b_log := interaction.guild.get_channel(Config.B_LOG_CHANNEL_ID):
                blacklist_log_embed = discord.Embed(
                    title="⛔ Blacklist Entry",
                    color=discord.Color.dark_red(),
                    timestamp=datetime.now(timezone.utc)
                )
                
                blacklist_log_embed.add_field(name="Issuer:", value=interaction.user.mention, inline=False)
                
                # List all blacklisted members with Roblox IDs if available
                for member in blacklisted_members:
                    cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
                    
                    # Try to get Roblox ID from database before removing
                    roblox_id = None
                    try:
                        def _get_roblox_id():
                            sup = self.bot.db.supabase
                            res = sup.table('users').select('roblox_id').eq('user_id', str(member.id)).execute()
                            if getattr(res, 'data', None) and len(res.data) > 0:
                                return res.data[0].get('roblox_id')
                            return None
                        
                        roblox_id = await self.bot.db.run_query(_get_roblox_id)
                    except Exception as e:
                        logger.error(f"Error getting Roblox ID for {cleaned_nickname}: {e}")
                    
                    member_info = f"**Name:** {cleaned_nickname}\n**Discord:** {member.mention}"
                    if roblox_id:
                        member_info += f"\n**Roblox ID:** {roblox_id}"
                    
                    blacklist_log_embed.add_field(
                        name=cleaned_nickname,
                        value=member_info,
                        inline=False
                    )
                
                blacklist_log_embed.add_field(name="Duration:", value=blacklist_duration, inline=False)
                blacklist_log_embed.add_field(name="Reason:", value=reason, inline=False)
                
                if ending_date:
                    blacklist_log_embed.add_field(
                        name="Ending date", 
                        value=f"<t:{int(ending_date.timestamp())}:D>",
                        inline=False
                    )
                
                blacklist_log_embed.add_field(name="Starting date", value=f"<t:{int(datetime.now(timezone.utc).timestamp())}:D>", inline=False)
                blacklist_log_embed.set_footer(text=f"Blacklisted by {interaction.user.display_name}")
                
                await b_log.send(embed=blacklist_log_embed)

        except Exception as e:
            logger.error(f"Blacklist command failed: {e}")
            await interaction.followup.send("❌ An error occurred while processing the blacklist.", ephemeral=True)



async def setup(bot):
    await bot.add_cog(ModerationCog(bot))

