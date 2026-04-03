import re
import discord
import asyncio
import logging
from datetime import datetime, timezone
from config import Config
from utils.decorators import min_rank_required
from discord import app_commands
from discord.ext import commands
from typing import Optional, Literal
from utils.helpers import clean_nickname, get_tier_info, make_progress_bar

logger = logging.getLogger(__name__)

class XPCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot 


    # Logs XP changes in logging channel
    async def log_xp_to_discord(
        self,
        admin: discord.User,
        user: discord.User,
        xp_change: int,
        new_total: int,
        reason: str
    ):
        """Log XP changes to a Discord channel instead of Supabase."""
        log_channel = self.bot.get_channel(Config.DEFAULT_LOG_CHANNEL)  # Replace with your channel ID
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

    async def handle_command_error(self, interaction: discord.Interaction, error: Exception):
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


    #/addxp Command
    @app_commands.command(name="add-xp", description="Add XP to a user")
    @app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
    @min_rank_required(Config.CSM_ROLE_ID)
    async def add_xp(self, interaction: discord.Interaction, user: discord.User, xp: int):
        async with self.bot.global_rate_limiter:
            # Validate XP amount
            if xp <= 0:
                await interaction.response.send_message(
                    "❌ XP amount must be positive.",
                    ephemeral=True
                )
                return
            if xp > Config.MAX_XP_PER_ACTION:
                await interaction.response.send_message(
                    f"❌ Cannot give more than {Config.MAX_XP_PER_ACTION} XP at once.",
                    ephemeral=True
                )
                return
        
            cleaned_name = clean_nickname(user.display_name)
            current_xp = await self.bot.db.get_user_xp(user.id)
            
            # Additional safety check
            if current_xp > 100000:  # Extreme value check
                await interaction.response.send_message(
                    "❌ User has unusually high XP. Contact admin.",
                    ephemeral=True
                )
                return
            
            success, new_total = await self.bot.db.add_xp(user.id, cleaned_name, xp)
            
            if success:
                await interaction.response.send_message(
                    f"✅ Added {xp} XP to {cleaned_name}. New total: {new_total} XP"
                )
                # Log the XP change
                await self.log_xp_to_discord(interaction.user, user, xp, new_total, "Manual Addition")
                 
            else:
                await interaction.response.send_message(
                    "❌ Failed to add XP. Notify admin.",
                    ephemeral=True
                )


    # /take-xp Command
    @app_commands.command(name="take-xp", description="Takes XP from user")
    @app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
    @min_rank_required(Config.CSM_ROLE_ID)
    async def take_xp(self, interaction: discord.Interaction, user: discord.User, xp: int):
        async with self.bot.global_rate_limiter:
            if xp <= 0:
                await interaction.response.send_message(
                    "❌ XP amount must be positive. Use /addxp to give XP.",
                    ephemeral=True
                )
                return
            if xp > Config.MAX_XP_PER_ACTION:
                await interaction.response.send_message(
                    f"❌ Cannot remove more than {Config.MAX_XP_PER_ACTION} XP at once.",
                    ephemeral=True
                )
                return
        
            cleaned_name = clean_nickname(user.display_name)
            current_xp = await self.bot.db.get_user_xp(user.id)
            
            if xp > current_xp:
                await interaction.response.send_message(
                    f"❌ User only has {current_xp} XP. Cannot take {xp}.",
                    ephemeral=True
                )
                return
            
            success, new_total = await self.bot.db.remove_xp(user.id, xp)
            
            if success:
                message = f"✅ Removed {xp} XP from {cleaned_name}. New total: {new_total} XP"
                if new_total == 0:
                    message += "\n⚠️ User's XP has reached 0"
                await interaction.response.send_message(message)
                # Log the XP change
                await self.log_xp_to_discord(interaction.user, user, -xp, new_total, "Manual Removal")
                
            else:
                await interaction.response.send_message(
                    "❌ Failed to take XP. Notify admin.",
                    ephemeral=True
                )


    # /xp Command
    @app_commands.command(name="xp", description="Check your XP or someone else's XP")
    @app_commands.describe(user="The user to look up (leave empty to view your own XP)")
    @app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
    @min_rank_required(Config.RMP_ROLE_ID)
    async def xp_command(self, interaction: discord.Interaction, user: Optional[discord.User] = None):
        try:
            await interaction.response.defer(ephemeral=True)
            target_user = user or interaction.user
            cleaned_name = clean_nickname(target_user.display_name)

            xp = await self.bot.db.get_user_xp(target_user.id)
            tier, current_threshold, next_threshold = get_tier_info(xp)  # Use fixed function

            result = await asyncio.to_thread(
                lambda: self.bot.db.supabase.table('users')
                .select("user_id", "xp")
                .order("xp", desc=True)
                .execute()
            )

            position = next(
                (idx for idx, entry in enumerate(result.data, 1)
                 if str(entry['user_id']) == str(target_user.id)),
                None
            )

            progress = make_progress_bar(xp, current_threshold, next_threshold)

            embed = discord.Embed(
                title=f"{cleaned_name}",
                color=discord.Color.green()
            ).set_thumbnail(url=target_user.display_avatar.url)

            embed.add_field(name="Current XP", value=f"```{xp}```", inline=True)
            embed.add_field(name="Tier", value=f"```{tier}```", inline=True)

            if position:
                embed.add_field(name="Leaderboard Position", value=f"```#{position}```", inline=True)

            embed.add_field(name="Progression", value=progress, inline=False)

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"XP command error: {str(e)}")
            await interaction.followup.send("❌ Failed to fetch XP data.", ephemeral=True)

    # Leadebaord Command
    @app_commands.command(name="leaderboard", description="View the top 15 users by XP")
    @app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
    @min_rank_required(Config.RMP_ROLE_ID)  
    async def leaderboard(self, interaction: discord.Interaction):
        try:
            async with self.bot.global_rate_limiter:
                # Rate limiting for Supabase
                await self.bot.rate_limiter.wait_if_needed(bucket="supabase_query")
                
                # Fetch top 15 users from Supabase
                result = self.bot.db.supabase.table('users') \
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
                        user = interaction.guild.get_member(user_id) or await self.bot.fetch_user(user_id)
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
    @app_commands.command(name="give-event-xp", description="Give XP to attendees mentioned in an event log message")
    @app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
    @min_rank_required(Config.CSM_ROLE_ID)
    async def give_event_xp(
        self,
        interaction: discord.Interaction,
        message_link: str,
        xp_amount: int,
        attendees_section: Literal["Attendees:", "Passed:"] = "Attendees:"
    ):
        async with self.bot.global_rate_limiter:
            # Rate Limit
            await self.bot.rate_limiter.wait_if_needed(bucket="global_xp_update")
            # Validate XP amount first
            if xp_amount <= 0:
                await interaction.response.send_message(
                    "❌ XP amount must be positive.",
                    ephemeral=True
                )
                return
            if xp_amount > Config.MAX_EVENT_XP_PER_USER:
                await interaction.response.send_message(
                    f"❌ Cannot give more than {Config.MAX_EVENT_XP_PER_USER} XP per user in events.",
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
                            self.bot.rate_limiter.wait_if_needed(bucket=f"give_xp_{interaction.user.id}"),
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
                    
                    if total_potential_xp > Config.MAX_EVENT_TOTAL_XP:
                        await initial_message.edit(
                            content=f"❌ Event would give {total_potential_xp} XP total (max is {Config.MAX_EVENT_TOTAL_XP}). Reduce XP or attendees."
                        )
                        return
                        
                    await initial_message.edit(content=f"🎯 Processing XP for {len(unique_mentions)} users...")
                    
                    success_count = 0
                    failed_users = []
                    processed_users = 0
                    
                    for i, user_id in enumerate(unique_mentions, 1):
                        try:
                            await initial_message.edit(
                                content=f"⏳ Processing {i}/{len(unique_mentions)} users ({success_count} successful)..."
                            )
                            
                            # Rate limit between users
                            if i > 1:
                                await asyncio.sleep(0.75)  # Slightly longer delay
                                
                            member = interaction.guild.get_member(int(user_id))
                            if not member:
                                try:
                                    member = await interaction.guild.fetch_member(int(user_id))
                                except discord.NotFound:
                                    failed_users.append(f"User {user_id} (not in guild)")
                                    continue
                                
                            try:
                                current_xp = await asyncio.wait_for(
                                    self.bot.db.get_user_xp(member.id),
                                    timeout=5.0
                                )
                                if current_xp + xp_amount > 100000:
                                    failed_users.append(f"{clean_nickname(member.display_name)} (would exceed max XP)")
                                    continue
                                    
                                success, new_total = await asyncio.wait_for(
                                    self.bot.db.add_xp(member.id, member.display_name, xp_amount),
                                    timeout=5.0
                                )
                                
                                if success:
                                    success_count += 1
                                    await interaction.followup.send(
                                        f"✨ **{clean_nickname(interaction.user.display_name)}** gave {xp_amount} XP to {member.mention} (New total: {new_total} XP)",
                                        silent=True
                                    )
                                    await self.log_xp_to_discord(interaction.user, member, xp_amount, new_total, f"Event: {message.jump_url}")
                
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
                        f"**__XP Distribution Completed**__",
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


async def setup(bot):
    await bot.add_cog(XPCog(bot))
