import re
import asyncio
import logging
import discord
from discord import app_commands
from discord.ext import commands
from datetime import datetime, timezone
from typing import Optional, Literal

from config import Config
from utils import embedBuilder
from utils.views import PageButtonView
from utils.decorators import has_modular_permission
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
        log_channel = self.bot.get_channel(Config.DEFAULT_LOG_CHANNEL)  

        if not log_channel:
            logger.error("Default Logging channel not found!")
            return False
        
        embed = discord.Embed(
            color=discord.Color.green() if xp_change > 0 else discord.Color.red(),
            timestamp=datetime.now(timezone.utc)
        )

        embed.set_author(
                name="XP Log",
                icon_url=Config.STAR_ICON
        )

        embed.add_field(name="Staff:", value=f"{clean_nickname(admin.display_name)}", inline=True)
        embed.add_field(name="User:", value=f"{clean_nickname(user.display_name)}", inline=True)
        embed.add_field(name=f"{'Added:' if xp_change>0 else 'Removed:'}", value=f"{abs(xp_change)}", inline=True)
        embed.add_field(name="Reason:", value=f"```{reason}```", inline=False)

        embed.set_footer(text=f"Staff ID: {admin.id} • User ID: {user.id}")
        
        try:
            await log_channel.send(embed=embed)
            return True
        except Exception as e:
            logger.error(f"Failed to log XP to Discord: {str(e)}")
            return False


    #/addxp Command
    @commands.hybrid_command(
            name="add-xp", 
            aliases=["axp"],
            usage="<member> <value>",
            description="Add XP to a user"
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("xp_rewards")
    async def add_xp(self, ctx: commands.Context, user: discord.User, xp: int):
        async with self.bot.global_rate_limiter:

            if ctx.author.id == user.id:
                return await ctx.send("```❌ You cannot modify your own XP.```")
            # Validate XP amount
            if xp <= 0:
                await ctx.send(
                    "```❌ XP amount must be positive.```",
                    ephemeral=True
                )
                return
            if xp > Config.MAX_XP_PER_ACTION:
                await ctx.send(
                    f"```❌ Cannot give more than {Config.MAX_XP_PER_ACTION} XP at once.```",
                    ephemeral=True
                )
                return
        
            cleaned_name = clean_nickname(user.display_name)
            current_xp = await self.bot.db.get_user_xp(user.id)
            
            if current_xp > 100000:  
                await ctx.send(
                    "```❌ User has unusually high XP. Contact admin.```",
                )
                return
            
            success, new_total = await self.bot.db.add_xp(user.id, cleaned_name, xp)
            
            if success:
                await ctx.send(
                    f"```✅ Added {xp} XP to {cleaned_name}. New total: {new_total} XP```"
                )
                # Log the XP change
                await self.log_xp_to_discord(ctx.author, user, xp, new_total, "Manual Addition")
                 
            else:
                await ctx.send(
                    "```❌ Failed to add XP.```",
                    ephemeral=True
                )


    # /take-xp Command
    @commands.hybrid_command(
            name="take-xp",
            aliases=["txp"],
            usage="<member> <value>",
            description="Takes XP from user"
    )
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("xp_rewards")
    async def take_xp(self, ctx: commands.Context, user: discord.User, xp: int):
        async with self.bot.global_rate_limiter:
            if ctx.author.id == user.id:
                return await ctx.send("```❌ You cannot modify your own XP.```")

            if xp <= 0:
                await ctx.send(
                    "```❌ XP amount must be positive. Use /addxp to give XP.```",
                )
                return
            if xp > Config.MAX_XP_PER_ACTION:
                await ctx.send(
                    f"```❌ Cannot remove more than {Config.MAX_XP_PER_ACTION} XP at once.```",
                )
                return
        
            cleaned_name = clean_nickname(user.display_name)
            current_xp = await self.bot.db.get_user_xp(user.id)
            
            if xp > current_xp:
                await ctx.send(
                    f"```❌ User only has {current_xp} XP. Cannot take {xp}.```",
                )
                return
            
            success, new_total = await self.bot.db.remove_xp(user.id, xp)
            
            if success:
                message = f"```✅ Removed {xp} XP from {cleaned_name}. New total: {new_total} XP```"
                await ctx.send(message)
                await self.log_xp_to_discord(ctx.author, user, -xp, new_total, "Manual Removal")
                
            else:
                await ctx.send(
                    "```❌ Failed to take XP. Notify admin.```",
                )


    # /xp Command
    @commands.hybrid_command(
            name="xp",
            usage="<member: (optional)>",
            description="Check your XP or someone else's XP"
    )
    @app_commands.describe(user="The user to look up (leave empty to view your own XP)")
    @commands.cooldown(1, 5.0, commands.BucketType.user)
    @has_modular_permission("general")
    async def xp_command(self, ctx: commands.Context, user: Optional[discord.User] = None):
        try:
            if ctx.interaction:
                await ctx.interaction.response.defer()

            target_user = user or ctx.author
            cleaned_name = clean_nickname(target_user.display_name)

            xp = await self.bot.db.get_user_xp(target_user.id)
            tier, current_threshold, next_threshold, colour = get_tier_info(xp)   

            all_users = await self.bot.db.get_leaderboard("XP")

            position = next(
                (idx for idx, entry in enumerate(all_users, 1)
                 if str(entry['user_id']) == str(target_user.id)),
                None
            )

            progress = make_progress_bar(xp, current_threshold, next_threshold)

            embed = discord.Embed(
                title=f"{cleaned_name}",
                color=colour
            ).set_thumbnail(url=target_user.display_avatar.url)

            embed.add_field(name="Current XP", value=f"```{xp}```", inline=True)
            embed.add_field(name="Tier", value=f"```{tier}```", inline=True)

            if position:
                embed.add_field(name="Leaderboard Position", value=f"```#{position}```", inline=True)

            embed.add_field(name="Progression", value=progress, inline=False)

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"XP command error: {str(e)}")
            await ctx.send("```❌ Failed to fetch XP data.```", ephemeral=True)

    # /profile Command
    @commands.hybrid_command(
            name="profile", 
            usage="<member: (optional)>",
            description="Check your profile or someone else's"
    )
    @app_commands.describe(user="The user to look up (leave empty to view your own)")
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("general")
    async def profile(self, ctx: commands.Context, user: Optional[discord.Member] = None):
        try:
            if ctx.interaction:
                await ctx.interaction.response.defer()

            target_user = user or ctx.author
             
            user_info = await self.bot.db.get_user(str(target_user.id))
            if not user_info:
                return await ctx.send("```❌ User data not found.```", ephemeral=True)

            army_rank = "Unknown"
            if user_info['roblox_id']:
                army_rank = await self.bot.roblox.get_group_rank(user_info['roblox_id'], Config.BRITISH_ARMY_GROUP_ID)
           
            embed = embedBuilder.build_profile_embed(user_info, target_user, army_rank)
            
            try:
                hr_role = ctx.guild.get_role(Config.HR_ROLE_ID)
                if hr_role and hr_role in target_user.roles:
                    hr_data = await self.bot.db.get_hr_info(target_user.id)
                    total_events = 0 
                    dep_points = 0
                    if hr_data: 
                        columns_to_sum = ['tryouts', 'events', 'phases', 'inspections', 'joint_events']
                        total_events = sum(hr_data.get(col, 0) for col in columns_to_sum)
                        dep_points = hr_data.get('courses', 0)
                    embed.add_field(name="EVENTS HOSTED:", value=f"```{total_events}```")
                    embed.add_field(name="DEP POINTS:", value=f"```{dep_points}```", inline=True)
                else:
                    lr_data = await self.bot.db.get_lr_info(target_user.id)
                    total_activity = 0
                    events_attended = 0
                    if lr_data: 
                        columns_to_sum = ['activity', 'time_guarded']
                        total_activity = sum(lr_data.get(col, 0) for col in columns_to_sum)
                        events_attended = lr_data.get('events_attended', 0)

                    embed.add_field(name="TOTAL ACTIVITY:", value=f"```{total_activity}```")
                    embed.add_field(name="EVENTS ATTENDED:", value=f"```{events_attended}```")
            except Exception as e:
                logger.error("Failed to retrieve hr/lr info of %s: %s", target_user.id, e)

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Profile command error: {e}")
            await ctx.send("```❌ Failed to generate profile.```", ephemeral=True)

    # Leadebaord Command
    @commands.hybrid_command(
            name="leaderboard",
            aliases=["lb"],
            usage="<category: xp/hr/activity/events/dep (optional)>",
            description="View Leaderboard for a category"
    )
    @app_commands.choices(category=[
        app_commands.Choice(name="XP", value="XP"),
        app_commands.Choice(name="HR", value="HR"),
        app_commands.Choice(name="LR Activity", value="LR Activity"),
        app_commands.Choice(name="LR Events", value="LR Events"),
        app_commands.Choice(name="Departments", value="Departments")
    ])
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("general")
    async def leaderboard(
            self, 
            ctx: commands.Context,
            category: Literal["XP", "HR", "LR Activity", "LR Events", "Departments", "xp", "hr","activity", "events", "dep"] = "XP"
    ):
        category_mapping = {
                "xp": "XP",
                "hr": "HR",
                "activity": "LR Activity",
                "events": "LR events",
                "dep": "Departments"
        }

        category_type = category_mapping.get(category, category)

        try:
            async with self.bot.global_rate_limiter:
                await self.bot.rate_limiter.wait_if_needed(bucket="supabase_query")
                
                data = await self.bot.db.get_leaderboard(category_type)
                
                if not data:
                    await ctx.send("```❌ No leaderboard data available.```", ephemeral=True)
                    return
                
                leaderboard_lines = []
                leaderboard_pages = []
                
                for idx, entry in enumerate(data, start=1):
                    try:
                        username = entry['username']  
                        score = entry['score']
                        leaderboard_lines.append(f"**#{idx}** - {username}: `{score}`")
                        if idx%30 == 0:
                            page = "\n".join(leaderboard_lines)
                            leaderboard_pages.append(page)
                            leaderboard_lines.clear()

                    except Exception as e:
                        continue
                
                if leaderboard_lines:
                    page = "\n".join(leaderboard_lines)
                    leaderboard_pages.append(page)
           
                embed = discord.Embed(
                    description=leaderboard_pages[0] or "No data available",
                    color=discord.Color.red()
                ).set_author(name=f"{category_type} Leaderboard", icon_url=Config.TROPHY_ICON)
               
                view = PageButtonView(ctx.author, 0, len(leaderboard_pages), leaderboard_pages)
                
                await ctx.send(embed=embed, view=view)
            
        except Exception as e:
            logger.error(f"Error in /leaderboard command: {str(e)}")
            await ctx.send(
                "❌ Failed to fetch leaderboard data. Please try again later.",
                ephemeral=True
            )


    # Give Event XP Command
    @commands.hybrid_command(
            name="give-event-xp",
            aliases=["gxp"],
            usage="<link> <xp> <section: a/p (optional)>",
            description="Give XP to attendees mentioned in an event log message"
    )
    @app_commands.choices(attendees_section=[
        app_commands.Choice(name="Attendees:", value="Attendees:"),
        app_commands.Choice(name="Passed:", value="Passed:")
    ])
    @app_commands.checks.cooldown(1, 5.0)
    @has_modular_permission("xp_rewards")
    async def give_event_xp(
        self,
        ctx: commands.Context,
        message_link: str,
        xp_amount: int,
        attendees_section: Literal["Attendees:", "Passed:", "a", "p"] = "Attendees:"
    ):
        async with self.bot.global_rate_limiter:
            await self.bot.rate_limiter.wait_if_needed(bucket="global_xp_update")
            if xp_amount <= 0:
                await ctx.send(
                    "❌ XP amount must be positive.",
                )
                return
            if xp_amount > Config.MAX_EVENT_XP_PER_USER:
                await ctx.send(
                    f"❌ Cannot give more than {Config.MAX_EVENT_XP_PER_USER} XP per user in events.",
                )
                return
        
            await ctx.defer()
            initial_message = await ctx.send("```⏳ Attempting to give XP...```")
            
            section_map = {
                    "a": "Attendees:",
                    "p": "Passed:",
            }
            section = section_map.get(attendees_section, attendees_section)

            try:
                async with asyncio.timeout(60):  
                    try:
                        await asyncio.wait_for(
                            self.bot.rate_limiter.wait_if_needed(bucket=f"give_xp_{ctx.author.id}"),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        await initial_message.edit(content="```⌛ Rate limit check timed out. Please try again.```")
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
                        await initial_message.edit(content="```❌ Invalid message link format.```")
                        return
                    
                    if guild_id != ctx.guild.id:
                        await initial_message.edit(content="```❌ Message must be from this server.```")
                        return
                        
                    # Fetch the message with timeout
                    try:
                        channel = ctx.guild.get_channel(channel_id)
                        if not channel:
                            await initial_message.edit(content="```❌ Channel not found.```")
                            return
                            
                        try:
                            message = await asyncio.wait_for(
                                channel.fetch_message(message_id),
                                timeout=10.0
                            )
                        except discord.NotFound:
                            await initial_message.edit(content="```❌ Message not found.```")
                            return
                        except discord.Forbidden:
                            await initial_message.edit(content="```❌ No permission to read that channel.```")
                            return
                    except asyncio.TimeoutError:
                        await initial_message.edit(content="```⌛ Timed out fetching message.```")
                        return
                        
                    # Process attendees section
                    content = message.content
                    section_index = content.find(section)
                    if section_index == -1:
                        await initial_message.edit(content=f"```❌ Could not find '{attendees_section}' in the message.``")
                        return
                        
                    mentions_section = content[section_index + len(attendees_section):]
                    mentions = re.findall(r'<@!?(\d+)>', mentions_section)

                    mentions = list(set(mentions))
                    
                    if not mentions:
                        await initial_message.edit(content=f"```❌ No user mentions found after '{attendees_section}'```")
                        return
                        
                    # Process users with progress updates
                    unique_mentions = list(set(mentions))
                    total_potential_xp = xp_amount * len(unique_mentions)
                    
                    if total_potential_xp > Config.MAX_EVENT_TOTAL_XP:
                        await initial_message.edit(
                            content=f"```❌ Event would give {total_potential_xp} XP total (max is {Config.MAX_EVENT_TOTAL_XP}). Reduce XP or attendees.```"
                        )
                        return
                        
                    await initial_message.edit(content=f"```🎯Processing XP for {len(unique_mentions)} users...```")
                    
                    successful_users = []
                    failed_users = []
                    
                    for i, user_id in enumerate(unique_mentions, 1):
                        try:
                            await initial_message.edit(
                                content=f"```⏳ Processing {i}/{len(unique_mentions)} users...```"
                            )
                            
                            if i > 1:
                                await asyncio.sleep(0.3) 
                                
                            member = ctx.guild.get_member(int(user_id))
                            if not member:
                                try:
                                    member = await ctx.guild.fetch_member(int(user_id))
                                except discord.NotFound:
                                    failed_users.append(f"User {user_id} (not in guild)")
                                    await initial_message.edit(
                                        content=f"```❌ Skipping {user_id} (Not in guild)```"
                                    )
                                    continue
                                

                            try:
                                cleaned_nickname = clean_nickname(member.display_name)

                                if ctx.author.id == member.id:
                                    failed_users.append(f"{cleaned_nickname} (You cannot modify your own XP)")
                                    await initial_message.edit(
                                        content=f"```❌ Skipping {cleaned_nickname} (Can't modify your own XP)```"
                                    )
                                    continue

                                current_xp = await self.bot.db.get_user_xp(member.id)

                                if current_xp + xp_amount > 100000:
                                    failed_users.append(f"{cleaned_nickname} (would exceed max XP)")
                                    continue
                                    
                                success, new_total = await self.bot.db.add_xp(member.id, member.display_name, xp_amount)
                                
                                if success:
                                    successful_users.append(f"• {cleaned_nickname}")
                                    await initial_message.edit(
                                        content=f"```✨ Gave {xp_amount} XP to {cleaned_nickname} (New total: {new_total} XP)```"
                                    )
                                    await self.log_xp_to_discord(ctx.author, member, xp_amount, new_total, f"Event: {message.jump_url}")
                
                                else:
                                    failed_users.append(f"• {cleaned_nickname}")
                                    
                            except asyncio.TimeoutError:
                                failed_users.append(f"• {cleaned_nickname} (timeout)")
                                continue
                                
                        except Exception as e:
                            logger.error(f"Error processing user {user_id}: {str(e)}")
                            failed_users.append(f"User {user_id} (error)")
                            continue
                            
                    # Final summary
                    result_message = [
                        f"**Given by:** {ctx.author.mention}",
                        f"**Successfully distributed {xp_amount} XP to:**\n{ "\n".join(successful_users)}"
                    ]
                    
                    if failed_users:
                        result_message.append(f"\n**Failed to distribute to:**\n{ "\n• ".join(failed_users)}")
                    
                    result = "\n".join(result_message)
                    embed = discord.Embed(
                            description=result, 
                            color=discord.Color.green()).set_author(
                                    name="XP DISTRIBUTION",
                                    icon_url=Config.CHECK_URL)

                    await initial_message.edit(content="", embed=embed)

            except asyncio.TimeoutError:
                await initial_message.edit(content="⌛ Command timed out. Some XP may have been awarded.")
            except Exception as e:
                logger.error(f"Error in give_event_xp: {str(e)}", exc_info=True)
                await initial_message.edit(content="❌ An unexpected error occurred. Please check logs.")


async def setup(bot):
    await bot.add_cog(XPCog(bot))
