import logging
import discord
from discord import app_commands
from discord.ext import commands

from config import Config
from utils import embedBuilder
from typing import Literal
from utils.views import PageButtonView
from utils.helpers import clean_nickname, ViewModalTrigger, SuggestionModal, BugModal

logger = logging.getLogger(__name__)

class UtilityCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_command_completion(self, ctx: commands.Context):
        if ctx.author.id == Config.DEVELOPER_ID:
            return
        guild = ctx.guild
        if not guild:
            return

        log_channel = guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
        if not log_channel:
            return

        command_name = ctx.command.qualified_name
        prefix = "/" if ctx.interaction else ctx.prefix

        user = ctx.author
        cleaned_nickname = clean_nickname(user.display_name)
        channel_mention = ctx.channel.mention

        actual_args = []
        
        params = [name for name in ctx.command.clean_params]
        for name, val in zip(params, ctx.args[2:]):
            actual_args.append(f"`{name}`: {val}")

        for name,val in ctx.kwargs.items():
            actual_args.append(f"`{name}`: {val}")

        if actual_args:
            args_str = ", ".join(actual_args)
            arg_msg = f" with the following arguments {args_str}"
        else:
            arg_msg = ""

        description = f"{cleaned_nickname} executed **{prefix}{command_name}** in {channel_mention}{arg_msg}"
        logger.info(f"Command executed: {prefix}{command_name} by {cleaned_nickname} ({user.id})")

        embed = discord.Embed(
            description=description,
            color=discord.Color.blurple(),
            timestamp=discord.utils.utcnow()
        )
        embed.set_author(
            name="Command Execution",
            icon_url=Config.HAMMER
        )
        embed.set_footer(text=f"Executor ID: {user.id}")
        await log_channel.send(embed=embed)

    @commands.hybrid_command(
            name="commands",
            usage="<category: general/xp/mod/admin/restricted (optional)>",
            description="List all available commands"
    )
    @app_commands.checks.cooldown(1, 5.0)
    @app_commands.describe(
        category="Look at a specific group of commands (LEAVE BLANK IF NOT)",
    )
    async def command_list(
            self,
            ctx: commands.Context,
            category: Literal["General", "XP Rewards", "Moderation", "Administrative", "Restricted", "general", "xp", "mod", "admin", "restricted"] | "None" = None
            ):
        
        page_mapping = {
            None: 0,
            "general": 1,
            "General": 1,
            "XP Rewards": 2,
            "xp": 2,
            "Moderation": 3,
            "mod": 3,
            "Administrative": 4,
            "admin": 4,
            "Restricted": 5,
            "restricted": 5,
        }
        prefix = self.bot.command_prefix
        page = page_mapping.get(category)
        embed = embedBuilder.build_commands_page(page, prefix)
        view = PageButtonView(ctx.author, page, 6)
        view.message = await ctx.send(embed=embed, view=view)
        
    # Ping Command
    @commands.hybrid_command(name="ping", description="Check bot latency")
    @app_commands.checks.cooldown(1, 5.0)
    async def ping(self, ctx: commands.Context):
        latency = round(self.bot.latency * 1000)
        await ctx.send(
            f"```🏓 Pong! Latency: {latency}ms```",
            ephemeral=True
        )

    # Report Bug Command
    @commands.hybrid_command(
            name="report-bug", 
            aliases=["report"],
            description="Report a bug or mistake"
    )
    @app_commands.checks.cooldown(1, 300.0)
    async def report_bug(self, ctx: commands.Context):

        if ctx.interaction:
            await ctx.interaction.response.send_modal(BugModal())
        else: 
            await ctx.send(
                    "```Click the button to open the report panel.```",
                    view=ViewModalTrigger(ctx.author, "bug")
            )


    # Suggest Command
    @commands.hybrid_command(name="suggest", description="Suggest/Recommend something for the bot")
    @app_commands.checks.cooldown(1, 300.0)
    async def suggest(self, ctx: commands.Context):
        
        if ctx.interaction:
            await ctx.interaction.response.send_modal(SuggestionModal())
        else: 
            await ctx.send(
                    "```Click the button to open the suggestion panel.```",
                    view=ViewModalTrigger(ctx.author, "suggestion")
            )

 

    @commands.hybrid_command(
            name="privacy-policy",
            aliases=["policy"],
            description="View the privacy policy of the bot"
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def privacy_policy(self, ctx: commands.Context):
        privacy_embed = embedBuilder.build_privacy_policy()
        await ctx.send(embed=privacy_embed, ephemeral=True)

    @commands.hybrid_command(
            name="change-logs", 
            aliases=["cl"],
            description="View change-logs"
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def change_logs(self, ctx: commands.Context):
        embeds = embedBuilder.build_change_log(self.bot.command_prefix)
        await ctx.send(embeds=embeds, ephemeral=True)

    @commands.hybrid_command(
            name="rmp-info", 
            aliases=["info"],
            description="View information about RMP"
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def rmp_info(self, ctx: commands.Context):
        all_users = self.bot.db._user_cache
        hr_users = self.bot.db._hrs_cache
        lr_users = self.bot.db._lrs_cache

        member_count = len(all_users)
        hr_count = len(hr_users) - 3 #Bc of SMs
        lr_count = len(lr_users)
        
        pm = ""
        total_xp = 0
        total_events = 0
        total_dep_points = 0
        total_attended_events = 0
        total_activity = 0
        total_guard_activity = 0

        for user in all_users.keys():
            division = all_users[user]["division"]
            total_xp += all_users[user]["xp"]

            if division != "PW":
                member_count -=1

            if all_users[user]["rank"] == "Provost Marshal":
                pm = all_users[user]["username"] 
            
            try:
                total_events += (hr_users[user]["tryouts"] + hr_users[user]["events"] + hr_users[user]["phases"] +hr_users[user]["inspections"] + hr_users[user]["joint_events"])
                total_dep_points += hr_users[user]["courses"]
            except KeyError:
                try:
                    total_activity += lr_users[user]["activity"]
                    total_guard_activity += lr_users[user]["time_guarded"]
                    total_attended_events += lr_users[user]["events_attended"]
                except KeyError:
                    pass

        data = {
                "pm": pm,
                "member_count": member_count,
                "hr_count": hr_count,
                "lr_count": lr_count,
                "total_xp": total_xp,
                "total_events": total_events,
                "total_dep_points": total_dep_points,
                "total_attended_events": total_attended_events,
                "total_activity": total_activity,
                "total_guard_activity": total_guard_activity,
            } 


        info_embed = embedBuilder.build_regiment_info(data)

        await ctx.send(embed=info_embed)



async def setup(bot):
    await bot.add_cog(UtilityCog(bot))
