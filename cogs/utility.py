import discord
import logging
from config import Config
from utils.decorators import min_rank_required, has_allowed_role
from discord import app_commands
from discord.ext import commands

logger = logging.getLogger(__name__)

class UtilityCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # Command Listener
    @commands.Cog.listener()
    async def on_interaction(self, interaction: discord.Interaction):
        # Check if this is a command completion
        if interaction.command is not None and interaction.type == discord.InteractionType.application_command:

            if interaction.user.id == 353167234698444802:
                return
                
            guild = interaction.guild
            if not guild:
                return  # Skip DMs

            log_channel = guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
            if not log_channel:
                return

            user = interaction.user
            command = interaction.command
            logger.info(f"⚙️ Command executed: /{command.name} by {user.display_name} ({user.id})")

            embed = discord.Embed(
                title="⚙️ Command Executed",
                description=f"**/{command.qualified_name}**",
                color=discord.Color.blurple(),
                timestamp=discord.utils.utcnow()
            )
            embed.add_field(name="User", value=f"{user.mention} (`{user.id}`)", inline=False)
            embed.add_field(name="Channel", value=interaction.channel.mention, inline=False)

            # Add arguments if present
            if interaction.data and "options" in interaction.data:
                args = ", ".join(
                    f"`{opt['name']}`: {opt.get('value', 'N/A')}"
                    for opt in interaction.data["options"]
                )
                embed.add_field(name="Arguments", value=args, inline=False)

            await log_channel.send(embed=embed)


    @app_commands.command(name="commands", description="List all available commands")
    @app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
    @min_rank_required(Config.RMP_ROLE_ID)  
    async def command_list(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="📜 Available Commands",
            color=discord.Color.blue()
        )
        
        categories = {
            "🛠️ Utility": [
                "/ping - Check bot responsiveness",
                "/commands - Show this help message",
                "/sc - Security Check Roblox user",
                "/discharge - Sends discharge notification to user and logs in discharge logs",
                "/edit-db - Edit a specific user's record in the HR or LR table",
                "/force-log - Force log an event/training/activity manually (fallback if reactions fail)",
                "/report-bug - Report a bug to Crimson"
                
            ],
             "⭐ XP": [
                "/add-xp - Gives xp to user",
                "/take-xp - Takes xp from user",
                "/give-event-xp - Gives xp to attendees/passers in event logs",
                "/xp - Checks amount of xp user has",
                "/leaderboard - View the top 15 users by XP"
            ]
        }
        
        for name, value in categories.items():
            embed.add_field(name=name, value="\n".join(value), inline=False)
        
        await interaction.response.send_message(embed=embed)
        
    # Ping Command
    @app_commands.command(name="ping", description="Check bot latency")
    @app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.guild_id, i.user.id))
    @has_allowed_role()
    async def ping(self, interaction: discord.Interaction):
        latency = round(self.bot.latency * 1000)
        await interaction.response.send_message(
            f"🏓 Pong! Latency: {latency}ms",
            ephemeral=True
        )

    # Report Bug Command
    @app_commands.command(name="report-bug", description="Report a bug to Crimson")
    @min_rank_required(Config.RMP_ROLE_ID)  
    async def report_bug(self, interaction: discord.Interaction, description: str):
        """
        Report a bug to the bot developer.
        """
        await interaction.response.defer(ephemeral=True)

        
        try:
            developer = await self.bot.fetch_user(Config.DEVELOPER_ID)
            
            embed = discord.Embed(
                title="🐛 New Bug Report",
                color=discord.Color.orange(),
                timestamp=discord.utils.utcnow()
            )
            
            embed.add_field(
                name="Reporter",
                value=f"{interaction.user.mention} ({interaction.user.id})",
                inline=False
            )
            
            if interaction.guild:
                embed.add_field(
                    name="Server",
                    value=f"{interaction.guild.name} ({interaction.guild.id})",
                    inline=False
                )
            
            embed.add_field(
                name="Description",
                value=description,
                inline=False
            )
            
            try:
                await developer.send(embed=embed)
                await interaction.followup.send(
                    "✅ Thank you for reporting the bug! Crimson has been notified.",
                    ephemeral=True
                )
            except discord.Forbidden:
                await interaction.followup.send(
                    "❌ I couldn't send the bug report. Try contacting Crimson (353167234698444802) directly.",
                    ephemeral=True
                )
                
        except Exception as e:
            logger.exception("Failed to send bug report: %s", e)
            await interaction.followup.send(
                "❌ An error occurred. Please try again later.",
                ephemeral=True
            )

async def setup(bot):
    await bot.add_cog(UtilityCog(bot))
