import discord
import asyncio
import logging
from config import Config
from datetime import datetime, timezone
from discord.ext import commands
from discord import app_commands
from utils.decorators import has_bg_role

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)

logger = logging.getLogger(__name__)

class ScCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.BGROUP_IDS = {Config.SI, Config.UBA, Config.PARAS, Config.HSD, Config.AAC, Config.RAR, Config.RTR, Config.UKSF}
        self.REQUIREMENTS = {'age': 90, 'friends': 7, 'groups': 5, 'badges': 120}
        self.DISCORD_RETRY_DELAY = 1.5

    def create_progress_bar(self, percentage: float, meets_req: bool) -> str:
        if percentage == 0:
            return ("▰") + ("▱"*9) # Due to font size on mobile

        filled = min(10, round(percentage / 10))
        return ("▰" * filled) + ("▱" * (10 - filled))

    async def animate_loading(self, message):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while True:
            try:
                embed = discord.Embed(
                    description=f"{frames[i % len(frames)]} Fetching Roblox data, please wait...",
                    color=discord.Color.blurple()
                ).set_footer(text="Try using their ID if it takes too long.")
                await message.edit(embed=embed)
                i += 1
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break

    @commands.hybrid_command(
            name="security-check", 
            aliases=["sc"],
            usage="<rbx username/id>",
            description="Security check a Roblox user"
    )
    @app_commands.describe(user="The Roblox username or userID to check")
    @app_commands.checks.cooldown(rate=1, per=5.0)
    @has_bg_role()
    async def sc(self, ctx: commands.Context, user: str):

        try:
            if ctx.interaction:
                await ctx.interaction.response.defer()

            initial_embed = discord.Embed(
                description="⠋ Fetching Roblox data, please wait...",
                color=discord.Color.blurple()
            )
            initial_embed.set_author(
                name="Security Check",
                icon_url=Config.USER_ICON
            )
            initial_embed.set_footer(text="This may take a few seconds.")

            loading_message = await  ctx.send(embed=initial_embed)
            
            animation_task = asyncio.create_task(self.animate_loading(loading_message))

            def error_embed(title: str, description: str) -> discord.Embed:
                embed = discord.Embed(description=description, color=discord.Color.red())
                embed.set_author(name=title, icon_url=Config.CANCEL_URL)
                return embed

            try:
                user_id = None
                if user.isdigit():
                    user_id = int(user)
                else:
                    user_id = await asyncio.wait_for(
                        self.bot.roblox.get_user_id(user),
                        timeout=15.0
                    )

                    if not user_id:
                        animation_task.cancel()
                        await asyncio.sleep(0.1)
                        return await loading_message.edit(
                            embed=error_embed("User Not Found", f"Could not find a Roblox user for **{user}**.")
                        )

                embed = await asyncio.wait_for(
                    self.compile_sc_embed(user_id),
                    timeout=30.0
                )

            except asyncio.TimeoutError:
                animation_task.cancel()
                await asyncio.sleep(0.1)
                return await loading_message.edit(
                    embed=error_embed(
                        "Request Timed Out",
                        "Roblox data took too long to load. Please try again in a moment."
                    )
                )

            animation_task.cancel()
            await asyncio.sleep(0.1)
            await loading_message.edit(embed=embed)

        except Exception as e:
            logger.exception("Unexpected error in /sc command: %s", e)
            await ctx.send("```❌ An unexpected error occurred.```", ephemeral=True)

    async def fetch_data (self, user_id: int):
        return await self.bot.roblox.fetch_sc_data(user_id, group_id=Config.BRITISH_ARMY_GROUP_ID)
        
    async def compile_sc_embed(self, user_id: int, data = None):
        if not data:
            data = await self.fetch_data(user_id)

        profile = data["user_info"]
        british_army_rank  = data["rank"]
        groups = data["groups"]
        friends_count = data["friends_count"]
        avatar = data["avatar_url"]
        badge_count = data["badge_count"]

        if profile is None or isinstance(profile, Exception) or not profile.get('name'):
            embed = discord.Embed(
                description="The specified Roblox user could not be found.",
                color=discord.Color.red()
            )
            embed.set_author(
                name="User Not Found",
                icon_url=Config.CANCEL_URL)

            return embed

        username = profile.get('name', 'Unknown')
        created_at = datetime.fromisoformat(profile['created'].replace('Z', '+00:00')) if profile.get('created') else None
        age_days = (datetime.now(timezone.utc) - created_at).days if created_at else 0

        british_army_rank = british_army_rank if british_army_rank else "Unknown"
        groups_count = len(groups) if not isinstance(groups, Exception) else 0
        warning = "Inventory is private" if badge_count == -1 else ""
        badge_count = 0 if badge_count<0 else badge_count

        metrics = {
            'age':{'value': age_days,'percentage': min(100, (age_days/self.REQUIREMENTS['age'])* 100),
            'meets_req': age_days>= self.REQUIREMENTS['age']},
            'friends': {'value': friends_count, 'percentage': min(100, (friends_count/self.REQUIREMENTS['friends'])*100),'meets_req': friends_count >= self.REQUIREMENTS['friends']},
            'groups':  {'value': groups_count, 'percentage': min(100, (groups_count/self.REQUIREMENTS['groups'])*100),  'meets_req': groups_count  >= self.REQUIREMENTS['groups']},
            'badges':  {'value': badge_count, 'percentage': min(100, (badge_count/max(1, self.REQUIREMENTS['badges']))*100), 'meets_req': badge_count>=self.REQUIREMENTS['badges']}
        }
        
        flagged_groups = []
        if groups:
            flagged_groups = [
                f"• {item['group']['name']}"
                for item in groups
                if item.get('group') and item['group'].get('id') in self.BGROUP_IDS
            ]
        
        fg_value= "YES" if flagged_groups else "CLEAR"

        criminal_records = await self.bot.db.get_criminal_record(user_id, username)

        if criminal_records:
            record_links = []
            char_count = 0
            
            for i, record in enumerate(criminal_records, start=1):
                entry = f"[Record {i}]({record["record"]})"
                
                record_links.append(entry)
                char_count += len(entry)
            
            records = ", ".join(record_links)
        else:
            records = "CLEAN"

        passed = all(m['meets_req'] for m in metrics.values()) and not bool(flagged_groups) and not criminal_records
        
        if warning:
            embed = discord.Embed(color=discord.Color.orange())
        else:
            embed = discord.Embed(color=discord.Color.green() if passed else discord.Color.red())
        
        embed.add_field(name="ROBLOX ID:", value=user_id, inline=True)
        embed.add_field(name="CREATED:", value=f"<t:{int(created_at.timestamp())}:R>", inline=True)
        embed.add_field(name="STATUS:", value=f"{"PASSED" if passed else "FAILED"}", inline=True)
        embed.add_field(name="ARMY RANK:", value=british_army_rank, inline=True)
        embed.add_field(name="CRIMINAL RECORD:", value=records, inline=True)
        embed.add_field(name="FLAGGED GROUPS:", value=fg_value, inline=True)

        for name, metric in metrics.items():
            status_icon = "✅" if metric['meets_req'] else "❌"
            progress_bar = self.create_progress_bar(metric['percentage'], metric['meets_req'])
            field_value = f"{progress_bar} {round(metric['percentage'])}%"
            if name == 'badges' and warning:
                field_value += f"\n{warning}"
                status_icon = "❔"
            embed.add_field(name=f"{name.upper()}: {status_icon}", value=field_value, inline=True)
        
        if flagged_groups:
            embed.add_field(name="FLAGGED GROUP LISTS:", value="\n".join(flagged_groups), inline=False)
        embed.set_author(
            name=username,
            url=f"https://www.roblox.com/users/{user_id}/profile",
            icon_url=avatar
        )

        return embed

async def setup(bot):
    await bot.add_cog(ScCog(bot))
