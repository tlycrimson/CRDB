import re
import logging
import discord
from config import Config
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class BlacklistData:
    def __init__(self, members, reason, duration_unit, duration_amount, evidence, member):
        self.members = members
        self.reason = reason
        self.unit = duration_unit
        self.duration = duration_amount
        self.evidence = evidence
        self.issuer = member

@dataclass
class MockPayload:
    guild_id: int
    channel_id: int
    message_id: int
    user_id: int
    emoji: discord.PartialEmoji
    member: discord.Member

class ViewModalTrigger(discord.ui.View):
    def __init__(self, author: discord.User, _type: str, timeout: float = 60.0):
        super().__init__(timeout=timeout)
        self.author = author
        self._type = _type

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author.id:
            await interaction.response.send_message(
                "```⛔ You cannot respond to this confirmation.```",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Open Panel", style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self._type == "bug":
            await interaction.response.send_modal(BugModal())
        elif self._type == "suggestion":
            await interaction.response.send_modal(SuggestionModal())

        self.stop()

    async def on_timeout(self):
        self.stop()
        for child in self.children:
            child.disabled = True


class BugModal(discord.ui.Modal, title='Bug Report'):
    bug_title = discord.ui.TextInput(
            label='Title', 
            placeholder='Title of your report',
            style=discord.TextStyle.short,
    )
    
    description = discord.ui.TextInput(
           label='Description', 
           placeholder='Describe the bug and how to reproduce it if possible',
           style=discord.TextStyle.paragraph,
    )

    async def on_submit(self, interaction: discord.Interaction):
        try:
            await interaction.response.send_message("```Thank you for reporting the bug.```", ephemeral=True)
            developer = await interaction.client.fetch_user(Config.DEVELOPER_ID)
            
            bug_embed = discord.Embed(
                title=f"{self.bug_title}",
                color=discord.Color.orange(),
                timestamp=discord.utils.utcnow()
            ).set_footer(text=f"Reporter ID: {interaction.user.id}")
            
            bug_embed.set_author(name=f"Bug Report By {interaction.user.name}", icon_url=Config.BUG_ICON)
            bug_embed.add_field(
                name="Description:",
                value=f"```{self.description}```",
                inline=False
            )
            
            bug_embed.set_thumbnail(url=Config.BOT_ICON)

            try:
                await developer.send(embed=bug_embed)
            except discord.Forbidden:
                await interaction.followup.send(
                    "```❌ I couldn't send the bug report. Try contacting Crimson (353167234698444802) directly.```",
                    ephemeral=True
                )
                
        except Exception as e:
            logger.exception("Failed to send bug report: %s", e)
            await interaction.response.send_message(
                "```❌ An error occurred. Please try again later.```",
                ephemeral=True
            )

        return

class SuggestionModal(discord.ui.Modal, title='Suggestion'):
    suggestion_title = discord.ui.TextInput(
            label='Title', 
            placeholder='Title of your suggestion',
            style=discord.TextStyle.short,
    )
    
    description = discord.ui.TextInput(
           label='Description', 
           placeholder='Describe your suggestion',
           style=discord.TextStyle.paragraph,
    )

    async def on_submit(self, interaction: discord.Interaction):
        try:
            await interaction.response.send_message("```Thank you for making a suggestion.```", ephemeral=True)
            developer = await interaction.client.fetch_user(Config.DEVELOPER_ID)
            
            s_embed = discord.Embed(
                title=f"{self.suggestion_title}",
                color=discord.Color.purple(),
                timestamp=discord.utils.utcnow()
            ).set_footer(text=f"Reporter ID: {interaction.user.id}")
            
            s_embed.set_author(name=f"Suggestion Made By {interaction.user.name}", icon_url=Config.BULB_ICON)
            s_embed.add_field(
                name="Description:",
                value=f"```{self.description}```",
                inline=False
            )
            
            s_embed.set_thumbnail(url=Config.BOT_ICON)

            try:
                await developer.send(embed=s_embed)
            except discord.Forbidden:
                await interaction.followup.send(
                    "```❌ I couldn't send suggestion. Try contacting Crimson (353167234698444802) directly.```",
                    ephemeral=True
                )
                
        except Exception as e:
            logger.exception("Failed to send suggestion: %s", e)
            await interaction.response.send_message(
                "```❌ An error occurred. Please try again later.```",
                ephemeral=True
            )

        return



def clean_nickname(nickname: str) -> str:
    """Remove tags like [RANK] from nicknames and clean whitespace"""
    if not nickname: 
        return "Unknown"
    cleaned = re.sub(r'\[.*?\]', '', nickname).strip()
    return cleaned or nickname  

class ViewTrigger(discord.ui.View):
    def __init__(self, author: discord.User, timeout: float = 60.0):
        super().__init__(timeout=timeout)
        self.author = author

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author.id:
            await interaction.response.send_message(
                "```⛔ You cannot respond to this confirmation.```",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Report", style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = True
        await interaction.response.defer()
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = False
        await interaction.response.defer()
        self.stop()

    async def on_timeout(self):
        self.value = False
        self.stop()
        for child in self.children:
            child.disabled = True

# XP Tier configuration
TIERS = [
    (f"<:SRF:1489734622966321202> Immortal", 2000, discord.Color.red()),
    (f"<a:CoolBooster:1279544082301194322> Grandmaster", 1000, discord.Color.purple()),
    (f"☄️ Cosmos", 750, discord.Color.dark_red()),
    (f"🛡️ Guardian", 600, discord.Color.blue()),
    (f"🦾 Knight", 490, discord.Color.dark_embed()),
    (f"🌟 Platinum", 375, discord.Color.dark_gold()),
    (f"💚 Emerald", 300, discord.Color.dark_green()),
    (f"💎 Diamond", 200, discord.Color.dark_blue()),
    (f"🥇 Gold", 135, discord.Color.gold()),
    (f"🥈 Silver", 100, discord.Color.dark_grey()),
    (f"🗡️ Iron", 50, discord.Color.darker_grey()),
    (f"🥉 Bronze", 25, discord.Color.from_rgb(101, 67, 33)),
    (f"🧭 Novice", 10, discord.Color.light_grey()),
    (f"⚪ Unranked", 0, discord.Color.from_rgb(255, 255, 255)),
]

# Pre-sort tiers from highest to lowest threshold
TIERS_SORTED = sorted(TIERS, key=lambda x: x[1], reverse=True)

def get_tier_info(xp: int) -> Tuple[str, int, Optional[int], discord.Color]:
    """Return (tier_name, current_threshold, next_threshold, color)"""
    
    for i, (name, threshold, colour) in enumerate(TIERS_SORTED):
        if xp >= threshold:
            # i-1 works because TIERS_SORTED[0] is Immortal. 
            # If you are Immortal (i=0), there is no i-1, so next is None.
            next_threshold = TIERS_SORTED[i-1][1] if i > 0 else None
            return name, threshold, next_threshold, colour
    
    # Fallback to the last item in the list (Unranked)
    name, threshold, color = TIERS_SORTED[-1][0], TIERS_SORTED[-1][1], TIERS_SORTED[-1][2]
    next_threshold = TIERS_SORTED[-2][1] if len(TIERS_SORTED) > 1 else None
    return name, threshold, next_threshold, color

def get_tier_name(xp: int) -> str:
    """Return just the tier name"""
    tier_name, _, _, _ = get_tier_info(xp)
    return tier_name

# Progress bar
def make_progress_bar(xp: int, current: int, next_threshold: Optional[int]) -> str:
    if not next_threshold:  
        return "🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨 (MAX)"

    total_needed = next_threshold - current
    gained = xp - current
    filled = int((gained / total_needed) * 10)
    filled = min(filled, 10) 

    bar = "🟩" * filled + "⬛" * (10 - filled)
    return f"{bar} ({gained}/{total_needed} XP)"


def dict_to_embed(data: dict) -> discord.Embed:
    """Convert dictionary to Discord Embed object"""
    embed = discord.Embed(
        title=data.get('title', ''),
        description=data.get('description', ''),
        color=discord.Color.from_str(data.get('color', '#000000'))
    )
    
    if 'footer' in data:
        embed.set_footer(text=data['footer'])
    
    if 'thumbnail' in data:
        embed.set_thumbnail(url=data['thumbnail'])
    
    if 'image' in data:
        embed.set_image(url=data['image'])
    
    if 'fields' in data:
        for field in data['fields']:
            embed.add_field(
                name=field.get('name', ''),
                value=field.get('value', ''),
                inline=field.get('inline', False)
            )
    
    return embed

# Function: Given the user id and a sorted list, it returns the position of that user within the list
def get_user_rank(user_id: int, sorted_users: list) -> Optional[int]:
    """Get a user's rank position based on XP (lower number = higher rank)"""
    try:
        # Create a lookup dictionary for this specific call
        rank_lookup = {user_id_in_db: index + 1 
                      for index, (user_id_in_db, _) in enumerate(sorted_users)}
        
        user_id_str = str(user_id)
        return rank_lookup.get(user_id_str)
    except Exception:
        return None

def in_regiment(user_groups):
    if not isinstance(user_groups, list):
        return None

    regiments = [
        f"• {group['group']['name']}"
        for group in user_groups
        if group 
        and group.get('group', {}).get('id') in Config.REGIMENTS
        and group.get('role', {}).get('rank') != 1
    ]
    
    return "\n".join(regiments) if regiments else None
