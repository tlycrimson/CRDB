import re
import discord
from typing import Optional, Tuple

# Function: Cleans the display name for their ROBLOX username
def clean_nickname(nickname: str) -> str:
    """Remove tags like [INS] from nicknames and clean whitespace"""
    if not nickname:  # Handle None or empty string
        return "Unknown"
    cleaned = re.sub(r'\[.*?\]', '', nickname).strip()
    return cleaned or nickname  # Fallback to original if empty after cleaning

# XP Tier configuration
TIERS = [
    ("🌟 Platinum", 800),
    ("💎 Diamond", 400),
    ("🥇 Gold", 200),
    ("🥈 Silver", 135),
    ("🥉 Bronze", 100),
    ("⚪ Unranked", 0),
]

# Pre-sort tiers from highest to lowest threshold
TIERS_SORTED = sorted(TIERS, key=lambda x: x[1], reverse=True)

def get_tier_info(xp: int) -> Tuple[str, int, Optional[int]]:
    """Return (tier_name, current_threshold, next_threshold)"""
    # Use pre-sorted tiers
    for i, (name, threshold) in enumerate(TIERS_SORTED):
        if xp >= threshold:
            # Get next tier's threshold (if exists)
            next_threshold = TIERS_SORTED[i-1][1] if i > 0 else None
            return name, threshold, next_threshold
    
    # Fallback (should never reach here with 0 threshold)
    return "⚪ Unranked", 0, TIERS_SORTED[-2][1] if len(TIERS_SORTED) > 1 else 100
    
def get_tier_name(xp: int) -> str:
    """Return just the tier name"""
    tier_name, _, _ = get_tier_info(xp)
    return tier_name

# Progress bar
def make_progress_bar(xp: int, current: int, next_threshold: Optional[int]) -> str:
    if not next_threshold:  # Already at max tier
        return "🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨 (MAX)"

    total_needed = next_threshold - current
    gained = xp - current
    filled = int((gained / total_needed) * 10)
    filled = min(filled, 10) #Safety guard | In case above produces a value>10

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


