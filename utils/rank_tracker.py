import discord
import logging
from utils.helpers import clean_nickname
from config import Config

class RankTracker:
    """Tracks and updates member ranks and divisions in the database"""
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(__name__)
        self.startup_completed = False  
        self._rank_role_ids = self._build_rank_role_set()

    def _build_rank_role_set(self) -> set:
        return {
            *Config.PW_LR_IDS, *Config.PW_HR_IDS,
            Config.HQ_ROLE_ID,
        }
        
    async def get_member_info(self, member: discord.Member, force_refresh: bool = False) -> dict:
        """Get comprehensive member information including rank and division"""
        user_id = str(member.id)

        if not force_refresh:
            cached_data = self.bot.db._user_cache.get(user_id)
            if cached_data:
                return cached_data
        
        member_role_ids = [role.id for role in member.roles]

        is_hr = Config.HR_ROLE_ID in member_role_ids

        division, rank = await self._determine_division_rank(member_role_ids, is_hr)
        
        member_info = {
            'username': clean_nickname(member.display_name),
            'division': division,
            'rank': rank,
        }
        
        self.bot.db._user_cache[user_id] = member_info
        return member_info
    
    async def _determine_division_rank(self, member_role_ids: list, is_hr: bool) -> tuple:
        """Determine member's division (HQ, SOR, PW, or Unknown)"""
        division = "PW"

        if Config.HQ_ROLE_ID in member_role_ids:
            return "HQ", "Headquarters"
    
        if Config.PW_ROLE_ID in member_role_ids:
            rank = await self._determine_rank(member_role_ids, is_hr)
            return division, rank
   
        return "Unknown", "Unknown"

    
    async def _determine_rank(self, member_role_ids: list, is_hr: bool) -> str:
        if is_hr:
            for role_id in Config.PW_HR_IDS:
                    if role_id in member_role_ids:
                        return Config.PW_HR_RANKS.get(role_id, "Unknown")
        else:
            for role_id in Config.PW_LR_IDS:
                if role_id in member_role_ids:
                    return Config.PW_LR_RANKS.get(role_id, "Unknown")

        return "Unknown"
    
    async def update_member_in_database(self, member: discord.Member) -> bool:
        """Update member's rank and division in the database"""
        try:
            member_info = await self.get_member_info(member, force_refresh=True)
            username = member_info['username']
            division = member_info['division']
            rank = member_info['rank']
            
            success = await self.bot.db.update_rank_div(member.id, username, rank, division)
            if success:
                self.logger.info(f"Updated info for %s: Division=%s, Rank=%s", username, division, rank)
            return True
        except Exception as e:
            self.logger.error(f"Failed to update member {member.id} in database: {e}")
            return False
    
    def _is_rank_related_role(self, role_id: int) -> bool:
        """Check if a role ID is related to rank/division determination"""
        return role_id in self._rank_role_ids
    
    def _roles_changed_affect_rank(self, before_roles: list, after_roles: list) -> bool:
        """Check if role changes affect rank/division determination"""
        before_set = set(before_roles)
        after_set = set(after_roles)
        
        # Get roles that were added or removed
        added_roles = after_set - before_set
        removed_roles = before_set - after_set
        
        # Check if any added/removed roles are rank-related
        for role_id in added_roles.union(removed_roles):
            if self._is_rank_related_role(role_id):
                return True
        
        return False


