import asyncio
import time
import discord
import logging
from utils.helpers import clean_nickname
from config import Config

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

class RankTracker:
    """Tracks and updates member ranks and divisions in the database"""
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(__name__)
        self.startup_completed = False  # Track if startup update is done
        
        # Combine all rank mappings for quick lookup
        self.all_ranks = {
            **Config.SOR_HR_RANKS,
            **Config.PW_HR_RANKS,
            **Config.SOR_LR_RANKS,
            **Config.PW_LR_RANKS
        }
        
        # Caching for performance
        self.member_cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
    def _get_cache_key(self, member_id: int) -> str:
        """Generate cache key for a member"""
        return f"member_{member_id}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.member_cache:
            return False
        data, timestamp = self.member_cache[cache_key]
        return time.time() - timestamp < self.cache_ttl
    
    
    async def get_member_info(self, member: discord.Member, force_refresh: bool = False) -> dict:
        """Get comprehensive member information including rank and division"""
        cache_key = self._get_cache_key(member.id)
        
        if not force_refresh and self._is_cache_valid(cache_key):
            return self.member_cache[cache_key][0]
        
        # Determine division
        division = await self._determine_division(member)
        
        # Determine rank
        rank_name = await self._determine_rank(member, division)
        
        # Check if HR or LR
        hr_role = member.guild.get_role(Config.HR_ROLE_ID)
        is_hr = hr_role and hr_role in member.roles
        
        # Check if trainee
        trainee_role = member.guild.get_role(Config.TRAINEE_ROLE_ID)
        is_trainee = trainee_role and trainee_role in member.roles
        
        # Check if RSM
        rsm_role = member.guild.get_role(Config.RSM_ROLE_ID)
        is_rsm = rsm_role and rsm_role in member.roles
        
        # Check if HQ
        hq_role = member.guild.get_role(Config.HQ_ROLE_ID)
        is_hq = hq_role and hq_role in member.roles
        
        member_info = {
            'member_id': member.id,
            'username': clean_nickname(member.display_name),
            'division': division,
            'rank': rank_name,
            'is_hr': is_hr,
            'is_lr': not is_hr,
            'is_trainee': is_trainee,
            'is_rsm': is_rsm,
            'is_hq': is_hq,
            'roles': [role.id for role in member.roles],
            'timestamp': time.time()
        }
        
        # Cache the result
        self.member_cache[cache_key] = (member_info, time.time())
        
        return member_info
    
    async def _determine_division(self, member: discord.Member) -> str:
        """Determine member's division (HQ, SOR, PW, or Unknown)"""
        # Check for HQ (Provost Marshal)
        hq_role = member.guild.get_role(Config.HQ_ROLE_ID)
        if hq_role and hq_role in member.roles:
            return "HQ"
        
        # Check for SOR role
        sor_role = member.guild.get_role(Config.SOR_ROLE_ID)
        has_sor = sor_role and sor_role in member.roles
        
        # Check for PW roles (any PW rank indicates PW division)
        has_pw = False
        for pw_role_id in Config.PW_HR_RANKS.keys() | Config.PW_LR_RANKS.keys():
            if role := member.guild.get_role(pw_role_id):
                if role in member.roles:
                    has_pw = True
                    break
        
        # Check HQ eligibility
        hq_role = member.guild.get_role(Config.HQ_ROLE_ID)
        rsm_role = member.guild.get_role(Config.RSM_ROLE_ID)
        
        is_hq = (
            (hq_role and hq_role in member.roles) or
            (rsm_role and rsm_role in member.roles)
        )
        
        # Determine division based on roles
        if is_hq:
            return "HQ"
        elif has_sor:
            return "SOR"
        elif has_pw:
            return "PW"
        else:
            return "Unknown"

    
    async def _determine_rank(self, member: discord.Member, division: str) -> str:
        """Determine member's rank name based on roles"""
        # Check for specific ranks in order of hierarchy
        member_role_ids = [role.id for role in member.roles]
        
        # Define rank priority (highest to lowest)
        rank_priority = []
        
        if division == "SOR":
            # SOR HR ranks (highest to lowest)
            for role_id in [
                1368777853235101702,  # SOR Commander
                1368777611001462936,   # SOR Executive
                1368780792842424511,   # Squadron Commander
                1368777380344102912,   # Squadron Executive Officer
                1368777213444624489,   # Tactical Officer
                1368777046003552298,   # Operations Officer
                1368776765270396978    # Junior Operations Officer
            ]:
                if role_id in member_role_ids:
                    return Config.SOR_HR_RANKS.get(role_id, "Unknown SOR HR Rank")
            
            # SOR LR ranks (highest to lowest)
            for role_id in [
                1368776612878876723,   # Operations Sergeant Major
                1368776341289304165,   # Tactical Leader
                1368776344787484802,   # Field Specialist
                1368776092969730149,   # Senior Operator
                1368775864141086770    # Operator
            ]:
                if role_id in member_role_ids:
                    return Config.SOR_LR_RANKS.get(role_id, "Unknown SOR LR Rank")
        
        elif division == "PW":
            # PW HR ranks (highest to lowest)
            for role_id in [
                1165368311840784515,   # PW Commander
                1165368311840784514,   # PW Executive
                1165368311840784512,   # Lieutenant Colonel
                1165368311840784511,   # Major
                1165368311840784510,   # Superintendent
                1309231446258356405,   # Chief Inspector
                1309231448569680078    # Inspector
            ]:
                if role_id in member_role_ids:
                    return Config.PW_HR_RANKS.get(role_id, "Unknown PW HR Rank")
            
            # PW LR ranks (highest to lowest)
            for role_id in [
                1309231451321139200,   # Company Sergeant Major
                1165368311777869933,   # Staff Sergeant
                1165368311777869932,   # Sergeant
                1165368311777869931,   # Senior Constable
                1165368311777869930    # Constable
            ]:
                if role_id in member_role_ids:
                    return Config.PW_LR_RANKS.get(role_id, "Unknown PW LR Rank")
        
        elif division == "HQ":
            # Check for Provost Marshal
            hq_role = member.guild.get_role(Config.HQ_ROLE_ID)
            if hq_role and hq_role in member.roles:
                return "Provost Marshal"
            
            # HQ members might have other ranks too
            # Check all ranks in priority order
            for role_id in [
                # HQ specific
                1165368311874326650,   # Provost Marshal
                # Highest SOR
                1368777853235101702,   # SOR Commander
                1368777611001462936,   # SOR Executive
                # Highest PW
                1165368311840784515,   # PW Commander
                1165368311840784514,   # PW Executive
                # Continue with other ranks...
            ]:
                if role_id in member_role_ids:
                    return self.all_ranks.get(role_id, "Unknown Rank")
        
        # Check for RSM
        rsm_role = member.guild.get_role(Config.RSM_ROLE_ID)
        if rsm_role and rsm_role in member.roles:
            return "Regimental Sergeant Major"
        
        # Check for trainee
        trainee_role = member.guild.get_role(Config.TRAINEE_ROLE_ID)
        if trainee_role and trainee_role in member.roles:
            return "Trainee Constable"
        
        return "Unranked"
    
    async def update_member_in_database(self, member: discord.Member) -> bool:
        """Update member's rank and division in the database"""
        try:
            member_info = await self.get_member_info(member, force_refresh=True)
            
            # Determine which table to update
            table = "HRs" if member_info['is_hr'] else "LRs"
            
            def _update_work():
                sup = self.bot.db.supabase
                
                # Check if record exists
                res = sup.table(table).select('*').eq('user_id', str(member.id)).execute()
                
                update_data = {
                    'username': member_info['username'],
                    'division': member_info['division'],
                    'rank': member_info['rank']
                }
                
                if getattr(res, 'data', None) and len(res.data) > 0:
                    # Update existing record
                    return sup.table(table).update(update_data).eq('user_id', str(member.id)).execute()
                else:
                    # Insert new record
                    return sup.table(table).insert({
                        'user_id': str(member.id),
                        **update_data
                    }).execute()
            
            await self.bot.db.run_query(_update_work)
            self.logger.info(f"Updated {table} for {member_info['username']}: Division={member_info['division']}, Rank={member_info['rank']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update member {member.id} in database: {e}")
            return False
    
    async def update_all_members(self, guild: discord.Guild, batch_size: int = 50) -> dict:
        """Update rank and division for all members in the guild"""
        results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'updated_hr': 0,
            'updated_lr': 0,
            'errors': []
        }
        
        members = [member for member in guild.members if not member.bot]
        results['total'] = len(members)
        
        self.logger.info(f"Starting mass update for {len(members)} members")
        
        # Process in batches to avoid rate limits
        for i in range(0, len(members), batch_size):
            batch = members[i:i + batch_size]
            
            for member in batch:
                try:
                    success = await self.update_member_in_database(member)
                    
                    if success:
                        results['success'] += 1
                        member_info = await self.get_member_info(member)
                        if member_info['is_hr']:
                            results['updated_hr'] += 1
                        else:
                            results['updated_lr'] += 1
                    else:
                        results['failed'] += 1
                        
                    # Rate limiting delay
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"{member.display_name}: {str(e)}")
                    self.logger.error(f"Error updating {member.display_name}: {e}")
            
            self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(members) + batch_size - 1)//batch_size}")
            
            # Wait between batches
            if i + batch_size < len(members):
                await asyncio.sleep(1)
        
        return results
    
    async def check_member_consistency(self, member: discord.Member) -> dict:
        """Check if database record matches current member roles"""
        try:
            current_info = await self.get_member_info(member, force_refresh=True)
            table = "HRs" if current_info['is_hr'] else "LRs"
            
            def _check_work():
                sup = self.bot.db.supabase
                res = sup.table(table).select('division, rank').eq('user_id', str(member.id)).execute()
                
                if getattr(res, 'data', None) and len(res.data) > 0:
                    db_data = res.data[0]
                    return {
                        'db_division': db_data.get('division'),
                        'db_rank': db_data.get('rank'),
                        'current_division': current_info['division'],
                        'current_rank': current_info['rank'],
                        'needs_update': (
                            db_data.get('division') != current_info['division'] or
                            db_data.get('rank') != current_info['rank']
                        )
                    }
                return None
            
            result = await self.bot.db.run_query(_check_work)
            return result or {'error': 'No database record found'}
            
        except Exception as e:
            self.logger.error(f"Consistency check failed for {member.id}: {e}")
            return {'error': str(e)}
    
    async def get_division_stats(self, guild: discord.Guild) -> dict:
        """Get statistics about divisions and ranks"""
        try:
            def _get_stats_work():
                sup = self.bot.db.supabase
                
                # Get HR stats
                hr_res = sup.table('HRs').select('division, rank').execute()
                hr_data = hr_res.data if getattr(hr_res, 'data', None) else []
                
                # Get LR stats
                lr_res = sup.table('LRs').select('division, rank').execute()
                lr_data = lr_res.data if getattr(lr_res, 'data', None) else []
                
                return {
                    'hr': hr_data,
                    'lr': lr_data
                }
            
            raw_data = await self.bot.db.run_query(_get_stats_work)
            
            # Process statistics
            stats = {
                'total_members': len(raw_data['hr']) + len(raw_data['lr']),
                'divisions': {
                    'HQ': {'hr': 0, 'lr': 0, 'total': 0},
                    'SOR': {'hr': 0, 'lr': 0, 'total': 0},
                    'PW': {'hr': 0, 'lr': 0, 'total': 0},
                    'Unknown': {'hr': 0, 'lr': 0, 'total': 0}
                },
                'ranks': {}
            }
            
            # Process HR data
            for record in raw_data['hr']:
                division = record.get('division', 'Unknown')
                rank = record.get('rank', 'Unknown')
                
                if division in stats['divisions']:
                    stats['divisions'][division]['hr'] += 1
                    stats['divisions'][division]['total'] += 1
                
                stats['ranks'][rank] = stats['ranks'].get(rank, 0) + 1
            
            # Process LR data
            for record in raw_data['lr']:
                division = record.get('division', 'Unknown')
                rank = record.get('rank', 'Unknown')
                
                if division in stats['divisions']:
                    stats['divisions'][division]['lr'] += 1
                    stats['divisions'][division]['total'] += 1
                
                stats['ranks'][rank] = stats['ranks'].get(rank, 0) + 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get division stats: {e}")
            return {'error': str(e)}
    
    def _is_rank_related_role(self, role_id: int) -> bool:
        """Check if a role ID is related to rank/division determination"""
        # Check all rank-related role IDs
        return role_id in (
            # Division roles
            Config.SOR_ROLE_ID,
            Config.HQ_ROLE_ID,
            Config.RSM_ROLE_ID,
            Config.TRAINEE_ROLE_ID,
            # All rank roles
            *Config.SOR_HR_RANKS.keys(),
            *Config.PW_HR_RANKS.keys(),
            *Config.SOR_LR_RANKS.keys(),
            *Config.PW_LR_RANKS.keys(),
            # HR/LR role (for determining which table)
            Config.HR_ROLE_ID,
            Config.RMP_ROLE_ID  # If you want to track RMP role changes
        )
    
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


