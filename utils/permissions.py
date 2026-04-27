import asyncio
import logging

logger = logging.getLogger(__name__)

class PermissionsCache:
    """In-memory cache for command permisssions"""
    def __init__(self, bot):
        self.bot = bot
        self._cache = {}  
        self._lock = asyncio.Lock()
        self._initialised = False
        self._table = 'command_permissions'
        
    async def initialise(self):
        """Load all role permissions from database on startup"""
        if self._initialised:
            return
            
        async with self._lock:
            try:
                groups = await self._load_all_from_database()

                if not groups:
                    logger.error("Failed to initialise command permissions as there is none")
                    return
                    
                for group in groups:
                    group_type = group["group_type"]
                    allowed_roles = group["allowed_roles"]
                    self._cache[group_type] = allowed_roles
                
                logger.info(f"Loaded {len(self._cache)} out of 5 category permissions into cache.")
                self._initialised = True
            except Exception as e:
                logger.error("Failed to initialise command permissions in to cache: %s", e)
    
    async def _load_all_from_database(self):
        try:
            result = await self.bot.db.supabase.table(self._table)\
                    .select("*")\
                    .execute()

            if result and result.data:
                return result.data
            
            return None
        except Exception as e:
            logger.error("Failed to load permissions from database: %s", e)
            return None 

    async def _load_from_database(self, group_type: str):
        """Load permissions for a specific type of group of commands from database"""
        try:
            result = await self.bot.db.supabase.table(self._table) \
                .select('allowed_roles') \
                .eq('group_type', group_type) \
                .execute()
            
            if result.data:
                return {
                    'allowed_roles': result.data[0].get('allowed_roles', []),
                }
            return {'allowed_roles': []} 
        except Exception as e:
            logger.error("Error loading %s from database: %s", group_type, e)
            return None


    async def get(self, group_type: str, refresh: bool = False):
        """
        Get a group type permissions from from cache.
        If refresh=True or not in cache, load from database.
        """
        async with self._lock:
            if refresh or group_type not in self._cache:
                data = await self._load_from_database(group_type)
                if data:
                    self._cache[group_type] = data
            
            return self._cache.get(group_type)

    async def update(self, group_type: str, allowed_ids: list[int]) -> bool:
        try:
            allowed_ids = list(set(allowed_ids))
            await self.bot.db.supabase.table(self._table) \
                    .update({'allowed_roles': allowed_ids}) \
                    .eq('group_type', group_type).execute()
            
            async with self._lock:
                self._cache[group_type] = {'allowed_roles': allowed_ids}
            
            return True
        except Exception as e:
            logger.error("Error updating permissions for %s: %s", group_type, e)
            return False
