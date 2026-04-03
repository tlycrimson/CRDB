import re
import asyncio
import discord
import logging
from typing import Literal
from discord import app_commands
from discord.ext import commands
from datetime import datetime, timezone, timedelta

from config import Config
from utils.decorators import min_rank_required
from utils.views import ConfirmView
from utils.roblox import RobloxAPI
from utils.helpers import clean_nickname, dict_to_embed 


logger = logging.getLogger(__name__)

class WelcomeMessageCache:
    """In-memory cache for welcome messages with multiple embed support"""
    
    def __init__(self, bot):
        self.bot = bot
        self._cache = {}  # message_type -> {"embeds": [], "metadata": {}}
        self._lock = asyncio.Lock()
        self._initialized = False
        
    async def initialize(self):
        """Load all welcome messages from database on startup"""
        if self._initialized:
            return
            
        async with self._lock:
            try:
                for msg_type in ['hr_welcome', 'rmp_welcome']:
                    data = await self._load_from_database(msg_type)
                    if data:
                        self._cache[msg_type] = data
                
                logger.info(f"✅ Loaded {len(self._cache)} welcome messages into cache")
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize welcome message cache: {e}")
                # Initialize with empty cache, will use original functions as fallback
    
    async def _load_from_database(self, message_type: str):
        """Load a specific message type from database"""
        def _work():
            result = self.bot.db.supabase.table('welcome_messages') \
                .select('*') \
                .eq('message_type', message_type) \
                .eq('is_active', True) \
                .limit(1) \
                .execute()
            
            if result.data and len(result.data) > 0:
                return {
                    'id': result.data[0]['id'],
                    'message_type': result.data[0]['message_type'],
                    'embeds': result.data[0]['embeds'],
                    'last_updated': result.data[0]['last_updated'],
                    'updated_by': result.data[0]['updated_by'],
                    'version': result.data[0].get('version', 1)
                }
            return None
        
        try:
            return await self.bot.db._run_sync(_work)
        except Exception as e:
            logger.error(f"Error loading {message_type} from database: {e}")
            return None
    
    async def get(self, message_type: str, refresh: bool = False):
        """
        Get a welcome message from cache.
        If refresh=True or not in cache, load from database.
        """
        async with self._lock:
            if refresh or message_type not in self._cache:
                data = await self._load_from_database(message_type)
                if data:
                    self._cache[message_type] = data
            
            return self._cache.get(message_type)
    
    async def update(self, message_type: str, embeds_data: list, updated_by: str):
        """
        Update welcome message with multiple embeds.
        """
        async with self._lock:
            try:
                # 1. Mark old message as inactive
                def _deactivate_old():
                    self.bot.db.supabase.table('welcome_messages') \
                        .update({'is_active': False}) \
                        .eq('message_type', message_type) \
                        .eq('is_active', True) \
                        .execute()
                
                await self.bot.db._run_sync(_deactivate_old)
                
                # 2. Get next version number
                def _get_next_version():
                    result = self.bot.db.supabase.table('welcome_messages') \
                        .select('version') \
                        .eq('message_type', message_type) \
                        .order('version', desc=True) \
                        .limit(1) \
                        .execute()
                    
                    if result.data and len(result.data) > 0:
                        return result.data[0]['version'] + 1
                    return 1
                
                next_version = await self.bot.db._run_sync(_get_next_version)
                
                # 3. Insert new active message
                def _insert_new():
                    new_message = {
                        'message_type': message_type,
                        'version': next_version,
                        'embeds': embeds_data,
                        'updated_by': updated_by,
                        'is_active': True
                    }
                    
                    result = self.bot.db.supabase.table('welcome_messages') \
                        .insert(new_message) \
                        .execute()
                    
                    return result.data[0] if result.data else None
                
                new_message = await self.bot.db._run_sync(_insert_new)
                
                if new_message:
                    # 4. Update cache
                    self._cache[message_type] = {
                        'id': new_message['id'],
                        'message_type': new_message['message_type'],
                        'embeds': new_message['embeds'],
                        'last_updated': new_message['last_updated'],
                        'updated_by': new_message['updated_by'],
                        'version': new_message['version']
                    }
                    
                    logger.info(f"✅ Updated {message_type} welcome message with {len(embeds_data)} embeds")
                    return True, self._cache[message_type]
                else:
                    logger.error(f"Failed to insert new {message_type} message")
                    return False, None
                    
            except Exception as e:
                logger.error(f"Error updating welcome message {message_type}: {e}")
                return False, None


class WelcomeCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.welcome_cache = WelcomeMessageCache(bot)
        self.ctx_menu = app_commands.ContextMenu(
                name="Preview Welcome For",
                callback=self.preview_welcome_context
        )
        self.bot.tree.add_command(self.ctx_menu)

    # HR Welcome Message
    async def send_hr_welcome(self, member: discord.Member):
        """Send HR welcome message using cached template"""
        if not (welcome_channel := member.guild.get_channel(Config.HR_CHAT_CHANNEL_ID)):
            logger.warning("HR welcome channel not found!")
            return
        
        # Get message data from cache
        message_data = await self.welcome_cache.get('hr_welcome')
        
        if not message_data or 'embeds' not in message_data:
            logger.error("No HR welcome message found in cache! Using fallback.")
            await self._send_fallback_hr_welcome(member, welcome_channel)
            return
        
        try:
            # Create Discord embed objects
            discord_embeds = []
            for embed_data in message_data['embeds']:
                embed = dict_to_embed(embed_data)
                discord_embeds.append(embed)
            
            await welcome_channel.send(content=member.mention, embeds=discord_embeds)
            logger.info(f"✅ Sent HR welcome with {len(discord_embeds)} embeds to {member.display_name}")
            
        except Exception as e:
            logger.error(f"Failed to send HR welcome: {e}")
            await self._send_fallback_hr_welcome(member, welcome_channel)

    async def _send_fallback_hr_welcome(self,member: discord.Member, channel):
        """Fallback HR welcome if cache fails"""
        embed = discord.Embed(
            title="🎉 Welcome to the HR Team!",
            description=(
                f"{member.mention}\n\n"
                "**Please note the following:**\n"
                "• Request for document access in [HR Documents](https://discord.com/channels/1165368311085809717/1165368317532438646).\n"
                "• You are exempted from quota this week only - you start next week ([Quota Info](https://discord.com/channels/1165368311085809717/1206998095552978974)).\n"
                "• Uncomplete quota = strike.\n"
                "• One failed tryout allowed if your try quota portion ≥2.\n"
                "• Ask for help anytime - we're friendly!\n"
                "• Are you Lieutenant+ in BA? Apply for the Education Department!\n"
                "• Are you Captain+ in BA? Apply for both departments: [Applications](https://discord.com/channels/1165368311085809717/1165368316970405916)."
            ),
            color=discord.Color.gold(),
            timestamp=datetime.now(timezone.utc)
        )
        embed.set_footer(text="We're excited to have you on board!")
        await channel.send(content=member.mention, embed=embed)
        

    # RMP Welcome Message
    async def send_rmp_welcome(self, member: discord.Member):
        """Send RMP welcome message with multiple embeds using cached template"""
        
        # Get message data from cache (no database hit)
        message_data = await self.welcome_cache.get('rmp_welcome')
        
        if not message_data or 'embeds' not in message_data:
            logger.error("No RMP welcome message found in cache! Using original function.")
            # Fallback to original function
            await self._send_original_rmp_welcome(member)
            return
        
        try:
            # Create Discord embed objects from cached data
            discord_embeds = []
            for embed_data in message_data['embeds']:
                embed = dict_to_embed(embed_data)
                discord_embeds.append(embed)
            
            # Send all embeds
            try:
                await member.send(embeds=discord_embeds)
                logger.info(f"✅ Sent {len(discord_embeds)} RMP welcome embeds to {member.display_name}")
            except discord.Forbidden:
                # Try public channel as fallback
                if welcome_channel := member.guild.get_channel(722002957738180620):
                    await welcome_channel.send(content=member.mention, embeds=discord_embeds)
                    logger.info(f"✅ Sent RMP welcome to {member.display_name} in public channel")
            
        except Exception as e:
            logger.error(f"Failed to send RMP welcome: {e}")
            # Fallback to original
            await self._send_original_rmp_welcome(member)

    # Keep your original RMP welcome function but rename it
    async def _send_original_rmp_welcome(self, member: discord.Member):
        """Original RMP welcome function as fallback"""
        embed1 = discord.Embed(
            title="👮| Welcome to the Royal Military Police",
            description="Congratulations on passing your security check, you're officially a TRAINING member of the police force. Please be sure to read the information found below.\n\n> ** 1.** Make sure to read all of the rules found in <#1165368313925353580> and in the brochure found below.\n\n> **2.** You **MUST** read the RMP main guide and MSL before starting your duties.\n\n> **3.** You can't use your L85 unless you are doing it for Self-Militia or enforcing the PD rules. (Self-defence)\n\n> **4.** Make sure to follow the Chain Of Command. 2nd Lieutenant > Lieutenant > Captain > Major > Lieutenant Colonel > Colonel > Brigadier > Major General\n\n> **5.** For phases, you may wait for one to be hosted in <#1207367013698240584> or request the phase you need in <#1270700562433839135>.\n\n> **6.** All the information about the Defence School of Policing and Guarding is found in both <#1237062439720452157> and <#1207366893631967262>\n\n> **7.** Choose your timezone here https://discord.com/channels/1165368311085809717/1165368313925353578\n\n**Besides that, good luck with your phases!**",
            color=discord.Color.from_str("#330000") 
        )

        special_embed = discord.Embed(
            title="Special Roles",
            description="> Get your role pings here <#1196085670360404018> and don't forget the Game Night role RMP always hosts fun events, don't miss out!",
            color=discord.Color.from_str("#330000")
        )
        
        embed2 = discord.Embed(
            title="Trainee Constable Brochure",
            color=discord.Color.from_str("#660000")
        )
        
        embed2.add_field(
            name="**TOP 5 RULES**",
            value="> **1**. You **MUST** read the RMP main guide and MSL before starting your duties.\n> **2**. You **CANNOT** enforce the MSL. Only the Parade Deck (PD) rules **AFTER** you pass your phase 1.\n> **3**. You **CANNOT** use your bike on the PD or the pavements.\n> **4**. You **MUST** use good spelling and grammar to the best of your ability.\n> **5**. You **MUST** remain mature and respectful at all times.",
            inline=False
        )
        
        embed2.add_field(
            name="**WHO'S ALLOWED ON THE PD AT ALL TIMES?**",
            value="> ↠ Royal Army Medical Corps,\n> ↠ Royal Military Police,\n> ↠ Intelligence Corps.\n> ↠ Royal Family.",
            inline=False
        )
        
        embed2.add_field(
            name="**WHO'S ALLOWED ON THE PD WHEN CARRYING OUT THEIR DUTIES?**",
            value="> ↠ United Kingdom Special Forces,\n> ↠ Grenadier Guards,\n> ↠ Foreign Relations,\n> ↠ Royal Logistic Corps,\n> ↠ Adjutant General's Corps,\n> ↠ High Ranks, RSM, CSM and ASM hosting,\n> ↠ Regimental personnel watching one of their regiment's events inside Pad area.",
            inline=False
        )
        
        embed2.add_field(
            name="**HOW DO I ENFORCE PD RULES ON PEOPLE NOT ALLOWED ON IT?**",
            value="> 1. Give them their first warning to get off the PD, \"W1, off the PD!\"\n> 2. Wait 3-5 seconds for them to listen; if they don't, give them their second warning, \"W2, off the PD!\"\n> 3. Wait 3-5 seconds for them to listen; if they don't kill them.",
            inline=False
        )
        
        embed2.add_field(
            name="**WHO'S ALLOWED __ON__ THE ACTUAL STAGE AT ALL TIMES**",
            value="> ↠ Major General and above,\n> ↠ Royal Family (they should have a purple name tag),\n> ↠ Those who have been given permission by a Lieutenant General.",
            inline=False
        )
        
        embed2.add_field(
            name="**WHO'S ALLOWED TO PASS THE RED LINE IN-FRONT OF THE STAGE?**",
            value="> ↠ Major General and above,\n> ↠ Royal Family,\n> ↠ Those who have been given permission by a Lieutenant General,\n> ↠ COMBATIVE Home Command Regiments:\n> - Royal Military Police,\n> - United Kingdom Forces,\n> - Household Division.\n> **Kill those not allowed who touch or pass the red line.**",
            inline=False
        )
        
        embed2.add_field(
            name="\u200b",  
            value="**LASTLY, IF YOU'RE UNSURE ABOUT SOMETHING, ASK SOMEONE USING THE CHAIN OF COMMAND BEFORE TAKING ACTION!**",
            inline=False
        )

        try:
            await member.send(embeds=[embed1, special_embed, embed2])
        except discord.Forbidden:
            if welcome_channel := member.guild.get_channel(722002957738180620):
                await welcome_channel.send(f"{member.mention}", embeds=[embed1, special_embed, embed2])
                logger.info(f"Sending welcome message to {member.display_name} ({member.id})")
        except discord.HTTPException as e:
            logger.error(f"Failed to send welcome message: {e}")

    async def load_test_messages(self):
        """Fetches test messages (like HR welcome) from Supabase for configurable templates."""
        if not self.bot.db.supabase:
            return

        try:
            result = self.bot.db.supabase.table("Messages_Test").select("*").execute()
            self.bot.test_messages = {item["key"]: item["content"] for item in result.data}
            logger.info(f"✅ Loaded {len(self.bot.test_messages)} test messages from Supabase.")
        except Exception as e:
            logger.error(f"❌ Failed to load test messages: {e}")


    @commands.Cog.listener()
    async def on_member_update(self, before: discord.Member, after: discord.Member):
        async with self.bot.global_rate_limiter:
            # ===== CHECK FOR RMP ROLE REMOVAL (DESERTION/REMOVAL) =====
            rmp_role = after.guild.get_role(Config.RMP_ROLE_ID)
            
            if rmp_role and rmp_role in before.roles and rmp_role not in after.roles:
                # Member had RMP role but lost it - handle desertion/removal
                cleaned_nickname = re.sub(r'\[.*?\]', '', after.display_name).strip() or after.name
                
                try:
                    await self.bot.db.discharge_user(str(after.id), cleaned_nickname, after.guild)
                    logger.info(f"Removed {cleaned_nickname} ({after.id}) from database due to RMP role removal")
                except Exception as e:
                    logger.error(f"Error removing {cleaned_nickname} ({after.id}) from DB: {e}")
                
                # Optionally send alert about role removal
                try:
                    alert_channel = after.guild.get_channel(Config.HR_CHAT_CHANNEL_ID)
                    if alert_channel:
                        embed = discord.Embed(
                            title="⚠️ RMP Role Removed",
                            description=f"{after.mention} no longer has the RMP role.",
                            color=discord.Color.orange(),
                            timestamp=datetime.now(timezone.utc)
                        )
                        embed.add_field(name="User", value=f"{cleaned_nickname} ({after.id})", inline=True)
                        embed.add_field(name="Action", value="Database record removed", inline=True)
                        await alert_channel.send(embed=embed)
                except Exception as e:
                    logger.error(f"Failed to send RMP removal alert: {e}")
            
            # ===== WELCOME MESSAGES =====

            #HR Welcome
            hr_role = after.guild.get_role(Config.HR_ROLE_ID)
            if hr_role and hr_role not in before.roles and hr_role in after.roles:
                cleaned_nickname = re.sub(r'\[.*?\]', '', after.display_name).strip() or after.name
            
                try:
                    await self.bot.db.remove_from_lr(str(after.id))
             
                    await self.bot.db.add_to_hr(
                        user_id=str(after.id),
                        username=cleaned_nickname,
                        guild=after.guild
                    )
            
                    logger.info(
                        f"🔁 {cleaned_nickname} ({after.id}) moved from LRs → HRs in database"
                    )
            
                except Exception as e:
                    logger.error(f"❌ Failed HR DB transfer for {after.id}: {e}")
            
                await self.send_hr_welcome(after)
            
            
            # Check for RMP role addition
            if rmp_role and rmp_role not in before.roles and rmp_role in after.roles:
                # Look up Roblox ID before passing to DB
                roblox_api = RobloxAPI(self.bot)
                cleaned_name = clean_nickname(after.display_name)
                roblox_id = await roblox_api.get_roblox_id_from_username(cleaned_name)
                
                await self.bot.db.create_or_update_user_in_db(str(after.id), after.display_name, after.guild, roblox_id)
                await self.send_rmp_welcome(after)


            # ===== RANK TRACKING =====
            #Check if roles changed
            if set(before.roles) == set(after.roles):
                return  # No role changes, exit early
            
            # Get role IDs for comparison
            before_role_ids = [role.id for role in before.roles]
            after_role_ids = [role.id for role in after.roles]
            
            # Check if rank-related roles changed (only update if they did)
            if hasattr(self.bot, 'rank_tracker') and self.bot.rank_tracker:
                if not self.bot.rank_tracker._roles_changed_affect_rank(before_role_ids, after_role_ids):
                    return  # Role changes don't affect rank, exit early
            
            # Only update rank if member has RMP or HR role
            rmp_role = after.guild.get_role(Config.RMP_ROLE_ID)  # Re-fetch in case we need it
            hr_role = after.guild.get_role(Config.HR_ROLE_ID)
            
            if not ((rmp_role and rmp_role in after.roles) or (hr_role and hr_role in after.roles)):
                return  # Not an RMP or HR member, exit early
            
            # Update rank in database (with slight delay to prevent rapid updates)
            try:
                await asyncio.sleep(0.5)  # Small delay
                
                if hasattr(self.bot, 'rank_tracker') and self.bot.rank_tracker:
                    success = await self.bot.rank_tracker.update_member_in_database(after)
                    
                    if success:
                        member_info = await self.bot.rank_tracker.get_member_info(after)
                        logger.info(
                            f"🔄 Auto-updated rank for {after.display_name}: "
                            f"Division={member_info['division']}, Rank={member_info['rank']}"
                        )
                    else:
                        logger.warning(f"Failed to auto-update rank for {after.display_name}")
                else:
                    logger.warning("Rank tracker not initialized, cannot update rank")
                    
            except Exception as e:
                logger.error(f"Error in on_member_update for {after.display_name}: {e}")
                
    @commands.Cog.listener()
    async def on_member_remove(self, member: discord.Member):
        async with self.bot.global_rate_limiter:
            guild = member.guild
            if not (deserter_role := guild.get_role(Config.RMP_ROLE_ID)):
                return
        
            if deserter_role not in member.roles:
                return
                
            if not (alert_channel := guild.get_channel(Config.HR_CHAT_CHANNEL_ID)):
                return

            # Clean nickname for logs + DB
            cleaned_nickname = re.sub(r'\[.*?\]', '', member.display_name).strip() or member.name
            
            # 🔍 FIRST: Get Roblox ID from database BEFORE removing user
            roblox_id = None
            try:
                def _get_roblox_id():
                    sup = self.bot.db.supabase
                    res = sup.table('users').select('roblox_id').eq('user_id', str(member.id)).execute()
                    if getattr(res, 'data', None) and len(res.data) > 0:
                        return res.data[0].get('roblox_id')
                    return None
                
                roblox_id = await self.bot.db.run_query(_get_roblox_id)
                logger.info(f"Retrieved Roblox ID for deserter {cleaned_nickname}: {roblox_id}")
            except Exception as e:
                logger.error(f"Error getting Roblox ID for {cleaned_nickname}: {e}")

            # 🚨 Alert embed (send immediately)
            embed = discord.Embed(
                title="🚨 Deserter Alert",
                description=f"{member.mention} just deserted! Please log this in BA.",
                color=discord.Color.red()
            )
            embed.set_thumbnail(url=member.display_avatar.url)
            
            await alert_channel.send(
                content=f"<@&{Config.HIGH_COMMAND_ROLE_ID}>",
                embed=embed
            )

            # Current date for blacklist calculation
            current_date = datetime.now(timezone.utc)    
            hr_role = guild.get_role(Config.HR_ROLE_ID)  
            
            if hr_role and hr_role in member.roles:
                ending_date = current_date + timedelta(days=180)  
            else:
                ending_date = current_date + timedelta(days=90)

            # 🔧 Dishonourable discharge embed WITH Roblox ID
            dishonourable_embed = discord.Embed(
                title="Discharge Log",
                color=discord.Color.red(),
                timestamp=current_date
            )
            dishonourable_embed.add_field(name="Type", value="🚨 Dishonourable Discharge", inline=False)
            dishonourable_embed.add_field(name="Reason", value="```Desertion.```", inline=False)
            
            # Build member info with Roblox ID if available [Discontinued| will probably readd later on"
            member_info = f"{member.mention} | {cleaned_nickname}"
            
            dishonourable_embed.add_field(name="Discharged Members", value=member_info, inline=False)
            dishonourable_embed.add_field(name="Discharged By", value="None - Unauthorised", inline=True)
            dishonourable_embed.set_footer(text="Desertion Monitor System")

            # ⛔ Blacklist embed
            blacklist_embed = discord.Embed(
                title="⛔ Blacklist",
                color=discord.Color.red(),
                timestamp=current_date
            )
            blacklist_embed.add_field(name="Issuer:", value="MP Assistant", inline=False)
            blacklist_embed.add_field(name="Name:", value=cleaned_nickname, inline=False)
            
            # Add Roblox ID to blacklist embed if available
            if roblox_id:
                blacklist_embed.add_field(name="Roblox ID:", value=str(roblox_id), inline=False)
            
            blacklist_embed.add_field(name="Starting date", value=f"<t:{int(current_date.timestamp())}:D>", inline=False)
            blacklist_embed.add_field(name="Ending date", value=f"<t:{int(ending_date.timestamp())}:D>", inline=False)
            blacklist_embed.add_field(name="Reason:", value="Desertion.", inline=False)
            blacklist_embed.set_footer(text="Desertion Monitor System")

            # 📝 SECOND: Now send the embeds to the logs
            try:
                d_log = self.bot.get_channel(Config.D_LOG_CHANNEL_ID)
                b_log = self.bot.get_channel(Config.B_LOG_CHANNEL_ID)
                
                if d_log and b_log:
                    await d_log.send(embed=dishonourable_embed)
                    await b_log.send(embed=blacklist_embed)
                    logger.info(f"Logged deserted member, {cleaned_nickname}. Roblox ID: {roblox_id}")
                else:
                    alt_channel = self.bot.get_channel(1165368316970405917)
                    if alt_channel:
                        await alt_channel.send(f"⚠️ Failed to log deserter discharge for {cleaned_nickname}")
                        logger.error(f"Failed to log deserted member {cleaned_nickname} - main channel not found")
            except Exception as e:
                logger.error(f"Error logging deserter discharge: {str(e)}")
            
            # 🗑️ THIRD: Only AFTER sending embeds, remove from database
            try:
                await self.bot.db.discharge_user(str(member.id), cleaned_nickname, guild)
                logger.info(f"Removed {cleaned_nickname} ({member.id}) from database")
            except Exception as e:
                logger.error(f"Error removing deserter {cleaned_nickname} ({member.id}) from DB: {e}")





        #Edit-Welcome Messages Commands
    @app_commands.command(name="edit-welcome", description="Edit welcome messages for HR or new RMP members")
    @min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
    async def edit_welcome(
        self,
        interaction: discord.Interaction,
        message_type: Literal["HR Welcome", "RMP Welcome"],
        action: Literal["View Full", "Edit Title", "Edit Description", "Edit Color", "Add Field", "Remove Field", "Add Embed", "Remove Embed", "Reset to Default"] = "View Full"
    ):
        """Main command for editing welcome messages"""
        
        # Map display name to database type
        type_mapping = {
            "HR Welcome": "hr_welcome",
            "RMP Welcome": "rmp_welcome"
        }
        db_type = type_mapping.get(message_type)
        
        if not db_type:
            await interaction.response.send_message("❌ Invalid message type.", ephemeral=True)
            return
        
        # Get current message
        message_data = await self.bot.db.get_welcome_message(db_type)
        if not message_data or 'embeds' not in message_data:
            await interaction.response.send_message(
                f"❌ No {message_type} found in database.", 
                ephemeral=True
            )
            return
        
        embeds_data = message_data['embeds']
        
        # Handle different actions
        if action == "View Full":
            await self._view_full_welcome(interaction, db_type, message_type, embeds_data)
        
        elif action == "Edit Title":
            await self._edit_title_menu(interaction, db_type, message_type, embeds_data)
        
        elif action == "Edit Description":
            await self._edit_description_menu(interaction, db_type, message_type, embeds_data)
        
        elif action == "Edit Color":
            await self._edit_color_menu(interaction, db_type, message_type, embeds_data)
        
        elif action == "Add Field":
            await self._add_field_menu(interaction, db_type, message_type, embeds_data)
        
        elif action == "Remove Field":
            await self._remove_field_menu(interaction, db_type, message_type, embeds_data)
        
        elif action == "Add Embed":
            await self._add_embed_menu(interaction, db_type, message_type, embeds_data)
        
        elif action == "Remove Embed":
            await self._remove_embed_menu(interaction, db_type, message_type, embeds_data)
        
        elif action == "Reset to Default":
            await self._reset_to_default(interaction, db_type, message_type)

    async def _view_full_welcome(self, interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
        """View the full welcome message with all embeds"""
        if not embeds_data:
            await interaction.response.send_message(
                f"❌ {display_name} has no embeds.", 
                ephemeral=True
            )
            return
        
        # Defer since we might send multiple messages
        await interaction.response.defer(ephemeral=True)
        
        # Create Discord embed objects
        discord_embeds = []
        for embed_data in embeds_data:
            discord_embeds.append(dict_to_embed(embed_data))
        
        # Send all embeds (Discord allows up to 10 per message)
        for i, chunk in enumerate([discord_embeds[j:j+10] for j in range(0, len(discord_embeds), 10)]):
            if i == 0:
                content = f"**{display_name} - Full Preview**\nTotal embeds: {len(discord_embeds)}"
            else:
                content = f"**{display_name} - Continued (Part {i+1})**"
            
            await interaction.followup.send(
                content=content,
                embeds=chunk,
                ephemeral=True
            )

    async def _edit_title_menu(self, interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
        """Menu to select which embed title to edit - WORKING VERSION"""
        
        class TitleSelectView(discord.ui.View):
            def __init__(self, db_type, display_name, embeds_data, bot):
                super().__init__(timeout=60)
                self.bot = bot
                self.db_type = db_type
                self.display_name = display_name
                self.embeds_data = embeds_data
            
            @discord.ui.select(
                placeholder="Select embed to edit title...",
                options=[
                    discord.SelectOption(
                        label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                        value=str(i),
                        description=embed.get('description', '')[:100] or "No description"
                    )
                    for i, embed in enumerate(embeds_data)
                ]
            )
            async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
                embed_index = int(select.values[0])
                current_title = self.embeds_data[embed_index].get('title', '')
                
                # Modal is defined INSIDE the callback to capture embed_index
                class TitleModal(discord.ui.Modal, title=f"Edit Title - Embed {embed_index + 1}"):
                    def __init__(self, embed_index, embeds_data, db_type, bot):
                        super().__init__()
                        self.bot = bot
                        self.embed_index = embed_index
                        self.embeds_data = embeds_data
                        self.db_type = db_type
                        
                        self.title_input = discord.ui.TextInput(
                            label="New Title",
                            default=current_title,
                            max_length=256,
                            required=True
                        )
                        self.add_item(self.title_input)
                    
                    async def on_submit(self, modal_interaction: discord.Interaction):
                        await modal_interaction.response.defer(ephemeral=True)
                        
                        new_title = self.title_input.value
                        updated_embeds = self.embeds_data.copy()
                        old_title = updated_embeds[self.embed_index].get('title', 'Untitled')
                        updated_embeds[self.embed_index]['title'] = new_title
                        
                        # SIMPLIFIED CALL - remove admin_user and change_details
                        success, new_data = await self.bot.db.update_welcome_message(
                            self.db_type,
                            updated_embeds,
                            f"{modal_interaction.user.name} ({modal_interaction.user.id})"
                        )
                        
                        if success:
                            await modal_interaction.followup.send(
                                f"✅ Updated title of Embed {self.embed_index + 1} to: **{new_title}**",
                                ephemeral=True
                            )
                        else:
                            await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
                
                # Create and show the modal
                modal = TitleModal(embed_index, self.embeds_data, self.db_type, self.bot)
                await select_interaction.response.send_modal(modal)
        
        view = TitleSelectView(db_type, display_name, embeds_data, self.bot)
        await interaction.response.send_message(
            f"Select which embed of **{display_name}** to edit title:",
            view=view,
            ephemeral=True
        )

    async def _edit_description_menu(self, interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
        """Menu to select which embed description to edit"""
        
        class DescriptionSelectView(discord.ui.View):
            def __init__(self, db_type, display_name, embeds_data, bot):
                super().__init__(timeout=60)
                self.bot = bot
                self.db_type = db_type
                self.display_name = display_name
                self.embeds_data = embeds_data
            
            @discord.ui.select(
                placeholder="Select embed to edit description...",
                options=[
                    discord.SelectOption(
                        label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                        value=str(i),
                        description="Has description" if embed.get('description') else "No description"
                    )
                    for i, embed in enumerate(embeds_data)
                ]
            )
            async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
                embed_index = int(select.values[0])
                current_desc = self.embeds_data[embed_index].get('description', '')
                
                class DescriptionModal(discord.ui.Modal, title=f"Edit Description - Embed {embed_index + 1}"):
                    def __init__(self, embed_index, embeds_data, db_type, bot):
                        super().__init__()
                        self.bot = bot
                        self.embed_index = embed_index
                        self.embeds_data = embeds_data
                        self.db_type = db_type
                        
                        self.desc_input = discord.ui.TextInput(
                            label="New Description",
                            default=current_desc,
                            style=discord.TextStyle.paragraph,
                            max_length=4000,
                            required=False
                        )
                        self.add_item(self.desc_input)
                    
                    async def on_submit(self, modal_interaction: discord.Interaction):
                        await modal_interaction.response.defer(ephemeral=True)
                        
                        new_desc = self.desc_input.value
                        updated_embeds = self.embeds_data.copy()
                        updated_embeds[self.embed_index]['description'] = new_desc
                        
                        success, _ = await self.bot.db.update_welcome_message(
                            self.db_type,
                            updated_embeds,
                            f"{modal_interaction.user.name} ({modal_interaction.user.id})"
                        )
                        
                        if success:
                            preview = new_desc[:100] + "..." if len(new_desc) > 100 else new_desc
                            await modal_interaction.followup.send(
                                f"✅ Updated description of Embed {self.embed_index + 1}\n**Preview:** {preview}",
                                ephemeral=True
                            )
                        else:
                            await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
                
                modal = DescriptionModal(embed_index, self.embeds_data, self.db_type, self.bot)
                await select_interaction.response.send_modal(modal)
        
        view = DescriptionSelectView(db_type, display_name, embeds_data, self.bot)
        await interaction.response.send_message(
            f"Select which embed of **{display_name}** to edit description:",
            view=view,
            ephemeral=True
        )


    async def _edit_color_menu(self, interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
        """Menu to select which embed color to edit"""
        
        class ColorSelectView(discord.ui.View):
            def __init__(self, db_type, display_name, embeds_data, bot):
                super().__init__(timeout=60)
                self.bot = bot
                self.db_type = db_type
                self.display_name = display_name
                self.embeds_data = embeds_data
            
            @discord.ui.select(
                placeholder="Select embed to edit color...",
                options=[
                    discord.SelectOption(
                        label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                        value=str(i),
                        description=f"Color: {embed.get('color', '#000000')}"
                    )
                    for i, embed in enumerate(embeds_data)
                ]
            )
            async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
                embed_index = int(select.values[0])
                current_color = self.embeds_data[embed_index].get('color', '#000000')
                
                class ColorModal(discord.ui.Modal, title=f"Edit Color - Embed {embed_index + 1}"):
                    def __init__(self, embed_index, embeds_data, db_type, bot):
                        super().__init__()
                        self.bot = bot
                        self.embed_index = embed_index
                        self.embeds_data = embeds_data
                        self.db_type = db_type
                        
                        self.color_input = discord.ui.TextInput(
                            label="New Color (hex format, e.g., #FFD700)",
                            default=current_color,
                            max_length=7,
                            required=True
                        )
                        self.add_item(self.color_input)
                    
                    async def on_submit(self, modal_interaction: discord.Interaction):
                        await modal_interaction.response.defer(ephemeral=True)
                        
                        new_color = self.color_input.value.upper()
                        
                        # Validate hex color
                        if not re.match(r'^#[0-9A-F]{6}$', new_color):
                            await modal_interaction.followup.send(
                                "❌ Invalid color format. Use hex format like #FFD700",
                                ephemeral=True
                            )
                            return
                        
                        updated_embeds = self.embeds_data.copy()
                        updated_embeds[self.embed_index]['color'] = new_color
                        
                        success, _ = await self.bot.db.update_welcome_message(
                            self.db_type,
                            updated_embeds,
                            f"{modal_interaction.user.name} ({modal_interaction.user.id})"
                        )
                        
                        if success:
                            await modal_interaction.followup.send(
                                f"✅ Updated color of Embed {self.embed_index + 1} to: **{new_color}**",
                                ephemeral=True
                            )
                        else:
                            await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
                
                modal = ColorModal(embed_index, self.embeds_data, self.db_type, self.bot)
                await select_interaction.response.send_modal(modal)
        
        view = ColorSelectView(db_type, display_name, embeds_data, self.bot)
        await interaction.response.send_message(
            f"Select which embed of **{display_name}** to edit color:",
            view=view,
            ephemeral=True
        )


    async def _add_field_menu(self, interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
        """Menu to add a field to an embed"""
        
        class AddFieldSelectView(discord.ui.View):
            def __init__(self, db_type, display_name, embeds_data, bot):
                super().__init__(timeout=60)
                self.bot = bot
                self.db_type = db_type
                self.display_name = display_name
                self.embeds_data = embeds_data
            
            @discord.ui.select(
                placeholder="Select embed to add field to...",
                options=[
                    discord.SelectOption(
                        label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                        value=str(i),
                        description=f"Fields: {len(embed.get('fields', []))}"
                    )
                    for i, embed in enumerate(embeds_data)
                ]
            )
            async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
                embed_index = int(select.values[0])
                
                class AddFieldModal(discord.ui.Modal, title=f"Add Field - Embed {embed_index + 1}"):
                    def __init__(self, embed_index, embeds_data, db_type, bot):
                        super().__init__()
                        self.bot = bot
                        self.embed_index = embed_index
                        self.embeds_data = embeds_data
                        self.db_type = db_type
                        
                        # Create form fields
                        self.field_name = discord.ui.TextInput(
                            label="Field Name",
                            placeholder="Enter field name...",
                            max_length=256,
                            required=True
                        )
                        self.field_value = discord.ui.TextInput(
                            label="Field Value",
                            placeholder="Enter field value...",
                            style=discord.TextStyle.paragraph,
                            max_length=1024,
                            required=True
                        )
                        self.inline_input = discord.ui.TextInput(
                            label="Inline? (true/false)",
                            placeholder="true or false",
                            default="false",
                            max_length=5,
                            required=False
                        )
                        
                        self.add_item(self.field_name)
                        self.add_item(self.field_value)
                        self.add_item(self.inline_input)
                    
                    async def on_submit(self, modal_interaction: discord.Interaction):
                        await modal_interaction.response.defer(ephemeral=True)
                        
                        # Parse inline boolean
                        inline_bool = self.inline_input.value.lower() == 'true'
                        
                        # Create new field
                        new_field = {
                            'name': self.field_name.value,
                            'value': self.field_value.value,
                            'inline': inline_bool
                        }
                        
                        # Update embeds
                        updated_embeds = self.embeds_data.copy()
                        if 'fields' not in updated_embeds[self.embed_index]:
                            updated_embeds[self.embed_index]['fields'] = []
                        
                        updated_embeds[self.embed_index]['fields'].append(new_field)
                        
                        success, _ = await self.bot.db.update_welcome_message(
                            self.db_type,
                            updated_embeds,
                            f"{modal_interaction.user.name} ({modal_interaction.user.id})"
                        )
                        
                        if success:
                            await modal_interaction.followup.send(
                                f"✅ Added field **{self.field_name.value}** to Embed {self.embed_index + 1}",
                                ephemeral=True
                            )
                        else:
                            await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
                
                modal = AddFieldModal(embed_index, self.embeds_data, self.db_type, self.bot)
                await select_interaction.response.send_modal(modal)
        
        view = AddFieldSelectView(db_type, display_name, embeds_data, self.bot)
        await interaction.response.send_message(
            f"Select which embed of **{display_name}** to add a field to:",
            view=view,
            ephemeral=True
        )


    async def _remove_field_menu(self, interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
        """Menu to remove a field from an embed - SIMPLIFIED VERSION"""
        
        # First, find embeds that have fields
        embeds_with_fields = []
        for i, embed in enumerate(embeds_data):
            if embed.get('fields'):
                embeds_with_fields.append((i, embed))
        
        if not embeds_with_fields:
            await interaction.response.send_message(
                f"❌ No embeds in {display_name} have fields to remove.",
                ephemeral=True
            )
            return
        
        class RemoveFieldSelectView(discord.ui.View):
            def __init__(self, db_type, display_name, embeds_with_fields, embeds_data, bot):
                super().__init__(timeout=60)
                self.bot = bot
                self.db_type = db_type
                self.display_name = display_name
                self.embeds_with_fields = embeds_with_fields
                self.embeds_data = embeds_data
            
            @discord.ui.select(
                placeholder="Select embed with field to remove...",
                options=[
                    discord.SelectOption(
                        label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                        value=str(i),
                        description=f"{len(embed.get('fields', []))} fields"
                    )
                    for i, embed in embeds_with_fields
                ]
            )
            async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
                embed_index = int(select.values[0])
                
                # Show field selection
                fields = self.embeds_data[embed_index].get('fields', [])
                
                class FieldSelectView(discord.ui.View):
                    def __init__(self, embed_index, db_type, embeds_data, bot):
                        super().__init__(timeout=60)
                        self.bot = bot
                        self.embed_index = embed_index
                        self.db_type = db_type
                        self.embeds_data = embeds_data
                    
                    @discord.ui.select(
                        placeholder="Select field to remove...",
                        options=[
                            discord.SelectOption(
                                label=f"Field {j+1}: {field.get('name', f'Field {j+1}')[:100]}",
                                value=str(j),
                                description=field.get('value', '')[:50]
                            )
                            for j, field in enumerate(fields)
                        ]
                    )
                    async def field_callback(self, field_interaction: discord.Interaction, field_select: discord.ui.Select):
                        field_index = int(field_select.values[0])
                        field_name = fields[field_index].get('name', f'Field {field_index + 1}')
                        
                        # Confirm removal
                        confirm_view = ConfirmView(field_interaction.user)
                        await field_interaction.response.send_message(
                            f"⚠️ Remove field **{field_name}** from Embed {embed_index + 1}?",
                            view=confirm_view,
                            ephemeral=True
                        )
                        
                        await confirm_view.wait()
                        
                        if confirm_view.value:
                            updated_embeds = self.embeds_data.copy()
                            updated_embeds[self.embed_index]['fields'].pop(field_index)
                            
                            # Clean up if no fields left
                            if not updated_embeds[self.embed_index]['fields']:
                                updated_embeds[self.embed_index].pop('fields', None)
                            
                            success, _ = await self.bot.db.update_welcome_message(
                                self.db_type,
                                updated_embeds,
                                f"{field_interaction.user.name} ({field_interaction.user.id})"
                            )
                            
                            if success:
                                await field_interaction.followup.send(
                                    f"✅ Removed field **{field_name}**",
                                    ephemeral=True
                                )
                            else:
                                await field_interaction.followup.send("❌ Failed to save.", ephemeral=True)
                        else:
                            await field_interaction.followup.send("❌ Cancelled.", ephemeral=True)
                
                field_view = FieldSelectView(embed_index, self.db_type, self.embeds_data, self.bot)
                await select_interaction.response.send_message(
                    f"Select which field to remove from Embed {embed_index + 1}:",
                    view=field_view,
                    ephemeral=True
                )
        
        view = RemoveFieldSelectView(db_type, display_name, embeds_with_fields, embeds_data, self.bot)
        await interaction.response.send_message(
            f"Select which embed of **{display_name}** has the field to remove:",
            view=view,
            ephemeral=True
        )

    async def _add_embed_menu(self, interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
        """Add a new embed to the welcome message"""
        
        class AddEmbedModal(discord.ui.Modal, title=f"Add New Embed to {display_name}"):
            def __init__(self, db_type, embeds_data, bot):
                super().__init__()
                self.bot = bot
                self.db_type = db_type
                self.embeds_data = embeds_data
                
                self.title_input = discord.ui.TextInput(
                    label="Embed Title",
                    placeholder="Enter embed title...",
                    max_length=256,
                    required=True
                )
                self.desc_input = discord.ui.TextInput(
                    label="Embed Description",
                    placeholder="Enter embed description...",
                    style=discord.TextStyle.paragraph,
                    max_length=4000,
                    required=False
                )
                self.color_input = discord.ui.TextInput(
                    label="Embed Color (hex)",
                    placeholder="#3498db",
                    default="#3498db",
                    max_length=7,
                    required=False
                )
                
                self.add_item(self.title_input)
                self.add_item(self.desc_input)
                self.add_item(self.color_input)
            
            async def on_submit(self, modal_interaction: discord.Interaction):
                await modal_interaction.response.defer(ephemeral=True)
                
                # Validate color
                new_color = self.color_input.value.upper()
                if not re.match(r'^#[0-9A-F]{6}$', new_color):
                    new_color = "#3498db"
                
                # Create new embed
                new_embed = {
                    'title': self.title_input.value,
                    'description': self.desc_input.value,
                    'color': new_color
                }
                
                # Add to existing embeds
                updated_embeds = self.embeds_data.copy()
                updated_embeds.append(new_embed)
                
                success, _ = await self.bot.db.update_welcome_message(
                    self.db_type,
                    updated_embeds,
                    f"{modal_interaction.user.name} ({modal_interaction.user.id})"
                )
                
                if success:
                    await modal_interaction.followup.send(
                        f"✅ Added new embed **{self.title_input.value}** to {display_name}\nTotal embeds: {len(updated_embeds)}",
                        ephemeral=True
                    )
                else:
                    await modal_interaction.followup.send("❌ Failed to save changes.", ephemeral=True)
        
        modal = AddEmbedModal(db_type, embeds_data, self.bot)
        await interaction.response.send_modal(modal)

    async def _remove_embed_menu(self, interaction: discord.Interaction, db_type: str, display_name: str, embeds_data: list):
        """Remove an embed from the welcome message"""
        
        if len(embeds_data) <= 1:
            await interaction.response.send_message(
                f"❌ Cannot remove embed. {display_name} must have at least 1 embed.",
                ephemeral=True
            )
            return
        
        class RemoveEmbedSelectView(discord.ui.View):
            def __init__(self, db_type, display_name, embeds_data, bot):
                super().__init__(timeout=60)
                self.bot = bot
                self.db_type = db_type
                self.display_name = display_name
                self.embeds_data = embeds_data
            
            @discord.ui.select(
                placeholder="Select embed to remove...",
                options=[
                    discord.SelectOption(
                        label=f"Embed {i+1}: {embed.get('title', f'Embed {i+1}')[:100]}",
                        value=str(i),
                        description="Click to remove"
                    )
                    for i, embed in enumerate(embeds_data)
                ]
            )
            async def select_callback(self, select_interaction: discord.Interaction, select: discord.ui.Select):
                embed_index = int(select.values[0])
                embed_title = self.embeds_data[embed_index].get('title', f'Embed {embed_index + 1}')
                
                # Confirm removal
                confirm_view = ConfirmView(select_interaction.user)
                await select_interaction.response.send_message(
                    f"⚠️ Remove **{embed_title}** from {self.display_name}?\nThis will leave {len(self.embeds_data)-1} embeds.",
                    view=confirm_view,
                    ephemeral=True
                )
                
                await confirm_view.wait()
                
                if confirm_view.value:
                    updated_embeds = self.embeds_data.copy()
                    updated_embeds.pop(embed_index)
                    
                    success, _ = await self.bot.db.update_welcome_message(
                        self.db_type,
                        updated_embeds,
                        f"{select_interaction.user.name} ({select_interaction.user.id})"
                    )
                    
                    if success:
                        await select_interaction.followup.send(
                            f"✅ Removed **{embed_title}**\nRemaining: {len(updated_embeds)} embeds",
                            ephemeral=True
                        )
                    else:
                        await select_interaction.followup.send("❌ Failed to save.", ephemeral=True)
                else:
                    await select_interaction.followup.send("❌ Cancelled.", ephemeral=True)
        
        view = RemoveEmbedSelectView(db_type, display_name, embeds_data, self.bot)
        await interaction.response.send_message(
            f"⚠️ **Remove Embed from {display_name}**\nSelect which embed to remove:",
            view=view,
            ephemeral=True
        )
        

    async def _reset_to_default(self, interaction: discord.Interaction, db_type: str, display_name: str):
        """Reset welcome message to default"""
        # Defer first since we're showing a view
        await interaction.response.defer(ephemeral=True)
        
        # Get default messages
        default_messages = await self._get_default_messages()
        default_embeds = default_messages.get(db_type, [])
        
        if not default_embeds:
            await interaction.followup.send(
                f"❌ No default found for {display_name}.",
                ephemeral=True
            )
            return
        
        # Show confirmation
        confirm_view = ConfirmView(interaction.user)
        await interaction.followup.send(
            f"⚠️ **Reset {display_name} to Default**\nThis will restore the original welcome message.\n\n**Are you sure?**",
            view=confirm_view,
            ephemeral=True
        )
        
        await confirm_view.wait()
        
        if confirm_view.value:
            # Save defaults to database
            success, new_data = await self.bot.db.update_welcome_message(
                db_type,
                default_embeds,
                f"{interaction.user.name} ({interaction.user.id}) - Reset to default",
            )
            
            if success:
                await interaction.followup.send(
                    f"✅ Reset {display_name} to default configuration\nEmbeds restored: {len(default_embeds)}",
                    ephemeral=True
                )
            else:
                await interaction.followup.send("❌ Failed to reset to default.", ephemeral=True)
        else:
            await interaction.followup.send("❌ Reset cancelled.", ephemeral=True)

    async def _get_default_messages(self):
        """Get default welcome messages"""
        # These should match what you inserted into the database
        return {
            'hr_welcome': [
                {
                    'title': '🎉 Welcome to the HR Team!',
                    'description': '**Please note the following:**\n• Request for document access in [HR Documents](https://discord.com/channels/1165368311085809717/1165368317532438646).\n• You are exempted from quota this week only - you start next week ([Quota Info](https://discord.com/channels/1165368311085809717/1206998095552978974)).\n• Uncomplete quota = strike.\n• One failed tryout allowed if your try quota portion ≥2.\n• Ask for help anytime - we\'re friendly!\n• Are you Lieutenant+ in BA? Apply for the Education Department!\n• Are you Captain+ in BA? Apply for both departments: [Applications](https://discord.com/channels/1165368311085809717/1165368316970405916).',
                    'color': '#FFD700',
                    'footer': 'We\'re excited to have you on board!'
                }
            ],
            'rmp_welcome': [
                {
                    'title': '👮| Welcome to the Royal Military Police',
                    'description': 'Congratulations on passing your security check, you\'re officially a TRAINING member of the police force. Please be sure to read the information found below.\n\n> ** 1.** Make sure to read all of the rules found in <#1165368313925353580> and in the brochure found below.\n\n> **2.** You **MUST** read the RMP main guide and MSL before starting your duties.\n\n> **3.** You can\'t use your L85 unless you are doing it for Self-Militia or enforcing the PD rules. (Self-defence)\n\n> **4.** Make sure to follow the Chain Of Command. 2nd Lieutenant > Lieutenant > Captain > Major > Lieutenant Colonel > Colonel > Brigadier > Major General\n\n> **5.** For phases, you may wait for one to be hosted in <#1207367013698240584> or request the phase you need in <#1270700562433839135>.\n\n> **6.** All the information about the Defence School of Policing and Guarding is found in both <#1237062439720452157> and <#1207366893631967262>\n\n> **7.** Choose your timezone here https://discord.com/channels/1165368311085809717/1165368313925353578\n\n**Besides that, good luck with your phases!**',
                    'color': '#330000'
                },
                {
                    'title': 'Special Roles',
                    'description': '> Get your role pings here <#1196085670360404018> and don\'t forget the Game Night role RMP always hosts fun events, don\'t miss out!',
                    'color': '#330000'
                },
                {
                    'title': 'Trainee Constable Brochure',
                    'color': '#660000',
                    'fields': [
                        {
                            'name': '**TOP 5 RULES**',
                            'value': '> **1**. You **MUST** read the RMP main guide and MSL before starting your duties.\n> **2**. You **CANNOT** enforce the MSL. Only the Parade Deck (PD) rules **AFTER** you pass your phase 1.\n> **3**. You **CANNOT** use your bike on the PD or the pavements.\n> **4**. You **MUST** use good spelling and grammar to the best of your ability.\n> **5**. You **MUST** remain mature and respectful at all times.'
                        },
                        {
                            'name': '**WHO\'S ALLOWED ON THE PD AT ALL TIMES?**',
                            'value': '> ↠ Royal Army Medical Corps,\n> ↠ Royal Military Police,\n> ↠ Intelligence Corps.\n> ↠ Royal Family.'
                        }
                    ]
                }
            ]
        }


    # Preview Welcome Message Command
    @app_commands.command(name="preview-welcome", description="Preview welcome messages as they appear to new members")
    @min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
    async def preview_welcome(
        self,
        interaction: discord.Interaction,
        message_type: Literal["HR Welcome", "RMP Welcome"],
        target_user: discord.Member = None
    ):
        """Preview welcome message exactly as a new member would see it"""
        
        await interaction.response.defer(ephemeral=True)
        
        type_mapping = {
            "HR Welcome": "hr_welcome",
            "RMP Welcome": "rmp_welcome"
        }
        db_type = type_mapping.get(message_type)
        
        if not db_type:
            await interaction.followup.send("❌ Invalid message type.", ephemeral=True)
            return
        
        message_data = await self.bot.db.get_welcome_message(db_type)
        if not message_data or 'embeds' not in message_data:
            await interaction.followup.send(f"❌ No {message_type} found.", ephemeral=True)
            return
        
        # Create Discord embeds
        discord_embeds = []
        for embed_data in message_data['embeds']:
            discord_embeds.append(dict_to_embed(embed_data))
        
        # Send preview
        preview_text = f"**Preview: {message_type}**"
        if target_user:
            preview_text += f"\n*Simulating send to {target_user.mention}*"
        
        await interaction.followup.send(
            content=preview_text,
            embeds=discord_embeds,
            ephemeral=True
        )
        
        # Log the preview
        logger.info(f"{interaction.user} previewed {message_type}")

    @preview_welcome.autocomplete('message_type')
    async def preview_welcome_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str
    ):
        """Autocomplete for preview welcome"""
        types = ["HR Welcome", "RMP Welcome"]
        return [
            discord.app_commands.Choice(name=msg_type, value=msg_type)
            for msg_type in types
            if current.lower() in msg_type.lower()
        ][:25]


    #Welcome Message(s) History Command
    @app_commands.command(name="welcome-history", description="View history of welcome message changes")
    @min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
    async def welcome_history(
        self,
        interaction: discord.Interaction,
        message_type: Literal["HR Welcome", "RMP Welcome"],
        limit: app_commands.Range[int, 1, 10] = 5
    ):
        """View historical versions of welcome messages"""
        
        await interaction.response.defer(ephemeral=True)
        
        type_mapping = {
            "HR Welcome": "hr_welcome",
            "RMP Welcome": "rmp_welcome"
        }
        db_type = type_mapping.get(message_type)
        
        if not db_type:
            await interaction.followup.send("❌ Invalid message type.", ephemeral=True)
            return
        
        history = await self.bot.db.get_welcome_message_history(db_type, limit)
        
        if not history:
            await interaction.followup.send(f"❌ No history found for {message_type}.", ephemeral=True)
            return
        
        embed = discord.Embed(
            title=f"📜 History: {message_type}",
            description=f"Last {len(history)} versions (newest first)",
            color=discord.Color.blue()
        )
        
        for i, version in enumerate(history, 1):
            timestamp = version.get('last_updated')
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = f"<t:{int(dt.timestamp())}:R>"
                except:
                    time_str = timestamp
            elif hasattr(timestamp, 'timestamp'):
                time_str = f"<t:{int(timestamp.timestamp())}:R>"
            else:
                time_str = "Unknown"
            
            embed_count = len(version.get('embeds', []))
            active_status = "✅ Active" if version.get('is_active') else "📁 Archived"
            
            embed.add_field(
                name=f"Version {version.get('version', i)} - {active_status}",
                value=(
                    f"**When:** {time_str}\n"
                    f"**By:** {version.get('updated_by', 'Unknown')}\n"
                    f"**Embeds:** {embed_count}\n"
                    f"**ID:** `{version.get('id')}`"
                ),
                inline=False
            )
        
        await interaction.followup.send(embed=embed, ephemeral=True)

    @min_rank_required(Config.HIGH_COMMAND_ROLE_ID)
    async def preview_welcome_context(self, interaction: discord.Interaction, member: discord.Member):
        """Preview welcome message for a specific member"""
        await interaction.response.defer(ephemeral=True)
        
        # Determine which welcome based on member's roles
        hr_role = interaction.guild.get_role(Config.HR_ROLE_ID)
        rmp_role = interaction.guild.get_role(Config.RMP_ROLE_ID)
        
        if hr_role and hr_role in member.roles:
            message_type = "HR Welcome"
            db_type = "hr_welcome"
        elif rmp_role and rmp_role in member.roles:
            message_type = "RMP Welcome"
            db_type = "rmp_welcome"
        else:
            await interaction.followup.send(
                "This member doesn't have HR or RMP roles.",
                ephemeral=True
            )
            return
        
        message_data = await self.bot.db.get_welcome_message(db_type)
        if not message_data or 'embeds' not in message_data:
            await interaction.followup.send(f"No {message_type} configured.", ephemeral=True)
            return
        
        # Create preview
        discord_embeds = []
        for embed_data in message_data['embeds']:
            discord_embeds.append(dict_to_embed(embed_data))
        
        await interaction.followup.send(
            f"**Preview for {member.mention}**\n{message_type}",
            embeds=discord_embeds,
            ephemeral=True
        )

    async def cog_unload(self):
        self.bot.tree.remove_command(self.ctx_menu.name, type=self.ctx_menu.type)

    #Error handling for Welcome Message Commands
    @edit_welcome.error
    @preview_welcome.error
    @welcome_history.error
    async def welcome_commands_error(self, interaction: discord.Interaction, error):
        """Handle errors in welcome commands"""
        try:
            if isinstance(error, discord.app_commands.errors.MissingPermissions):
                await interaction.response.send_message(
                    "❌ You don't have permission to use welcome commands.",
                    ephemeral=True
                )
            elif isinstance(error, discord.app_commands.errors.CommandOnCooldown):
                await interaction.response.send_message(
                    f"⏳ Please wait {error.retry_after:.1f}s before editing welcome messages again.",
                    ephemeral=True
                )
            else:
                logger.error(f"Welcome command error: {type(error).__name__}: {error}")
                await interaction.response.send_message(
                    "❌ An error occurred. Please try again or contact admin.",
                    ephemeral=True
                )
        except discord.NotFound:
            pass  # Interaction already handled
        except Exception as e:
            logger.error(f"Error handler error: {e}")




async def setup(bot):
    await bot.add_cog(WelcomeCog(bot))
