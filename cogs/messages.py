import re
import random
import asyncio
import discord
import logging
from config import Config
from discord.ext import commands, tasks
from utils.embedBuilder import build_change_log
from utils.helpers import clean_nickname
from utils.views import RequestView, DischargeView, save_pending_checks, PENDING_CHECK_TABLE, PENDING_BL_TABLE
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

class MessageLoggerCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.cr_table = self.bot.db.crs_table
        self.prompt_users.start()
 
    # Clean old processed records in db
    async def start_cleanup_task(self):
        """Clean up old database entries periodically"""
        async def cleanup_loop():
            while True:
                try:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=2)
                    await self.bot.db.supabase.table(self.cr_table)\
                        .delete()\
                        .lt('logged_at', cutoff.isoformat())\
                        .execute()

                    cutoff = datetime.now(timezone.utc) - timedelta(days=20)
                    await self.bot.db.supabase.table(PENDING_BL_TABLE)\
                            .delete()\
                            .lt('created_at', cutoff.isoformat())\
                            .execute()
 
                    res = await self.bot.db.supabase.table(PENDING_CHECK_TABLE)\
                            .delete()\
                            .lt('created_at', cutoff.isoformat())\
                            .execute()

                    if res.data:
                        await self.cleanup_pending_checks(res.data)

                    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                    res = await self.bot.db.supabase.table(self.bot.db.s_users_table)\
                            .delete()\
                            .lt('last_updated', cutoff.isoformat())\
                            .execute()

                    if res.data:
                        await self.log_deleted_users(res.data)

                    logger.info("Cleaned up old Criminal Records, Stored Users & Requests")
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

                await asyncio.sleep(86400) #Daily 

        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def log_deleted_users(self, users):
        users_str = "\n".join([f'- {user['username']} ({user['user_id']})' for user in users])
        logger.info(f"Successfully deleted the following users from archive: {users_str}")

        color = discord.Color.green()
        title = "Archive Removal"
        icon_url = Config.TRASH_ICON
        description = f"The following user(s) have been **removed** from the user archive and cannot be **retrieved** anymore:\n"\
                f"```{users_str}```"

        log_channel = self.bot.get_channel(Config.DEFAULT_LOG_CHANNEL)
        if log_channel:
            try:
                embed = discord.Embed(description=description, color=color) 
                embed.set_author(name=title, icon_url=icon_url)
                await log_channel.send(embed=embed)
            except Exception as log_error:
                logger.error("Failed to send archive removal log embed: %s", log_error)

 

    async def cleanup_pending_checks(self, panels):
        for panel in panels:
            channel = self.bot.get_channel(int(panel["channel_id"]))

            if channel:
                try:
                    panel_msg = await channel.fetch_message(int(panel['panel_id']))
                    await panel_msg.delete()
                except Exception:
                    pass 

    @commands.Cog.listener()
    async def on_ready(self):
        if not hasattr(self, "_cleanup_task"):
            await self.start_cleanup_task()

    
    async def send_change_log(self):
        key_name = 'send_change_log'
        column_name =  'send'
        main_channel = self.bot.get_channel(Config.MAIN_COMMS_CHANNEL_ID)
        log_channel = self.bot.get_channel(Config.DEFAULT_LOG_CHANNEL)
        
        res = await self.bot.db.supabase.table('bot_state')\
                .select(column_name)\
                .eq('key', key_name)\
                .execute()
        
        if res.data:
            send = res.data[0][column_name]
            if not send:
                return

        prefix = self.bot.command_prefix
        cl_embeds = build_change_log(prefix, 0)

        try: 
            if main_channel:
                await main_channel.send(embeds=cl_embeds)
        except Exception:
            pass
        
        await asyncio.sleep(0.3)

        try:
            if log_channel:
                await log_channel.send(embeds=cl_embeds)
        except Exception:
            pass

        await self.bot.db.supabase.table('bot_state')\
            .upsert(
                {'key': key_name, column_name: False},
                on_conflict='key'
            )\
            .execute()
               
    @tasks.loop(hours=24)  
    async def prompt_users(self):
        channel = self.bot.get_channel(Config.MAIN_COMMS_CHANNEL_ID)
        if not channel:
            return

        result = await self.bot.db.supabase.table('bot_state')\
            .select('value')\
            .eq('key', 'last_prompt_time')\
            .execute()

        if result.data:
            last_time = datetime.fromisoformat(result.data[0]['value'])
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last_time < timedelta(days=5):
                return

        PROMPTS = [
            "Have a good suggestion to improve me? Use the suggest command.",
            "Use the report command to flag any bugs/mistakes I have.",
            "Type `!commands` to see what I can do.",
            "Use the change-log command to view recent updates made to me.",
            "You can use commands using my prefix '!', use !commands to learn more."
        ]

        await channel.send(random.choice(PROMPTS))

        await self.bot.db.supabase.table('bot_state')\
            .upsert(
                {'key': 'last_prompt_time', 'value': datetime.now(timezone.utc).isoformat()},
                on_conflict='key'
            )\
            .execute()
    
    @prompt_users.before_loop
    async def before_prompt_users(self):
        await self.bot.wait_until_ready()


    async def case_logs(self, message: discord.Message, manual: bool = False, user = None):
        suspect = re.search(r'suspect:\s*([^\s(]+)', message.content, re.IGNORECASE)
        
        if not suspect:
            return False
    
        suspect_user = str(suspect.group(1)) 
        
        if not suspect_user:
            return False

        try: 
            suspect_id = await self.bot.roblox.get_user_id(suspect_user)

            if suspect_id:
                if user :
                    if user.isnumeric() and user != suspect_id:
                        return False
                    else:
                        if user != suspect_user:
                            return False 

                await self.bot.db.add_criminal_record(message.id, suspect_id, suspect_user, message.jump_url)
                logger.info(f"Criminal record logged: rbxID=%s, username=%s, msgID=%s", suspect_id, suspect_user, message.id)
            try:
                await message.add_reaction("✅")
                if manual:
                    return suspect_user
            except Exception as e:
                logger.error("Failed to react to message: %s", e)
            
            try:
                log_channel = message.guild.get_channel(Config.DEFAULT_LOG_CHANNEL)
                if log_channel: 
                    color = discord.Color.green()
                    title = "Case Logged"
                    icon_url = Config.CUFFS_ICON
                    description = f"**{suspect_user}'s** [arrest log]({message.jump_url}) has been successfully logged in the database."
                    embed = discord.Embed(description=description, color=color).set_footer(text="Will be deleted in 2 days. This can be managed by using /manage-case-log.")
                    embed.set_author(name=title, icon_url=icon_url)
                    await log_channel.send(embed=embed)

                    logger.info(f"Processed Case Log | MsgID=%s", message.id)
                    return True
            except Exception as e:
                logger.error("Failed to send embed for arrest log on %s: %s", suspect_user, e)
                return False
 
        except Exception as e:       
            logger.error(f"MessageLogger.case_logs failed: {e}", exc_info=True)
    
    async def security_check(self, message: discord.Message):
        content = message.content

        sc_pattern = r"Roblox\s+profile\s+link:\s*https?://(?:www\.)?roblox\.com/users/(\d+)"

        link_match = re.search(sc_pattern, content, re.IGNORECASE)
        
        user_id = -2
        if link_match:
            user_id = int(link_match.group(1))
        else:
            user_match = re.search(r'Username:\s*(\w+)', content, re.IGNORECASE)
            if user_match:
                username = user_match.group(1)
                try:
                    user_id = await self.bot.roblox.get_user_id(username)
                except Exception:
                    pass

        has_format = any(re.search(p, content, re.IGNORECASE) for p in [
            r'Username:',
            r"Host's Username:",
            r"Name of Application Reader:",
            r"Application Result Link:",
            r'Tryout Date:',
            r'Roblox Profile Link:',
            r'Division:',
            r'Ping:',
        ])

        if not has_format:
            return
        
        cleaned_nickname = clean_nickname(message.author.display_name)

        sc_msg = None
        sc_embed = None
        description = f"**Manage their request via the buttons below.**"

        view = RequestView(message, self.bot, Config.BG_CHECKER_ROLE_ID) 
        panel_embed = discord.Embed(
            description=description,
            color=discord.Color.blue()
        ).set_author(name=f"SECURITY CHECK PANEL FOR {cleaned_nickname.upper()}", icon_url=Config.ID_ICON)
        panel_embed.set_footer(text="Fetching data...")
       
        
        await message.add_reaction("⌛")
        await asyncio.sleep(0.1)

        try:
            sc_request = await message.reply(embed=panel_embed, view=view, allowed_mentions=discord.AllowedMentions.none())
        except Exception as e:
            logger.error(f"Failed to send sc panel embed: {e}")
            return
            
        try:
            await save_pending_checks(
                    self.bot, 
                    str(sc_request.id), 
                    str(Config.BG_CHECKER_ROLE_ID), 
                    str(message.id), 
                    str(message.channel.id)
            )

            sc_cog = self.bot.get_cog("ScCog")
            data = await sc_cog.fetch_data(user_id, include_badges=True)

            check = data.get('badge_count')

            if check is None:
                check = len(data.get('badges') or [])

            if check == -1:
                await message.reply("```Inventory is private, please publicise it before someone conducts a security check on you.```")
            if check == -2:
                await message.reply("```Invalid user. Please double check the link.```")
            
            caution = ""
            if sc_cog and check != -2:
                field1 = f"[Jump to Request]({message.jump_url})\n"
                field2 = f"[Jump to Panel]({sc_request.jump_url})"
                
                try:
                    link_id = await self.bot.roblox.get_user_id(cleaned_nickname)
                except Exception:
                    link_id = None

                if link_id != user_id:
                   caution = "**Caution:** Roblox Profile provided does not match their Discord's display name."


                title_embed = discord.Embed(description=f"{caution if caution else ""}", color=discord.Color.blurple())
                title_embed.add_field(name=f"View request", value=field1)
                title_embed.add_field(name="Manage request", value=field2, inline=True)
                title_embed.set_author(name=f"{cleaned_nickname} Has Requested a Security Check",
                                       icon_url=Config.NOTIF_URL)

                sc_embed = await sc_cog.compile_sc_embed(user_id, data)
                badge_data = (200, data.get("badges") or [])

                ld_channel = self.bot.get_channel(Config.LD_CHANNEL_ID)
                if ld_channel and sc_embed:
                    embeds_to_send = [title_embed, sc_embed]

                    sc_msg = await ld_channel.send(embeds=embeds_to_send)
                    msg_content = f"[View information for {cleaned_nickname}]({sc_msg.jump_url})" 
                    panel_embed.description = description + "\n\n" + msg_content
                    panel_embed.set_footer(text="")

                    await asyncio.sleep(0.3)
                    await sc_request.edit(embed=panel_embed)

                    username = data["user_info"].get("name")
                    bh_embed, bh_file = await sc_cog.compile_badge_history_embed(user_id, username, data=badge_data)

                    await asyncio.sleep(0.3)

                    if bh_embed:
                        embeds_to_send.append(bh_embed)
                        await sc_msg.edit(
                                embeds=embeds_to_send,
                                attachments=[bh_file] if bh_file else []
                        )

            logger.info("Processed Security Check | MsgID={%s}", message.id)

        except Exception as e:
            logger.error("Failed to check sc request: %s", e)
            try:
                if sc_request:
                    panel_embed.set_footer(text="Could not fetch data.")
                    await sc_request.edit(embed=panel_embed)
            except Exception as inner_e:
                logger.error(f"Failed to update panel embed after error:{inner_e}")
                pass


    async def discharge_request(self, message: discord.Message):
        reason_search = re.search(r'reason for discharge:\s*(.+)', message.content, re.IGNORECASE)

        if not reason_search:
            reason_search = re.search(r'discharge:\s*(.+)', message.content, re.IGNORECASE)

        if reason_search:
            reason = reason_search.group(1).strip()       
        else: 
            return

        reason = str(reason_search.group(1))

        cleaned_nickname = clean_nickname(message.author.display_name)
        view = DischargeView(message, reason) 

        embed = discord.Embed(
            description="**How should they be discharged?**",
            color=discord.Color.blue()
        ).set_author(name=f"DISCHARGE REQUEST PANEL FOR {cleaned_nickname.upper()}", icon_url=Config.LOGOUT_ICON) 
        embed.set_footer(text="We're sad to see you go.")

        discharge_panel = await message.reply(embed=embed, view=view, allowed_mentions=discord.AllowedMentions.none())
        await message.add_reaction("⌛")
        
        #The reason is saved as as request message id to cut corners until I decide to create more tables (in db) and functions
        try:
            await self.bot.db.supabase.table(PENDING_CHECK_TABLE).insert({
                "panel_id": str(discharge_panel.id), 
                "request_id": str(message.id),
                "channel_id": str(message.channel.id),
                "interaction_user": str(reason) 
            }).execute()
            logger.info("Processed Discharge Request | MsgID={%s}", message.id)
        except Exception as e:
            logger.error(f"DB Update failed: {e}")        
 

    async def induct_request(self, message: discord.Message):
        content = message.content
        guild = message.guild

        link_match = re.search(r'Username:\s*([^\s(]+)', content, re.IGNORECASE)
        user_match = not link_match and re.search(r'Inductee\(s\):\s*(\w+)', content, re.IGNORECASE)
        match = link_match or user_match
        
        if not match:
            has_format = any(re.search(p, content, re.IGNORECASE) for p in [
                r'Course Selected:',
                r'Requirement Proof:',
                r'Ping:',
            ])

            if not has_format:
                return
     
        inductor = clean_nickname(message.author.display_name)
        raw_inductee = match.group(1).strip("<@>") if match else None
        
        if raw_inductee and raw_inductee.isnumeric():
            member = guild.get_member(int(raw_inductee))
            inductee_name = clean_nickname(member.display_name) if member else raw_inductee
        elif raw_inductee:
            inductee_name = raw_inductee
        else:
            inductee_name = inductor
        
        try: 
            description = (
                    f"**{inductor}** is requesting an induction on behalf of" 
                    f" **{inductee_name if inductor != inductee_name else "themself"}**."
                    f"\n\n**Manage their request via the buttons below.**"
            )

            view = RequestView(message, self.bot, Config.LA_ROLE_ID) 
            embed = discord.Embed(
                description=description,
                color=discord.Color.pink()
            ).set_author(name=f"INDUCTION REQUESTED PANEL FOR {inductee_name.upper()}", icon_url=Config.ID_ICON)
            
            induct_panel = await message.reply(embed=embed, view=view, allowed_mentions=discord.AllowedMentions.none())
            await message.add_reaction("⌛")
           
            await save_pending_checks(self.bot, str(induct_panel.id), str(Config.LA_ROLE_ID), str(message.id), str(message.channel.id))
            logger.info("Processed Induction Request | MsgID={%s}", message.id)
        except Exception as e:
            logger.error("Failed to check induction request: %s", e)


    # --- Event listener ---
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        await self.bot.rate_limiter.wait_if_needed(bucket="message_log")

        if message.channel.id == Config.CASE_LOGS_CHANNEL_ID:
            return await self.case_logs(message)

        if message.channel.id == Config.SC_CHANNEL_ID:
            return await self.security_check(message)
        
        if message.channel.id == Config.DISCHARGE_REQUEST_CHANNEL_ID:
            return await self.discharge_request(message)

        if message.channel.id == Config.LA_INDUCTION_CHANNEL_ID:
            return await self.induct_request(message)
 
            
    @commands.Cog.listener()
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent):

        await self.bot.rate_limiter.wait_if_needed(bucket="message_log")

        message_id = payload.message_id
        channel_id = payload.channel_id

        monitored_channels = [Config.SC_CHANNEL_ID, Config.DISCHARGE_REQUEST_CHANNEL_ID, Config.LA_INDUCTION_CHANNEL_ID]
        bl_req =[Config.B_LOG_CHANNEL_ID, Config.HR_CHAT_CHANNEL_ID] 

        if channel_id in monitored_channels:
            res = await self.bot.db.supabase.table(PENDING_CHECK_TABLE).select("*").eq("request_id", str(message_id)).maybe_single().execute()
            
            if res and res.data:
                data = res.data
                channel = self.bot.get_channel(channel_id)
                if channel:
                    try:
                        msg = await channel.fetch_message(int(data['panel_id']))
                        await msg.delete()
                    except Exception:
                        pass # Message might already be gone
                
                logger.info("Deleting a pending_check as a result of a deleted message=%s.", message_id)
                await self.bot.db.supabase.table(PENDING_CHECK_TABLE).delete().eq("request_id", str(message_id)).execute()
            
            else:
                res = await self.bot.db.supabase.table(PENDING_CHECK_TABLE).select("*").eq("panel_id", str(message_id)).maybe_single().execute()
                if res and res.data:
                    logger.info("Deleting a pending_check as a result of a deleted message=%s.", message_id)
                    await self.bot.db.supabase.table(PENDING_CHECK_TABLE).delete().eq("panel_id", str(message_id)).execute()

            return
        
        if channel_id in bl_req:
            if payload.cached_message and (payload.cached_message.author.bot != False):
                return
            await self.bot.db.supabase.table(PENDING_BL_TABLE).delete().eq("message_id", str(message_id)).execute()

        if channel_id == Config.CASE_LOGS_CHANNEL_ID:
            return await self.bot.db.remove_criminal_record(message_id)
            

async def setup(bot):
    await bot.add_cog(MessageLoggerCog(bot))
