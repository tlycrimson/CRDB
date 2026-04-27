import re
import asyncio
import discord
import mimetypes
import logging
from datetime import datetime, timezone
from config import Config
from types import SimpleNamespace
from utils import embedBuilder
from utils.decorators import min_rank_required2, has_role
from utils.helpers import clean_nickname, MockPayload

logger =  logging.getLogger(__name__)

# ----- TABLE NAMES ----
PENDING_BL_TABLE = "pending_blacklist_requests"
PENDING_CHECK_TABLE = "pending_checks"

class ConfirmView(discord.ui.View):
    def __init__(self, author: discord.User, *, timeout: float = 30.0):
        super().__init__(timeout=timeout)
        self.author = author
        self.value: bool | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author.id:
            await interaction.response.send_message(
                "```⛔ You cannot respond to this confirmation.```",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.green)
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


class PageButtonView(discord.ui.View):
    def __init__(self, user: discord.User, current_page, total_page, data = None, timeout: float = 300.0):
        super().__init__(timeout=timeout)
        self.user = user
        self.current_page = current_page
        self.total_page = total_page
        self.data = data
        self.message = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user.id:
            await interaction.response.send_message(
                "```⛔ You cannot respond to this confirmation.```",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="←", style=discord.ButtonStyle.red)
    async def left(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page =  (self.current_page - 1) % self.total_page
        if self.data:
            embed = interaction.message.embeds[0]
            embed.description = self.data[self.current_page]
        else:
            embed = embedBuilder.build_commands_page(self.current_page, interaction.client.command_prefix)

        try:
            await interaction.message.edit(embed=embed)
        except Exception:
            pass

    @discord.ui.button(label="→", style=discord.ButtonStyle.red)
    async def right(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page =  (self.current_page + 1) % self.total_page
        if self.data:
            embed = interaction.message.embeds[0]
            embed.description = self.data[self.current_page]
        else:
            embed = embedBuilder.build_commands_page(self.current_page, interaction.client.command_prefix)

        try:
            await interaction.message.edit(embed=embed)
        except Exception:
            pass
        
    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass


class DischargeView(discord.ui.View):
    def __init__(self, discharge_req_msg, reason_for_discharging):
        super().__init__(timeout=None)
        self._processing = False
        self.discharge_req_msg = discharge_req_msg
        self.reason = reason_for_discharging

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not (await min_rank_required2(Config.HIGH_COMMAND_ROLE_ID, interaction)):
            if self._processing:
                return False

            await interaction.response.send_message(
                "```⛔ You cannot respond to this confirmation.```",
                ephemeral=True
            )
            return False

        if interaction.user.id == self.discharge_req_msg.author.id:
            await interaction.response.send_message(
                "```❌ You cannot discharge yourself.```",
                ephemeral=True
            )
            return False

        return True

    @discord.ui.select(
            cls=discord.ui.Select,
            placeholder="Select a discharge type",
            options = [
            discord.SelectOption(label="Honourable", description="Honourable Discharge Type"),
            discord.SelectOption(label="General", description="General Discharge Type"),
            ],
            custom_id="select_discharge"
    )
    async def my_select_callback(self, interaction: discord.Interaction, select):
        admincog = interaction.client.get_cog("AdminCog")
        requster_user = self.discharge_req_msg.author.id
        success = False
        try:
            ctx = await interaction.client.get_context(interaction)
            success = await admincog.discharge.callback(admincog, ctx, f"<@{requster_user}>", select.values[0], self.reason)
        except Exception as e:
            logger.error(f"Error occured while trying to discharge {requster_user}: {e}")
            await interaction.response.send_message("```❌ An error occured while trying to discharge them.```", ephemeral=True)
        
        if not success:
            return
        try:
            await self.discharge_req_msg.remove_reaction("⌛", interaction.client.user)
            await self.discharge_req_msg.add_reaction("🟢")
        except Exception:
            pass

        await delete_pending_checks(interaction.client, interaction.message.id)
        await interaction.message.delete()

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red, custom_id="deny_discharge")
    async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(ReasonModal(self.discharge_req_msg, interaction))


class ReasonModal(discord.ui.Modal, title='Response'):
    def __init__(self, req_msg: discord.Message, button_interaction: discord.Interaction, _return: bool = False):
        super().__init__()
        self.msg = req_msg
        self.interaction = button_interaction
        self._return = _return

    reason = discord.ui.TextInput(label='Reason', style=discord.TextStyle.paragraph)

    async def on_submit(self, interaction: discord.Interaction):
        if self._return:
            await interaction.response.send_message("```✅ Reason recorded.```", ephemeral=True)
            return

        embed = discord.Embed(
            color=discord.Color.red(),
            timestamp=datetime.now(timezone.utc)
        ).set_footer(text=f"Denied by {clean_nickname(interaction.user.display_name)}")

        embed.add_field(name="Reason for Denial:", value=self.reason.value, inline=False)
        embed.set_author(name="DISCHARGE REQUEST DENIED", icon_url=Config.CANCEL_URL)
        embed.set_thumbnail(url=Config.RMP_URL)

        try:
            channel = interaction.guild.get_channel(Config.MAIN_COMMS_CHANNEL_ID)
            await channel.send(content=self.msg.author.mention,embed=embed)
            await interaction.response.send_message(
            f"```✅ I have notified them of their denial.```",
            ephemeral=True
        )
        except discord.Forbidden as e:
            logger.error("Failed to send discharge request denial embed: %s", e)
            await interaction.response.send_message(
                    f"```❌ I couldn't notfiy them of their denial."
                    ,ephemeral=True)
       
        try:
            await self.msg.remove_reaction("⌛", interaction.client.user)
            await self.msg.add_reaction("🔴")
        except Exception:
            pass

        await asyncio.sleep(0.5)
        
        await self.interaction.message.delete()
        await delete_pending_checks(interaction.client, self.interaction.message.id)


class BlacklistReasonModal(discord.ui.Modal, title='Response'):
    def __init__(self, interaction_data, view, button_interaction):
        super().__init__()
        self.interaction_data = interaction_data
        self.view = view
        self.interaction = button_interaction

    reason = discord.ui.TextInput(label='Reason', style=discord.TextStyle.paragraph)

    async def on_submit(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="Blacklist Request Denied",
            color=discord.Color.red(),
            timestamp=datetime.now(timezone.utc)
        )

        embed.add_field(name="Reason for Denial:", value=self.reason.value, inline=False)
        embed.add_field(
            name="Members you wanted to blacklist:",
            value=(','.join(interaction.guild.get_member(member).display_name for member in self.interaction_data.members))
        )
        embed.add_field(name=f"For Duration ({self.interaction_data.unit}):", value=self.interaction_data.duration, inline=False)

        embed.add_field(name="Reason you wanted to blacklist them:", value=self.interaction_data.reason, inline=False)

        if self.interaction_data.evidence:
            embed.add_field(name="With Evidence:", value=f"[Attachment Link]({self.interaction_data.evidence.url})", inline=False)
            mime, _ = mimetypes.guess_type(self.interaction_data.evidence.filename)
            if mime and mime.startswith("image/"):
                embed.set_image(url=self.interaction_data.evidence.url)

        try:
            await self.interaction_data.issuer.send(embed=embed)
        except discord.Forbidden:
            if channel := interaction.guild.get_channel(Config.HR_CHAT_CHANNEL_ID):
                await channel.send(f"{self.interaction_data.issuer.mention}", embed=embed)

        await interaction.response.send_message(
            f"```✅ {self.interaction_data.issuer.display_name} has been notified of the denial.```",
            ephemeral=True
        )

        await self.interaction.message.delete()

        # Clean up the DB row now that the request is resolved
        await delete_pending_blacklist(self.view.bot, self.interaction.message.id)

        self.view.stop()

class HaltReasonModal(discord.ui.Modal, title='Reason for Halt'):
    HALT_MESSAGES = {
        "security": (
            "Your Royal Military Police background check has been halted for the following reasons:\n{reason}"
            "\n\n**You have {hours} hours to comply. Once you have, use the button to let the checker know.**"
        ),
        "induction": (
            "Your Royal Military Police induction check has been halted for the following reasons:\n{reason}"
            "\n\n**You have {hours} hours to comply. Once you have, use the button to let the checker know.**"
        ),
    }

    TIMEOUT_BASE = 3600

    def __init__(self, original_msg, bot, checker_type):
        super().__init__()
        self.original_msg = original_msg
        self.bot = bot
        self.checker_type = checker_type

        self.hours = discord.ui.TextInput(
            label='Hours to Comply',
            placeholder='Time in Hours',
            min_length=1,
            max_length=3,
            required=True,
        )
        self.reason = discord.ui.TextInput(
            label='Reason',
            style=discord.TextStyle.paragraph,
            required=True,
        )

        self.add_item(self.hours)
        self.add_item(self.reason)

    def _get_default_message(self, hours, reason) -> str:
        if self.checker_type == Config.BG_CHECKER_ROLE_ID:
            return self.HALT_MESSAGES["security"].format(hours=hours, reason=reason)
        elif self.checker_type == Config.LA_ROLE_ID:
            return self.HALT_MESSAGES["induction"].format(hours=hours, reason=reason)
        return ""

    async def on_submit(self, interaction: discord.Interaction):
        selected_hours = self.hours.value.strip()

        if int(selected_hours)>120:
            await interaction.response.send_message(
                "```❌ Time cannot exceed 5 days.```",
                ephemeral=True,
            )
            return

        timeout = self.TIMEOUT_BASE * int(selected_hours)

        embed = embedBuilder.build_request_respsone(
            accepted=False,
            halted=True,
            reason=self._get_default_message(selected_hours, self.reason.value),
            username=interaction.user.display_name,
            check_type=self.checker_type
        )

        halt_response_msg = await self.original_msg.reply(embed=embed)
        notif_channel = interaction.guild.get_channel(Config.MAIN_COMMS_CHANNEL_ID)

        view = ComplyView(
            self.bot,
            interaction.user.id,
            self.original_msg.author.id,
            notif_channel,
            halt_response_msg,
            self.original_msg,
            self.checker_type,
            timeout,
            timeout=timeout,
        )
        await halt_response_msg.edit(view=view)
        await asyncio.sleep(0.1)
        
        try:
            await self.original_msg.remove_reaction("⌛", self.bot.user)
            await self.original_msg.add_reaction("🟡")
            await asyncio.sleep(0.1)
            await interaction.response.send_message("```✅ Successfully halted the check.```", ephemeral=True)
        except Exception:
            pass

        try:
            await (
                self.bot.db.supabase
                .table(PENDING_CHECK_TABLE)
                .update({
                    "interaction_user": str(interaction.user.id),
                    "halt_msg_id": str(halt_response_msg.id),
                    "complying_user": str(self.original_msg.author.id),
                    "time_given": str(timeout)
                })
                .eq("request_id", str(self.original_msg.id))
                .execute()
            )
        except Exception as e:
            logger.error(f"DB Update failed: {e}")

# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
async def save_pending_blacklist(bot, message_id: int, channel_id: int, interaction_data) -> None:
    await bot.db.supabase.table(PENDING_BL_TABLE).insert({
        "message_id":        message_id,
        "channel_id":        channel_id,
        "issuer_id":         interaction_data.issuer.id,
        "member_ids":        interaction_data.members,          # list[int]
        "reason":            interaction_data.reason,
        "duration_unit":     interaction_data.unit,
        "duration_amount":   interaction_data.duration,
        "evidence_url":      interaction_data.evidence.url      if interaction_data.evidence else None,
        "evidence_filename": interaction_data.evidence.filename if interaction_data.evidence else None,
    }).execute()

async def save_pending_checks(bot, panel_id: str, allowed_to_interact: str, request_id: str, channel_id: str) -> None:
    await bot.db.supabase.table(PENDING_CHECK_TABLE).insert({
        "panel_id": panel_id,
        "allowed_to_interact": allowed_to_interact,
        "request_id": request_id,
        "channel_id": channel_id,
    }).execute()


async def delete_pending_blacklist(bot, message_id: int) -> None:
    """Remove a resolved request from Supabase."""
    await bot.db.supabase.table(PENDING_BL_TABLE) \
        .delete() \
        .eq("message_id", str(message_id)) \
        .execute()

async def delete_pending_checks(bot, panel_id: int) -> None:
    """Remove a resolved check from Supabase."""
    await bot.db.supabase.table(PENDING_CHECK_TABLE) \
        .delete() \
        .eq("panel_id", str(panel_id)) \
        .execute()

async def load_pending_blacklists(bot) -> list[dict]:
    """Fetch all unresolved blacklist requests from Supabase."""
    res = await bot.db.supabase.table(PENDING_BL_TABLE).select("*").execute()
    return res.data or []

async def load_pending_checks(bot) -> list[dict]:
    """Fetch all unresolved blacklist requests from Supabase."""
    res = await bot.db.supabase.table(PENDING_CHECK_TABLE).select("*").execute()
    return res.data or []


# ---------------------------------------------------------------------------
# ApprovalView
# ---------------------------------------------------------------------------
class ApprovalView(discord.ui.View):
    def __init__(self, interaction_data, adminCog, bot=None):
        super().__init__(timeout=None)
        self.interaction_data = interaction_data
        self.adminCog = adminCog
        self.bot = bot or adminCog.bot
        self._processing = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not (await min_rank_required2(Config.PM_ROLE_ID, interaction)):
            if self._processing:
                return False
            await interaction.response.send_message(
                "```⛔ You cannot respond to this confirmation.```",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green, custom_id="blacklist_approve")
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._processing = True
        await interaction.response.defer(ephemeral=True)
        await delete_pending_blacklist(self.bot, interaction.message.id)
        await interaction.message.delete()
        await self.adminCog.blacklist_members(self.interaction_data, interaction)


    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red, custom_id="blacklist_deny")
    async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._processing = True
        await interaction.response.send_modal(BlacklistReasonModal(self.interaction_data, self, interaction))
    

# ---------------------------------------------------------------------------
# ApprovalView without ReasonModal
# ---------------------------------------------------------------------------
class AoDView(discord.ui.View):
    def __init__(self, interaction_data, welcomeCog, bot=None):
        super().__init__(timeout=None)
        self.interaction_data = interaction_data
        self.wCog = welcomeCog
        self.bot = bot or welcomeCog.bot
        self._processing = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self._processing:
            return False
        if not (await min_rank_required2(Config.HIGH_COMMAND_ROLE_ID, interaction)):
            await interaction.response.send_message(
                "```⛔ You cannot respond to this confirmation.```",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green, custom_id="approve_blacklist")
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._processing = True
        await interaction.response.defer(ephemeral=True)
        await delete_pending_blacklist(self.bot, interaction.message.id)
        await self.wCog.register_deserter(self.interaction_data, interaction)
        self.stop()

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red, custom_id="deny_blacklist")
    async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._processing = True
        await interaction.response.defer(ephemeral=True)
        await interaction.message.delete()
        self.stop()

# ---------------------------------------------------------------------------
# Request View Approval
# ---------------------------------------------------------------------------
class RequestView(discord.ui.View):
    def __init__(self, original_msg, bot, allowed_to_interact, interaction_user = None):
        super().__init__(timeout=None)
        self.original_msg = original_msg
        self.bot = bot
        self._processing = False
        self.interaction_user = interaction_user
        self.checker_type = allowed_to_interact
        self.rl = self.bot.get_cog("ReactionLoggerCog")
        if not self.rl:
            logger.error("Could not find ReactionLogger Cog.")
            raise Exception
 
    async def interaction_check(self, interaction: discord.Interaction) -> bool:    
        if self._processing:
            await interaction.response.send_message("```⛔ You cannot respond to this.```", ephemeral=True)
            return False

        if not (await has_role(self.checker_type, interaction, self.interaction_user)):
            await interaction.response.send_message("```⛔ You cannot respond to this.```", ephemeral=True)
            return False

        return True

    @discord.ui.button(label="Accept", style=discord.ButtonStyle.green, custom_id="accept_req")
    async def approve_sc(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._processing = True
        await interaction.response.defer(ephemeral=True)
        
        embed = embedBuilder.build_request_respsone(True, interaction.user.display_name, self.checker_type)
       
        await delete_pending_checks(self.bot, interaction.message.id)

        await interaction.message.delete()
        try:
            await self.original_msg.remove_reaction("⌛", self.bot.user)
            await asyncio.sleep(0.3)
            await self.original_msg.remove_reaction("🟡", self.bot.user)
            await asyncio.sleep(0.5)
            await self.original_msg.add_reaction("🟢")
        except Exception:
            pass

        await self.original_msg.reply(embed=embed)

       
        if self.checker_type == Config.LA_ROLE_ID:
            emoji = discord.PartialEmoji.from_str("🟢")
            payload = MockPayload(
                guild_id=interaction.guild.id,
                channel_id=interaction.channel.id,
                message_id=self.original_msg.id,
                user_id=interaction.user.id,
                emoji=emoji,
                member=interaction.user
            )
            await self.rl._log_la_and_examiner_impl(payload, interaction.guild, interaction.user)
        
        self.stop()

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red, custom_id="deny_req")
    async def deny_sc(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._processing = True
        
        reason = ""
        if self.checker_type == Config.LA_ROLE_ID:
            modal = ReasonModal(self.original_msg, interaction, True)
            await interaction.response.send_modal(modal)
            await modal.wait()
            if modal.reason:
                reason = modal.reason
            else:
                return
        else:
            await interaction.response.defer(ephemeral=True)

        embed = embedBuilder.build_request_respsone(False, interaction.user.display_name, self.checker_type, False, reason)
        await delete_pending_checks(self.bot, interaction.message.id)

        await interaction.message.delete()
        try:
            await self.original_msg.remove_reaction("⌛", self.bot.user)
            await asyncio.sleep(0.3)
            await self.original_msg.remove_reaction("🟡", self.bot.user)
            await asyncio.sleep(0.5)
            await self.original_msg.add_reaction("🔴")
        except Exception:
            pass

        await self.original_msg.reply(embed=embed)

        await delete_pending_checks(self.bot, self.original_msg.id)
        
        if self.checker_type == Config.LA_ROLE_ID:
            emoji = discord.PartialEmoji.from_str("🔴")
            payload = MockPayload(
                guild_id=interaction.guild.id,
                channel_id=interaction.channel.id,
                message_id=self.original_msg.id,
                user_id=interaction.user.id,
                emoji=emoji,
                member=interaction.user
            )
            await self.rl._log_la_and_examiner_impl(payload, interaction.guild, interaction.user)
 
        self.stop()

    @discord.ui.button(label="Halt", style=discord.ButtonStyle.grey, custom_id="halt_req")
    async def halt_sc(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._processing = True
        modal = HaltReasonModal(self.original_msg, self.bot, self.checker_type)
        await interaction.response.send_modal(modal)
        await modal.wait()
        self._processing = False 
        


class ComplyView(discord.ui.View):
    def __init__(self, bot, checker_user, complying_user, notif_channel, halt_msg, request_msg, check_type, time_given, timeout=172800):
        super().__init__(timeout=timeout)
        self.bot = bot
        self.checker_user_id = checker_user
        self.complying_user_id = complying_user
        self.notif_channel = notif_channel
        self.request_msg = request_msg # The original messsage (the one the user sent)
        self.halt_msg = halt_msg # The message this view is on
        self.check_type = check_type #This is An Int, cba to specify in function param
        self.update_type = "Security Check"
        self.time_given = round(time_given/3600)

        if self.check_type == Config.LA_ROLE_ID:
            self.update_type = "Induction Request"


    async def interaction_check(self, interaction: discord.Interaction) -> bool:    
        if interaction.user.id != self.complying_user_id:
            await interaction.response.send_message("```⛔ You cannot respond to this confirmation.```", ephemeral=True)
            return False

        return True

    async def _clear_pending_status(self):
        try:
            await (
                self.bot.db.supabase
                .table(PENDING_CHECK_TABLE)
                .update({
                    "halt_msg_id": None,
                    "complying_user": None,
                    "time_given": None
                })
                .eq("request_id", str(self.request_msg.id))
                .execute()
            )
        except Exception as e:
            logger.error(f"_clear_pending_status failed: {e}")

    async def on_timeout(self):
        try:
            embed = self.halt_msg.embeds[0].add_field(name="Updated Status: Not Complied", value=f"\u200B")
            await self.halt_msg.edit(embed=embed, view=None)
        except Exception:
            pass
        
        msg = f"<@{self.complying_user_id}> has failed to comply with the instructions provided when you halted their [check]({self.request_msg.jump_url}) in the {self.time_given} hour period given."

        notif_embed = embedBuilder.build_comply_notif(msg, self.update_type) 

        await self.notif_channel.send(content=f"<@{self.checker_user_id}>", embed=notif_embed)    
        await self._clear_pending_status()

    @discord.ui.button(label="I have Complied", style=discord.ButtonStyle.green, custom_id="complied")
    async def approve_sc(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        
        embed = interaction.message.embeds[0].add_field(name="Updated Status: Complied", value=f"\u200B")
        await interaction.message.edit(embed=embed, view=None)

        msg = f"{interaction.user.mention} has said they have followed the instructions you gave when you halted their [check]({self.request_msg.jump_url})."

        notif_embed = embedBuilder.build_comply_notif(msg, self.update_type) 
        
        try: 
            await self.notif_channel.send(content=f"<@{self.checker_user_id}>", embed=notif_embed)    
            await interaction.followup.send("```✅ I have let them know you have complied.```", ephemeral=True)
        except Exception:
            await interaction.followup.send("```❎ I could not notify them that you have complied. Please contact them yourself.```", ephemeral=True)
            pass
        
        await self._clear_pending_status()

        self.stop()

async def restore_approval_views(bot) -> None:
    """
    Re-register every ApprovalView that was pending when the bot last shut down.
    """
    blacklists = await load_pending_blacklists(bot)
    checks = await load_pending_checks(bot)  
    
    if checks:
        for row in checks:
            try:
                channel_id = int(row["channel_id"])
                panel_id = int(row["panel_id"]) 
                request_id = int(row['request_id'])
                allowed_to_interact = int(row["allowed_to_interact"]) if row["allowed_to_interact"] else None


                channel = bot.get_channel(channel_id)
                notif_channel = bot.get_channel(Config.MAIN_COMMS_CHANNEL_ID)
                await asyncio.sleep(0.2)

                #Storing reason under interaction_user for this view until im bothered enough
                if channel_id == Config.DISCHARGE_REQUEST_CHANNEL_ID:
                    reason = row["interaction_user"]
                    msg = await channel.fetch_message(request_id)
                    view = DischargeView(msg, reason)
                else:
                    sc_request_msg = await channel.fetch_message(request_id) 
                    view = RequestView(original_msg=sc_request_msg, bot=bot, allowed_to_interact=allowed_to_interact)

                bot.add_view(view, message_id=panel_id)

                if row["halt_msg_id"]:
                    sc_request_msg = await channel.fetch_message(request_id) 
                    halt_msg_id = int(row["halt_msg_id"])
                    halt_msg = await channel.fetch_message(halt_msg_id)
                    await asyncio.sleep(0.1)
                    checker_user = int(row["interaction_user"])
                    complying_user = int(row["complying_user"])

                    passed = (discord.utils.utcnow() - halt_msg.created_at).total_seconds()
                    time_given = int(row["time_given"])
                    remaining = max((time_given - passed), 120)
                    restored_embed = halt_msg.embeds[0]
                    if remaining >= 3600:
                        val = round(remaining / 3600)
                        unit = "hour" if val == 1 else "hours"
                    else:
                        val = round(remaining / 60)
                        unit = "minute" if val == 1 else "minutes"

                    time_str = f"{val} {unit}"
                    restored_embed.description = re.sub(r'\d+ (hour|minute)s?', time_str, restored_embed.description)

                    view = ComplyView(bot, checker_user, complying_user, notif_channel, halt_msg, sc_request_msg, allowed_to_interact, time_given, timeout=remaining)
                    await halt_msg.edit(embed=restored_embed, view=view)


            except Exception as e:
                logger.error(f"Failed to restore message {row.get('panel_id')}: {e}")

    if blacklists:
        admin_cog = bot.cogs.get("AdminCog")
        welcome_cog = bot.cogs.get("WelcomeCog")

        for row in blacklists:
            try:
                evidence = None
                if row["evidence_url"]:
                    evidence = SimpleNamespace(
                        url=row["evidence_url"],
                        filename=row["evidence_filename"] or "attachment",
                    )

                guild = bot.guilds[0]  
                try:
                    issuer = await guild.fetch_member(int(row["issuer_id"]))
                except discord.NotFound:
                    issuer = await bot.fetch_user(int(row["issuer_id"]))

                interaction_data = SimpleNamespace(
                    members=row["member_ids"],          
                    reason=row["reason"],
                    unit=row["duration_unit"],
                    duration=row["duration_amount"],
                    evidence=evidence,
                    issuer=issuer,
                )
                
                if row["channel_id"] == Config.B_LOG_CHANNEL_ID:
                    view = ApprovalView(interaction_data, admin_cog, bot=bot)
                else:
                    view = AoDView(interaction_data, welcome_cog, bot=bot)

                bot.add_view(view, message_id=int(row["message_id"]))

            except Exception as e:
                logger.error(f"[restore_approval_views] Failed to restore message {row.get('message_id')}: {e}")
