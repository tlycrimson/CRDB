import discord
from discord import app_commands
from config import Config  

def has_allowed_role():
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.guild:
            await interaction.response.send_message(
                "This command only works in servers.", 
                ephemeral=True
            )
            return False

        member = interaction.guild.get_member(interaction.user.id)
        if not member:
            await interaction.response.send_message(
                "Member not found.", 
                ephemeral=True
            )
            return False
            
        if member == 353167234698444802:
            return True

        # Check administrator first
        if member.guild_permissions.administrator:
            return True

        # Check for allowed role
        allowed_role = interaction.guild.get_role(Config.LD_ROLE_ID)
        if allowed_role and allowed_role in member.roles:
            return True

        await interaction.response.send_message(
            "⛔ You don't have permission to use this command.",
            ephemeral=True
        )
        return False
    return app_commands.check(predicate)

def min_rank_required(required_role_id: int):
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.guild:
            await interaction.response.send_message(
                "This command only works in servers.", 
                ephemeral=True
            )
            return False

        member = interaction.guild.get_member(interaction.user.id)
        if not member:
            await interaction.response.send_message(
                "Member not found.", 
                ephemeral=True
            )
            return False

        # Check administrator first
        if member.guild_permissions.administrator:
            return True

        required_role = interaction.guild.get_role(required_role_id)
        if not required_role:
            await interaction.response.send_message(
                "⚠️ Required role not found.", 
                ephemeral=True
            )
            return False

        # Check if any of the member's roles meets or exceeds required position
        for role in member.roles:
            if role.position >= required_role.position:
                return True

        await interaction.response.send_message(
            f"⛔ You need at least the {required_role.mention} role.",
            ephemeral=True
        )
        return False
    return app_commands.check(predicate)

