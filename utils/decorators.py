import discord
from typing import Union
from config import Config  
from discord.ext import commands

def is_admin_or_dev():
    async def predicate(ctx: Union[commands.Context, discord.Interaction]) -> bool:
        member = ctx.author if isinstance(ctx, commands.Context) else ctx.user

        is_dev = member.id == Config.DEVELOPER_ID
        is_admin = ctx.guild and member.guild_permissions.administrator
        
        if is_dev or is_admin:
            return True

        return False
    return commands.check(predicate)

async def has_modular_permission_check(ctx: Union[commands.Context, discord.Interaction], group_type: str):
    member = ctx.author if isinstance(ctx, commands.Context) else ctx.user
    user_id = member.id

    if user_id == Config.DEVELOPER_ID:
        return True
    
    if ctx.guild and member.guild_permissions.administrator:
        return True

    allowed_ids = await ctx.bot.permissions.get(group_type)
    if not allowed_ids:
        return False 

    member_role_ids = {role.id for role in member.roles}
    
    if not member_role_ids.isdisjoint(allowed_ids):
        return True

    return False


def has_modular_permission(group_type: str):
    async def predicate(ctx: Union[commands.Context, discord.Interaction]) -> bool:
        member = ctx.author if isinstance(ctx, commands.Context) else ctx.user
        user_id = member.id

        if user_id == Config.DEVELOPER_ID:
            return True
        
        if ctx.guild and member.guild_permissions.administrator:
            return True

        allowed_ids = await ctx.bot.permissions.get(group_type)
        if not allowed_ids:
            return False 


        member_role_ids = {role.id for role in member.roles}
        
        if not member_role_ids.isdisjoint(allowed_ids):
            return True

        return False

    return commands.check(predicate)


def has_bg_role():
    async def predicate(ctx: Union[commands.Context, discord.Interaction]) -> bool:

        member = ctx.author if isinstance(ctx, commands.Context) else ctx.user
        user_id = member.id


        if user_id == Config.DEVELOPER_ID :
            return True

        if ctx.guild and member.guild_permissions.administrator:
            return True

        member_role_ids = {role.id for role in member.roles}

        if Config.BG_CHECKER_ROLE_ID in member_role_ids:
            return True

        return False
    return commands.check(predicate)

def min_rank_required(required_role_id: int):
    async def predicate(ctx: Union[commands.Context, discord.Interaction]) -> bool:

        member = ctx.author if isinstance(ctx, commands.Context) else ctx.user

        if member.id == Config.DEVELOPER_ID:
            return True

        if member.guild_permissions.administrator:
            return True

        required_role = ctx.guild.get_role(required_role_id)

        if not required_role:
            return False

        if any(role.position >= required_role.position for role in member.roles):
            return True

        return False
    return commands.check(predicate)


async def min_rank_required2(required_role_id: int, ctx: Union[commands.Context, discord.Interaction]) -> bool:

    member = ctx.author if isinstance(ctx, commands.Context) else ctx.user

    if member.id == Config.DEVELOPER_ID:
        return True

    if member.guild_permissions.administrator:
        return True

    required_role = ctx.guild.get_role(required_role_id)
    if not required_role:
        return False

    if any(role.position >= required_role.position for role in member.roles):
        return True

    return False

async def has_role(role_id: int, ctx: Union[commands.Context, discord.Interaction], occupying_user = None) -> bool:

    member = ctx.author if isinstance(ctx, commands.Context) else ctx.user

    if member.id == Config.DEVELOPER_ID :
        return True

    if not isinstance(member, discord.Member):
        return False

    if member.guild_permissions.administrator:
        return True
    
    member_role_ids = {role.id for role in member.roles}

    if Config.HIGH_COMMAND_ROLE_ID in member_role_ids:
        return True
    
    if occupying_user and member.id != occupying_user:
        return False

    return role_id in member_role_ids


