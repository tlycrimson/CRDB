from discord.ext import commands

def has_allowed_role():
    async def predicate(ctx):
        allowed_role_id = 1335394269535666216  # Your allowed role ID
        if any(role.id == allowed_role_id for role in getattr(ctx.author, "roles", [])):
            return True
        await ctx.send("⛔ You don't have permission to use this command.")
        return False
    return commands.check(predicate)

def min_rank_required(required_rank_name: str):
    async def predicate(ctx):
        member_roles = ctx.author.roles
        guild_roles = ctx.guild.roles

        required_role = discord.utils.get(guild_roles, name=required_rank_name)
        if not required_role:
            await ctx.send(f"⚠️ Role `{required_rank_name}` not found.")
            return False

        if any(role.position >= required_role.position for role in member_roles):
            return True

        await ctx.send(f"⛔ You need at least the `{required_rank_name}` role to use this command.")
        return False
    return commands.check(predicate)