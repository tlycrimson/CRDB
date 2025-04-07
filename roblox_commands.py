import discord
import socket
import requests
import json
from datetime import datetime
from discord.ext import commands
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
USER_AGENT = "RMPBot/1.0 (+https://github.com/tlycrimson/RMP-Discord-Bot)"
FALLBACK_IP = "172.253.118.95"  # Update this if needed
GROUP_IDS = [32578828, 4219097]  # Groups to check membership

def get_roblox_ip():
    """Get current IP for Roblox API with fallback"""
    try:
        return socket.gethostbyname('api.roblox.com')
    except socket.gaierror:
        return FALLBACK_IP

def create_session():
    """Configure requests session with retries"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.verify = False  # Required for IP direct access
    return session

@commands.command(name="sc")
async def sc(ctx, user_id: int):
    """Fetch and display Roblox user info"""
    try:
        async with ctx.typing():
            # Initialize session and headers
            session = create_session()
            headers = {
                "Host": "api.roblox.com",
                "User-Agent": USER_AGENT,
                "Accept": "application/json"
            }
            roblox_ip = get_roblox_ip()
            base_url = f"https://{roblox_ip}"

            # 1. Fetch main user data
            user_url = f"{base_url}/users/{user_id}"
            print(f"[DEBUG] Fetching user data from: {user_url}")
            
            try:
                response = session.get(user_url, headers=headers, timeout=15)
                response.raise_for_status()
                user_data = response.json()
            except json.JSONDecodeError:
                print(f"[ERROR] Invalid JSON received:\n{response.text[:500]}")
                return await ctx.send("❌ Roblox API returned invalid data. Try again later.")
            
            if "errorMessage" in user_data:
                return await ctx.send(f"❌ Roblox API error: {user_data['errorMessage']}")

            # 2. Process user data
            username = user_data["username"]
            created_at = datetime.strptime(user_data["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
            account_age = (datetime.utcnow() - created_at).days // 365

            # 3. Fetch badges
            badges_url = f"{base_url}/users/{user_id}/badges"
            print(f"[DEBUG] Fetching badges from: {badges_url}")
            badges_data = session.get(badges_url, headers=headers, timeout=10).json()
            total_badges = len(badges_data)
            free_badges = [b for b in badges_data if b.get("isFree")]

            # 4. Check group memberships
            groups = []
            for group_id in GROUP_IDS:
                group_url = f"{base_url}/users/{user_id}/groups/{group_id}"
                print(f"[DEBUG] Checking group: {group_url}")
                group_data = session.get(group_url, headers=headers, timeout=10).json()
                if group_data.get("success"):
                    groups.append(str(group_id))

            # 5. Build and send embed
            embed = discord.Embed(
                title=f"Roblox User: {username}",
                color=discord.Color.blue()
            )
            embed.add_field(name="Account Age", value=f"{account_age} years", inline=False)
            embed.add_field(name="Total Badges", value=total_badges, inline=False)
            embed.add_field(name="Free Badges", value=len(free_badges), inline=False)
            embed.add_field(name="Groups", value=", ".join(groups) if groups else "None", inline=False)
            
            await ctx.send(embed=embed)

    except requests.exceptions.Timeout:
        await ctx.send("⏳ Roblox API timed out. Try again later.")
    except requests.exceptions.RequestException as e:
        print(f"[NETWORK ERROR] {str(e)}")
        await ctx.send(f"🔴 Network error: {str(e)}")
    except Exception as e:
        print(f"[UNEXPECTED ERROR] {type(e).__name__}: {str(e)}")
        await ctx.send(f"❌ Unexpected error: {type(e).__name__}")

def setup(bot):
    bot.add_command(sc)
