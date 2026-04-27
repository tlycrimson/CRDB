import discord
from config import Config
from utils.helpers import clean_nickname, get_tier_info, make_progress_bar
from datetime import datetime, timezone 


def build_comply_notif(msg: str, update_type):

        notif_embed = discord.Embed(
                description=msg, 
                color=discord.Color.dark_orange())
        notif_embed.set_author(
                name=f"{update_type} Update",
                icon_url=Config.NOTIF_URL)

        return notif_embed


def build_regiment_info(data: dict):
        info_embed = discord.Embed(color=discord.Color.red())\
                        .set_author(name="Royal Military Police",
                                    icon_url=Config.INFO_ICON,
                                    url=Config.RMP_GROUP_URL
                                    ).set_thumbnail(url=Config.RMP_URL)
                        
        info_embed.add_field(name="OVERSIGHT", value=f"───────────────────", inline=False)
        info_embed.add_field(name="Overseer:", value="dominqsss")
        info_embed.add_field(name="Provost Marshal:", value=data["pm"])


        info_embed.add_field(name="GENERAL STATS", value=f"───────────────────", inline=False)
        info_embed.add_field(name="Members:", value=data["member_count"])
        info_embed.add_field(name="Total XP:", value=data["total_xp"])

        info_embed.add_field(name="HR STATS", value=f"───────────────────", inline=False)
        info_embed.add_field(name="HRs:", value=data["hr_count"])
        info_embed.add_field(name="Hosted:", value=data["total_events"])
        info_embed.add_field(name="Dep Points:", value=data["total_dep_points"])

        info_embed.add_field(name="LR STATS", value=f"───────────────────", inline=False)
        info_embed.add_field(name="LRs:", value=data["lr_count"])
        info_embed.add_field(name="Events Attended:", value=data["total_attended_events"])
        info_embed.add_field(name="Activity:", value=data["total_activity"])
        info_embed.add_field(name="Guard Activity:", value=data["total_guard_activity"])

        return info_embed

def build_change_log(prefix):
        title = "MP Assistant v1.3 Change Logs"
        description = "Below features the recent changes made to the bot. If you wish to make a suggestion to improve the bot, use the /suggest command.\n\n"
        footer = "Last Updated: April 2026"
        updates = (
                          f"- **Commands can now be executed with the bot's prefix ({prefix}) and name or their shorthand: !lb**\n"
                          "- Set-permissions introduces granular control over who can run specific categories of commands.\n"
                          "- Improved automation for security checks, discharge requests and induction requests.\n"
                          "- The blacklist command initiated by those below the rank of PM requires manual approving.\n"
                          "- The !commands has a refined UI with button interactions.\n"
                          "- Case logs get automatically recorded and can be managed via !manage-case-logs(mcl).\n"
                          "- The security check command now accepts the username and ID along with an improved display.\n"
                          "- Changed Response & Logs Layout.\n"
                          "- Leaderboard command now displays the full rankings with different categories to choose from.\n"
                          "- Improved bug report interface.\n"
                          "- Anyone can now suggest new ideas or changes for the bot.\n"
                          "- RMP members can view their data via the profile command.\n"
                          "- Info and Stats about RMP can now be displayed using !rmp-info (info).\n"
                          "- The bot now has a privacy policy to explain how user data is handled and stored."
                        
        )
        new_commands = (
                        "- /set-permissions\n"
                        "- /restore-user\n"
                        "- /remove-user\n"
                        "- /profile\n"
                        "- /manage-case-logs\n"
                        "- /policy\n" 
                        "- /rmp-info\n"
                        "- /suggest\n"
                        "- /change-logs"
        )

        acknowledgements = "- Most icons used by the bot are from [icons8](https://icons8.com/icons).\n- AI was used in the production of the bot."

        top_embed = discord.Embed(description=description, color=discord.Color.dark_red())
        top_embed.set_author(name=title, icon_url=Config.OPEN_BOOK_ICON)
        top_embed.set_image(url=Config.BOT_BANNER_CROPPED)
        top_embed.set_thumbnail(url=Config.BOT_ICON)

        embed = discord.Embed(description=updates,color=discord.Color.dark_red())
        embed.add_field(name="New Commands", value=new_commands,inline=False)
        embed.add_field(name="Acknowledgements", value=acknowledgements, inline=False)
        embed.set_author(name="Change Logs",icon_url=Config.BOT_ICON)
        embed.set_footer(text=footer)

        return [top_embed, embed]


def build_privacy_policy():
        title = "MP Assistant Privacy Policy"
        footer = "Last Updated: April 2026"
        member_records = (
                          "Once assigned the role that designates you an official regimental member in the server, the following pieces of information will be stored: your Discord ID, Roblox ID, Roblox username, your XP, and your regiment's rank and division. The information will be used to manage and track your membership, rank progression, and other server statistics. This data will be removed upon removal of the role or manually removed by permitted members. A backup copy of your data will be stored for a thirty-day period in case of a reinstatement, after which it will be deleted permanently."
        )
        arrest_logs = (
                        " When an official member provides an arrest log related to a particular Roblox user, the information on their Roblox ID, Roblox username, and the link to their log message will be stored. This data can be used as part of a background check conducted by official members. The information will be automatically and permanently deleted after two days or manually removed by the members."
        )
        data_usage = (
                        f" Your information is used exclusivelys for the running of the regiment system and upholding the integrity of the roleplay on this server. Your information will not be shared, sold, or used for advertising purposes.\n"
                        f"In accordance with your rights under GDPR and other data protection laws, you are entitled to:\n"
                        f"- Request a copy of the data we hold about you (for members you can view most by /profile)\n"
                        f"- Request correction of inaccurate data\n"
                        f"- Request deletion of your data (subject to our legitimate moderation interests)\n"
                        f"To exercise any of these rights, please contact the Developer ({Config.DEVELOPER_ID})."
        )
        data_retention_summary = (
                        f"Member records (active) - Duration of membership\n"
                        f"Member records (backup) - 30 days post-discharge\n"
                        f"Arrest logs - 2 days\n\n"
                        f"We do not knowingly store data relating to anyone under the age of 13, in line with Discord's Terms of Service."
                        ) 
        embed = discord.Embed(color=discord.Color.dark_red()).set_footer(text=footer)
        
        embed.set_author(name=title, icon_url=Config.PRIVACY_ICON)
        embed.add_field(name="What data we collect and why",value=member_records+arrest_logs, inline=False)
        embed.add_field(name="How Your Data Is Used", value=data_usage, inline=False)
        embed.add_field(name="Data Retention Summary", value=data_retention_summary)
        return embed

def build_commands_page(page: int, prefix):
        title = "Commands"
        description= (  
                        f"**The bot's prefix is {prefix}**\n\n"
                        "Permissions for each group apart from the last can be changed by using the /set-permissions command."
                        f" Please do note that the names of the categories are generic and do not correspond to actual Discord permissions.\n\n"
        )
        footer = f"Shortcuts are shown in brackets. Example: Use {prefix}lb for leaderboard."
        description += footer
        icon_url = Config.SCROLL_ICON

        embed = discord.Embed(
            color=discord.Color.red()
         )
        
        categories = None
        if page != 0:
                categories = {
                    1: {
                            'title': "General Commands",
                            'description':"*Generic commands for all users.*",
                            'icon': Config.COG_ICON,
                            'content': 
                                [
                                        "/privacy-policy (policy)- View the privacy policy of the bot (can be used by anyone)",
                                        "/commands - Show this help message (can be used by anyone)",
                                        "/ping - Check bot responsiveness (can be used by anyone)",
                                        "/report-bug (report) - Report a bug (can be used by anyone)",
                                        "/suggest - Make a suggestion for the bot (can be used by anyone)",
                                        "/rmp-info (info) - View information about RMP (can be used by anyone)",
                                        "/xp - Checks amount of xp user has",
                                        "/profile - display a user's profile",
                                        "/leaderboard (lb) - View leaderboard for a category",
                                ]
                    },
                    2: {
                            'title': "XP Reward Commands",
                            'description':"*Manage progression and event rewards.*",
                            'icon': Config.STAR_ICON,
                            'content': 
                                [
                                        "/add-xp (axp) - Gives xp to user",
                                        "/take-xp (txp) - Takes xp from user",
                                        "/give-event-xp (gxp) - Gives xp to attendees/passers in event logs",
                                ]
                    },
                    3: {
                            'title': "Moderation Commands",
                            'description':"*Tools for oversight and user management.*",
                            'icon': Config.SHIELD_WARNING_ICON,
                            'content': 
                                [
                                        "/security-check (sc) - Security Check Roblox user (Only for Security checkers and HI-COM)",
                                        "/force-log (fl | log) - Force log an event/training/activity/etc manually",
                                        "/edit-user (eu) - Edit a user in the database",
                                        "/save-roles (sr) - Save requestable roles for a user in the database", 
                                        "/restore-roles (rr) - Generate a Dyno role format to restore a user's requestable roles",
                                        "/restore-user (restore) - Restore a user's data who has been recently removed from the database",
                                        "/remove-user (remove) - Remove a user from the database",
                                        "/manage-case-logs (mcl) - Manage a user's case-log by viewing their record or adding/deleting a record.",
                                ]
                    },
                    4: {
                            'title': "Administrative",
                            'description':"*High-level utilities for management (HI-COM).*",
                            'icon': Config.STAFF_BADGE_ICON,
                            'content': 
                                [
                                        "/reset-db (rdb) - Reset the LR and HR section of the database",
                                        "/discharge - Sends discharge notification to user and logs in discharge logs",
                                        "/blacklist (bl) - Request or blacklist a user which sends a notification to them and logs the blacklist in discharge logs",
                                        "/welcome-history (wl) - View history of welcome message changes",
                                        "/preview-welcome (preview) - Preview welcome messages as they appear to members",
                                ]
                    },
                    5: {
                            'title': "Restricted Commands",
                            'description':"*Restricted utilities for server administrators and permissions cannot be changed*",
                            'icon': Config.CROWN_ICON,
                            'content': 
                                [
                                        "/set-permissions (sp)- View or toggle role permissions for command categories",
                                        "/edit-welcome - Edit one of the welcome messages [Cannot be used with a prefix]",
                                ] 
                        },
                }
                
                category = categories.get(page)
                title = category['title']
                description = category['description']
                icon_url = category['icon']
                content = category['content']

                for line in content:
                        embed.add_field(name=f"↠ {line}", value="\u200b",inline=False)

                embed.set_footer(text=footer)
                    
        embed.description = description
        embed.set_author(name=title, icon_url=icon_url)

        return embed

def build_request_respsone(accepted: bool, username: str, check_type:int, halted: bool | None = None, reason: str | None = None):
        if halted:
                color = discord.Color.orange()
                title = "HALTED"
                icon_url = Config.PRIVATE_URL

                embed = discord.Embed(
                        description=reason,
                        color=color).set_author(name=title, icon_url=icon_url).set_thumbnail(url=Config.RMP_URL)

                embed.set_footer(text=f"Halted by {clean_nickname(username)}")
                return embed
        

        color = discord.Color.green() if accepted else discord.Color.red()
        title = "ACCEPTED" if accepted else "DENIED"
        icon_url = Config.CHECK_URL if accepted else Config.CANCEL_URL
        denied_msg = ("To submit another background check, you"
                 " must pass another tryout or submit a High Rank application.\n **We are not"
                 " permitted to disclose any reason for the denial of your security check. Apologies for the"
                 " inconvenience.**" 
        )
        accepted_msg = "**You have hereby passed the Royal Military Police Security Check and are awaiting ranking. Welcome to RMP!**"
        
        
        status_msg = f"{accepted_msg if accepted else denied_msg}"
        if check_type == Config.LA_ROLE_ID:
                status_msg = f"Induction request {'accepted' if accepted else 'denied'}."
                if reason:
                        status_msg += f"\n{reason}"
       
        embed = discord.Embed(
                        description=status_msg,
                        color=color).set_author(name=title, icon_url=icon_url).set_thumbnail(url=Config.RMP_URL)
        
        embed.set_footer(text=f"Check conducted by {clean_nickname(username)}")
        
        return embed

def build_welcome_log(username, w_type):
        color = discord.Color.green()
        title = "Welcome Message Sent"
        icon_url = Config.CHECK_URL
        description = f"Sent a {w_type} welcome message to **{username}**."
        
        return discord.Embed(description=description, color=color).set_author(name=title, icon_url=icon_url)

def build_profile_embed(user_info, discord_user, army_rank):
        username = user_info['username']
        division = user_info['division']
        rank = user_info['rank']
        xp = user_info['xp']

        tier, current_threshold, next_threshold, color = get_tier_info(xp)

        progress = make_progress_bar(xp, current_threshold, next_threshold)
        
        embed = discord.Embed(
                title=f"{username.upper()} PROFILE",
                color=color
        ).set_thumbnail(url=discord_user.display_avatar.url)
        

        embed.add_field(name="USER INFO", value=f"───────────────────", inline=False)
        embed.add_field(name="ARMY RANK:", value=f"```{army_rank}```", inline=True)
        embed.add_field(name="REGIMENTAL RANK:", value=f"```{rank}```", inline=True)
        embed.add_field(name="DIVISION:", value=f"```{division}```", inline=True)
        embed.add_field(name="XP STATS", value=f"───────────────────", inline=False)
        embed.add_field(name="XP:", value=f"```{xp}```", inline=True)
        embed.add_field(name="TIER:", value=f"```{tier}```", inline=True)
        embed.add_field(name=f"\u200b", value=f"\u200b", inline=True)
        embed.add_field(name="PROGRESSION", value=progress, inline=False)

        embed.add_field(name="EVENT & ACTIVITY  STATS", value=f"───────────────────", inline=False)
       
        return embed

def build_discharge_log(d_type, members, issuer, reason, sub_type=None, blacklist_duration=None, ending_date=None, evidence=None, color=None):
        d_log = discord.Embed(
            title="Discharge Log",
            color=color if color else discord.Color.red(),
            timestamp=datetime.now(timezone.utc)
        )

        d_log.add_field(name="Type:", value=d_type, inline=False)
        if sub_type:
                d_log.add_field(name="Sub-Type:", value=sub_type, inline=False)
                d_log.add_field(name="Blacklist Duration:", value=blacklist_duration, inline=True)
                d_log.add_field(name="Starting date:", value=f"<t:{int(datetime.now(timezone.utc).timestamp())}:D>", inline=True)
                
                value = f"<t:{int(ending_date.timestamp())}:D>" if ending_date else "NONE"
                d_log.add_field(name="Ending date:", value=value, inline=True)
        
        if isinstance(members, str):
                    members = [members] 

        if members and isinstance(members, list):
                value = "\n".join(members) or "None"
       
        d_log.add_field(
                        name=f"Discharged Member{'s' if len(members)>1 else ''}:",
            value=value,
            inline=False
        )
        
        if isinstance(issuer, discord.Member):
                d_log.add_field(name="Discharged By:", value=clean_nickname(issuer.display_name), inline=False)
        else:
                d_log.add_field(name="Discharged By:", value=issuer, inline=False)
                d_log.set_footer(text="Desertion Monitor System")

        d_log.add_field(name="Reason:", value=f"```{reason}```", inline=False)

        if evidence:
            d_log.add_field(name="Evidence:", value=f"[View Attachment]({evidence.url})", inline=True)

        return d_log


def build_blacklist_log(issuer, members, duration, reason, ending_date=None):
        blacklist_d_log = discord.Embed(
                    title="Blacklist Log",
                    color=discord.Color.dark_purple(),
                    timestamp=datetime.now(timezone.utc)
        )
         
        if isinstance(issuer, discord.Member):
                blacklist_d_log.add_field(name="Issuer:", value=clean_nickname(issuer.display_name), inline=False)
        else:
                blacklist_d_log.add_field(name="Issuer:", value="MP Assistant", inline=False)

        if isinstance(members, str):
                    members = [members] 

        if members and isinstance(members, list):
                value = "\n".join(members) or "None"

        blacklist_d_log.add_field(
                name="Blacklisted:",
                value=value,
                inline=False
            )
        
        blacklist_d_log.add_field(name="Duration:", value=duration, inline=True)
        
        blacklist_d_log.add_field(name="Starting date:", value=f"<t:{int(datetime.now(timezone.utc).timestamp())}:D>", inline=True)
        
        value = f"<t:{int(ending_date.timestamp())}:D>" if ending_date else "NONE"
        blacklist_d_log.add_field(
                name="Ending date:", 
                value=value,
                inline=True
        )

        blacklist_d_log.add_field(name="Reason:", value=f"```{reason}```", inline=False)
        
        return blacklist_d_log

def build_blacklist_request(members, duration, reason, requester, evidence = None):
        embed = discord.Embed(
            color=discord.Color.dark_gold(),
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.set_author(name="Blacklist Request")
        
        embed.add_field(name="Members to be blacklisted:", value=members, inline=False)
        embed.add_field(name="Duration:", value=duration, inline=False)
        embed.add_field(name="Reason:", value=reason, inline=False)
        embed.add_field(name="Requested by:", value=requester, inline=False)

        if evidence:
                embed.add_field(name="Evidence:",value=f"[View Evidence]({evidence.url})")
        
        embed.add_field(name="Authorised Approvers:", value="Provost Marshal and Above")
        embed.set_footer(text=f"Requested by {requester}")

        return embed

def build_event_log(member, message, host, event_type: str, attendees=None, excluded_attendee_count=None) -> discord.Embed:

       embed = discord.Embed(
            color=discord.Color.random(),
            timestamp=datetime.now(timezone.utc)
        )
 
       name = f"Event Log • {event_type}"
       icon_url = Config.DOC_URL
        
       if event_type == "TC Supervision":
               name_host = "Supervisor:"
       else:
               name_host = "Host:"

       embed.add_field(name=name_host, value=clean_nickname(host.display_name), inline=True)

       embed.set_author(
             name=name,
             icon_url=icon_url 
        )
       
        
       embed.add_field(name="Original Log:", value=f"[Jump to Log]({message.jump_url})", inline=True)
       embed.add_field(name="Logged By:", value=clean_nickname(member.display_name), inline=True)
       if attendees:
          embed.add_field(name="Attendees Logged:", value=attendees, inline=True)
       if excluded_attendee_count:
          embed.add_field(name="Excluded Attendees:", value=excluded_attendee_count, inline=True)
        
       embed.set_footer(text=f"Logger: {member.id} • Host: {host.id}")
        
       return embed


def build_activity_log(member, message, constable, time, isTimeGuarded: bool, points=None) -> discord.Embed:

       embed = discord.Embed(
            color=discord.Color.dark_gold(),
            timestamp=datetime.now(timezone.utc)
        )
 
       name = f"{"Guard Log" if isTimeGuarded else "Activity Log"}"  
       icon_url = Config.TIME_ICON

       embed.add_field(name="Constable:", value=clean_nickname(constable.display_name), inline=True)
       embed.add_field(name=f"{"Time Guarded:" if isTimeGuarded else "Time Active:"}", value=time, inline=True)
       if points:
        embed.add_field(name="XP Awarded for Time:", value=points, inline=True)
       
       embed.set_author(
             name=name,
             icon_url=icon_url 
        )
        
        
       embed.add_field(name="Original Log:", value=f"[Jump to Log]({message.jump_url})", inline=True)
       embed.add_field(name="Logged By:", value=clean_nickname(member.display_name), inline=True)
       embed.set_footer(text=f"Logger: {member.id} • Constable: {constable.id}")
        
       return embed



def build_db_logger_record(member, message, points, emoji) -> discord.Embed:
        embed = discord.Embed(
            color=discord.Color.greyple(),
            timestamp=datetime.now(timezone.utc)
        )

        name = "Database Logger Activity Record"
        icon_url = Config.LOG_ICON 

        embed.set_author(
             name=name,
             icon_url=icon_url 
        )
        
        embed.add_field(name="Logger:", value=clean_nickname(member.display_name), inline=False)
        embed.add_field(name="Message:", value=f"[Jump to Message]({message.jump_url})", inline=True)
        embed.add_field(name="Action:", value=str(emoji), inline=True)
        embed.add_field(name="Points Awarded:", value=points, inline=True)

        embed.set_footer(text=f"Logger: {member.id}")
       
        return embed


def build_inductor_record(inductor, logger, message, points, emoji) -> discord.Embed:
        embed = discord.Embed(
            color=discord.Color.dark_magenta(),
            timestamp=datetime.now(timezone.utc)
        )

        name = "Inductor Activity Record"
        icon_url = Config.ID_ICON

        embed.set_author(
             name=name,
             icon_url=icon_url 
        )

        embed.add_field(name="Inductor:", value=clean_nickname(inductor.display_name), inline=True)
        embed.add_field(name="Logger:", value=clean_nickname(logger.display_name), inline=True)
        embed.add_field(name="Message Request:", value=f"[Jump to Request]({message.jump_url})", inline=True)
        embed.add_field(name="Status:", value=str(emoji), inline=True)
        embed.add_field(name="Points Awarded:", value=points, inline=True)


        embed.set_footer(text=f"Logger:{logger.id} • Inductor: {inductor.id}")
        
        return embed

def build_examiner_record(examiner, logger, message, points) -> discord.Embed:
        embed = discord.Embed(
            color=discord.Color.dark_magenta(),
            timestamp=datetime.now(timezone.utc)
        )


        name = "Examiner Activity Record"
        icon_url = Config.WRITE_ICON
        embed.set_author(
             name=name,
             icon_url=icon_url 
        )

        embed.add_field(name="Examiner:", value=clean_nickname(examiner.display_name), inline=True)
        embed.add_field(name="Logger:", value=clean_nickname(logger.display_name), inline=True)
        embed.add_field(name="Type:", value=message.channel.name, inline=True)
        embed.add_field(name="Exam Report:", value=f"[Jump to Report]({message.jump_url})", inline=True)
        embed.add_field(name="Points Awarded:", value=points, inline=True)

        embed.set_footer(text=f"Logger:{logger.id} • Examiner: {examiner.id}")
        return embed

def build_sc_check_log(approver, member, checks, points, message) -> discord.Embed:
        embed = discord.Embed(
                color=discord.Color.blue(),
                timestamp=datetime.now(timezone.utc)
        )

        name = "Security Check Log Approved"
        icon_url = Config.ID_CHECKED_ICON
        embed.set_author(
             name=name,
             icon_url=icon_url 
        )


        embed.add_field(name="Logger:", value=f"{clean_nickname(member.display_name)}", inline=True)
        embed.add_field(name="Amount of Checks:", value=checks, inline=True)
        embed.add_field(name="Points Awarded:", value=points, inline=True)
        embed.add_field(name="Log:", value=f"[Jump to Log]({message.jump_url})", inline=True)
        embed.add_field(name="Approved by:", value=f"{clean_nickname(approver.display_name)} ", inline=True)

        embed.set_footer(text=f"Logger: {member.id} • Approver: {approver.id}")
        return embed
                

