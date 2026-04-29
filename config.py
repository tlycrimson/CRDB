class Config:

    #  ===== MISCELLANEOUS =====
    POINTS_PER_ACTIVITY = 0.5
    DEVELOPER_ID = 353167234698444802
    IGNORED_EMOJI = "✅"
    TRACKED_REACTIONS = {"✅", "❌", "☑️", "🟢", "🔴", "<:green:1168511080746729512>", "<:red:1168511120949121115>"}
    RMP_GROUP_URL = "https://www.roblox.com/communities/4972920/BA-Royal-Military-Police-Regiment#!/about"

    #  ===== XP LIMIT CONFIGURATION =====
    # XP Limit Configuration
    MAX_XP_PER_ACTION =  20  # Maximum XP that can be given/taken in a single action
    MAX_EVENT_XP_PER_USER = 20 # Maximum XP per user in event distributions
    MAX_EVENT_TOTAL_XP = 5000  # Maximum total XP for entire event distribution
    
    #  ===== GLOBAL RATE LIMITER CONFIGURATION =====
    GLOBAL_RATE_LIMIT = 15  # requests per minute
    COMMAND_COOLDOWN = 10    # seconds between command uses per user
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    # ===== CHANNELS =====
    DEFAULT_LOG_CHANNEL = 1224765477935386676
    D_LOG_CHANNEL_ID = 1165368315791806552 #Discharge
    B_LOG_CHANNEL_ID = 1165368317532438640 #Blacklist
    W_EVENT_LOG_CHANNEL_ID = 1165368314915192955
    EVENT_LOG_CHANNEL_ID = 1165368317930913947
    PHASE_LOG_CHANNEL_ID = 1165368317930913945
    TRYOUT_LOG_CHANNEL_ID = 1165368317930913946
    COURSE_LOG_CHANNEL_ID = 1165368316500656241
    COURSE_GRADES_CHANNEL_ID = 1165368316970405910
    ACTIVITY_LOG_CHANNEL_ID = 1165368316123152389
    SC_CHANNEL_ID = 1165368313925353577  
    SC_LOGS_CHANNEL_ID = 1224763414153531472
    HR_CHAT_CHANNEL_ID = 1165368316970405917
    BGC_LOGS_CHANNEL = 1224763414153531472 
    LA_INDUCTION_CHANNEL_ID = 1165368316123152392
    TC_SUPERVISION_CHANNEL_ID = 1412139766136176700
    PUBLIC_CHAT_CHANNEL_ID = 1219410104240050236
    MAIN_COMMS_CHANNEL_ID = 1165368314915192958 
    INACTIVTY_R_CHANNEL_ID = 1165368315791806549
    ROLE_REQUEST_CHANNEL_ID = 1165368316123152385
    DSPG_GRADES_CHANNEL_ID = 1207367396424425483
    DEGREE_GRADES_CHANNEL_ID = 1267563275223040111
    INDOC_CHANNEL_ID = 1165368317532438639
    CASE_LOGS_CHANNEL_ID = 1165368315326251114 
    LD_CHANNEL_ID = 1360422355255689276  
    DISCHARGE_REQUEST_CHANNEL_ID = 1165368315791806551

    # ===== CHANNEL LISTS =====
    DEFAULT_MONITOR_CHANNELS = {
        EVENT_LOG_CHANNEL_ID,
        W_EVENT_LOG_CHANNEL_ID,
        PHASE_LOG_CHANNEL_ID,
        TRYOUT_LOG_CHANNEL_ID,
        INDOC_CHANNEL_ID, 
        INACTIVTY_R_CHANNEL_ID,
        ROLE_REQUEST_CHANNEL_ID, 
        ACTIVITY_LOG_CHANNEL_ID, 
        COURSE_LOG_CHANNEL_ID, 
        COURSE_GRADES_CHANNEL_ID, 
        SC_LOGS_CHANNEL_ID, 
        DEGREE_GRADES_CHANNEL_ID, 
        DSPG_GRADES_CHANNEL_ID,
        LA_INDUCTION_CHANNEL_ID, 
        TC_SUPERVISION_CHANNEL_ID,
    }

    EXAM_AND_INDUCTION_MONITOR_CHANNELS = {
        COURSE_GRADES_CHANNEL_ID,  
        DEGREE_GRADES_CHANNEL_ID, 
        DSPG_GRADES_CHANNEL_ID, 
        LA_INDUCTION_CHANNEL_ID, 
    }
    
    IGNORED_CHANNELS = { 
        INDOC_CHANNEL_ID,
        COURSE_GRADES_CHANNEL_ID,
        SC_CHANNEL_ID,
    }

    # ===== ROBLOX GROUP IDS =====
    BRITISH_ARMY_GROUP_ID = 4972535
    RMP = 4972920
    SI = 32578828 
    UBA = 6447250 
    PARAS = 4973512 
    HSD = 14286518 
    IC = 32014700 
    RAMC = 15229694 
    AAC = 15224554 
    RAR = 14557406
    RTR = 14609194 
    UKSF = 5029915 
    ETS = 5237367
    MCS = 13906055
    QMC = 14317225
    RMAS = 4994558
    AMC = 7293600
    PATROL = 5786049
    SI = 32578828
    REGIMENTS = {
            SI,
            PARAS, 
            HSD,
            IC,
            RAMC, 
            AAC,
            RAR,
            RTR,
            UKSF,
            ETS,
            MCS,
            QMC,
            RMAS,
            AMC,
            PATROL,
    }

    # ===== DIVISION & RANK ROLES =====
    RMP_ROLE_ID = 1165368311727521795
    HR_ROLE_ID = 1165368311840784507
    HIGH_COMMAND_ROLE_ID = 1165368311840784508

    # Department Roles
    BG_CHECKER_ROLE_ID = 1435045180557230184 
    DB_LOGGER_ROLE_ID = 1435045193048133732  
    LA_ROLE_ID = 1165368311660429380 
    
    # Division Roles
    PW_ROLE_ID = 1309237502909087824 

    
    # ===== PW HR RANKS (Highest to Lowest) =====
    HQ_ROLE_ID = 1165368311874326655  
    PM_ROLE_ID = 1165368311874326650  
    PW_COMMANDER_ROLE_ID = 1165368311840784515
    PW_EXECUTIVE_ROLE_ID = 1165368311840784514
    LIEUTENANT_COLONEL_ROLE_ID = 1165368311840784512
    MAJOR_ROLE_ID = 1165368311840784511
    SUPERINTENDENT_ROLE_ID = 1165368311840784510
    CHIEF_INSPECTOR_ROLE_ID = 1309231446258356405
    INSPECTOR_ROLE_ID = 1309231448569680078
    PW_HR_IDS = {
        PM_ROLE_ID,
        PW_COMMANDER_ROLE_ID,
        PW_EXECUTIVE_ROLE_ID,
        LIEUTENANT_COLONEL_ROLE_ID,
        MAJOR_ROLE_ID,
        SUPERINTENDENT_ROLE_ID,
        CHIEF_INSPECTOR_ROLE_ID,
        INSPECTOR_ROLE_ID,
    }

    # ===== PW LR RANKS (Highest to Lowest) =====
    RSM_ROLE_ID = 1309231453816492104   
    COMPANY_SERGEANT_MAJOR_ROLE_ID = 1309231451321139200
    STAFF_SERGEANT_ROLE_ID = 1165368311777869933
    SERGEANT_ROLE_ID = 1165368311777869932
    SENIOR_CONSTABLE_ROLE_ID = 1165368311777869931
    CONSTABLE_ROLE_ID = 1165368311777869930
    TRAINEE_CONSTABLE_ROLE_ID = 1165368311777869924  
    PW_LR_IDS = {
        COMPANY_SERGEANT_MAJOR_ROLE_ID,
        STAFF_SERGEANT_ROLE_ID,
        SERGEANT_ROLE_ID,
        SENIOR_CONSTABLE_ROLE_ID,
        CONSTABLE_ROLE_ID,
        TRAINEE_CONSTABLE_ROLE_ID,
    }
 
    # ===== RANK MAPPING DICTIONARIES =====
    OVERSIGHT_RANKS = {
        HQ_ROLE_ID: "Headquarters",
        PM_ROLE_ID: "Provost Marshal",
        RSM_ROLE_ID: "Regimental Sergeant Major"
    }
   
    PW_HR_RANKS = {
        PM_ROLE_ID: "Provost Marshal",
        PW_COMMANDER_ROLE_ID: "PW Commander",
        PW_EXECUTIVE_ROLE_ID: "PW Executive",
        LIEUTENANT_COLONEL_ROLE_ID: "Lieutenant Colonel",
        MAJOR_ROLE_ID: "Major",
        SUPERINTENDENT_ROLE_ID: "Superintendent",
        CHIEF_INSPECTOR_ROLE_ID: "Chief Inspector",
        INSPECTOR_ROLE_ID: "Inspector"
    }
   
    PW_LR_RANKS = {
        RSM_ROLE_ID: "Regimental Sergeant Major",
        COMPANY_SERGEANT_MAJOR_ROLE_ID: "Company Sergeant Major",
        STAFF_SERGEANT_ROLE_ID: "Staff Sergeant",
        SERGEANT_ROLE_ID: "Sergeant",
        SENIOR_CONSTABLE_ROLE_ID: "Senior Constable",
        CONSTABLE_ROLE_ID: "Constable",
        TRAINEE_CONSTABLE_ROLE_ID: "Trainee Constable",
    }

    # ===== AWARDS & MISCELLANEOUS ROLES =====
    TRACKED_ROLE_IDS = {
        1344753293766819961,  # Soundboard Access
        1165368311727521799,  # Picture Permissions
        1165368311727521800,  # Mic Checked ✔️
        1430316487561842822,  # Hall of Shame permission
        1165368311681388613,  # GMT
        1165368311681388611,  # EST
        1165368311681388612,  # AEST
        1165368311681388607,  # Gamenight Ping
        1165368311681388608,  # Movie Night Ping
        1165368311681388609,  # QOTD Ping
        1165368311681388610,  # DJ
        1165368311618469968,  # 🥇 Distinguished Conduct Medal
        1165368311618469966,  # 🏆 Gamenight Winner
        1312843022324596776,  # 🏅 Loyal Service Award
        1165368311618469964,  # 🎖️ Accolade of Honour
        1378450077760360568,  # 🎖️ Officer Guardian Service Award
        1378450228344258631,  # 🎖️ Constable Guardian Service Award
        1378450356987625582,  # 🎖️ Officer Sentinel Service Award
        1378450486734098442,  # 🎖️ Constable Sentinel Service Award
        1378450498171961557,  # 🎖️ Officer Vanguard Service Medal
        1378450782898094121,  # 🎖️ Constable Vanguard Service Award
        1165368311618469963,  # 🎖️ Veteran's Accolade
        1165368311618469962,  # 🎖️ Conspicuous Activity Award
        1378450794663116809,  # 🎖️ Constable Resolute Duty Award
        1378450791731429427,  # 🎖️ Officer Resolute Duty Award
        1165368311618469961,  # 🎖️ Distinguished Service Accolade
        1378450943837868084,  # 🎖️ Veteran's Honour Guard
        1165368311618469960,  # 🎖️ Standard Service Accolade
        1238202822110085130,  # 🎖️ Honourable Lawman
        1165368311618469959,  # 🎖️ Platoon Leader Certified
        1165368311584919651,  # 🎖️ Inspection Enthusiast
        1393298109416607834,  # 🎖️ SRF Hall of Fame
        1393298323141693613,  # 🎖️ Elite Operator Award
        1393298280284164267,  # 🎖️ Soulbound Award
        1165368311584919650,  # 💬 Communications Award
        1165368311584919647,  # 🎩 Disciplinary Award
        1165368311584919649,  # 🗣️ Inspection Certified
        1165368311584919648,  # 🏹 Gallantry Award
        1357209259082911744,  # ✈️ Aviation Award
        1357209582325207101,  # ⚔️ Parry Award
        1346632492089868410,  # 🔎 Investigator Award
        1238203867032715375,  # 🥷 Specialist Award
        1308102956612321291,  # ❤️‍🩹 Soulbound Award (Alternative)
        1378451395853680692,  # 🎖️ Sergeant Major's Combat Commendation
        1378451391571296299,  # 🎖️ Tactical Communicator Award
        1378451387456815315,  # 🎖️ Adaptive Maneuver Award
        1378451385041031219,  # 🎖️ Golden Cross for Medical Valor
        1378451382352220302,  # 🎖️ Precision Marksmanship Badge
        1378451379672059914,  # 🎖️ Crisis Response Citation
        1378451351373352970,  # 🎖️ Sergeant Major's Enforcement Commendation
        1378451348185546923,  # 🎖️ Surveillance & Security Medal
        1378451344704147618,  # 🎖️ Patrol Excellence Ribbon
        1378451325620322435,  # 🎖️ Law and Order Commendation
        1378451309350486026,  # 🎖️ Provost Excellence Medal
        1180512772820320427,  # ⭐⏱️ Advanced Military Drills Award
        1180511093181919303,  # ⭐🚓 Advanced General Service Award
        1180512757750185984,  # ⭐🤸‍♂️ Advanced Agility Award
        1180512815317012611,  # ⭐👮 Advanced Leadership Award
        1180512840835141643,  # ⭐🛵 Advanced Motorcyclist Award
        1180512860598718484,  # ⭐🏠 Advanced Structure Breaching Award
        1180512624719437893,  # ⭐🛡️ Advanced Close Security Award
        1337219449278828594,  # ⭐🎓 Advanced Information Award
        1165368311085809721,  # ⭐⚔️ Advanced Combatant Award
        1165368311085809720,  # ⭐🩺 Advanced Medical Award
        1165368311584919644,  # ⏱️ Military Drills Award
        1165368311584919646,  # 🚓 General Service Award
        1165368311584919645,  # 🤸‍♂️ Agility Award
        1165368311584919643,  # 👮 Leadership Award
        1165368311584919642,  # 🛵 Motorcyclist Award
        1165368311085809723,  # 🏠 Structure Breaching Award
        1165368311085809722,  # 🛡️ Close Security Award
        1324062837907394651,  # 🎓 Information Award
        1180513363881644084,  # ⚔️ Combatant Award
        1180513399398989844,  # 🩺 Medical Award
        1450546812099366932,  # 📝Task Efficiency Award 
        1450902236006711390,  # 📝Advanced Task Efficiency Award    
        1165368311618469968,  # 🥇I Distinguished Conduct Medal 
        1312843022324596776,  # 🏅 I Loyal Service Award 
        1165368311618469964,  # 🎖️I Accolade of Honour 
        1452729923104931860,  # 🥇I Monumental Leadership Award 
        1452730974654693527,  # 🥇I Crimson Provost Distinction
        1452730978563915889,  # 🥇I Regimental Valor & Conduct Award
    }
    #  ===== ICON URLS =====
    RMP_URL = "https://i.gyazo.com/368e44df13132bb63e699a56350733f7.png"
    CHECK_URL = "https://img.icons8.com/?size=100&id=ThDU3IFKpFui&format=png&color=000000"
    CANCEL_URL = "https://img.icons8.com/?size=100&id=DXECg4JU1n2x&format=png&color=000000"
    PRIVATE_URL = "https://img.icons8.com/?size=100&id=M5MHhpYjEJ0j&format=png&color=000000"
    ALERT_URL = "https://img.icons8.com/?size=100&id=5PexHbhSpPcd&format=png&color=000000"
    NOTIF_URL = "https://img.icons8.com/?size=100&id=oStKNuPVzV7A&format=png&color=000000"
    DOC_URL  = "https://img.icons8.com/?size=100&id=DHOunydDcKfC&format=png&color=000000"
    TIME_ICON = "https://img.icons8.com/?size=100&id=4i5bTF9azXVR&format=png&color=000000"
    LOG_ICON = "https://img.icons8.com/?size=100&id=BVOPSSK89IdK&format=png&color=000000"
    ID_ICON  = "https://img.icons8.com/?size=100&id=dGAwDTFCLptB&format=png&color=000000"
    WRITE_ICON = "https://img.icons8.com/?size=100&id=AFI73eah0Ict&format=png&color=000000"
    ID_CHECKED_ICON = "https://img.icons8.com/?size=100&id=FcImsbcafkas&format=png&color=000000"
    STAFF_BADGE_ICON = "https://img.icons8.com/?size=100&id=Zk5UTKNPbUev&format=png&color=000000"
    SHIELD_WARNING_ICON = "https://img.icons8.com/?size=100&id=zu4gAvVemf2r&format=png&color=000000"
    ALERTING_NOTIF_ICON = "https://img.icons8.com/?size=100&id=GuiYMUpQWuOG&format=png&color=000000"
    STAR_ICON = "https://img.icons8.com/?size=100&id=KS3ARUzPXuOj&format=png&color=000000"
    USER_ICON = "https://img.icons8.com/?size=100&id=NcQNyxjmHvuB&format=png&color=000000"
    SCROLL_ICON = "https://img.icons8.com/?size=100&id=RbbYFLGNyoFg&format=png&color=000000"
    TROPHY_ICON = "https://img.icons8.com/?size=100&id=loPah3jTaW8a&format=png&color=000000"
    COG_ICON = "https://img.icons8.com/?size=100&id=xyFoc6U1Hu3c&format=png&color=000000"
    CROWN_ICON = "https://img.icons8.com/?size=100&id=h1WG6I4MUqdj&format=png&color=000000"
    LOGOUT_ICON = "https://img.icons8.com/?size=100&id=5HW1YsFkzHio&format=png&color=000000"
    TRASH_ICON = "https://img.icons8.com/?size=100&id=CzTISLkmHrKE&format=png&color=000000"
    CUFFS_ICON = "https://img.icons8.com/?size=100&id=JYaD93Fha6RH&format=png&color=000000"
    PRIVACY_ICON = "https://img.icons8.com/?size=100&id=JNxUx3djiocR&format=png&color=000000"
    CELEBRATE_ICON = "https://img.icons8.com/?size=100&id=85nCYFk9TW9d&format=png&color=000000"
    INFO_ICON =  "https://img.icons8.com/?size=100&id=40JxrZB76JLv&format=png&color=000000"
    RESET_ICON = "https://img.icons8.com/?size=100&id=Rb3DR2iW56nj&format=png&color=000000"
    OPEN_BOOK_ICON = "https://img.icons8.com/?size=100&id=n9PJmcajBrm4&format=png&color=000000"
    BUG_ICON = "https://img.icons8.com/?size=100&id=MC4pt1iFw8Aw&format=png&color=000000"
    BULB_ICON = "https://img.icons8.com/?size=100&id=qeWJ5AQYaoau&format=png&color=000000"
    BOT_ICON = "https://i.gyazo.com/dfda0ab8b533bcedd069701adf7bca8e.png"
    BOT_BANNER = "https://i.gyazo.com/a60a9ff595fef8b22644f6fcce6ecc6a.gif"
    BOT_BANNER_CROPPED = "https://i.gyazo.com/950dda1dc9d3bced7cdf673488acd001.gif"
    HAMMER =  "https://img.icons8.com/?size=100&id=cwUfaMWCLeL8&format=png&color=000000"
