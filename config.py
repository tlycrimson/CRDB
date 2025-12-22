class Config:
    HIGH_COMMAND_ROLE_ID = 1165368311840784508
    BG_CHECKER_ROLE_ID = 1435045180557230184 
    DB_LOGGER_ROLE_ID = 1435045193048133732  
    LA_ROLE_ID = 1165368311660429380 
    HR_ROLE_ID = 1165368311840784507
    RMP_ROLE_ID = 1165368311727521795
    SOR_ROLE_ID = 1282454489084858490
    CSM_ROLE_ID = 1309231451321139200
    TRACKED_REACTIONS = {"✅", "❌", "☑️", "🟢", "🔴", "<:green:1168511080746729512>", "<:red:1168511120949121115>"}
    DEFAULT_MONITOR_CHANNELS = {
        1165368317930913947, # Event log 
        1165368314915192955, # Wide event log
        1165368317930913945, # Phase Log 
        1165368317930913946, # Tryout Log
        1165368317532438639, #
        1165368315791806549, # Inactivity Req
        1165368316123152385, # Role Request
        1165368316123152389, # Activity Log
        1244451957884715049, # 
        1165368316500656241, # Course Log
        1165368316970405910, # Course Grades
        1165368316123152393, #
        1165368313925353577, # Security Check Channel (SC Log)
        1267563275223040111, #Degree Grades
        1207367396424425483, #DSPG Grades
        1165368316123152392, # Induction Request
    }

    EXAM_MONITOR_CHANNELS = {
        1165368316970405910, #Course Grades 
        1267563275223040111, #Degree Grades
        1207367396424425483, #DSPG Grades
        1165368316123152392, #Induction Request
    }
    
    DEFAULT_LOG_CHANNEL = 1224765477935386676
    IGNORED_CHANNELS = { 
        1165368317532438639,
        1165368316970405910,
        1165368313925353577,
    }
    IGNORED_EMOJI = "✅"
    D_LOG_CHANNEL_ID = 1165368315791806552
    B_LOG_CHANNEL_ID = 1165368317532438640
    W_EVENT_LOG_CHANNEL_ID = 1165368314915192955
    EVENT_LOG_CHANNEL_ID = 1165368317930913947
    PHASE_LOG_CHANNEL_ID = 1165368317930913945
    TRYOUT_LOG_CHANNEL_ID = 1165368317930913946
    COURSE_LOG_CHANNEL_ID = 1165368316500656241
    ACTIVITY_LOG_CHANNEL_ID = 1165368316123152389
    SC_CHANNEL_ID = 1165368313925353577 
    HR_CHAT_CHANNEL_ID = 1165368316970405917
    BGC_LOGS_CHANNEL = 1224763414153531472 
    LA_INDUCTION_CHANNEL_ID = 1165368316123152392

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
        1378451320637227110,  # 🎖️ Rapid Response Medal
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
        
    # ===== DIVISION & RANK ROLES =====
    
    # Division Roles
    SOR_ROLE_ID = 1368777853235101702  # SOR role (any SOR member has this)
    
    # HQ Roles
    HQ_ROLE_ID = 1165368311874326650  # Provost Marshal
    
    # RSM Role
    RSM_ROLE_ID = 1309231453816492104  # Regimental Sergeant Major
    
    # Trainee Role
    TRAINEE_ROLE_ID = 1165368311777869924  # Trainee Constable
    
    # ===== SOR HR RANKS (Highest to Lowest) =====
    SOR_COMMANDER_ROLE_ID = 1368777853235101702
    SOR_EXECUTIVE_ROLE_ID = 1368777611001462936
    SQUADRON_COMMANDER_ROLE_ID = 1368780792842424511
    SQUADRON_EXECUTIVE_OFFICER_ROLE_ID = 1368777380344102912
    TACTICAL_OFFICER_ROLE_ID = 1368777213444624489
    OPERATIONS_OFFICER_ROLE_ID = 1368777046003552298
    JUNIOR_OPERATIONS_OFFICER_ROLE_ID = 1368776765270396978
    
    # ===== PW HR RANKS (Highest to Lowest) =====
    PW_COMMANDER_ROLE_ID = 1165368311840784515
    PW_EXECUTIVE_ROLE_ID = 1165368311840784514
    LIEUTENANT_COLONEL_ROLE_ID = 1165368311840784512
    MAJOR_ROLE_ID = 1165368311840784511
    SUPERINTENDENT_ROLE_ID = 1165368311840784510
    CHIEF_INSPECTOR_ROLE_ID = 1309231446258356405
    INSPECTOR_ROLE_ID = 1309231448569680078
    
    # ===== SOR LR RANKS (Highest to Lowest) =====
    OPERATIONS_SERGEANT_MAJOR_ROLE_ID = 1368776612878876723
    TACTICAL_LEADER_ROLE_ID = 1368776341289304165
    FIELD_SPECIALIST_ROLE_ID = 1368776344787484802
    SENIOR_OPERATOR_ROLE_ID = 1368776092969730149
    OPERATOR_ROLE_ID = 1368775864141086770
    
    # ===== PW LR RANKS (Highest to Lowest) =====
    COMPANY_SERGEANT_MAJOR_ROLE_ID = 1309231451321139200
    STAFF_SERGEANT_ROLE_ID = 1165368311777869933
    SERGEANT_ROLE_ID = 1165368311777869932
    SENIOR_CONSTABLE_ROLE_ID = 1165368311777869931
    CONSTABLE_ROLE_ID = 1165368311777869930
    
    # ===== RANK MAPPING DICTIONARIES =====
    # These are used by the RankTracker class
    SOR_HR_RANKS = {
        SOR_COMMANDER_ROLE_ID: "SOR Commander",
        SOR_EXECUTIVE_ROLE_ID: "SOR Executive",
        SQUADRON_COMMANDER_ROLE_ID: "Squadron Commander",
        SQUADRON_EXECUTIVE_OFFICER_ROLE_ID: "Squadron Executive Officer",
        TACTICAL_OFFICER_ROLE_ID: "Tactical Officer",
        OPERATIONS_OFFICER_ROLE_ID: "Operations Officer",
        JUNIOR_OPERATIONS_OFFICER_ROLE_ID: "Junior Operations Officer"
    }
    
    PW_HR_RANKS = {
        PW_COMMANDER_ROLE_ID: "PW Commander",
        PW_EXECUTIVE_ROLE_ID: "PW Executive",
        LIEUTENANT_COLONEL_ROLE_ID: "Lieutenant Colonel",
        MAJOR_ROLE_ID: "Major",
        SUPERINTENDENT_ROLE_ID: "Superintendent",
        CHIEF_INSPECTOR_ROLE_ID: "Chief Inspector",
        INSPECTOR_ROLE_ID: "Inspector"
    }
    
    SOR_LR_RANKS = {
        OPERATIONS_SERGEANT_MAJOR_ROLE_ID: "Operations Sergeant Major",
        TACTICAL_LEADER_ROLE_ID: "Tactical Leader",
        FIELD_SPECIALIST_ROLE_ID: "Field Specialist",
        SENIOR_OPERATOR_ROLE_ID: "Senior Operator",
        OPERATOR_ROLE_ID: "Operator"
    }
    
    PW_LR_RANKS = {
        COMPANY_SERGEANT_MAJOR_ROLE_ID: "Company Sergeant Major",
        STAFF_SERGEANT_ROLE_ID: "Staff Sergeant",
        SERGEANT_ROLE_ID: "Sergeant",
        SENIOR_CONSTABLE_ROLE_ID: "Senior Constable",
        CONSTABLE_ROLE_ID: "Constable"
    }

   # MESSAGE_TRACKER_CHANNELS = {
    #    1165368316500656241, # Course-log
    #    1165368316970405910, # Course grades
   #     1267563275223040111, # Degree grades
    #    1207367396424425483, # DSPG grades
    #    1165368316123152392, # LA Induction
   # }  
   # MESSAGE_TRACKER_LOG_CHANNEL = 1224764125150707904  
   # MESSAGE_TRACKER_ROLE_ID = 1224738140082798682 

   # LD_ROLE_ID = 1224736326566547556
   # LD_HEAD_ROLE_ID = 1224732743036833802
   # LD_DEP_HEAD_ROLE_ID = 1224732812020813935
    # MAX_MONITORED_CHANNELS = 15





























