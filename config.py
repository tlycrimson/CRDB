class Config:
    HIGH_COMMAND_ROLE_ID = 1165368311840784508
    BG_CHECKER_ROLE_ID = 1435045180557230184 
    DB_LOGGER_ROLE_ID = 1435045193048133732  
    HR_ROLE_ID = 1165368311840784507
    RMP_ROLE_ID = 1165368311727521795
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
        1165368316970405910, # 
        1165368316123152393, #
        1165368313925353577, # Security Check Channel (SC Log)
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























