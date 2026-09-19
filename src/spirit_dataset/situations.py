EVERTALK_OPENING_SITUATION = "(구원자에게 먼저 말을 건다)"
TRIP_OPENING_SITUATION = "(구원자와 함께 놀러 나와 있다)"
TRIP_SHARED_KEYWORD_SITUATION = "(구원자와 놀러 나와서: {keyword})"
TRIP_PERSONAL_KEYWORD_SITUATION = "(구원자와 놀다가 '{keyword}' 이야기를 꺼낸다)"
OUTING_PROPOSAL_SITUATION = "(구원자가 데이트를 청한다)"
OUTING_PARTING_SITUATION = "(구원자와 놀다가 헤어질 시간이 되었다)"
OUTING_GROUP_ROLE_MODULUS = 1000
OUTING_TALK_TYPES: tuple[int, ...] = (9, 59, 97, 98, 99, 999)
OUTING_GROUP_SITUATIONS: dict[int, str] = {
    0: OUTING_PROPOSAL_SITUATION,
    1: TRIP_OPENING_SITUATION,
    3: TRIP_OPENING_SITUATION,
    59: OUTING_PROPOSAL_SITUATION,
    97: OUTING_PARTING_SITUATION,
    98: OUTING_PARTING_SITUATION,
    99: OUTING_PARTING_SITUATION,
    999: OUTING_PARTING_SITUATION,
}
OUTING_AFFECTION_GROUP_ROLES: frozenset[int] = frozenset({99, 999})
OUTING_AFFECTION_TOPIC = "affection"
OUTING_SITUATION_TOPICS: dict[str, str] = {
    OUTING_PROPOSAL_SITUATION: "outing",
    OUTING_PARTING_SITUATION: "outing_parting",
}
TOWN_LOST_ITEM_OPENING_SITUATION = "(잃어버린 물건 이야기를 한다)"
LOVE_LEVEL_MEMORY = "구원자와의 인연 레벨은 {level}"

HERO_DESC_SITUATIONS: dict[str, str] = {
    "greeting": "(구원자에게 인사한다)",
    "contract_line": "(구원자와 처음 계약을 맺었다)",
    "introduction": "(구원자에게 자기소개를 한다)",
}
HERO_COMMENT_SITUATION = "({name}에 대해 이야기한다)"

LOBBY_TYPE_SITUATIONS: dict[str, str] = {
    "Greeting": "(구원자가 찾아왔다)",
    "GreetingSp": "({month}월 {day}일, 구원자가 찾아왔다)",
    "Season": "({month}월 {day}일, 구원자와 함께 있다)",
    "Normal": "(구원자와 함께 있다)",
    "TouchSp": "(구원자가 특별하게 쓰다듬는다)",
    "TouchSp2": "(구원자가 다시 특별하게 쓰다듬는다)",
    "Love1": "(구원자와의 인연이 깊어졌다)",
    "Love2": "(구원자와의 인연이 더 깊어졌다)",
    "Love3": "(구원자와의 인연이 가장 깊어졌다)",
    "Leave": "(구원자가 한동안 내버려 두었다)",
}
LOBBY_GAME_FEATURE_TYPES: frozenset[str] = frozenset(
    {"Mail", "Achievements", "AutoHunt", "Hero", "Town"}
)

BUBBLE_COLUMN_SITUATIONS: dict[str, str] = {
    "BubbleTalk1": "(혼잣말을 한다)",
    "BubbleTalk2": "(혼잣말을 한다)",
    "BubbleTalk3": "(혼잣말을 한다)",
    "SadTalk1": "(기분이 가라앉아 있다)",
    "SadTalk2": "(기분이 가라앉아 있다)",
    "SadTalk3": "(기분이 가라앉아 있다)",
    "HappyTalk1": "(기분이 좋다)",
    "HappyTalk2": "(기분이 좋다)",
    "HappyTalk3": "(기분이 좋다)",
    "LovelyTalk": "(구원자에게 애정을 느낀다)",
    "ArbeitStart": "(일을 하러 나선다)",
    "ArbeitFinish": "(일을 마치고 돌아왔다)",
}
BUBBLE_COLUMN_EMOTIONS: dict[str, str] = {
    "SadTalk1": "Sad",
    "SadTalk2": "Sad",
    "SadTalk3": "Sad",
    "HappyTalk1": "Happy",
    "HappyTalk2": "Happy",
    "HappyTalk3": "Happy",
    "LovelyTalk": "Lovely",
}
BUBBLE_BATTLE_COLUMNS: frozenset[str] = frozenset(
    {
        "HitReaction1",
        "HitReaction2",
        "HitReaction3",
        "HitReactionStrong1",
        "HitReactionStrong2",
        "HitReactionStrong3",
    }
)
CANONICAL_SITUATION_TOPICS: dict[str, str] = {
    **OUTING_SITUATION_TOPICS,
    LOBBY_TYPE_SITUATIONS["Greeting"]: "visit_greeting",
    HERO_DESC_SITUATIONS["greeting"]: "visit_greeting",
    LOBBY_TYPE_SITUATIONS["Normal"]: "idle_chat",
    BUBBLE_COLUMN_SITUATIONS["BubbleTalk1"]: "idle_chat",
    LOBBY_TYPE_SITUATIONS["Leave"]: "reunion",
    LOBBY_TYPE_SITUATIONS["TouchSp"]: "head_pat",
    LOBBY_TYPE_SITUATIONS["TouchSp2"]: "head_pat",
    BUBBLE_COLUMN_SITUATIONS["SadTalk1"]: "cheer_up",
    BUBBLE_COLUMN_SITUATIONS["HappyTalk1"]: "good_mood",
    BUBBLE_COLUMN_SITUATIONS["LovelyTalk"]: "affection",
    BUBBLE_COLUMN_SITUATIONS["ArbeitStart"]: "work_leave",
    BUBBLE_COLUMN_SITUATIONS["ArbeitFinish"]: "work_done",
}
NIGHT_TOGETHER_SITUATION = "(구원자와 단둘이 밤을 보낸다)"
BATH_TOGETHER_SITUATION = "(구원자와 함께 씻는다)"
INTIMACY_SITUATION_TOPICS: dict[str, str] = {
    NIGHT_TOGETHER_SITUATION: "night_together",
    BATH_TOGETHER_SITUATION: "bath_together",
    TRIP_OPENING_SITUATION: "outing_together",
    LOBBY_TYPE_SITUATIONS["Greeting"]: "visit_greeting",
    LOBBY_TYPE_SITUATIONS["Love1"]: "affection",
    LOBBY_TYPE_SITUATIONS["Normal"]: "idle_chat",
}
BUBBLE_STRING_TABLE = "StringTalk"
TRIP_KEYWORD_STRING_TABLE = "StringUI"
