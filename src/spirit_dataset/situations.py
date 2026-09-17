EVERTALK_OPENING_SITUATION = "(구원자에게 에버톡 메시지를 보낸다)"
TRIP_OPENING_SITUATION = "(구원자와 함께 여행 중이다)"
TRIP_SHARED_KEYWORD_SITUATION = "(구원자와 여행 중: {keyword})"
TRIP_PERSONAL_KEYWORD_SITUATION = "(여행 중 '{keyword}' 이야기를 꺼낸다)"
TOWN_LOST_ITEM_OPENING_SITUATION = "(영지에서 잃어버린 물건 이야기를 한다)"
LOVE_LEVEL_MEMORY = "구원자와의 인연 레벨은 {level}"

HERO_DESC_SITUATIONS: dict[str, str] = {
    "greeting": "(구원자에게 인사한다)",
    "contract_line": "(구원자와 처음 계약을 맺었다)",
    "introduction": "(구원자에게 자기소개를 한다)",
}
HERO_COMMENT_SITUATION = "({name}에 대해 이야기한다)"

LOBBY_TYPE_SITUATIONS: dict[str, str] = {
    "Greeting": "(구원자가 로비에 찾아왔다)",
    "GreetingSp": "({month}월 {day}일, 구원자가 로비에 찾아왔다)",
    "Season": "({month}월 {day}일, 구원자와 로비에 있다)",
    "Normal": "(구원자와 로비에 함께 있다)",
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
    "BubbleTalk1": "(로비에서 혼잣말을 한다)",
    "BubbleTalk2": "(로비에서 혼잣말을 한다)",
    "BubbleTalk3": "(로비에서 혼잣말을 한다)",
    "SadTalk1": "(기분이 가라앉아 있다)",
    "SadTalk2": "(기분이 가라앉아 있다)",
    "SadTalk3": "(기분이 가라앉아 있다)",
    "HappyTalk1": "(기분이 좋다)",
    "HappyTalk2": "(기분이 좋다)",
    "HappyTalk3": "(기분이 좋다)",
    "LovelyTalk": "(구원자에게 애정을 느낀다)",
    "ArbeitStart": "(아르바이트를 시작한다)",
    "ArbeitFinish": "(아르바이트를 마쳤다)",
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
BUBBLE_STRING_TABLE = "StringTalk"
TRIP_KEYWORD_STRING_TABLE = "StringUI"
