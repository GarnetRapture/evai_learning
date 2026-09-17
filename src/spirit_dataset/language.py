"""Language-specific renderings of the fixed contract, never new canonical facts."""

from common.errors import EvaiError

LANGUAGE_CODES = {"kr": "ko", "en": "en", "zh_tw": "zh_tw"}

# The Korean keys are the canonical meanings owned by profile/situations.
TRANSLATIONS: dict[str, tuple[str, str]] = {
    "나는 방주의 인공 정령": ("I'm an artificial Ark Soul.", "我是方舟的人造精靈"),
    "나는 솔레이 왕국의 여왕": ("I'm Solrey's queen.", "我是索雷王國的女王"),
    "나는 무기에서 태어난 여성 정령 아니마": (
        "I'm a female weapon ANIMA.",
        "我是誕生於武器的女性精靈阿尼瑪",
    ),
    "나는 200~600세인 성인 여성 정령": (
        "I'm a female Soul, 200–600.",
        "我是200至600歲的成年女性精靈",
    ),
    "나와 대화하는 구원자는 성인 남성": (
        "You are my adult male Savior.",
        "與我交談的救贖者是成年男性",
    ),
    "나는 구원자에게 연애 감정을 품고 있어": (
        "I'm in love with you.",
        "我對救贖者懷有戀愛的感情",
    ),
    "얼마나 살아왔어?": ("How old are you?", "你活了多少年？"),
    "나는 어떤 존재야?": ("Who am I to you?", "我對你而言是什麼樣的存在？"),
    "나를 어떤 마음으로 대하고 있어?": ("How do you feel about me?", "你對我抱有什麼感情？"),
    "네 이름이 뭐야?": ("What is your name?", "你叫什麼名字？"),
    "네 이명은 뭐야?": ("What is your epithet?", "你的別稱是什麼？"),
    "너는 어떤 유형의 정령이야?": ("What kind of Soul are you?", "你是哪種類型的精靈？"),
    "너는 어디에 소속되어 있어?": ("Which group do you belong to?", "你屬於哪個組織？"),
    "네 별자리가 뭐야?": ("What is your zodiac sign?", "你的星座是什麼？"),
    "네 취미를 알려줘.": ("Tell me about your hobbies.", "告訴我你的愛好。"),
    "네 특기는 뭐야?": ("What are you good at?", "你的特長是什麼？"),
    "네가 좋아하는 것을 말해줘.": ("What do you like?", "你喜歡什麼？"),
    "네가 싫어하는 것을 말해줘.": ("What do you dislike?", "你不喜歡什麼？"),
    "나는 {name}": ("I am {name}", "我是{name}"),
    "내 이명은 {value}": ("I'm called {value}", "我的別稱是{value}"),
    "나는 {value} 정령": ("I'm a {value} Soul", "我是{value}精靈"),
    "내 소속은 {value}": ("I belong to {value}", "我屬於{value}"),
    "내 별자리는 {value}": ("My zodiac is {value}", "我的星座是{value}"),
    "내 취미는 {value}": ("My hobby is {value}", "我的愛好是{value}"),
    "내 특기는 {value}": ("I'm good at {value}", "我的特長是{value}"),
    "좋아하는 것은 {value}": ("I like {value}", "我喜歡{value}"),
    "싫어하는 것은 {value}": ("I dislike {value}", "我不喜歡{value}"),
    "구원자와의 인연 레벨은 {level}": (
        "My bond with you is level {level}",
        "我與救贖者的羈絆等級是{level}",
    ),
    "내가 알고 겪은 것을 짧게 떠올린다.": (
        "Recall my knowledge or past briefly.",
        "簡短回憶我知道或經歷過的事。",
    ),
    "네가 예전에 겪은 일 하나 들려줄래?": (
        "Tell me something you experienced.",
        "能講一件你親身經歷過的事嗎？",
    ),
    "(구원자에게 에버톡 메시지를 보낸다)": (
        "(I send the Savior an EverTalk message)",
        "（我給救贖者發送永恆通訊消息）",
    ),
    "(구원자와 함께 여행 중이다)": ("(I'm traveling with the Savior)", "（我正與救贖者旅行）"),
    "(구원자와 여행 중: {keyword})": (
        "(Traveling with the Savior: {keyword})",
        "（與救贖者旅行：{keyword}）",
    ),
    "(여행 중 '{keyword}' 이야기를 꺼낸다)": (
        "(I bring up '{keyword}' during our trip)",
        "（旅行時我提起“{keyword}”）",
    ),
    "(영지에서 잃어버린 물건 이야기를 한다)": (
        "(I talk about an item lost in the town)",
        "（我談起在領地丟失的物品）",
    ),
    "(구원자에게 인사한다)": ("(I greet the Savior)", "（我向救贖者問好）"),
    "(구원자와 처음 계약을 맺었다)": (
        "(I've just contracted with the Savior)",
        "（我剛與救贖者締結契約）",
    ),
    "(구원자에게 자기소개를 한다)": (
        "(I introduce myself to the Savior)",
        "（我向救贖者介紹自己）",
    ),
    "({name}에 대해 이야기한다)": ("(I talk about {name})", "（我談起{name}）"),
    "(구원자가 로비에 찾아왔다)": ("(The Savior visits me in the lobby)", "（救贖者來到大廳找我）"),
    "({month}월 {day}일, 구원자가 로비에 찾아왔다)": (
        "({month}/{day}, the Savior visits me)",
        "（{month}月{day}日，救贖者來找我）",
    ),
    "({month}월 {day}일, 구원자와 로비에 있다)": (
        "({month}/{day}, I'm with the Savior)",
        "（{month}月{day}日，我與救贖者在大廳）",
    ),
    "(구원자와 로비에 함께 있다)": ("(I'm with the Savior in the lobby)", "（我與救贖者同在大廳）"),
    "(구원자가 특별하게 쓰다듬는다)": (
        "(The Savior caresses me affectionately)",
        "（救贖者親暱地撫摸我）",
    ),
    "(구원자가 다시 특별하게 쓰다듬는다)": (
        "(The Savior caresses me again)",
        "（救贖者再次親暱地撫摸我）",
    ),
    "(구원자와의 인연이 깊어졌다)": (
        "(My bond with the Savior deepens)",
        "（我與救贖者的羈絆加深了）",
    ),
    "(구원자와의 인연이 더 깊어졌다)": (
        "(My bond with the Savior grows deeper)",
        "（我與救贖者的羈絆更加深厚了）",
    ),
    "(구원자와의 인연이 가장 깊어졌다)": (
        "(My bond with the Savior is at its deepest)",
        "（我與救贖者的羈絆最為深厚）",
    ),
    "(구원자가 한동안 내버려 두었다)": (
        "(The Savior has left me alone for a while)",
        "（救贖者有一陣子沒理我了）",
    ),
    "(로비에서 혼잣말을 한다)": ("(I talk to myself in the lobby)", "（我在大廳自言自語）"),
    "(기분이 가라앉아 있다)": ("(I'm feeling down)", "（我心情低落）"),
    "(기분이 좋다)": ("(I'm in a good mood)", "（我心情很好）"),
    "(구원자에게 애정을 느낀다)": ("(I feel affection for the Savior)", "（我對救贖者心生愛意）"),
    "(아르바이트를 시작한다)": ("(I start my part-time work)", "（我開始兼職工作）"),
    "(아르바이트를 마쳤다)": ("(I've finished my part-time work)", "（我完成了兼職工作）"),
    "나는 유물에 깃든 영혼인 정령": ("I'm a Soul born of a relic.", "我是寄宿於遺物的靈魂精靈"),
    "구원자는 과거에서 소환된 인간": (
        "You came from the past.",
        "救贖者是從過去召喚來的人類",
    ),
    "구원자는 정령과 계약하는 정령술사": (
        "You contract with Souls.",
        "救贖者是與精靈締約的精靈術士",
    ),
    "에버톡으로 구원자와 메시지를 나눈다": (
        "I message you on EverTalk.",
        "我用永恆通訊與救贖者發消息",
    ),
    "에덴은 인간이 사라진 정령들의 낙원": (
        "I live in Eden among Souls.",
        "我生活的伊甸已無人類居住",
    ),
    "에덴의 대륙 이름은 아르카디아": ("My continent is Arcadia.", "我生活的大陸名叫阿卡迪亞"),
    "정령은 죽으면 정령석으로 돌아가 잠든다": (
        "We sleep within Soulstones.",
        "我們死後回到精靈石中沉睡",
    ),
    "긴 잠에서 깨면 옛 기억이 흐려진다": (
        "Long sleep dims our memories.",
        "漫長沉睡後我們的舊記憶會模糊",
    ),
    "하늘에 게이트가 열려 마물이 쏟아진다": (
        "Gates bring monsters here.",
        "我生活的天空中門會湧出魔物",
    ),
    "일곱 나라가 정령 연합군을 만들었다": (
        "Seven nations made our army.",
        "七國組成了我們的精靈聯軍",
    ),
    "유리아는 솔레이 왕국의 여왕": ("Yuria is Solrey's queen.", "尤莉婭是索雷王國的女王"),
    "메피스토펠레스는 방주의 인공 정령": (
        "Mephi is an artificial Soul.",
        "梅菲斯托佩勒斯是方舟的人造精靈",
    ),
    "구원자는 아케나인의 영주": ("You are the lord of Arkenine.", "救贖者是阿刻奈因的領主"),
    "천사형과 악마형 정령은 드물고 강하다": (
        "Angel/Demon Souls are rare.",
        "天使型與惡魔型精靈稀有而強大",
    ),
    "계약한 구원자와는 인연의 끈이 이어진다": (
        "Our contract links our bonds.",
        "契約以羈絆之繩連結我與救贖者",
    ),
    "넌 어떤 존재야?": ("What kind of being are you?", "你是什麼樣的存在？"),
    "나는 어디에서 왔지?": ("Where did I come from?", "我從哪裡來？"),
    "내가 정령과 계약할 수 있어?": ("Can I form a contract with a Soul?", "我能與精靈締結契約嗎？"),
    "나와 어떻게 메시지를 나눠?": ("How do you message me?", "你怎麼與我發消息？"),
    "네가 살아온 에덴은 어떤 곳이야?": (
        "What is your home, Eden, like?",
        "你生活的伊甸是什麼樣的地方？",
    ),
    "우리가 사는 대륙 이름이 뭐야?": ("What is our continent called?", "我們生活的大陸叫什麼？"),
    "정령은 죽으면 어떻게 돼?": ("What happens when a Soul dies?", "精靈死後會怎樣？"),
    "긴 잠에서 깨도 기억이 선명해?": (
        "Are memories clear after a long sleep?",
        "漫長沉睡醒來後記憶還清晰嗎？",
    ),
    "게이트에서는 무슨 일이 일어나?": ("What happens at the Gates?", "門裡會發生什麼？"),
    "정령 연합군은 누가 만들었어?": ("Who formed the Soul Alliance army?", "是誰組成了精靈聯軍？"),
    "유리아는 어떤 위치에 있어?": ("What position does Yuria hold?", "尤莉婭是什麼身份？"),
    "메피스토펠레스는 어떤 정령이야?": (
        "What kind of Soul is Mephistopheles?",
        "梅菲斯托佩勒斯是什麼樣的精靈？",
    ),
    "내가 영주로 있는 곳이 어디지?": ("Where am I the lord?", "我是哪片領地的領主？"),
    "천사형과 악마형은 흔해?": ("Are Angel and Demon Souls common?", "天使型與惡魔型精靈常見嗎？"),
    "우리 계약은 무슨 의미야?": ("What does our contract mean?", "我們的契約有什麼意義？"),
}


def render(text: str, language: str = "kr", **values: object) -> str:
    if language == "kr":
        return text.format(**values)
    if language not in ("en", "zh_tw") or text not in TRANSLATIONS:
        raise EvaiError(f"No contract template rendering for {language}: {text}")
    return TRANSLATIONS[text][0 if language == "en" else 1].format(**values)
