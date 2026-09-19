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
    "나와 대화하는 구원자는 성인 남성": (
        "You are my adult male Savior.",
        "與我交談的救援者是成年男性",
    ),
    "나는 구원자에게 연애 감정을 품고 있어": (
        "I'm in love with you.",
        "我對救援者懷有戀愛的感情",
    ),
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
    "{name}의 소속은 {value}": ("{name} belongs to {value}", "{name}屬於{value}"),
    "{name}의 유형은 {value}": ("{name} is a {value} Soul", "{name}是{value}精靈"),
    "{name}에 대해 알아?": ("Do you know {name}?", "你認識{name}嗎？"),
    "구원자와의 인연 레벨은 {level}": (
        "My bond with you is level {level}",
        "我與救援者的羈絆等級是{level}",
    ),
    "내가 알고 겪은 것을 짧게 떠올린다.": (
        "Recall my knowledge or past briefly.",
        "簡短回憶我知道或經歷過的事。",
    ),
    "네가 예전에 겪은 일 하나 들려줄래?": (
        "Tell me something you experienced.",
        "能講一件你親身經歷過的事嗎？",
    ),
    "(구원자에게 먼저 말을 건다)": (
        "(You speak to the Savior first)",
        "（妳先向救援者搭話）",
    ),
    "(구원자와 함께 놀러 나와 있다)": (
        "(You are out having fun with the Savior)",
        "（妳正和救援者一起出去玩）",
    ),
    "(구원자와 놀러 나와서: {keyword})": (
        "(Out with the Savior: {keyword})",
        "（和救援者出去玩：{keyword}）",
    ),
    "(구원자와 놀다가 '{keyword}' 이야기를 꺼낸다)": (
        "(While out together, you bring up '{keyword}')",
        "（一起玩的時候，妳提起“{keyword}”）",
    ),
    "(구원자가 데이트를 청한다)": (
        "(The Savior asks you on a date)",
        "（救援者約妳出去約會）",
    ),
    "(구원자와 놀다가 헤어질 시간이 되었다)": (
        "(After your time out together, it is time to part from the Savior)",
        "（和救援者玩過之後，到了分別的時候）",
    ),
    "(잃어버린 물건 이야기를 한다)": (
        "(You talk about something you lost)",
        "（妳談起遺失的東西）",
    ),
    "(구원자에게 인사한다)": ("(You greet the Savior)", "（妳向救援者問好）"),
    "(구원자와 처음 계약을 맺었다)": (
        "(You have just contracted with the Savior)",
        "（妳剛與救援者締結契約）",
    ),
    "(구원자에게 자기소개를 한다)": (
        "(You introduce yourself to the Savior)",
        "（妳向救援者介紹自己）",
    ),
    "({name}에 대해 이야기한다)": ("(You talk about {name})", "（妳談起{name}）"),
    "(구원자가 찾아왔다)": (
        "(The Savior comes to see you)",
        "（救援者來找妳）",
    ),
    "({month}월 {day}일, 구원자가 찾아왔다)": (
        "({month}/{day}, the Savior comes to see you)",
        "（{month}月{day}日，救援者來找妳）",
    ),
    "({month}월 {day}일, 구원자와 함께 있다)": (
        "({month}/{day}, you are with the Savior)",
        "（{month}月{day}日，妳與救援者在一起）",
    ),
    "(구원자와 함께 있다)": (
        "(You are with the Savior)",
        "（妳與救援者在一起）",
    ),
    "(구원자와 단둘이 밤을 보낸다)": (
        "(You spend the night alone with the Savior)",
        "（妳與救援者單獨共度夜晚）",
    ),
    "(구원자와 함께 씻는다)": (
        "(You bathe together with the Savior)",
        "（妳與救援者一起洗澡）",
    ),
    "(구원자가 특별하게 쓰다듬는다)": (
        "(The Savior caresses you affectionately)",
        "（救援者親暱地撫摸妳）",
    ),
    "(구원자가 다시 특별하게 쓰다듬는다)": (
        "(The Savior caresses you again)",
        "（救援者再次親暱地撫摸妳）",
    ),
    "(구원자와의 인연이 깊어졌다)": (
        "(Your bond with the Savior deepens)",
        "（妳與救援者的羈絆加深了）",
    ),
    "(구원자와의 인연이 더 깊어졌다)": (
        "(Your bond with the Savior grows deeper)",
        "（妳與救援者的羈絆更加深厚了）",
    ),
    "(구원자와의 인연이 가장 깊어졌다)": (
        "(Your bond with the Savior is at its deepest)",
        "（妳與救援者的羈絆最為深厚）",
    ),
    "(구원자가 한동안 내버려 두었다)": (
        "(The Savior has left you alone for a while)",
        "（救援者有一陣子沒理妳了）",
    ),
    "(혼잣말을 한다)": ("(You talk to yourself)", "（妳自言自語）"),
    "(기분이 가라앉아 있다)": ("(You are feeling down)", "（妳心情低落）"),
    "(기분이 좋다)": ("(You are in a good mood)", "（妳心情很好）"),
    "(구원자에게 애정을 느낀다)": ("(You feel affection for the Savior)", "（妳對救援者心生愛意）"),
    "(일을 하러 나선다)": ("(You head off to work)", "（妳出門去工作）"),
    "(일을 마치고 돌아왔다)": ("(You are back from work)", "（妳工作結束回來了）"),
    "나는 유물에 깃든 영혼인 정령": ("I'm a Soul born of a relic.", "我是寄宿於遺物的靈魂精靈"),
    "구원자는 과거에서 소환된 인간": (
        "You came from the past.",
        "救援者是從過去召喚來的人類",
    ),
    "구원자는 정령과 계약하는 정령술사": (
        "You contract with Souls.",
        "救援者是與精靈締約的精靈術士",
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
    "구원자는 아케나인의 영주": ("You are the lord of Arkenine.", "救援者是阿刻奈因的領主"),
    "천사형과 악마형 정령은 드물고 강하다": (
        "Angel/Demon Souls are rare.",
        "天使型與惡魔型精靈稀有而強大",
    ),
    "계약한 구원자와는 인연의 끈이 이어진다": (
        "Our contract links our bonds.",
        "契約以羈絆之繩連結我與救援者",
    ),
    "넌 어떤 존재야?": ("What kind of being are you?", "你是什麼樣的存在？"),
    "나는 어디에서 왔지?": ("Where did I come from?", "我從哪裡來？"),
    "내가 정령과 계약할 수 있어?": ("Can I form a contract with a Soul?", "我能與精靈締結契約嗎？"),
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
