"""Small pure-function tests: cleanup, kana helpers, frequency value parsing."""

from i1sort.dictdb import parse_freq_value, pos_family
from i1sort.parse.clean import clean_text, normalize_for_tokenizer
from i1sort.textutil import is_kana, kata_to_hira, readings_match


def test_kana_helpers():
    assert kata_to_hira("カワイイ") == "かわいい"
    assert is_kana("すごーい") and not is_kana("凄い")


def test_readings_match_rules():
    assert readings_match("はんばいがいしゃ", "はんばいかいしゃ")  # rendaku ignored
    assert readings_match("すごーい", "すごおい")  # ー matches any vowel
    assert readings_match("みかづき", "みかずき")
    assert not readings_match("ににんさんきゃく", "ふたりさんきゃく")


def test_clean_speaker_labels_and_readings():
    c = clean_text("（藤本(ふじもと)）　君こそ理想の人材だ　草薙(くさなぎ)少佐")
    assert c.text == "君こそ理想の人材だ 草薙少佐"
    assert "草薙" in c.annotated
    assert clean_text("（カズマ）１つだけ…").text == "１つだけ…"
    assert clean_text("死にたくない（笑）").text == "死にたくない"


def test_clean_html_and_furigana():
    assert clean_text("海賊[かいぞく]だー！<br>ok").text == "海賊だー！ ok"
    assert clean_text("[sound:x.mp3]<b>今日</b>&nbsp;は").text == "今日 は"


def test_clean_music():
    c = clean_text("♪ きっと このまま ♪")
    assert c.has_music and c.text == "きっと このまま"


def test_normalize_small_vowels():
    assert normalize_for_tokenizer("すごぉい") == "すごーい"
    assert normalize_for_tokenizer("ねぇ") == "ねー"
    assert normalize_for_tokenizer("お邪魔しま～す") == "お邪魔しまーす"


def test_parse_freq_value_formats():
    assert parse_freq_value(123) == ("", 123.0, False)
    assert parse_freq_value("1234") == ("", 1234.0, False)
    assert parse_freq_value({"value": 614, "displayValue": "614㋕"}) == ("", 614.0, True)
    assert parse_freq_value({"reading": "たべる", "frequency": 199}) == ("たべる", 199.0, False)
    assert parse_freq_value({"reading": "たべる", "frequency": {"value": 184, "displayValue": "184"}}) == ("たべる", 184.0, False)
    assert parse_freq_value("★★★★★ (1)")[1] == 1.0


def test_pos_family():
    assert pos_family("v5r") == "verb" and pos_family("vs") == "vs" and pos_family("adj-ix") == "adj-i"


def test_clean_keeps_fully_bracketed_line():
    from i1sort.parse.clean import clean_text
    assert clean_text("（大丈夫ですか）").text == "大丈夫ですか"
    assert clean_text("（リムル）何やら 胸騒ぎがする").text == "何やら 胸騒ぎがする"
    assert clean_text("人数集めの一環だろうか （うん 次言っとくね）").text == "人数集めの一環だろうか"


def test_long_stretched_shout_is_bounded():
    # 18 ー in one chunk used to build 3^18 spelling variants (OOM)
    import time
    from i1sort.config import load_config
    from i1sort.parse.pipeline import Parser
    p = Parser(load_config(), use_cache=False)
    t = time.time()
    p.parse_words("うわーぁーぁーぁーぁーぁーぁーぁーぁーぁーぁーぁーぁーぁーぁーぁーぁーぁ")
    assert time.time() - t < 5


def test_clean_broadcast_markup():
    from i1sort.parse.clean import clean_text
    assert clean_text("《テル：あっ　これ➨").text.startswith("あっ これ")
    assert clean_text("呼ばないといけないパターンだ…》").text == "呼ばないといけないパターンだ…"
    assert clean_text("[無線]（コンラッド）うむ").text == "うむ"
    assert clean_text("反対する同志 ｢ﾘﾋﾞｭｰﾏﾝ｣｡").text == "反対する同志 「リビューマン」。"
    assert clean_text("今日は休日です＞").text == "今日は休日です"


def test_clean_keeps_bracketed_speech_not_captions():
    from i1sort.parse.clean import clean_text
    assert clean_text("（銃声）（悲鳴）").text == ""
    assert clean_text("（大丈夫ですか）").text == "大丈夫ですか"
