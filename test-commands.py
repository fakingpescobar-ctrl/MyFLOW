"""Временный тест классификации голосовых команд (без серверов/LLM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from myflow.commands import classify, is_llm_intent_candidate, VoiceCommand


def check(name, text, expect_kind=None, expect_text="", expect_rephrase=False):
    cmd = classify(text)
    if expect_kind is None:
        assert cmd is None, f"{name}: ждали None, получили {cmd}"
        print(f"  {name} — OK (не команда)")
    else:
        assert cmd is not None, f"{name}: команда не распознана"
        assert cmd.kind == expect_kind, f"{name}: kind {cmd.kind}, ждали {expect_kind}"
        assert cmd.text == expect_text, f"{name}: text={cmd.text!r}, ждали {expect_text!r}"
        assert cmd.rephrase_last == expect_rephrase, f"{name}: rephrase={cmd.rephrase_last}"
        print(f"  {name} — OK ({cmd.kind}"
              + (f" <{cmd.text}>" if cmd.text else "")
              + (f", rephrase_last" if cmd.rephrase_last else "") + ")")


if __name__ == "__main__":
    # 1) точные фразы
    check("ровный «стоп»", "стоп", "stop")
    check("«скип» с пунктуацией", "скип.", "skip")
    check("«удали последнее»", "удали последнее", "delete_last")
    check("голый «отправь»", "отправь", "send", "", True)
    check("голый «перескажи»", "перескажи", "summarize", "", True)
    check("голый «списком»", "списком", "list", "", True)

    # 2) префиксы с аргументом
    check("«перескажи про митинг»", "перескажи про митинг", "summarize", "про митинг")
    check("«список задач на завтра» -> текст (не команда — «задач» в safelist)",
          "список задач на завтра", None)
    check("«списком поставщиков цены» (не из safelist) -> команда",
          "списком поставщиков цены", "list", "поставщиков цены")

    # 3) «отправь X» — НЕ команда (обычная диктовка)
    check("«отправь документы Тане» -> текст", "отправь документы Тане", None)

    # 4) кандидаты LLM-интента
    assert is_llm_intent_candidate("удали последнюю фразу"), "маркер «удали» не найден"
    assert is_llm_intent_candidate("кратко резюмируй встречу"), "маркер «кратко» не найден"
    assert is_llm_intent_candidate("пиши как список"), "маркер «спис» не найден"
    assert not is_llm_intent_candidate("завтра еду в аэропорт"), "ложный маркер"
    print("  4) LLM-интент кандидаты — OK")

    # 5) обычный текст не триггерится
    check("обычная диктовка", "сегодня хорошая погода едем на дачу", None)
    check("«список» внутри фразы (не в начале)", "дай мне список продуктов", None)

    print("COMMANDS TEST: ALL OK")