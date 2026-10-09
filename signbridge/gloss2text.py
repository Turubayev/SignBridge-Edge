"""Глоссы → связный текст на русском и казахском.

Жестовый язык — самостоятельный язык со своей грамматикой, поэтому последовательность
глоссов нельзя просто склеить. Здесь два переводчика с общим интерфейсом:

* `RuleTranslator` — правила и шаблоны для бытовых сценариев MVP (ЦОН, поликлиника),
  падежи русских существительных и казахские аффиксы с учётом сингармонизма. Работает
  полностью офлайн и предсказуемо — важно для киоска.
* `LocalLLMTranslator` — компактная локальная языковая модель через OpenAI-совместимый
  сервер на том же устройстве (llama.cpp, Ollama). Данные не покидают устройство.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

VOCAB_DIR = Path(__file__).resolve().parent.parent / "data" / "vocab"

# Формы русских существительных: винительный, дательный, родительный; род.
RU_NOUNS = {
    "ПАСПОРТ": ("паспорт", "паспорту", "паспорта", "m"), "УДОСТОВЕРЕНИЕ": ("удостоверение личности", "удостоверению личности", "удостоверения личности", "n"),
    "СПРАВКА": ("справку", "справке", "справки", "f"), "ДОКУМЕНТ": ("документ", "документу", "документа", "m"),
    "ПОДПИСЬ": ("подпись", "подписи", "подписи", "f"), "ОКНО": ("окно", "окну", "окна", "n"), "НОМЕР": ("номер", "номеру", "номера", "m"),
    "ОЧЕРЕДЬ": ("очередь", "очереди", "очереди", "f"), "ЦОН": ("ЦОН", "ЦОНу", "ЦОНа", "m"), "БАНК": ("банк", "банку", "банка", "m"),
    "ДЕНЬГИ": ("деньги", "деньгам", "денег", "pl"), "ВРАЧ": ("врача", "врачу", "врача", "m"), "БОЛЬНИЦА": ("больницу", "больнице", "больницы", "f"),
    "АПТЕКА": ("аптеку", "аптеке", "аптеки", "f"), "ЛЕКАРСТВО": ("лекарство", "лекарству", "лекарства", "n"),
    "ВОДА": ("воду", "воде", "воды", "f"), "ТУАЛЕТ": ("туалет", "туалету", "туалета", "m"), "ВХОД": ("вход", "входу", "входа", "m"),
    "ВЫХОД": ("выход", "выходу", "выхода", "m"), "ПЕРЕВОДЧИК": ("сурдопереводчика", "сурдопереводчику", "сурдопереводчика", "m"),
    "ТЕЛЕФОН": ("телефон", "телефону", "телефона", "m"), "АДРЕС": ("адрес", "адресу", "адреса", "m"), "ИМЯ": ("имя", "имени", "имени", "n"),
    "ТЕМПЕРАТУРА": ("температуру", "температуре", "температуры", "f"), "ДАВЛЕНИЕ": ("давление", "давлению", "давления", "n"),
}
RU_NEEDED = {"m": "нужен", "f": "нужна", "n": "нужно", "pl": "нужны"}
BODY = {"ГОЛОВА", "ЖИВОТ", "ЗУБ"}
TIME = {"СЕГОДНЯ", "ЗАВТРА", "ВЧЕРА"}

_BACK = set("аоұыуяёю")
_FRONT = set("әөүіеэи")
_VOICELESS = set("кқпстфхцчшщ")


def _last_vowel_back(word: str) -> bool:
    for ch in reversed(word.lower()):
        if ch in _BACK:
            return True
        if ch in _FRONT:
            return False
    return True


def kk_dative(word: str) -> str:
    """Барыс септік: -ға/-ге/-қа/-ке/-а/-е (упрощённо)."""
    last = word[-1].lower()
    back = _last_vowel_back(word)
    if last in "кг":  # заимствования на -к/-г: банкке, паркке
        back = False
    elif last in "қғ":
        back = True
    if last in _VOICELESS:
        return word + ("қа" if back else "ке")
    return word + ("ға" if back else "ге")


def kk_accusative(word: str) -> str:
    """Табыс септік: -ны/-ні после гласной, -ты/-ті после глухих, -ды/-ді после звонких и сонорных."""
    last = word[-1].lower()
    back = _last_vowel_back(word)
    if last in _BACK | _FRONT:
        return word + ("ны" if back else "ні")
    if last in _VOICELESS:
        return word + ("ты" if back else "ті")
    return word + ("ды" if back else "ді")


def load_vocab() -> tuple[dict, dict]:
    g = json.loads((VOCAB_DIR / "glosses.json").read_text(encoding="utf-8"))["glosses"]
    d = json.loads((VOCAB_DIR / "dactyl_kk.json").read_text(encoding="utf-8"))["letters"]
    return {x["id"]: x for x in g}, {x["id"]: x for x in d}


@dataclass
class Utterance:
    glosses: list[str]
    question: bool = False  # немануальный маркер: поднятые брови
    negation: bool = False  # покачивание головой
    meta: dict = field(default_factory=dict)


class RuleTranslator:
    def __init__(self):
        self.g, self.d = load_vocab()

    # --- служебное ---
    def _merge_dactyl(self, glosses: list[str]) -> list[str]:
        """Подряд идущие буквы дактиля склеиваются в слово: DACT_* … → «@Әлия»."""
        out, word = [], ""
        for x in glosses:
            if x in self.d:
                word += self.d[x]["char"]
            else:
                if word:
                    out.append("@" + word.capitalize())
                    word = ""
                out.append(x)
        if word:
            out.append("@" + word.capitalize())
        return out

    def _ru(self, x: str) -> str:
        return x[1:] if x.startswith("@") else self.g.get(x, {}).get("ru", x.lower())

    def _kk(self, x: str) -> str:
        return x[1:] if x.startswith("@") else self.g.get(x, {}).get("kk", x.lower())

    # --- перевод ---
    def translate(self, utt: Utterance | list[str]) -> dict:
        if isinstance(utt, list):
            utt = Utterance(utt)
        gl = self._merge_dactyl(utt.glosses)
        q = utt.question or any(x in ("ГДЕ", "КОГДА", "СКОЛЬКО", "ЧТО") for x in gl)
        neg = utt.negation
        ru, kk = self._rules(gl, neg)
        if ru is None:
            ru = " ".join(self._ru(x) for x in gl if x not in ("Я",))
            kk = " ".join(self._kk(x) for x in gl if x not in ("Я",))
        end = "?" if q else "."
        return {"ru": _sentence(ru, end), "kk": _sentence(kk, end), "glosses": utt.glosses}

    def _rules(self, gl: list[str], neg: bool):
        s = [x for x in gl if x not in ("ПОЖАЛУЙСТА",)]
        polite = "ПОЖАЛУЙСТА" in gl
        nouns = [x for x in s if x in RU_NOUNS or x.startswith("@")]
        times = [x for x in s if x in TIME]
        t_ru = (self._ru(times[0]) + " ") if times else ""
        t_kk = (self._kk(times[0]) + " ") if times else ""
        has = set(s)

        if has & {"ИМЯ"} and any(x.startswith("@") for x in s):
            name = next(x for x in s if x.startswith("@"))[1:]
            if "ТЫ" in has:
                return "Как вас зовут", "Сіздің атыңыз кім"
            return f"Меня зовут {name}", f"Менің атым {name}"
        if {"ИМЯ", "ТЫ"} <= has:
            return "Как вас зовут", "Сіздің атыңыз кім"
        if "НЕ-ПОНИМАТЬ" in has:
            extra_ru = ", повторите, пожалуйста, медленнее" if has & {"ПОВТОРИТЬ", "МЕДЛЕННО"} else ""
            extra_kk = ", баяу қайталаңызшы" if has & {"ПОВТОРИТЬ", "МЕДЛЕННО"} else ""
            return "Я не понимаю" + extra_ru, "Мен түсінбеймін" + extra_kk
        if has & {"ПОВТОРИТЬ", "МЕДЛЕННО"}:
            return "Повторите, пожалуйста, медленнее", "Баяу қайталаңызшы"
        if "ГДЕ" in has and nouns:
            n = nouns[0]
            ru_nom = self._ru(n)
            return f"Где {ru_nom}", f"{self._kk(n).capitalize()} қайда"
        if "БОЛЕТЬ" in has:
            part = next((x for x in s if x in BODY), None)
            if part:
                return f"{t_ru}У меня болит {self._ru(part)}".strip(), f"{t_kk}Менің {self._kk(part)} ауырады".strip()
            return f"{t_ru}Я болею".strip(), f"{t_kk}Мен ауырып тұрмын".strip()
        if "ЗАПИСАТЬСЯ" in has:
            tgt = next((x for x in nouns if x in RU_NOUNS), "ВРАЧ")
            ru = f"{'Не х' if neg else 'Х'}очу записаться {t_ru}к {RU_NOUNS[tgt][1]}"
            kk = f"{t_kk}{kk_dative(self._kk(tgt))} жазыл{'ғым келмейді' if neg else 'ғым келеді'}"
            return ru.replace("  ", " "), kk
        if "ПОЛУЧИТЬ" in has and nouns:
            n = nouns[0]
            ru = f"{'Я не хочу' if neg else 'Я хочу'} получить {RU_NOUNS.get(n, (self._ru(n),))[0]}"
            kk = f"Мен {kk_accusative(self._kk(n))} {'алғым келмейді' if neg else 'алғым келеді'}"
            return ru, kk
        if "НУЖНО" in has or "ХОТЕТЬ" in has:
            if nouns:
                n = nouns[0]
                form = RU_NOUNS.get(n, (self._ru(n), "", "", "m"))
                if "ХОТЕТЬ" in has:
                    return f"{'Я не хочу' if neg else 'Я хочу'} {form[0]}", f"Мен {kk_accusative(self._kk(n))} {'қаламаймын' if neg else 'қалаймын'}"
                nom = self._ru(n)
                return (f"Мне {'не ' if neg else ''}{RU_NEEDED[form[3]]} {nom}", f"Маған {self._kk(n)} {'керек емес' if neg else 'керек'}")
            if "ПОМОЧЬ" in has or "ПЕРЕВОДЧИК" in has:
                return "Мне нужна помощь", "Маған көмек керек"
        if "ПОМОЧЬ" in has:
            return ("Помогите, пожалуйста" if polite or "Я" in has else "Помогите"), "Көмектесіңізші"
        if "ГЛУХОЙ" in has and "Я" in has:
            ru = "Я глухой (глухая), мне нужен сурдопереводчик" if "ПЕРЕВОДЧИК" in has else "Я глухой (глухая)"
            kk = "Мен саңыраумын, маған сурдоаудармашы керек" if "ПЕРЕВОДЧИК" in has else "Мен саңыраумын"
            return ru, kk
        if "СКОЛЬКО" in has:
            if "ЖДАТЬ" in has:
                return "Сколько ждать", "Қанша күту керек"
            if "ВРЕМЯ" in has:
                return "Сколько сейчас времени", "Сағат қанша"
            if "ДЕНЬГИ" in has:
                return "Сколько стоит", "Қанша тұрады"
        if "КОГДА" in has and nouns:
            return f"Когда {self._ru(nouns[0])}", f"{self._kk(nouns[0]).capitalize()} қашан"
        return None, None


def _sentence(text: str, end: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return text[0].upper() + text[1:] + ("" if text[-1] in ".?!" else end)


class LocalLLMTranslator:
    """Перевод через локальную LLM (OpenAI-совместимый endpoint на этом же устройстве).

    Пример: `llama-server -m qwen2.5-1.5b-instruct-q4.gguf --port 8080`
    """

    def __init__(self, url: str = "http://127.0.0.1:8080/v1/chat/completions", model: str = "local",
                 fallback: RuleTranslator | None = None):
        if not re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?/", url):
            raise ValueError("Разрешён только локальный сервер: видео и тексты не должны покидать устройство")
        self.url, self.model = url, model
        self.fallback = fallback or RuleTranslator()

    def translate(self, utt: Utterance | list[str]) -> dict:
        import requests

        if isinstance(utt, list):
            utt = Utterance(utt)
        prompt = ("Переведи последовательность глоссов казахско-русского жестового языка в грамматически правильное "
                  "предложение. Верни JSON {\"ru\": ..., \"kk\": ...}. Глоссы: " + " ".join(utt.glosses)
                  + (" [вопрос]" if utt.question else "") + (" [отрицание]" if utt.negation else ""))
        try:
            r = requests.post(self.url, json={"model": self.model, "temperature": 0,
                                              "messages": [{"role": "user", "content": prompt}]}, timeout=10)
            js = json.loads(re.search(r"\{.*\}", r.json()["choices"][0]["message"]["content"], re.S).group(0))
            return {"ru": js["ru"], "kk": js["kk"], "glosses": utt.glosses}
        except Exception:  # noqa: BLE001 — нет сервера или ответ не JSON: правила надёжнее пустоты
            return self.fallback.translate(utt)
