"""Демо с веб-камеры: жесты → глоссы → текст (RU/KZ) → речь. Полностью офлайн.

    python run_demo.py --model models/krsl_isolated.onnx
    python run_demo.py --model models/krsl_isolated.onnx --tts --camera 1

Клавиши: Q — выход, C — очистить фразу, L — язык озвучивания (ru/kk).
Видео не сохраняется и не передаётся: в модель идут только координаты ключевых точек.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from signbridge.features import normalize  # noqa: E402
from signbridge.gloss2text import RuleTranslator, Utterance  # noqa: E402
from signbridge.landmarks import SL_LHAND, SL_POSE, SL_RHAND, HolisticExtractor  # noqa: E402
from signbridge.model import OnnxClassifier  # noqa: E402
from signbridge.segmenter import SignSegmenter  # noqa: E402

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Speaker:
    """Офлайн-озвучивание через голоса ОС (pyttsx3). Казахский голос нужно установить в системе."""

    def __init__(self):
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
        except Exception:  # noqa: BLE001
            self.engine = None

    def say(self, text: str, lang: str = "ru"):
        if not self.engine or not text:
            return
        keys = {"ru": ("russian", "ru_", "ru-", "irina", "pavel"), "kk": ("kazakh", "kk_", "kk-")}[lang]
        for v in self.engine.getProperty("voices"):
            if any(k in f"{v.id} {v.name}".lower() for k in keys):
                self.engine.setProperty("voice", v.id)
                break
        self.engine.say(text)
        self.engine.runAndWait()


def draw(frame, lms, cv2):
    h, w = frame.shape[:2]
    for sl, color in ((SL_LHAND, (80, 200, 120)), (SL_RHAND, (60, 140, 255))):
        hand = lms[sl]
        if not np.isfinite(hand[0, 0]):
            continue
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y, _ in lms[SL_POSE]:
        if np.isfinite(x):
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (200, 200, 200), -1)


def put_text(frame, text, org, scale=0.8, color=(255, 255, 255)):
    """Кириллица в OpenCV не рисуется шрифтами Hershey — используем PIL, если он есть."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(frame[:, :, ::-1])
        d = ImageDraw.Draw(img)
        font = None
        for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                font = ImageFont.truetype(name, int(28 * scale))
                break
            except OSError:
                continue
        d.text(org, text, fill=color[::-1], font=font or ImageFont.load_default())
        frame[:] = np.array(img)[:, :, ::-1]
    except ImportError:
        import cv2

        cv2.putText(frame, text.encode("ascii", "replace").decode(), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/krsl_isolated.onnx")
    ap.add_argument("--holistic", default="models/holistic_landmarker.task")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.7)
    a = ap.parse_args()
    if not .5 <= a.min_conf <= 1: ap.error('min-conf must be between .5 and 1')

    import cv2

    clf = OnnxClassifier(a.model)
    if not clf.translation_allowed:
        ap.error('This model is synthetic or lacks verified real-data/signer-disjoint evaluation. Translation disabled; inspect the browser pipeline demo instead.')
    info = {x["id"]: x for x in json.loads(Path(a.model).with_suffix(".labels.json").read_text(encoding="utf-8"))["info"]}
    ext = HolisticExtractor(a.holistic)
    seg = SignSegmenter()
    tr = RuleTranslator()
    tts = Speaker() if a.tts else None
    lang = "ru"
    cap = cv2.VideoCapture(a.camera)
    glosses, last_pred, sentence = [], "", {"ru": "", "kk": ""}
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # зеркально для пользователя; координаты MediaPipe — с исходного кадра
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        lms = ext(rgb, int((time.time() - t0) * 1000))
        ev = seg.push(lms)
        if ev and "segment" in ev:
            t1 = time.perf_counter()
            top = clf.top_k(normalize(ev["segment"]), 3)
            dt = (time.perf_counter() - t1) * 1000
            if top[0][1] >= a.min_conf and len(top) >= 2 and top[0][1] - top[1][1] >= .15:
                glosses.append(top[0][0])
            last_pred = " | ".join(f"{info[g]['ru']} {p:.0%}" for g, p in top) + f"  ({dt:.0f} мс)"
        if ev and ev.get("end_of_sentence") and glosses:
            sentence = tr.translate(Utterance(glosses, **seg.nonmanual()))
            if tts:
                tts.say(sentence[lang], lang)
            glosses = []
        view = frame.copy()
        draw(view, np.where(np.isfinite(lms), lms * [[-1, 1, 1]] + [[1, 0, 0]], np.nan), cv2)
        cv2.rectangle(view, (0, view.shape[0] - 110), (view.shape[1], view.shape[0]), (30, 30, 30), -1)
        put_text(view, "Жесты: " + " ".join(info[g]["ru"] for g in glosses), (12, view.shape[0] - 105), 0.7)
        put_text(view, sentence.get(lang, ""), (12, view.shape[0] - 70), 1.0, (120, 230, 255))
        put_text(view, last_pred, (12, view.shape[0] - 30), 0.6, (180, 180, 180))
        cv2.imshow("SignBridge-Edge (Q — выход)", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            glosses, sentence = [], {"ru": "", "kk": ""}
        if k == ord("l"):
            lang = "kk" if lang == "ru" else "ru"
    cap.release()
    ext.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
