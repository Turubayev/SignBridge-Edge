"""Перевод видеодатасета (например, K-RSL, Назарбаев Университет) в ключевые точки проекта.

Ожидаемая структура: <src>/<класс>/<видео>.mp4 (структуру K-RSL при необходимости
приведите к ней; датасет предоставляется авторами по запросу и на их условиях).
Имя исполнителя берётся из имени файла по шаблону --signer-regex.

    python scripts/prepare_krsl.py --src D:/K-RSL/videos --dst data/recordings --signer-regex "^(\\w+?)_"
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signbridge.dataset import save_sample  # noqa: E402
from signbridge.landmarks import HolisticExtractor, extract_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/recordings")
    ap.add_argument("--signer-regex", default=r"^([A-Za-z0-9]+)")
    ap.add_argument("--label-map", help="JSON {имя_класса: ГЛОСС} для сопоставления со словарём проекта")
    a = ap.parse_args()
    import json

    lmap = json.loads(Path(a.label_map).read_text(encoding="utf-8")) if a.label_map else {}
    rx = re.compile(a.signer_regex)
    videos = sorted(p for p in Path(a.src).rglob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
    print(f"Видео: {len(videos)}")
    for i, v in enumerate(videos):
        ext = HolisticExtractor(ROOT / "models" / "holistic_landmarker.task")  # новый трекер на видео: метки времени с нуля
        seq = extract_video(v, ext)
        ext.close()
        if len(seq) < 8:
            continue
        m = rx.search(v.stem)
        label = lmap.get(v.parent.name, v.parent.name)
        save_sample(a.dst, label, seq, {"signer_id": m.group(1) if m else "unknown", "source": "K-RSL", "file": v.name})
        if i % 50 == 0:
            print(f"{i}/{len(videos)}", flush=True)


if __name__ == "__main__":
    main()
