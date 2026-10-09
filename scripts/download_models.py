"""Скачивает модель MediaPipe Holistic Landmarker (≈ 13 МБ, Apache 2.0) в models/.

    python scripts/download_models.py
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signbridge.landmarks import MODEL_URL  # noqa: E402


def main():
    out = ROOT / "models" / "holistic_landmarker.task"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        print("Уже есть:", out)
        return
    print("Загрузка", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, out)
    print(f"Готово: {out} ({out.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
