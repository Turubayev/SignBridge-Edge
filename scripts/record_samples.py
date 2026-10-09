"""Запись примеров жестов с веб-камеры (сохраняются только ключевые точки, не видео).

    python scripts/record_samples.py --signer S001 --consent C2026-001 --gloss СПАСИБО --n 10 \
        --skin-tone III-IV --lighting хорошее --device веб-камера --age 30–55 --hand right
    python scripts/record_samples.py --list                 # что уже записано
    python scripts/record_samples.py --revoke C2026-001     # отзыв согласия: удалить все записи

Перед записью участник подписывает информированное согласие; его номер заносится в
data/consent_registry.csv. Без действующего согласия запись не начнётся.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from signbridge.dataset import save_sample  # noqa: E402

REGISTRY = ROOT / "data" / "consent_registry.csv"  # шаблон: data/consent_registry.template.csv
REC = ROOT / "data" / "recordings"


def consent_ok(consent_id: str) -> bool:
    if not REGISTRY.exists():
        return False
    with open(REGISTRY, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["consent_id"] == consent_id and not r.get("revoked_date"):
                return True
    return False


def revoke(consent_id: str):
    n = 0
    for f in REC.rglob("*.npz"):
        meta = json.loads(str(np.load(f)["meta"]))
        if meta.get("consent_id") == consent_id:
            f.unlink()
            n += 1
    rows = list(csv.DictReader(open(REGISTRY, encoding="utf-8")))
    for r in rows:
        if r["consent_id"] == consent_id:
            r["revoked_date"] = time.strftime("%Y-%m-%d")
    with open(REGISTRY, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Согласие {consent_id} отозвано, удалено записей: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer")
    ap.add_argument("--consent")
    ap.add_argument("--gloss")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--skin-tone", default="")
    ap.add_argument("--lighting", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--age", default="")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--revoke")
    a = ap.parse_args()
    if a.revoke:
        return revoke(a.revoke)
    if a.list:
        for d in sorted(p for p in REC.iterdir() if p.is_dir()):
            print(f"{d.name}: {len(list(d.glob('*.npz')))}")
        return
    if not (a.signer and a.consent and a.gloss):
        ap.error("нужны --signer, --consent, --gloss")
    if not consent_ok(a.consent):
        sys.exit(f"Согласие {a.consent} не найдено в {REGISTRY} или отозвано — запись невозможна.")

    import cv2

    from signbridge.landmarks import HolisticExtractor

    ext = HolisticExtractor(ROOT / "models" / "holistic_landmarker.task")
    cap = cv2.VideoCapture(a.camera)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    meta = {"signer_id": a.signer, "consent_id": a.consent, "skin_tone": a.skin_tone, "lighting": a.lighting,
            "device": a.device, "age": a.age, "hand": a.hand, "fps": fps}
    t0 = time.time()
    for k in range(a.n):
        for c in (3, 2, 1):  # обратный отсчёт
            end = time.time() + 0.7
            while time.time() < end:
                ok, fr = cap.read()
                cv2.putText(fr, f"{a.gloss}: {k + 1}/{a.n}  ...{c}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
                cv2.imshow("record", cv2.flip(fr, 1))
                cv2.waitKey(1)
        frames, end = [], time.time() + a.seconds
        while time.time() < end:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(ext(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), int((time.time() - t0) * 1000)))
            cv2.circle(fr, (30, 30), 12, (0, 0, 255), -1)
            cv2.imshow("record", cv2.flip(fr, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
        path = save_sample(REC, a.gloss, np.stack(frames), meta)
        print("сохранено", path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
