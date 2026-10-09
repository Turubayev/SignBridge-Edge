"""Хранение записей и разбиение по исполнителям.

Формат записи: data/recordings/<ГЛОСС>/<signer>_<n>.npz
  * landmarks — (T, K, 3) float32, координаты MediaPipe (кадр видео НЕ сохраняется);
  * meta — JSON-строка: signer_id, hand (right/left), skin_tone, lighting, device, age, consent_id.

Согласие участников ведётся в data/consent_registry.csv (consent_id, дата, объём согласия,
отзыв). Записи с отозванным согласием удаляются `scripts/record_samples.py --revoke <consent_id>`.

Проверка качества всегда signer-independent: исполнители из валидации не встречаются в обучении,
иначе точность завышается.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_sample(root: str | Path, label: str, landmarks: np.ndarray, meta: dict) -> Path:
    d = Path(root) / label
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob(f"{meta['signer_id']}_*.npz")))
    path = d / f"{meta['signer_id']}_{n:03d}.npz"
    np.savez_compressed(path, landmarks=landmarks.astype(np.float32), meta=json.dumps(meta, ensure_ascii=False))
    return path


def load_dir(root: str | Path) -> tuple[list, list[str]]:
    """→ ([(seq, label_idx, meta), …], labels)."""
    root = Path(root)
    labels = sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.npz")))
    data = []
    for i, lab in enumerate(labels):
        for f in sorted((root / lab).glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            data.append((z["landmarks"], i, json.loads(str(z["meta"]))))
    return data, labels


def signer_split(data: list, val_frac: float = 0.2, seed: int = 0):
    signers = sorted({m["signer_id"] for _, _, m in data})
    rng = np.random.default_rng(seed)
    rng.shuffle(signers)
    n_val = max(1, int(round(len(signers) * val_frac)))
    val = set(signers[:n_val])
    return [x for x in data if x[2]["signer_id"] not in val], [x for x in data if x[2]["signer_id"] in val]


def group_report(y_true: np.ndarray, y_pred: np.ndarray, metas: list[dict], groups=("skin_tone", "lighting", "device", "age", "hand")) -> dict:
    """Точность по группам и максимальный разрыв (п.п.) — для Model Card."""
    out = {}
    for g in groups:
        vals = [m.get(g) if g != "hand" else ("left" if m.get("left_handed") or m.get("hand") == "left" else "right") for m in metas]
        if all(v is None for v in vals):
            continue
        acc = {}
        for v in sorted({v for v in vals if v is not None}):
            idx = np.array([x == v for x in vals])
            acc[v] = {"accuracy": round(float(np.mean(y_true[idx] == y_pred[idx])), 4), "n": int(idx.sum())}
        gap = 100 * (max(a["accuracy"] for a in acc.values()) - min(a["accuracy"] for a in acc.values()))
        out[g] = {"by_group": acc, "gap_pp": round(gap, 1), "within_target_5pp": bool(gap <= 5.0)}
    return out
