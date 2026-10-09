"""Нормализация последовательности ключевых точек и признаки для модели.

* Центр — середина плеч, масштаб — ширина плеч: признаки не зависят от расстояния до камеры
  и положения человека в кадре.
* Кисти дополнительно описываются в собственной системе (относительно запястья, масштаб —
  размер ладони): так конфигурация пальцев (важна для дактиля) не теряется на фоне движения руки.
* Длина приводится к T кадрам линейной интерполяцией (жесты выполняются с разной скоростью).
* Зеркалирование (для левшей) — перестановка рук и смена знака x.

Та же нормализация реализована в web/app.js для браузерной версии.
"""
from __future__ import annotations

import numpy as np

from .landmarks import K, N_HAND, P_LSH, P_RSH, SL_FACE, SL_LHAND, SL_POSE, SL_RHAND

T_FIXED = 48
FEAT_DIM = K * 3 + 2 * N_HAND * 3 + 2  # глобальные точки + локальные кисти + флаги наличия рук


def interpolate_missing(seq: np.ndarray) -> np.ndarray:
    """Заполняет короткие пропуски (NaN) линейной интерполяцией по времени; длинные — нулями позже."""
    out = seq.copy()
    T = len(seq)
    t = np.arange(T)
    flat = out.reshape(T, -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = np.isfinite(col)
        if ok.sum() >= 2 and not ok.all():
            flat[:, j] = np.interp(t, t[ok], col[ok], left=np.nan, right=np.nan)
    return flat.reshape(seq.shape)


def resample(seq: np.ndarray, T: int = T_FIXED) -> np.ndarray:
    if len(seq) == T:
        return seq
    src = np.linspace(0, len(seq) - 1, T)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(seq) - 1)
    w = (src - i0)[:, None, None]
    a, b = seq[i0], seq[i1]
    return np.where(np.isfinite(a) & np.isfinite(b), a * (1 - w) + b * w, np.where(np.isfinite(a), a, b))


def mirror(seq: np.ndarray) -> np.ndarray:
    """Зеркальное отражение: x → 1 − x, левая и правая стороны тела меняются местами."""
    f = seq.copy()
    f[..., 0] = 1.0 - f[..., 0]
    out = f.copy()
    out[:, SL_LHAND], out[:, SL_RHAND] = f[:, SL_RHAND], f[:, SL_LHAND]
    for sl, pairs in ((SL_POSE, ((1, 2), (3, 4), (5, 6), (7, 8))),  # плечи, локти, запястья, бёдра
                      (SL_FACE, ((0, 5), (1, 4), (2, 3), (6, 9), (7, 8), (10, 11)))):  # брови, глаза, уголки губ
        part = f[:, sl].copy()
        for a, b in pairs:
            part[:, [a, b]] = part[:, [b, a]]
        out[:, sl] = part
    return out


def normalize(seq: np.ndarray, T: int = T_FIXED) -> np.ndarray:
    """(T0, K, 3) сырые координаты MediaPipe → (T, FEAT_DIM) признаки."""
    seq = interpolate_missing(np.asarray(seq, np.float32))
    seq = resample(seq, T)
    pose = seq[:, SL_POSE]
    ls, rs = pose[:, P_LSH], pose[:, P_RSH]
    center = np.nanmean((ls + rs) / 2, axis=0)
    width = np.nanmean(np.linalg.norm((ls - rs)[:, :2], axis=-1))
    if not np.isfinite(width) or width < 1e-3:
        width = 0.25
    if not np.isfinite(center).all():
        center = np.array([0.5, 0.5, 0.0], np.float32)
    g = (seq - center) / width
    feats = [np.nan_to_num(g, nan=0.0).reshape(T, -1)]
    present = []
    for sl in (SL_LHAND, SL_RHAND):
        hand = seq[:, sl]
        wrist = hand[:, :1]
        size = np.linalg.norm((hand[:, 9] - hand[:, 0])[:, :2], axis=-1)[:, None, None]  # запястье → основание среднего пальца
        local = (hand - wrist) / np.where(np.isfinite(size) & (size > 1e-4), size, 1.0)
        feats.append(np.nan_to_num(local, nan=0.0).reshape(T, -1))
        present.append(np.isfinite(hand[:, 0, 0]).astype(np.float32)[:, None])
    return np.concatenate(feats + present, axis=1).astype(np.float32)


def augment(seq: np.ndarray, rng: np.random.Generator, p_mirror: float = 0.5) -> np.ndarray:
    """Аугментация сырых координат: зеркало, масштаб, поворот, сдвиг, шум, скорость, выпадение кадров."""
    s = seq.copy()
    if rng.random() < p_mirror:
        s = mirror(s)
    T0 = len(s)
    speed = rng.uniform(0.8, 1.25)
    s = resample(s, max(8, int(T0 * speed)))
    ang = np.radians(rng.normal(0, 7))
    c, si = np.cos(ang), np.sin(ang)
    xy = s[..., :2] - 0.5
    s[..., 0] = 0.5 + rng.uniform(0.9, 1.1) * (c * xy[..., 0] - si * xy[..., 1]) + rng.normal(0, 0.03)
    s[..., 1] = 0.5 + rng.uniform(0.9, 1.1) * (si * xy[..., 0] + c * xy[..., 1]) + rng.normal(0, 0.03)
    s = s + rng.normal(0, 0.004, s.shape).astype(np.float32)
    drop = rng.random(len(s)) < 0.05
    s[drop, SL_LHAND if rng.random() < 0.5 else SL_RHAND] = np.nan
    return s.astype(np.float32)
