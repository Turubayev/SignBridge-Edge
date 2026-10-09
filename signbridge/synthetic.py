"""Процедурные «жесты» для проверки конвейера без реальных записей.

ВАЖНО: это не КРЖЯ. Каждому классу случайно (но воспроизводимо) назначаются место
исполнения, траектория рук, конфигурация пальцев, участие второй руки и мимика.
Синтетика нужна, чтобы обучение, экспорт, браузерная версия и тесты работали «из коробки»;
модель, обученная на ней, не распознаёт настоящие жесты. Для реальной модели нужны K-RSL
и записи носителей языка (scripts/record_samples.py, scripts/prepare_krsl.py).

Генератор моделирует разных «исполнителей»: пропорции тела, положение в кадре, скорость,
левшей/правшей, дрожание и пропуски трекинга, а также метаданные групп (освещение,
устройство, тон кожи) — чтобы отработать отчёт о справедливости.
"""
from __future__ import annotations

import numpy as np

from .landmarks import K, N_HAND, SL_FACE, SL_LHAND, SL_POSE, SL_RHAND

# Скелет кисти: 5 пальцев × 4 сустава + запястье (порядок MediaPipe).
_FINGER_BASE_ANGLES = np.radians([-50, -20, 0, 18, 36])  # большой … мизинец, от оси ладони
_FINGER_LEN = np.array([[0.35, 0.3, 0.25, 0.2], [0.9, 0.45, 0.28, 0.22], [0.9, 0.5, 0.3, 0.24],
                        [0.85, 0.46, 0.28, 0.22], [0.8, 0.38, 0.22, 0.18]])


def hand_shape(curl: np.ndarray, spread: float, wrist: np.ndarray, rot: float, size: float, side: int) -> np.ndarray:
    """21 точка кисти по вектору сгибания пальцев curl (5 значений 0..1)."""
    pts = np.zeros((N_HAND, 3), np.float32)
    pts[0] = wrist
    for f in range(5):
        ang = rot + side * _FINGER_BASE_ANGLES[f] * (1 + spread)
        p = wrist.copy()
        d = np.array([np.sin(ang) * side, -np.cos(ang), 0.0])
        for j in range(4):
            bend = curl[f] * (0.6 + 0.5 * j)
            dirj = np.array([d[0] * np.cos(bend), d[1] * np.cos(bend), -np.sin(bend)])
            p = p + dirj * _FINGER_LEN[f, j] * size
            pts[1 + 4 * f + j] = p
    return pts


class GestureSpec:
    """Случайное, но фиксированное для класса описание жеста."""

    LOCATIONS = {"neutral": (0.0, 0.9), "chest": (0.0, 0.5), "face": (0.0, -0.6), "chin": (0.0, -0.35),
                 "side": (0.8, 0.6), "forehead": (0.0, -1.0)}

    def __init__(self, class_id: int, static: bool = False):
        r = np.random.default_rng(1000 + class_id)
        self.loc = np.array(self.LOCATIONS[r.choice(list(self.LOCATIONS))]) + r.normal(0, 0.15, 2)
        self.amp = (0.05 if static else r.uniform(0.2, 0.7)) * r.uniform(0.6, 1.0, 2)
        self.freq = r.integers(1, 3, 2)
        self.phase = r.uniform(0, 2 * np.pi, 2)
        self.curl0, self.curl1 = r.uniform(0, 1.4, 5), r.uniform(0, 1.4, 5)
        if static:
            self.curl1 = self.curl0 + r.normal(0, 0.1, 5)
        self.spread = r.uniform(-0.3, 0.6)
        self.rot = r.uniform(-0.6, 0.6)
        self.two_hands = (not static) and r.random() < 0.4
        self.symmetric = r.random() < 0.6
        self.brow = r.random() < 0.15  # подъём бровей (вопрос)
        self.head_shake = r.random() < 0.08


def render(spec: GestureSpec, rng: np.random.Generator, n_frames: int | None = None, left_handed: bool = False,
           signer: dict | None = None) -> np.ndarray:
    """Последовательность (T, K, 3) в нормированных координатах изображения (0..1)."""
    T = n_frames or int(rng.integers(30, 70))
    sg = signer or {}
    cx, cy = sg.get("cx", 0.5), sg.get("cy", 0.55)
    sw = sg.get("shoulder", 0.22)
    t = np.linspace(0, 1, T)
    env = np.sin(np.pi * t) ** 0.7  # подъём руки → жест → опускание
    seq = np.full((T, K, 3), np.nan, np.float32)
    side_dom = -1 if left_handed else 1
    for i in range(T):
        pose = np.zeros((9, 3), np.float32)
        pose[0] = [cx, cy - 1.15 * sw, 0]
        pose[1], pose[2] = [cx + sw / 2, cy - 0.5 * sw, 0], [cx - sw / 2, cy - 0.5 * sw, 0]
        pose[7], pose[8] = [cx + sw * 0.4, cy + 1.3 * sw, 0], [cx - sw * 0.4, cy + 1.3 * sw, 0]
        hands = {}
        for side, active in ((side_dom, True), (-side_dom, spec.two_hands)):
            rest = np.array([cx - side * 0.35 * sw, cy + 1.4 * sw, 0.0])
            sx = side if (side == side_dom or spec.symmetric) else -side
            path = spec.amp * np.sin(2 * np.pi * spec.freq * t[i] + spec.phase)
            target = np.array([cx - side * 0.1 * sw - sx * (spec.loc[0] + path[0]) * sw,
                               cy + (spec.loc[1] + path[1]) * sw, -0.05])
            if not active:
                target = rest
            wrist = rest + (target - rest) * env[i]
            c = spec.curl0 + (spec.curl1 - spec.curl0) * t[i]
            hands[side] = hand_shape(c * env[i] + 0.2 * (1 - env[i]), spec.spread, wrist, spec.rot, 0.33 * sw, side)
        # MediaPipe: «левая рука» — левая для человека (в кадре справа, x больше).
        lh, rh = hands[-1], hands[1]
        pose[5], pose[6] = lh[0], rh[0]
        pose[3] = (pose[1] + lh[0]) / 2 + [0.02, 0.03, 0]
        pose[4] = (pose[2] + rh[0]) / 2 + [-0.02, 0.03, 0]
        face = np.zeros((18, 3), np.float32)
        hx = pose[0, 0] + (0.03 * np.sin(6 * np.pi * t[i]) if spec.head_shake else 0)
        hy = pose[0, 1]
        brow = -0.03 * sw / 0.22 if spec.brow else 0.0
        fx = np.array([-.09, -.06, -.02, .02, .06, .09, -.08, -.03, .03, .08, -.05, .05, 0, 0, 0, 0, 0, 0]) * sw / 0.22
        fy = np.array([-.09, -.1, -.09, -.09, -.1, -.09, -.06, -.06, -.06, -.06, .06, .06, .05, .08, .055, .065, 0, .13]) * sw / 0.22
        fy[:6] += brow
        face[:, 0], face[:, 1] = hx + fx, hy + fy
        seq[i, SL_POSE] = pose
        seq[i, SL_LHAND] = lh
        seq[i, SL_RHAND] = rh
        seq[i, SL_FACE] = face
    seq += rng.normal(0, 0.003, seq.shape).astype(np.float32)
    # Трекер теряет руки у края кадра / при перекрытии.
    for sl in (SL_LHAND, SL_RHAND):
        if rng.random() < 0.3:
            a = int(rng.integers(0, T))
            seq[a : a + int(rng.integers(1, 6)), sl] = np.nan
    return seq


GROUPS = {
    "skin_tone": ["I-II", "III-IV", "V-VI"],
    "lighting": ["хорошее", "тусклое", "контровое"],
    "device": ["смартфон", "веб-камера", "планшет"],
    "age": ["до 30", "30–55", "55+"],
}


def make_signers(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        out.append({
            "signer_id": f"S{k:03d}",
            "cx": rng.uniform(0.4, 0.6), "cy": rng.uniform(0.5, 0.62), "shoulder": rng.uniform(0.16, 0.3),
            "left_handed": bool(rng.random() < 0.12), "speed": rng.uniform(0.75, 1.3),
            **{g: str(rng.choice(v)) for g, v in GROUPS.items()},
        })
    return out


def generate_dataset(labels: list[str], static_mask: list[bool], signers: list[dict], reps: int = 3, seed: int = 0):
    """Возвращает список (seq, label_idx, signer_meta)."""
    rng = np.random.default_rng(seed)
    specs = [GestureSpec(i, static=s) for i, s in enumerate(static_mask)]
    data = []
    for sg in signers:
        # Условия съёмки ухудшают трекинг (имитация смещения, которое надо измерять).
        jitter = {"хорошее": 0.0, "тусклое": 0.006, "контровое": 0.01}[sg["lighting"]]
        for ci, spec in enumerate(specs):
            for _ in range(reps):
                T = int(rng.integers(30, 70) / sg["speed"])
                seq = render(spec, rng, T, sg["left_handed"], sg)
                seq += rng.normal(0, jitter, seq.shape).astype(np.float32)
                data.append((seq, ci, sg))
    return data
