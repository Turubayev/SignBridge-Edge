"""Выделение отдельных жестов в потоке кадров по движению рук.

Жест начинается, когда кисть поднимается над линией бёдер и/или движется, и заканчивается
после паузы (руки опущены или неподвижны ~0.4 с). Готовый фрагмент отправляется в классификатор.
Немануальные маркеры (подъём бровей, покачивание головой) накапливаются за всё высказывание.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .landmarks import P_LHIP, P_LSH, P_NOSE, P_RHIP, P_RSH, SL_FACE, SL_LHAND, SL_POSE, SL_RHAND


class SignSegmenter:
    def __init__(self, fps: float = 25.0, min_frames: int = 10, max_frames: int = 90, pause_s: float = 0.4,
                 sentence_pause_s: float = 2.0):
        self.min_frames, self.max_frames = min_frames, max_frames
        self.pause_frames = int(pause_s * fps)
        self.sentence_frames = int(sentence_pause_s * fps)
        self.buf: list[np.ndarray] = []
        self.idle = 0
        self.since_last_sign = 0
        self.brow_hist: deque = deque(maxlen=int(fps * 10))
        self.nose_x: deque = deque(maxlen=int(fps * 2))

    @staticmethod
    def _active(f: np.ndarray) -> bool:
        pose = f[SL_POSE]
        hip_y = np.nanmean([pose[P_LHIP, 1], pose[P_RHIP, 1]])
        sh_y = np.nanmean([pose[P_LSH, 1], pose[P_RSH, 1]])
        if not np.isfinite(hip_y):
            hip_y = sh_y + 0.3 if np.isfinite(sh_y) else 0.9
        for sl in (SL_LHAND, SL_RHAND):
            wrist = f[sl][0]
            if np.isfinite(wrist[1]) and wrist[1] < hip_y - 0.05:
                return True
        return False

    def push(self, frame_lms: np.ndarray) -> dict | None:
        """Добавляет кадр (K, 3). Возвращает {"segment": (T, K, 3)} при завершении жеста,
        {"end_of_sentence": True} после длинной паузы, иначе None."""
        face = frame_lms[SL_FACE]
        sh = frame_lms[SL_POSE][[P_LSH, P_RSH]]
        width = np.linalg.norm(sh[0, :2] - sh[1, :2]) if np.isfinite(sh).all() else np.nan
        if np.isfinite(face[:6, 1]).all() and np.isfinite(face[6:10, 1]).all() and np.isfinite(width) and width > 0:
            self.brow_hist.append((face[6:10, 1].mean() - face[:6, 1].mean()) / width)  # бровь–глаз
        self.nose_x.append(frame_lms[SL_POSE][P_NOSE, 0])

        if self._active(frame_lms):
            self.buf.append(frame_lms)
            self.idle = 0
            self.since_last_sign = 0
            if len(self.buf) >= self.max_frames:
                return self._flush()
            return None
        self.since_last_sign += 1
        if self.buf:
            self.idle += 1
            self.buf.append(frame_lms)
            if self.idle >= self.pause_frames:
                return self._flush()
        elif self.since_last_sign == self.sentence_frames:
            return {"end_of_sentence": True}
        return None

    def _flush(self):
        seg = np.stack(self.buf[: len(self.buf) - self.idle]) if len(self.buf) > self.idle else None
        self.buf, self.idle = [], 0
        if seg is None or len(seg) < self.min_frames:
            return None
        return {"segment": seg}

    def nonmanual(self) -> dict:
        """Подъём бровей (вопрос) и покачивание головой (отрицание) за последние секунды."""
        b = np.array(self.brow_hist, float)
        question = bool(len(b) > 20 and np.nanpercentile(b, 90) - np.nanmedian(b) > 0.06)
        x = np.array(self.nose_x, float)
        x = x[np.isfinite(x)]
        sign_changes = int(np.sum(np.diff(np.sign(np.diff(x))) != 0)) if len(x) > 5 else 0
        negation = bool(len(x) > 10 and np.ptp(x) > 0.03 and sign_changes >= 4)
        return {"question": question, "negation": negation}
