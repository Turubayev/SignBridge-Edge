"""Ключевые точки тела, рук и лица (MediaPipe Tasks — Holistic Landmarker).

На модель передаются только координаты, а не изображение: это снижает нагрузку
и повышает приватность — кадр видео не сохраняется и не покидает устройство.

Набор точек (K = 69):
* поза (9): нос, плечи, локти, запястья, бёдра;
* левая и правая кисть (2 × 21);
* лицо (18): брови, глаза, губы, кончик носа, подбородок — немануальные компоненты
  (вопрос, отрицание и т. п. выражаются мимикой и движением головы).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
FACE_IDX = [70, 105, 107, 336, 334, 300,  # брови
            33, 133, 362, 263,  # глаза
            61, 291, 0, 17, 13, 14,  # губы
            1, 152]  # нос, подбородок
N_POSE, N_HAND, N_FACE = len(POSE_IDX), 21, len(FACE_IDX)
K = N_POSE + 2 * N_HAND + N_FACE
SL_POSE = slice(0, N_POSE)
SL_LHAND = slice(N_POSE, N_POSE + N_HAND)
SL_RHAND = slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND)
SL_FACE = slice(N_POSE + 2 * N_HAND, K)
# Индексы внутри POSE_IDX
P_NOSE, P_LSH, P_RSH, P_LEL, P_REL, P_LWR, P_RWR, P_LHIP, P_RHIP = range(9)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")


def _to_array(lms, idx=None, n=None) -> np.ndarray:
    if lms and isinstance(lms[0], (list, tuple)):  # в части версий результат вложен по людям
        lms = lms[0]
    if not lms:
        return np.full((n if idx is None else len(idx), 3), np.nan, np.float32)
    pts = lms if idx is None else [lms[i] for i in idx]
    return np.array([[p.x, p.y, p.z] for p in pts], np.float32)


class HolisticExtractor:
    """Обёртка над MediaPipe HolisticLandmarker в режиме VIDEO (кадры с метками времени)."""

    def __init__(self, model_path: str | Path = "models/holistic_landmarker.task"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Нет {model_path}. Скачайте: python scripts/download_models.py")
        opts = vision.HolisticLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                                running_mode=vision.RunningMode.VIDEO)
        self._mp = mp
        self.landmarker = vision.HolisticLandmarker.create_from_options(opts)

    def __call__(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Кадр RGB (H, W, 3) → массив (K, 3); отсутствующие точки — NaN."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        r = self.landmarker.detect_for_video(img, int(timestamp_ms))
        pose = _to_array(r.pose_landmarks, POSE_IDX)
        lh = _to_array(r.left_hand_landmarks, n=N_HAND)
        rh = _to_array(r.right_hand_landmarks, n=N_HAND)
        face = _to_array(r.face_landmarks, FACE_IDX)
        return np.concatenate([pose, lh, rh, face])

    def close(self):
        self.landmarker.close()


def extract_video(path: str | Path, extractor: HolisticExtractor, max_frames: int = 300) -> np.ndarray:
    """Видео-файл → (T, K, 3)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, t = [], 0
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(extractor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(t * 1000 / fps)))
        t += 1
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, K, 3), np.float32)
