"""Классификатор изолированных жестов по последовательности ключевых точек.

Небольшой Transformer-энкодер (≈ 0.5 млн параметров): проекция признаков кадра →
позиционные эмбеддинги → 3 слоя self-attention → усреднение по времени → классы.
Модель экспортируется в ONNX и работает через ONNX Runtime (Python, Raspberry Pi,
браузер через onnxruntime-web). Для LiteRT/ExecuTorch см. scripts/export.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEAT_DIM, T_FIXED


def build_model(n_classes: int, d_model: int = 128, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.2):
    import torch
    import torch.nn as nn

    class SignTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.LayerNorm(FEAT_DIM), nn.Linear(FEAT_DIM, d_model), nn.GELU(), nn.Dropout(dropout))
            self.pos = nn.Parameter(torch.zeros(1, T_FIXED, d_model))
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

        def forward(self, x):  # x: (B, T, F)
            h = self.enc(self.inp(x) + self.pos)
            return self.head(h.mean(dim=1))

    return SignTransformer()


class OnnxClassifier:
    """Инференс без PyTorch: onnxruntime + метки классов."""

    def __init__(self, model_path: str | Path, labels_path: str | Path | None = None):
        import onnxruntime as ort

        model_path = Path(model_path)
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        lp = Path(labels_path) if labels_path else model_path.with_suffix(".labels.json")
        self.metadata = json.loads(lp.read_text(encoding="utf-8"))
        self.labels = self.metadata["labels"]

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        """feats: (T, F) или (B, T, F) → вероятности (B, C)."""
        x = feats[None] if feats.ndim == 2 else feats
        if x.ndim != 3 or x.shape[1:] != (T_FIXED, FEAT_DIM) or not np.isfinite(x).all():
            raise ValueError('Features must be finite (B, T_FIXED, FEAT_DIM)')
        logits = self.sess.run(None, {self.input: x.astype(np.float32)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def top_k(self, feats: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        p = self.predict_proba(feats)[0]
        idx = np.argsort(-p)[:k]
        return [(self.labels[i], float(p[i])) for i in idx]

    @property
    def translation_allowed(self):
        source = str(self.metadata.get('trained_on','')).lower()
        return bool(source and not any(x in source for x in ('синтет','synthetic','demo')) and
                    self.metadata.get('real_data_verified') is True and self.metadata.get('evaluation',{}).get('signer_disjoint') is True)
