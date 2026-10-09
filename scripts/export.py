"""Экспорт для мобильных устройств и замер задержки.

ONNX создаётся при обучении (scripts/train.py). Здесь — дополнительные форматы:

    python scripts/export.py --onnx models/krsl_isolated.onnx --benchmark          # задержка ONNX Runtime
    python scripts/export.py --onnx models/krsl_isolated.onnx --tflite             # LiteRT (через onnx2tf)
    python scripts/export.py --onnx models/krsl_isolated.onnx --int8               # динамическое квантование ONNX

* LiteRT (бывший TensorFlow Lite): pip install onnx2tf tensorflow → models/krsl_isolated.tflite
* ExecuTorch: экспорт из PyTorch-чекпойнта, см. docs/EXPORT.md.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signbridge.features import FEAT_DIM, T_FIXED  # noqa: E402


def bench(path: Path, n: int = 200) -> float:
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(1, T_FIXED, FEAT_DIM)).astype(np.float32)
    name = s.get_inputs()[0].name
    for _ in range(10):
        s.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(n):
        s.run(None, {name: x})
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/krsl_isolated.onnx")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--tflite", action="store_true")
    a = ap.parse_args()
    src = Path(a.onnx)
    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        dst = src.with_name(src.stem + "_int8.onnx")
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        print(f"INT8: {dst} ({dst.stat().st_size / 1024:.0f} КБ, исходная {src.stat().st_size / 1024:.0f} КБ)")
        if a.benchmark:
            print(f"Задержка INT8: {bench(dst):.2f} мс")
    if a.tflite:
        out_dir = src.with_suffix("")
        cmd = [sys.executable, "-m", "onnx2tf", "-i", str(src), "-o", str(out_dir), "-osd"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"LiteRT-модели в {out_dir}/ (float32 и float16)")
    if a.benchmark:
        print(f"Задержка ONNX Runtime (CPU, 1 поток вызова): {bench(src):.2f} мс на окно {T_FIXED} кадров")


if __name__ == "__main__":
    main()
