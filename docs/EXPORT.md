# Развёртывание на устройствах

| Платформа | Формат | Как получить | Примечание |
|---|---|---|---|
| ПК, киоск, Raspberry Pi 4/5 | ONNX | `scripts/train.py` | `run_demo.py`, ONNX Runtime (aarch64-колёса есть на PyPI) |
| Браузер (ЦОН, больница, акимат) | ONNX | копируется в `web/models` | ONNX Runtime Web (WASM/WebGPU) + MediaPipe Tasks for Web |
| Android / iOS | LiteRT (`.tflite`) | `scripts/export.py --tflite` (onnx2tf) | MediaPipe Holistic Landmarker есть в Tasks для Android/iOS |
| Android / iOS | ExecuTorch (`.pte`) | см. ниже | Альтернатива LiteRT |
| Слабые устройства | ONNX INT8 | `scripts/export.py --int8` | Размер ×4 меньше; проверьте точность после квантования |

## ExecuTorch
Чекпойнт `models/krsl_isolated.pt` сохраняется `scripts/train.py` при каждом обучении.
```python
import torch
from executorch.exir import to_edge
from signbridge.model import build_model
from signbridge.features import T_FIXED, FEAT_DIM

model = build_model(n_classes).eval()
model.load_state_dict(torch.load("models/krsl_isolated.pt"))
ep = torch.export.export(model, (torch.zeros(1, T_FIXED, FEAT_DIM),))
open("models/krsl_isolated.pte", "wb").write(to_edge(ep).to_executorch().buffer)
```

## Браузерная версия
Папка `web/` — статический сайт. Для GitHub Pages: Settings → Pages → Source: ветка `main`, папка `/web`
(или скопируйте `web/` в `docs/`). Камера в браузере доступна только по HTTPS или на `localhost`:
```bash
python -m http.server 8080 --directory web
```

## Raspberry Pi
```bash
sudo apt install python3-opencv
pip install onnxruntime mediapipe numpy pillow
python run_demo.py --model models/krsl_isolated.onnx
```
