"""Обучение классификатора изолированных жестов и экспорт в ONNX.

    # демо на процедурной синтетике (проверка конвейера, НЕ реальный КРЖЯ)
    python scripts/train.py --synthetic --out models/krsl_isolated.onnx
    # реальные записи (scripts/record_samples.py, scripts/prepare_krsl.py)
    python scripts/train.py --data data/recordings --out models/krsl_isolated.onnx

Проверка — по исполнителям, которых не было в обучении. Результаты: top-1/top-5,
задержка инференса, точность по группам (тон кожи, освещение, устройство, возраст, рука)
→ results/metrics.json и docs/MODEL_CARD.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from signbridge.dataset import group_report, load_dir, signer_split  # noqa: E402
from signbridge.features import FEAT_DIM, T_FIXED, augment, normalize  # noqa: E402
from signbridge.gloss2text import load_vocab  # noqa: E402


def featurize(data, rng=None, n_aug: int = 0):
    X, y, metas = [], [], []
    for seq, lab, meta in data:
        X.append(normalize(seq))
        y.append(lab)
        metas.append(meta)
        for _ in range(n_aug):
            X.append(normalize(augment(seq, rng)))
            y.append(lab)
            metas.append(meta)
    return np.stack(X), np.array(y), metas


def synthetic_data(n_signers: int, reps: int, seed: int = 0):
    from signbridge.synthetic import generate_dataset, make_signers

    g, d = load_vocab()
    labels = list(g) + list(d)
    static = [False] * len(g) + [True] * len(d)
    data = generate_dataset(labels, static, make_signers(n_signers, seed), reps=reps, seed=seed)
    return data, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/recordings")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--signers", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--aug", type=int, default=1, help="число аугментированных копий на пример")
    ap.add_argument("--out", default="models/krsl_isolated.onnx")
    a = ap.parse_args()

    import torch

    from signbridge.model import build_model

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    if a.synthetic:
        data, labels = synthetic_data(a.signers, a.reps)
        source = "синтетика (процедурные жесты, не КРЖЯ)"
    else:
        data, labels = load_dir(a.data)
        source = f"записи {a.data}"
    tr, va = signer_split(data)
    print(f"Классов: {len(labels)}; примеров: обучение {len(tr)}, проверка {len(va)} "
          f"(исполнителей в проверке: {len({m['signer_id'] for _, _, m in va})})", flush=True)
    Xtr, ytr, _ = featurize(tr, rng, a.aug)
    Xva, yva, mva = featurize(va)

    model = build_model(len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * ((len(Xtr) + 63) // 64))
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = perm[i : i + 64]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xva)).numpy()
        acc = float(np.mean(logits.argmax(1) == yva))
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"эпоха {ep + 1}/{a.epochs}: loss {loss.item():.3f}, top-1 (новые исполнители) {acc:.3f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xva)).numpy()
    pred = logits.argmax(1)
    top5 = np.mean([yva[i] in np.argsort(-logits[i])[:5] for i in range(len(yva))])

    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out.with_suffix(".pt"))  # для ExecuTorch / дообучения
    torch.onnx.export(model, torch.zeros(1, T_FIXED, FEAT_DIM), str(out), input_names=["features"], output_names=["logits"],
                      dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    g, d = load_vocab()
    lab_info = [{"id": lb, "ru": g[lb]["ru"] if lb in g else d[lb]["char"], "kk": g[lb]["kk"] if lb in g else d[lb]["char"],
                 "dactyl": lb in d} for lb in labels]
    out.with_suffix(".labels.json").write_text(json.dumps({"labels": labels, "info": lab_info, "T": T_FIXED, "feat_dim": FEAT_DIM,
                                                           "trained_on": source}, ensure_ascii=False, indent=1), encoding="utf-8")

    # Задержка ONNX Runtime на CPU (одно окно).
    from signbridge.model import OnnxClassifier

    clf = OnnxClassifier(out)
    x1 = Xva[:1]
    for _ in range(5):
        clf.predict_proba(x1)
    t0 = time.perf_counter()
    for _ in range(50):
        clf.predict_proba(x1)
    latency = (time.perf_counter() - t0) / 50 * 1000

    metrics = {"trained_on": source, "n_classes": len(labels), "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
               "top1": round(float(np.mean(pred == yva)), 4), "top5": round(float(top5), 4),
               "onnx_latency_ms_cpu": round(latency, 2), "model_size_kb": round(out.stat().st_size / 1024, 1),
               "fairness": group_report(yva, pred, mva)}
    # Копия для браузерной версии (web/ раздаётся как статический сайт, например GitHub Pages).
    import shutil

    web_models = ROOT / "web" / "models"
    web_models.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, web_models / "krsl_isolated.onnx")
    shutil.copy(out.with_suffix(".labels.json"), web_models / "krsl_isolated.labels.json")
    shutil.copy(ROOT / "data" / "vocab" / "glosses.json", web_models / "glosses.json")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_model_card(metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "fairness"}, ensure_ascii=False, indent=1))
    for gname, r in metrics["fairness"].items():
        print(f"  {gname}: разрыв {r['gap_pp']} п.п. {'✓' if r['within_target_5pp'] else '✗ (цель ≤ 5)'}  "
              + ", ".join(f"{k}: {v['accuracy']:.2f}" for k, v in r["by_group"].items()))


def write_model_card(m: dict):
    rows = []
    for g, r in m["fairness"].items():
        for k, v in r["by_group"].items():
            rows.append(f"| {g} | {k} | {v['accuracy']:.3f} | {v['n']} |")
        rows.append(f"| **{g}: разрыв** | | **{r['gap_pp']} п.п.** {'✓' if r['within_target_5pp'] else '✗'} | |")
    card = f"""# Model Card: krsl_isolated

**Назначение.** Распознавание изолированных жестов КРЖЯ и букв казахского дактиля по ключевым точкам
(MediaPipe Holistic) для коротких бытовых диалогов. Не заменяет сурдопереводчика.

**Данные обучения:** {m['trained_on']}.

> ⚠️ Если выше указана синтетика — модель демонстрирует работу конвейера и **не распознаёт реальные жесты**.
> Реальная модель обучается на K-RSL и записях носителей КРЖЯ с их информированного согласия.

## Качество (исполнители, не встречавшиеся в обучении)
| Метрика | Значение |
|---|---|
| Классов | {m['n_classes']} |
| Top-1 | {m['top1']:.3f} |
| Top-5 | {m['top5']:.3f} |
| Задержка ONNX Runtime (CPU, одно окно 48 кадров) | {m['onnx_latency_ms_cpu']} мс |
| Размер модели | {m['model_size_kb']} КБ |

## Справедливость: точность по группам (цель — разрыв ≤ 5 п.п.)
| Признак | Группа | Top-1 | Примеров |
|---|---|---|---|
{chr(10).join(rows)}

## Ограничения
* Словарь ограничен бытовыми темами MVP (60 глоссов + 42 буквы дактиля).
* Непрерывная речь не распознаётся; жесты разделяются паузами.
* Качество снижается при плохом освещении и перекрытии рук.
"""
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "MODEL_CARD.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
