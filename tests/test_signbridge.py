import json
from pathlib import Path

import numpy as np
import pytest

from signbridge.dataset import group_report, signer_split
from signbridge.features import FEAT_DIM, T_FIXED, augment, mirror, normalize
from signbridge.gloss2text import RuleTranslator, Utterance, kk_accusative, kk_dative, load_vocab
from signbridge.landmarks import K, SL_LHAND, SL_RHAND
from signbridge.segmenter import SignSegmenter
from signbridge.synthetic import GestureSpec, generate_dataset, make_signers, render

ROOT = Path(__file__).resolve().parent.parent


def test_vocab_full_kazakh_alphabet():
    d = json.loads((ROOT / "data/vocab/dactyl_kk.json").read_text(encoding="utf-8"))["letters"]
    chars = {x["char"] for x in d}
    assert len(d) == 42
    assert set("ӘҒҚҢӨҰҮҺІ") <= chars


def test_normalize_is_scale_and_position_invariant():
    rng = np.random.default_rng(0)
    seq = render(GestureSpec(3), rng, 40)
    f1 = normalize(seq)
    moved = seq.copy()
    moved[..., :2] = (moved[..., :2] - 0.5) * 0.6 + [0.6, 0.45]  # дальше от камеры и сдвинут
    moved[..., 2] *= 0.6
    f2 = normalize(moved)
    assert f1.shape == (T_FIXED, FEAT_DIM)
    assert np.allclose(f1[:, : K * 3], f2[:, : K * 3], atol=1e-3)


def test_mirror_twice_is_identity_and_swaps_hands():
    seq = render(GestureSpec(5), np.random.default_rng(1), 30)
    m = mirror(seq)
    assert np.allclose(mirror(m), seq, equal_nan=True)
    assert np.allclose(m[:, SL_LHAND, 0], 1 - seq[:, SL_RHAND, 0], equal_nan=True)


def test_augment_keeps_shape_after_normalize():
    seq = render(GestureSpec(7), np.random.default_rng(2), 50)
    a = augment(seq, np.random.default_rng(3))
    assert normalize(a).shape == (T_FIXED, FEAT_DIM)
    assert np.isfinite(normalize(a)).all()


def test_signer_split_is_disjoint():
    data = generate_dataset(["A", "B"], [False, True], make_signers(5), reps=1)
    tr, va = signer_split(data)
    assert not ({m["signer_id"] for *_, m in tr} & {m["signer_id"] for *_, m in va})


def test_group_report_gap():
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 1, 1, 1])
    metas = [{"lighting": "хорошее"}, {"lighting": "хорошее"}, {"lighting": "тусклое"}, {"lighting": "тусклое"}]
    r = group_report(y, p, metas, groups=("lighting",))
    assert r["lighting"]["gap_pp"] == 50.0 and not r["lighting"]["within_target_5pp"]


@pytest.mark.parametrize("glosses,ru,kk", [
    (["ГДЕ", "ОКНО"], "Где окно (обслуживания)?", "Терезе қайда?"),
    (["ГОЛОВА", "БОЛЕТЬ"], "У меня болит голова.", "Менің бас ауырады."),
    (["Я", "ХОТЕТЬ", "ЗАПИСАТЬСЯ", "ВРАЧ"], "Хочу записаться к врачу.", "Дәрігерге жазылғым келеді."),
    (["НУЖНО", "СПРАВКА"], "Мне нужна справка.", "Маған анықтама керек."),
    (["ПОЛУЧИТЬ", "ПАСПОРТ"], "Я хочу получить паспорт.", "Мен төлқұжатты алғым келеді."),
    (["НЕ-ПОНИМАТЬ", "ПОВТОРИТЬ"], "Я не понимаю, повторите, пожалуйста, медленнее.", "Мен түсінбеймін, баяу қайталаңызшы."),
])
def test_rule_translator(glosses, ru, kk):
    out = RuleTranslator().translate(glosses)
    assert out["ru"] == ru
    assert out["kk"] == kk


def test_dactyl_name_and_question_marker():
    _, d = load_vocab()
    by_char = {v["char"]: k for k, v in d.items()}
    name = [by_char[c] for c in "ӘЛИЯ"]
    out = RuleTranslator().translate(Utterance(["Я", "ИМЯ", *name]))
    assert out["ru"] == "Меня зовут Әлия." and out["kk"] == "Менің атым Әлия."
    q = RuleTranslator().translate(Utterance(["ВРАЧ", "ЗАВТРА"], question=True))
    assert q["ru"].endswith("?")


def test_kazakh_suffixes():
    assert kk_dative("дәрігер") == "дәрігерге"
    assert kk_dative("банк") == "банкке"
    assert kk_dative("аурухана") == "ауруханаға"
    assert kk_accusative("анықтама") == "анықтаманы"
    assert kk_accusative("төлқұжат") == "төлқұжатты"
    assert kk_accusative("құжат") == "құжатты"


def test_segmenter_finds_sign():
    rng = np.random.default_rng(4)
    seg = SignSegmenter(fps=25)
    sign = render(GestureSpec(10), rng, 40)
    rest = np.repeat(sign[:1], 30, axis=0)
    rest[:, SL_LHAND, 1] = rest[:, SL_RHAND, 1] = 0.99  # руки опущены
    events = [seg.push(f) for f in np.concatenate([rest, sign, rest])]
    segs = [e for e in events if e and "segment" in e]
    assert len(segs) == 1 and 15 <= len(segs[0]["segment"]) <= 45


def test_onnx_model_if_present():
    model = ROOT / "models" / "krsl_isolated.onnx"
    if not model.exists():
        pytest.skip("модель не обучена")
    pytest.importorskip("onnxruntime")
    from signbridge.model import OnnxClassifier

    clf = OnnxClassifier(model)
    seq = render(GestureSpec(0), np.random.default_rng(5), 45)
    top = clf.top_k(normalize(seq), 3)
    assert len(top) == 3 and abs(sum(clf.predict_proba(normalize(seq))[0]) - 1) < 1e-4
