// SignBridge-Edge, браузерная версия: MediaPipe Tasks (Holistic) + ONNX Runtime Web.
// Всё выполняется на устройстве; сервер только раздаёт статические файлы.
import { FilesetResolver, HolisticLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

// --- те же константы, что в signbridge/landmarks.py и features.py ---
const POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24];
const FACE_IDX = [70, 105, 107, 336, 334, 300, 33, 133, 362, 263, 61, 291, 0, 17, 13, 14, 1, 152];
const NP = 9, NH = 21, NF = FACE_IDX.length, K = NP + 2 * NH + NF, T = 48;
const LH0 = NP, RH0 = NP + NH, F0 = NP + 2 * NH;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

const $ = id => document.getElementById(id);
let landmarker, session, labels, info, running = false;
let buf = [], idle = 0, sinceSign = 0, glosses = [], lastTs = -1;

function frameToArray(r) {
  const out = new Float32Array(K * 3).fill(NaN);
  const put = (lms, idx, base) => {
    if (!lms || !lms.length) return;
    const pts = lms[0] ?? lms;
    (idx ?? pts.map((_, i) => i)).forEach((src, j) => {
      const p = pts[src]; if (!p) return;
      out.set([p.x, p.y, p.z], (base + j) * 3);
    });
  };
  put(r.poseLandmarks, POSE_IDX, 0);
  put(r.leftHandLandmarks, null, LH0);
  put(r.rightHandLandmarks, null, RH0);
  put(r.faceLandmarks, FACE_IDX, F0);
  return out;
}

// Порт features.normalize(): интерполяция пропусков, приведение к 48 кадрам, центр/масштаб по плечам,
// локальные координаты кистей, флаги наличия рук.
function normalize(seq) {
  const n = seq.length, D = K * 3;
  const flat = seq.map(f => Float32Array.from(f));
  for (let j = 0; j < D; j++) {               // линейная интерполяция внутренних пропусков
    let prev = -1;
    for (let t = 0; t < n; t++) {
      if (!Number.isNaN(flat[t][j])) {
        if (prev >= 0 && t - prev > 1) for (let u = prev + 1; u < t; u++) flat[u][j] = flat[prev][j] + (flat[t][j] - flat[prev][j]) * (u - prev) / (t - prev);
        prev = t;
      }
    }
  }
  const rs = [];
  for (let i = 0; i < T; i++) {                // ресэмплинг
    const s = (n - 1) * i / (T - 1), i0 = Math.floor(s), i1 = Math.min(i0 + 1, n - 1), w = s - i0;
    const f = new Float32Array(D);
    for (let j = 0; j < D; j++) {
      const a = flat[i0][j], b = flat[i1][j];
      f[j] = Number.isNaN(a) ? b : Number.isNaN(b) ? a : a * (1 - w) + b * w;
    }
    rs.push(f);
  }
  const mean = arr => { const v = arr.filter(x => !Number.isNaN(x)); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; };
  const c = [0, 1, 2].map(k => mean(rs.map(f => (f[1 * 3 + k] + f[2 * 3 + k]) / 2)));
  let width = mean(rs.map(f => Math.hypot(f[3] - f[6], f[4] - f[7])));
  if (!(width > 1e-3)) width = 0.25;
  if (c.some(Number.isNaN)) { c[0] = 0.5; c[1] = 0.5; c[2] = 0; }
  const FD = D + 2 * NH * 3 + 2;
  const out = new Float32Array(T * FD);
  rs.forEach((f, t) => {
    const o = t * FD;
    for (let k = 0; k < K; k++) for (let d = 0; d < 3; d++) {
      const v = (f[k * 3 + d] - c[d]) / width; out[o + k * 3 + d] = Number.isNaN(v) ? 0 : v;
    }
    [LH0, RH0].forEach((h0, hi) => {
      const wx = f[h0 * 3], wy = f[h0 * 3 + 1], wz = f[h0 * 3 + 2];
      let size = Math.hypot(f[(h0 + 9) * 3] - wx, f[(h0 + 9) * 3 + 1] - wy);
      if (!(size > 1e-4)) size = 1;
      for (let k = 0; k < NH; k++) {
        const base = o + D + hi * NH * 3 + k * 3;
        const v = [(f[(h0 + k) * 3] - wx) / size, (f[(h0 + k) * 3 + 1] - wy) / size, (f[(h0 + k) * 3 + 2] - wz) / size];
        v.forEach((x, d) => out[base + d] = Number.isNaN(x) ? 0 : x);
      }
      out[o + D + 2 * NH * 3 + hi] = Number.isNaN(wx) ? 0 : 1;
    });
  });
  return { data: out, dims: [1, T, FD] };
}

function active(f) {
  const hipY = (f[7 * 3 + 1] + f[8 * 3 + 1]) / 2;
  const lim = Number.isNaN(hipY) ? 0.9 : hipY - 0.05;
  return [LH0, RH0].some(h0 => !Number.isNaN(f[h0 * 3 + 1]) && f[h0 * 3 + 1] < lim);
}

async function classify(seg) {
  const { data, dims } = normalize(seg);
  const t0 = performance.now();
  const res = await session.run({ features: new ort.Tensor("float32", data, dims) });
  const logits = res.logits.data, m = Math.max(...logits);
  const ex = Array.from(logits, v => Math.exp(v - m)), s = ex.reduce((a, b) => a + b, 0);
  const top = ex.map((v, i) => [i, v / s]).sort((a, b) => b[1] - a[1]).slice(0, 3);
  $("pred").textContent = top.map(([i, p]) => `${info[i].ru} ${(p * 100).toFixed(0)}%`).join(" · ") + ` (${(performance.now() - t0).toFixed(0)} мс)`;
  if (top[0][1] > 0.35) { glosses.push(top[0][0]); renderGlosses(); }
}

// Упрощённый порт правил gloss2text.RuleTranslator для браузера.
function translate(ids, lang) {
  const words = []; let name = "";
  ids.forEach(i => { if (info[i].dactyl) name += info[i].ru; else { if (name) { words.push({ name }); name = ""; } words.push(info[i]); } });
  if (name) words.push({ name });
  const has = id => words.some(w => w.id === id), noun = words.find(w => w.name || ["cson", "place", "medical", "data", "everyday"].includes(topicOf(w.id)));
  const nm = w => w?.name ? w.name[0] + w.name.slice(1).toLowerCase() : w?.[lang];
  const q = ["ГДЕ", "КОГДА", "СКОЛЬКО", "ЧТО"].some(has);
  let s;
  if (has("ИМЯ") && words.some(w => w.name)) s = lang === "ru" ? `Меня зовут ${nm(words.find(w => w.name))}` : `Менің атым ${nm(words.find(w => w.name))}`;
  else if (has("НЕ-ПОНИМАТЬ")) s = lang === "ru" ? "Я не понимаю" : "Мен түсінбеймін";
  else if (has("ГДЕ") && noun) s = lang === "ru" ? `Где ${nm(noun)}` : `${nm(noun)} қайда`;
  else if (has("БОЛЕТЬ")) { const p = words.find(w => ["ГОЛОВА", "ЖИВОТ", "ЗУБ"].includes(w.id)); s = p ? (lang === "ru" ? `У меня болит ${p.ru}` : `Менің ${p.kk} ауырады`) : (lang === "ru" ? "Я болею" : "Мен ауырып тұрмын"); }
  else if (has("НУЖНО") && noun) s = lang === "ru" ? `Мне нужно: ${nm(noun)}` : `Маған ${nm(noun)} керек`;
  else s = words.filter(w => w.id !== "Я").map(nm).join(" ");
  return s ? s[0].toUpperCase() + s.slice(1) + (q ? "?" : ".") : "";
}
const topicOf = id => (window.__topics || {})[id];

function renderGlosses() { $("glosses").textContent = glosses.map(i => info[i].ru).join(" · "); }

function endSentence() {
  if (!glosses.length) return;
  const lang = $("lang").value, text = translate(glosses, lang);
  $("text").textContent = text;
  if ($("speak").checked && "speechSynthesis" in window) {   // встроенные голоса ОС, офлайн при наличии
    const u = new SpeechSynthesisUtterance(text); u.lang = lang === "kk" ? "kk-KZ" : "ru-RU"; speechSynthesis.speak(u);
  }
  glosses = []; renderGlosses();
}

async function loop() {
  if (!running) return;
  const v = $("video"), ts = performance.now();
  if (v.readyState >= 2 && ts !== lastTs) {
    lastTs = ts;
    const r = landmarker.detectForVideo(v, ts);
    draw(r);
    const f = frameToArray(r);
    if (active(f)) { buf.push(f); idle = 0; sinceSign = 0; if (buf.length >= 90) flush(); }
    else {
      sinceSign++;
      if (buf.length) { idle++; buf.push(f); if (idle >= 10) flush(); }
      else if (sinceSign === 50) endSentence();
    }
  }
  requestAnimationFrame(loop);
}
function flush() { const seg = buf.slice(0, buf.length - idle); buf = []; idle = 0; if (seg.length >= 10) classify(seg); }

let drawer;
function draw(r) {
  const c = $("overlay"), ctx = c.getContext("2d");
  c.width = $("video").videoWidth; c.height = $("video").videoHeight;
  ctx.clearRect(0, 0, c.width, c.height);
  drawer ??= new DrawingUtils(ctx);
  [[r.leftHandLandmarks, "#12b76a"], [r.rightHandLandmarks, "#2e90fa"]].forEach(([h, col]) => (h || []).forEach(lm => {
    drawer.drawConnectors(lm, HolisticLandmarker.HAND_CONNECTIONS, { color: col, lineWidth: 3 });
    drawer.drawLandmarks(lm, { color: "#fff", radius: 2 });
  }));
}

async function init() {
  $("status").textContent = "Загрузка моделей…";
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm");
  landmarker = await HolisticLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" }, runningMode: "VIDEO" });
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  session = await ort.InferenceSession.create("models/krsl_isolated.onnx", { executionProviders: ["wasm"] });
  const lj = await (await fetch("models/krsl_isolated.labels.json")).json();
  labels = lj.labels; info = lj.info;
  try { const g = await (await fetch("models/glosses.json")).json(); window.__topics = Object.fromEntries(g.glosses.map(x => [x.id, x.topic])); } catch (e) { /* необязательно */ }
  $("demo-warning").hidden = !String(lj.trained_on || "").startsWith("синтетика");
  $("status").textContent = `Готово: ${labels.length} классов. Покажите жест, затем опустите руки; пауза ~2 с завершает фразу.`;
}

$("start").onclick = async () => {
  if (running) return;
  if (!landmarker) await init();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  } catch (e) {
    $("status").textContent = "Нет доступа к камере: разрешите его в браузере (страница должна быть открыта по HTTPS или на localhost).";
    return;
  }
  $("video").srcObject = stream; await $("video").play();
  running = true; $("start").textContent = "Камера включена"; loop();
};
window.__signbridge = { normalize, translate };  // для тестов паритета с Python
$("clear").onclick = () => { glosses = []; renderGlosses(); $("text").textContent = ""; $("pred").textContent = ""; };
