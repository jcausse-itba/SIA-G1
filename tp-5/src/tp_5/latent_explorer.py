from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
import webbrowser

import numpy as np
from numpy.typing import NDArray

from tp_5.config.loader import load_and_merge_config
from tp_5.config.parser import build_parser
from tp_5.config.validator import AutoencoderConfig, DatasetBundle, build_config_object
from tp_5.engine.models.autoencoder import Autoencoder


@dataclass
class LatentSpaceExplorer:
    """
    Domain service encapsulating 2D point decoding and N-directional latent
    vector arithmetic for any injected Autoencoder/VariationalAutoencoder.
    """

    model: Autoencoder
    dataset: DatasetBundle
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.latent_coords: NDArray[np.float64] = self.model.encode(self.dataset.inputs)
        self.latent_dim: int = int(self.latent_coords.shape[1])
        self.latent_mean: NDArray[np.float64] = np.mean(self.latent_coords, axis=0)

    def get_initial_state(self) -> dict[str, Any]:
        eval_res = self.model.evaluate_reconstruction(
            self.dataset.inputs,
            self.dataset.inputs,
            threshold=self.threshold,
        )
        h, w = self.dataset.image_shape
        return {
            "model_name": type(self.model).__name__,
            "latent_dim": self.latent_dim,
            "image_shape": [h, w],
            "threshold": self.threshold,
            "labels": self.dataset.labels,
            "inputs": self.dataset.inputs.reshape(-1, h, w).tolist(),
            "reconstructions": eval_res["reconstructed_bin"].reshape(-1, h, w).tolist(),
            "pixel_errors": eval_res["pixel_errors"].tolist(),
            "latent_coords": self.latent_coords.tolist(),
        }

    def decode_latent_vector(
        self,
        z_vector: list[float] | NDArray[np.float64],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Decodes an arbitrary D-dimensional (or 2D projected) latent vector."""
        thr = self.threshold if threshold is None else float(threshold)
        z_arr = np.asarray(z_vector, dtype=np.float64).ravel()

        if len(z_arr) < self.latent_dim:
            full_z = self.latent_mean.copy()
            full_z[: len(z_arr)] = z_arr
            z_arr = full_z
        elif len(z_arr) > self.latent_dim:
            z_arr = z_arr[: self.latent_dim]

        probs = self.model.decode(z_arr.reshape(1, -1))[0]
        binary = (probs >= thr).astype(int)

        dists = np.linalg.norm(self.latent_coords - z_arr, axis=1)
        nearest_idx = np.argsort(dists)[:4]
        nearest = [
            {"index": int(idx), "label": self.dataset.labels[int(idx)], "distance": float(dists[int(idx)])}
            for idx in nearest_idx
        ]

        h, w = self.dataset.image_shape
        return {
            "z": z_arr.tolist(),
            "probs": probs.reshape(h, w).tolist(),
            "binary": binary.reshape(h, w).tolist(),
            "nearest": nearest,
        }

    def combine_directions(
        self,
        direction_vectors: list[list[float]],
        weights: list[float],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Computes the linear combination z = sum_i (weight_i * direction_i)
        in the D-dimensional latent space and decodes the resulting image.
        """
        if not direction_vectors:
            z_combined = np.zeros(self.latent_dim, dtype=np.float64)
        else:
            dirs = np.asarray(direction_vectors, dtype=np.float64)
            w_arr = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
            z_combined = np.sum(dirs * w_arr, axis=0)

        return self.decode_latent_vector(z_combined, threshold=threshold)

    def encode_custom_pattern(self, pixels: list[float]) -> dict[str, Any]:
        """Encodes a custom flat input image into the D-dimensional latent space."""
        x = np.asarray(pixels, dtype=np.float64).reshape(1, -1)
        z = self.model.encode(x)[0]
        return {"z": z.tolist()}


# =========================================================================
# EMBEDDED SINGLE-PAGE INTERACTIVE UI
# =========================================================================
EXPLORER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Latent Space Explorer — TP5</title>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --border: #334155;
    --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8; --accent2: #f43f5e;
  }
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); padding: 18px; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 12px; }
  h1 { margin: 0; font-size: 1.35rem; }
  .badge { background: #0284c7; padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
  .tabs { display: flex; gap: 10px; margin-bottom: 16px; }
  .tab-btn { background: var(--panel); color: var(--text); border: 1px solid var(--border); padding: 10px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; }
  .tab-btn.active { background: var(--accent); color: #0f172a; border-color: var(--accent); }
  .workspace { display: none; grid-template-columns: 1.25fr 1fr; gap: 18px; }
  .workspace.active { display: grid; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .card h2 { margin-top: 0; font-size: 1.05rem; color: var(--accent); }
  canvas.scatter { width: 100%; height: 480px; background: #090d16; border-radius: 8px; cursor: crosshair; border: 1px solid var(--border); }
  .outputs-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; text-align: center; }
  canvas.bitmap { width: 100%; max-width: 220px; aspect-ratio: 5 / 7; image-rendering: pixelated; border: 1px solid var(--border); border-radius: 6px; background: #000; margin: 0 auto; display: block; }
  .controls-grid { display: grid; gap: 10px; margin-top: 14px; }
  .slider-row { display: grid; grid-template-columns: 110px 1fr 70px; align-items: center; gap: 10px; background: #0f172a; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border); }
  input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
  input[type=number] { width: 100%; background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 4px 6px; text-align: right; }
  .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(62px, 1fr)); gap: 8px; max-height: 250px; overflow-y: auto; padding: 6px; background: #090d16; border-radius: 8px; border: 1px solid var(--border); }
  .thumb { border: 2px solid transparent; border-radius: 6px; padding: 4px; background: var(--panel); cursor: pointer; text-align: center; transition: 0.15s; }
  .thumb:hover { border-color: var(--muted); }
  .thumb.selected { border-color: var(--accent); background: #0c4a6e; }
  .thumb canvas { width: 40px; height: 56px; image-rendering: pixelated; display: block; margin: 0 auto 4px; }
  .thumb span { font-size: 0.75rem; font-weight: 700; }
  .btn-bar { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
  button.action { background: #334155; color: var(--text); border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.82rem; font-weight: 600; }
  button.action:hover { background: var(--accent); color: #0f172a; }
  .dir-card { display: grid; grid-template-columns: 46px 90px 1fr 68px auto; align-items: center; gap: 10px; background: #0f172a; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 8px; }
  .dir-card canvas { width: 35px; height: 49px; image-rendering: pixelated; border: 1px solid var(--border); }
  .mono { font-family: monospace; font-size: 0.82rem; color: var(--muted); word-break: break-all; }
  .neighbors { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .neighbor-pill { background: #0f172a; border: 1px solid var(--border); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Interactive Latent Space Explorer</h1>
    <div class="mono" id="subtitle">Loading model...</div>
  </div>
  <span class="badge" id="model-badge">Autoencoder</span>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('tab2d', this)">
    1. 2D Latent Space Point Explorer (Click & Drag)
  </button>
  <button class="tab-btn" onclick="switchTab('tabNd', this)">
    2. N-Image Directional Explorer (N Sliders in D-Dim Space)
  </button>
</div>

<!-- ================= TAB 1: 2D POINT EXPLORER ================= -->
<div id="tab2d" class="workspace active">
  <div class="card">
    <h2>2D Latent Space Plane (z₁, z₂) — Click or Drag Anywhere</h2>
    <canvas id="scatter2d" class="scatter" width="700" height="480"></canvas>
    <div class="controls-grid">
      <div class="slider-row">
        <label>Coord z₁:</label>
        <input type="range" id="z1-slider" min="-5" max="5" step="0.01" value="0" oninput="on2dSliderChange()">
        <input type="number" id="z1-num" step="0.05" value="0" onchange="on2dNumberChange()">
      </div>
      <div class="slider-row">
        <label>Coord z₂:</label>
        <input type="range" id="z2-slider" min="-5" max="5" step="0.01" value="0" oninput="on2dSliderChange()">
        <input type="number" id="z2-num" step="0.05" value="0" onchange="on2dNumberChange()">
      </div>
      <div class="slider-row">
        <label>Threshold:</label>
        <input type="range" id="thr-slider" min="0.05" max="0.95" step="0.01" value="0.5" oninput="onThresholdChange(this.value)">
        <input type="number" id="thr-num" step="0.05" value="0.5" readonly>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Decoded Output at Selected Point</h2>
    <div class="outputs-row">
      <div>
        <canvas id="out2d-prob" class="bitmap"></canvas>
        <p class="mono">Continuous Probability P(x=1)</p>
      </div>
      <div>
        <canvas id="out2d-bin" class="bitmap"></canvas>
        <p class="mono">Binarized Pattern (≥ Threshold)</p>
      </div>
    </div>
    <hr style="border-color: var(--border); margin: 16px 0;">
    <div>
      <div class="mono">Active Latent Vector z: <span id="z2d-readout"></span></div>
      <h3 style="font-size:0.9rem; margin:12px 0 6px;">Nearest Training Patterns in Latent Space:</h3>
      <div id="neighbors2d" class="neighbors"></div>
    </div>
  </div>
</div>

<!-- ================= TAB 2: N-IMAGE DIRECTIONAL EXPLORER ================= -->
<div id="tabNd" class="workspace">
  <div class="card">
    <h2>Select N Images as Latent Space Directions (Click to Add/Remove)</h2>
    <div id="gallery" class="gallery"></div>

    <div class="btn-bar">
      <button class="action" onclick="setPreset('midpoint')">Midpoint (1/N ... 1/N)</button>
      <button class="action" onclick="setPreset('sum')">Sum All (1, 1 ... 1)</button>
      <button class="action" onclick="setPreset('zero')">Reset Sliders (0 ... 0)</button>
      <button class="action" onclick="clearDirections()">Clear Loaded Images</button>
    </div>

    <h2>Active Direction Sliders: z = Σ αᵢ · encode(Imageᵢ)</h2>
    <div id="sliders-container"></div>
  </div>

  <div class="card">
    <h2>Generated Image from Linear Combination</h2>
    <div class="outputs-row">
      <div>
        <canvas id="outNd-prob" class="bitmap"></canvas>
        <p class="mono">Continuous Probability P(x=1)</p>
      </div>
      <div>
        <canvas id="outNd-bin" class="bitmap"></canvas>
        <p class="mono">Binarized Pattern (≥ Threshold)</p>
      </div>
    </div>
    <hr style="border-color: var(--border); margin: 14px 0;">
    <div class="mono" style="margin-bottom:8px;">Combined Latent Vector z: <span id="zNd-readout"></span></div>
    <h3 style="font-size:0.9rem; margin:8px 0 6px;">Nearest Training Patterns:</h3>
    <div id="neighborsNd" class="neighbors"></div>
    <h3 style="font-size:0.9rem; margin:14px 0 6px;">2D Projection of Basis Directions & Resultant Vector z:</h3>
    <canvas id="scatterNd" class="scatter" width="600" height="270" style="height:250px; cursor:default;"></canvas>
  </div>
</div>

<script>
let STATE = null;
let probe2D = [0, 0];
let isDragging2D = false;
let selectedDirections = []; // [{index, label, z, pixels, weight}]
let bounds = {minX: -3, maxX: 3, minY: -3, maxY: 3};

async function init() {
  const res = await fetch('/api/state');
  STATE = await res.json();

  document.getElementById('model-badge').textContent = `${STATE.model_name} (Latent Dim = ${STATE.latent_dim}D)`;
  document.getElementById('subtitle').textContent =
    `Dataset: ${STATE.labels.length} patterns (${STATE.image_shape[0]}x${STATE.image_shape[1]}) | Latent space: R^${STATE.latent_dim}`;

  computeBounds();
  setup2DCanvas();
  buildGallery();

  // Default: load first 2 characters (e.g. 'a' and 'b') with 0.5, 0.5 weights
  if (STATE.labels.length >= 3) {
    toggleDirection(1, 0.5);
    toggleDirection(2, 0.5);
  } else if (STATE.labels.length >= 2) {
    toggleDirection(0, 0.5);
    toggleDirection(1, 0.5);
  }

  probe2D = [STATE.latent_coords[0][0], STATE.latent_coords[0][1] || 0];
  sync2DInputs();
  update2DDecode();
}

function switchTab(tabId, btn) {
  document.querySelectorAll('.workspace').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');
  if (tabId === 'tab2d') drawScatter2D();
  else drawScatterNd();
}

function computeBounds() {
  const xs = STATE.latent_coords.map(c => c[0]);
  const ys = STATE.latent_coords.map(c => c[1] ?? 0);
  const padX = Math.max(0.5, (Math.max(...xs) - Math.min(...xs)) * 0.2);
  const padY = Math.max(0.5, (Math.max(...ys) - Math.min(...ys)) * 0.2);
  bounds.minX = Math.min(...xs) - padX;
  bounds.maxX = Math.max(...xs) + padX;
  bounds.minY = Math.min(...ys) - padY;
  bounds.maxY = Math.max(...ys) + padY;

  const s1 = document.getElementById('z1-slider');
  const s2 = document.getElementById('z2-slider');
  s1.min = bounds.minX.toFixed(2); s1.max = bounds.maxX.toFixed(2);
  s2.min = bounds.minY.toFixed(2); s2.max = bounds.maxY.toFixed(2);
}

function worldToCanvas(x, y, canvas) {
  const pad = 36;
  const w = canvas.width - 2 * pad;
  const h = canvas.height - 2 * pad;
  const cx = pad + ((x - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * w;
  const cy = canvas.height - pad - ((y - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * h;
  return [cx, cy];
}

function canvasToWorld(cx, cy, canvas) {
  const pad = 36;
  const w = canvas.width - 2 * pad;
  const h = canvas.height - 2 * pad;
  const x = bounds.minX + ((cx - pad) / w) * (bounds.maxX - bounds.minX);
  const y = bounds.minY + ((canvas.height - pad - cy) / h) * (bounds.maxY - bounds.minY);
  return [x, y];
}

function drawBitmap(canvasId, grid, isProb = false) {
  const canvas = document.getElementById(canvasId);
  const rows = grid.length;
  const cols = grid[0].length;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(cols, rows);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = Math.max(0, Math.min(1, grid[r][c]));
      const idx = (r * cols + c) * 4;
      if (isProb) {
        imgData.data[idx]     = Math.round(15 + v * 41);
        imgData.data[idx + 1] = Math.round(23 + v * 166);
        imgData.data[idx + 2] = Math.round(42 + v * 206);
        imgData.data[idx + 3] = 255;
      } else {
        const bw = v >= 0.5 ? 248 : 15;
        imgData.data[idx] = bw;
        imgData.data[idx + 1] = bw;
        imgData.data[idx + 2] = bw;
        imgData.data[idx + 3] = 255;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

// ================= TAB 1 LOGIC (2D EXPLORER) =================
function setup2DCanvas() {
  const canvas = document.getElementById('scatter2d');
  const pickPoint = (evt) => {
    const rect = canvas.getBoundingClientRect();
    const cx = (evt.clientX - rect.left) * (canvas.width / rect.width);
    const cy = (evt.clientY - rect.top) * (canvas.height / rect.height);
    probe2D = canvasToWorld(cx, cy, canvas);
    sync2DInputs();
    update2DDecode();
  };
  canvas.addEventListener('mousedown', (e) => { isDragging2D = true; pickPoint(e); });
  window.addEventListener('mousemove', (e) => { if (isDragging2D) pickPoint(e); });
  window.addEventListener('mouseup', () => { isDragging2D = false; });
}

function sync2DInputs() {
  document.getElementById('z1-slider').value = probe2D[0];
  document.getElementById('z2-slider').value = probe2D[1];
  document.getElementById('z1-num').value = probe2D[0].toFixed(3);
  document.getElementById('z2-num').value = probe2D[1].toFixed(3);
}

function on2dSliderChange() {
  probe2D[0] = parseFloat(document.getElementById('z1-slider').value);
  probe2D[1] = parseFloat(document.getElementById('z2-slider').value);
  document.getElementById('z1-num').value = probe2D[0].toFixed(3);
  document.getElementById('z2-num').value = probe2D[1].toFixed(3);
  update2DDecode();
}

function on2dNumberChange() {
  probe2D[0] = parseFloat(document.getElementById('z1-num').value) || 0;
  probe2D[1] = parseFloat(document.getElementById('z2-num').value) || 0;
  document.getElementById('z1-slider').value = probe2D[0];
  document.getElementById('z2-slider').value = probe2D[1];
  update2DDecode();
}

function onThresholdChange(val) {
  STATE.threshold = parseFloat(val);
  document.getElementById('thr-num').value = STATE.threshold.toFixed(2);
  update2DDecode();
  updateNdDecode();
}

async function update2DDecode() {
  drawScatter2D();
  const res = await fetch('/api/decode_point', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({z: probe2D, threshold: STATE.threshold})
  });
  const data = await res.json();
  drawBitmap('out2d-prob', data.probs, true);
  drawBitmap('out2d-bin', data.binary, false);
  document.getElementById('z2d-readout').textContent = `[${data.z.map(v => v.toFixed(3)).join(', ')}]`;
  document.getElementById('neighbors2d').innerHTML = data.nearest.map(
    n => `<span class="neighbor-pill"><b>${n.label}</b> (d=${n.distance.toFixed(2)})</span>`
  ).join('');
}

function drawScatter2D() {
  const canvas = document.getElementById('scatter2d');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Origin axes
  const [ox, oy] = worldToCanvas(0, 0, canvas);
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(canvas.width, oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, canvas.height); ctx.stroke();

  // Dataset points
  STATE.latent_coords.forEach((coord, i) => {
    const [cx, cy] = worldToCanvas(coord[0], coord[1] ?? 0, canvas);
    ctx.fillStyle = STATE.pixel_errors[i] <= 1 ? '#38bdf8' : '#f59e0b';
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(STATE.labels[i], cx + 7, cy - 5);
  });

  // Active probe cursor
  const [px, py] = worldToCanvas(probe2D[0], probe2D[1], canvas);
  ctx.strokeStyle = '#f43f5e';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = '#f43f5e';
  ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
}

// ================= TAB 2 LOGIC (N-DIRECTIONAL EXPLORER) =================
function buildGallery() {
  const gal = document.getElementById('gallery');
  gal.innerHTML = '';
  STATE.inputs.forEach((grid, idx) => {
    const div = document.createElement('div');
    div.className = 'thumb';
    div.id = `thumb-${idx}`;
    div.onclick = () => toggleDirection(idx);

    const c = document.createElement('canvas');
    c.id = `thumb-canvas-${idx}`;
    const lbl = document.createElement('span');
    lbl.textContent = STATE.labels[idx];

    div.appendChild(c);
    div.appendChild(lbl);
    gal.appendChild(div);
    drawBitmap(c.id, grid, false);
  });
}

function toggleDirection(idx, initialWeight = null) {
  const existingIdx = selectedDirections.findIndex(d => d.index === idx);
  if (existingIdx >= 0) {
    selectedDirections.splice(existingIdx, 1);
    document.getElementById(`thumb-${idx}`).classList.remove('selected');
  } else {
    const defaultW = initialWeight !== null ? initialWeight : (selectedDirections.length === 0 ? 1.0 : 0.5);
    selectedDirections.push({
      index: idx,
      label: STATE.labels[idx],
      z: STATE.latent_coords[idx],
      grid: STATE.inputs[idx],
      weight: defaultW
    });
    document.getElementById(`thumb-${idx}`).classList.add('selected');
  }
  renderDirectionSliders();
  updateNdDecode();
}

function renderDirectionSliders() {
  const container = document.getElementById('sliders-container');
  if (selectedDirections.length === 0) {
    container.innerHTML = '<p class="mono">Click one or more images above to add them as latent space directions.</p>';
    return;
  }
  container.innerHTML = '';
  selectedDirections.forEach((dir, i) => {
    const row = document.createElement('div');
    row.className = 'dir-card';
    row.innerHTML = `
      <canvas id="dir-canvas-${i}"></canvas>
      <div>
        <b>Dir #${i + 1}: '${dir.label}'</b>
        <div class="mono">α_${i + 1}</div>
      </div>
      <input type="range" min="-1.0" max="2.0" step="0.01" value="${dir.weight}"
             oninput="updateWeight(${i}, this.value, true)">
      <input type="number" step="0.05" value="${dir.weight.toFixed(2)}"
             onchange="updateWeight(${i}, this.value, false)">
      <div style="display:flex; gap:4px;">
        <button class="action" title="Solo this image (1, others 0)" onclick="soloDirection(${i})">Solo (1,0)</button>
        <button class="action" title="Remove" onclick="toggleDirection(${dir.index})">✕</button>
      </div>
    `;
    container.appendChild(row);
    drawBitmap(`dir-canvas-${i}`, dir.grid, false);
  });
}

function updateWeight(i, val, fromSlider) {
  selectedDirections[i].weight = parseFloat(val) || 0;
  renderDirectionSliders();
  updateNdDecode();
}

function soloDirection(targetIdx) {
  selectedDirections.forEach((d, i) => { d.weight = (i === targetIdx) ? 1.0 : 0.0; });
  renderDirectionSliders();
  updateNdDecode();
}

function setPreset(mode) {
  const n = selectedDirections.length;
  if (n === 0) return;
  selectedDirections.forEach(d => {
    if (mode === 'midpoint') d.weight = +(1.0 / n).toFixed(4);
    else if (mode === 'sum') d.weight = 1.0;
    else if (mode === 'zero') d.weight = 0.0;
  });
  renderDirectionSliders();
  updateNdDecode();
}

function clearDirections() {
  selectedDirections = [];
  document.querySelectorAll('.thumb').forEach(el => el.classList.remove('selected'));
  renderDirectionSliders();
  updateNdDecode();
}

async function updateNdDecode() {
  const res = await fetch('/api/combine_directions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      directions: selectedDirections.map(d => d.z),
      weights: selectedDirections.map(d => d.weight),
      threshold: STATE.threshold
    })
  });
  const data = await res.json();
  drawBitmap('outNd-prob', data.probs, true);
  drawBitmap('outNd-bin', data.binary, false);
  document.getElementById('zNd-readout').textContent = `[${data.z.map(v => v.toFixed(3)).join(', ')}]`;
  document.getElementById('neighborsNd').innerHTML = data.nearest.map(
    n => `<span class="neighbor-pill"><b>${n.label}</b> (d=${n.distance.toFixed(2)})</span>`
  ).join('');
  drawScatterNd(data.z);
}

function drawScatterNd(combinedZ = [0, 0]) {
  const canvas = document.getElementById('scatterNd');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const [ox, oy] = worldToCanvas(0, 0, canvas);
  ctx.strokeStyle = '#1e293b';
  ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(canvas.width, oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, canvas.height); ctx.stroke();

  // Faint dataset background points
  STATE.latent_coords.forEach((coord, i) => {
    const [cx, cy] = worldToCanvas(coord[0], coord[1] ?? 0, canvas);
    ctx.fillStyle = '#475569';
    ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.fill();
  });

  // Basis vectors from origin
  selectedDirections.forEach((dir) => {
    const [dx, dy] = worldToCanvas(dir.z[0], dir.z[1] ?? 0, canvas);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(dx, dy); ctx.stroke();
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath(); ctx.arc(dx, dy, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(`${dir.label} (α=${dir.weight.toFixed(2)})`, dx + 8, dy - 6);
  });

  // Combined resultant vector
  const [rx, ry] = worldToCanvas(combinedZ[0] ?? 0, combinedZ[1] ?? 0, canvas);
  ctx.strokeStyle = '#f43f5e';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(rx, ry); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f43f5e';
  ctx.beginPath(); ctx.arc(rx, ry, 7, 0, Math.PI * 2); ctx.fill();
}

init();
</script>
</body>
</html>
"""


def _make_handler(explorer: LatentSpaceExplorer) -> type[BaseHTTPRequestHandler]:
    class ExplorerRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass  # Keep terminal output clean during real-time slider dragging

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/state":
                self._send_json(explorer.get_initial_state())
            else:
                body = EXPLORER_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

            if self.path == "/api/decode_point":
                res = explorer.decode_latent_vector(payload.get("z", [0.0, 0.0]), payload.get("threshold"))
                self._send_json(res)
            elif self.path == "/api/combine_directions":
                res = explorer.combine_directions(
                    payload.get("directions", []),
                    payload.get("weights", []),
                    payload.get("threshold"),
                )
                self._send_json(res)
            elif self.path == "/api/encode_custom":
                res = explorer.encode_custom_pattern(payload.get("pixels", []))
                self._send_json(res)
            else:
                self.send_error(404)

    return ExplorerRequestHandler


def build_explorer(config: AutoencoderConfig) -> LatentSpaceExplorer:
    """Wires DatasetBundle and trained/loaded Autoencoder into LatentSpaceExplorer via DI."""
    dataset = config.build_dataset()
    model = config.build_model()

    if not config.load_model_path:
        print(
            f"\n=== Training {type(model).__name__} ({config.encoder_layers}) "
            f"for {config.max_epochs} epochs before launching Explorer ==="
        )
        model.fit(
            dataset.inputs,
            dataset.inputs,
            batch_size=config.batch_size,
            epochs=config.max_epochs,
            lr=config.learning_rate,
            print_every=max(1, config.max_epochs // 5),
        )
        if config.save_model_path:
            model.save(config.save_model_path)
            print(f"[OK] Saved trained model to: {config.save_model_path}")

    return LatentSpaceExplorer(model=model, dataset=dataset, threshold=config.threshold)


def main() -> None:
    parser = build_parser()
    parser.add_argument("--port", type=int, default=8765, help="Local port for the interactive explorer UI")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    raw_config = load_and_merge_config(parser)
    port = int(raw_config.pop("port", 8765))
    no_browser = bool(raw_config.pop("no_browser", False))

    config = build_config_object(raw_config)
    explorer = build_explorer(config)

    server = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(explorer))
    url = f"http://127.0.0.1:{port}"
    print(f"\n[OK] Interactive Latent Space Explorer running at: {url}")
    print("     Press Ctrl+C in the terminal to stop the server.")

    if not no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down Latent Space Explorer.")
        server.server_close()


if __name__ == "__main__":
    main()
