import base64
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import tempfile
from typing import Any
import webbrowser

import numpy as np
from numpy.typing import NDArray

from tp_5.config.validator import (
    MAX_CODE_LEN,
    PROJECT_ROOT,
    CodeDataset,
    DatasetBundle,
    build_config_object,
)
from tp_5.engine.models.autoencoder import Autoencoder

COMPILER_PATH = PROJECT_ROOT / "code_dataset" / "Flex-Bison-Compiler"
NUM_REGEX = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
HEX_COLOR_REGEX = re.compile(r"#[0-9a-fA-F]{6}")


class FractalCompilerService:
    """Invokes `code_dataset/Flex-Bison-Compiler` on fractal DSL code and returns the rendered BMP image."""

    def __init__(self, compiler_path: Path = COMPILER_PATH) -> None:
        self.compiler_path = compiler_path

    @staticmethod
    def _sanitize_dsl(code: str) -> str:
        """Normalizes fixed-width Mandelbrot formatting (unary '+' and zero-padded max) for Flex/Bison."""
        code = re.sub(r"([\[,\s])\+(?=\d)", r"\1", code)
        return re.sub(r"(max:\s*)0+(\d+)", r"\1\2", code)

    @staticmethod
    def _cap_preview_resolution(code: str, max_w: int = 640, max_h: int = 360) -> str:
        """Sets a fast real-time preview resolution and caps excessive IFS points for interactive dragging."""
        lines = [ln for ln in code.splitlines() if not ln.strip().startswith("size:")]
        capped_code = f"size: {max_w} {max_h}\n" + "\n".join(lines)
        return re.sub(
            r"(points:\s*)(\d+)",
            lambda m: f"{m.group(1)}{min(int(m.group(2)), 60000)}",
            capped_code,
        )

    @staticmethod
    def _to_wsl_path(win_path: Path) -> str:
        resolved = win_path.resolve()
        drive = resolved.drive.rstrip(":").lower()
        tail = resolved.as_posix().split(":", 1)[-1]
        return f"/mnt/{drive}{tail}"

    def compile_code(self, code: str, fast_preview: bool = True) -> dict[str, Any]:
        if not code.strip():
            return {"ok": False, "error": "Empty code buffer.", "image_b64": None}

        if not self.compiler_path.exists():
            return {
                "ok": False,
                "error": f"Compiler binary not found at: {self.compiler_path}",
                "image_b64": None,
            }

        sanitized = self._sanitize_dsl(code)
        source = self._cap_preview_resolution(sanitized) if fast_preview else sanitized

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_bmp = tmp_path / "preview.bmp"

            if os.name == "nt" and not self.compiler_path.suffix.endswith(".exe"):
                cmd = ["wsl", self._to_wsl_path(self.compiler_path), "preview.bmp"]
            else:
                cmd = [str(self.compiler_path), str(out_bmp)]

            candidates = [source]
            if re.search(r"\bescape:\s*0\b", source):
                candidates.append(re.sub(r"\bescape:\s*0\b", "escape: [0, 0]", source))

            proc = None
            for candidate in candidates:
                try:
                    proc = subprocess.run(
                        cmd,
                        input=candidate.encode("utf-8"),
                        cwd=str(tmp_path),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=8.0,
                    )
                except Exception as exc:
                    return {"ok": False, "error": str(exc), "image_b64": None}

                # Locate preview.bmp or any generated image in tmp_path
                bmp_files = [out_bmp] if out_bmp.exists() else list(tmp_path.glob("*.bmp"))
                if bmp_files and bmp_files[0].stat().st_size > 0:
                    raw_bytes = bmp_files[0].read_bytes()
                    b64 = base64.b64encode(raw_bytes).decode("ascii")
                    return {
                        "ok": True,
                        "error": None,
                        "image_b64": f"data:image/bmp;base64,{b64}",
                    }

            stderr_msg = proc.stderr.decode("utf-8", errors="replace").strip() if proc else ""
            stdout_msg = proc.stdout.decode("utf-8", errors="replace").strip() if proc else ""
            return {
                "ok": False,
                "error": stderr_msg or stdout_msg or f"Compiler exited with code {proc.returncode if proc else -1}",
                "image_b64": None,
            }


def blend_fractal_codes_ast(codes: list[str], weights: list[float]) -> str:
    """
    Interpolates numeric parameters and hex color palettes onto the dominant
    structural fractal template when raw ASCII character-level blending is mid-transition.
    """
    active = [(c, w) for c, w in zip(codes, weights) if abs(w) > 1e-6]
    if not active:
        return ""
    if len(active) == 1 and abs(active[0][1] - 1.0) < 1e-4:
        return active[0][0]

    # Pick structural template of highest-weighted code
    dominant_code, _ = max(active, key=lambda item: abs(item[1]))
    total_w = sum(abs(w) for _, w in active) or 1.0
    norm_weights = [w / total_w for _, w in active]

    # Blend hex colors if all active codes have 2 hex colors
    all_colors = [HEX_COLOR_REGEX.findall(c) for c, _ in active]
    blended_code = dominant_code
    if all(len(cols) >= 2 for cols in all_colors):
        new_colors = []
        for col_idx in range(2):
            rgb = [0.0, 0.0, 0.0]
            for (cols, w) in zip(all_colors, norm_weights):
                h = cols[col_idx].lstrip("#")
                for ch in range(3):
                    rgb[ch] += max(0.0, w) * int(h[ch * 2 : ch * 2 + 2], 16)
            clamped = [max(0, min(255, int(round(v)))) for v in rgb]
            new_colors.append(f"#{clamped[0]:02X}{clamped[1]:02X}{clamped[2]:02X}")

        color_iter = iter(new_colors)
        blended_code = HEX_COLOR_REGEX.sub(lambda m: next(color_iter, m.group(0)), blended_code, count=2)

    return blended_code


@dataclass
class ExplorerSession:
    """Holds the currently loaded Autoencoder/VAE model, dataset, and compiler service."""

    compiler: FractalCompilerService
    model: Autoencoder | None = None
    dataset: DatasetBundle | None = None
    model_path: str = ""
    threshold: float = 0.5
    latent_coords: NDArray[np.float64] | None = None
    latent_dim: int = 2
    input_dim: int = 35
    is_code_mode: bool = False
    raw_codes: list[str] | None = None

    def discover_models(self) -> list[str]:
        candidates: list[Path] = []
        for folder in ("saved_models", "optuna_outputs", "results"):
            root = PROJECT_ROOT / folder
            if root.exists():
                candidates.extend(root.rglob("*.pkl"))
        candidates.extend(PROJECT_ROOT.glob("*.pkl"))
        seen: set[str] = set()
        rel_paths: list[str] = []
        for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
            rel = str(p.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
            if rel not in seen:
                seen.add(rel)
                rel_paths.append(rel)
        return rel_paths

    def load_checkpoint(self, path_str: str) -> dict[str, Any]:
        ckpt_path = Path(path_str)
        if not ckpt_path.is_absolute():
            ckpt_path = (PROJECT_ROOT / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model file not found: {ckpt_path}")

        with ckpt_path.open("rb") as f:
            payload = pickle.load(f)

        model: Autoencoder = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        input_dim = int(model.encoder_sizes[0])

        # Determine dataset path from checkpoint metadata or input dimension
        ds_path_meta = payload.get("dataset_path") if isinstance(payload, dict) else None
        subset_meta = payload.get("subset_size") if isinstance(payload, dict) else None

        if ds_path_meta and Path(ds_path_meta).exists():
            ds_path = Path(ds_path_meta)
        elif ds_path_meta and (PROJECT_ROOT / "code_dataset" / Path(ds_path_meta).name).exists():
            ds_path = PROJECT_ROOT / "code_dataset" / Path(ds_path_meta).name
        elif input_dim == MAX_CODE_LEN:
            ds_path = PROJECT_ROOT / "code_dataset" / "code"
            subset_meta = subset_meta or 100
        else:
            ds_path = PROJECT_ROOT / "doc" / "font.h"

        cfg = build_config_object(
            {
                "dataset_path": str(ds_path),
                "subset_size": subset_meta,
                "encoder_layers": list(model.encoder_sizes),
            }
        )
        dataset = cfg.build_dataset()

        self.model = model
        self.dataset = dataset
        self.model_path = (
            str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            if ckpt_path.is_relative_to(PROJECT_ROOT)
            else str(ckpt_path)
        )
        self.latent_coords = model.encode(dataset.inputs)
        self.latent_dim = int(self.latent_coords.shape[1])
        self.input_dim = input_dim
        self.is_code_mode = input_dim == MAX_CODE_LEN

        if self.is_code_mode:
            self.raw_codes = [CodeDataset.decode_vector(row) for row in dataset.inputs]
        else:
            self.raw_codes = None

        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        available = self.discover_models()
        if self.model is None and available:
            try:
                return self.load_checkpoint(available[0])
            except Exception:
                pass

        if self.model is None or self.dataset is None or self.latent_coords is None:
            return {
                "loaded": False,
                "available_models": available,
            }

        eval_res = self.model.evaluate_reconstruction(
            self.dataset.inputs, self.dataset.inputs, self.threshold
        )
        h, w = self.dataset.image_shape
        return {
            "loaded": True,
            "available_models": available,
            "current_model_path": self.model_path,
            "model_name": type(self.model).__name__,
            "is_code_mode": self.is_code_mode,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "image_shape": [h, w],
            "threshold": self.threshold,
            "labels": self.dataset.labels,
            "pixel_errors": eval_res["pixel_errors"].tolist(),
            "latent_coords": self.latent_coords.tolist(),
            "inputs": None if self.is_code_mode else self.dataset.inputs.reshape(-1, h, w).tolist(),
            "codes": self.raw_codes if self.is_code_mode else None,
        }

    def decode_vector(
        self,
        z_vec: list[float] | NDArray[np.float64],
        threshold: float = 0.5,
        basis_indices: list[int] | None = None,
        basis_weights: list[float] | None = None,
        syntax_assist: bool = True,
    ) -> dict[str, Any]:
        if self.model is None or self.dataset is None or self.latent_coords is None:
            return {"error": "No model loaded."}

        z_arr = np.asarray(z_vec, dtype=np.float64).ravel()
        if len(z_arr) < self.latent_dim:
            padded = np.mean(self.latent_coords, axis=0)
            padded[: len(z_arr)] = z_arr
            z_arr = padded
        elif len(z_arr) > self.latent_dim:
            z_arr = z_arr[: self.latent_dim]

        probs = self.model.decode(z_arr.reshape(1, -1))[0]
        dists = np.linalg.norm(self.latent_coords - z_arr, axis=1)
        nearest_idx = np.argsort(dists)[:4]
        nearest = [
            {
                "index": int(i),
                "label": self.dataset.labels[int(i)],
                "distance": float(dists[int(i)]),
            }
            for i in nearest_idx
        ]

        if self.is_code_mode:
            raw_decoded_code = CodeDataset.decode_vector(probs)
            compile_res = self.compiler.compile_code(raw_decoded_code)
            used_syntax_fallback = False
            compiled_code = raw_decoded_code

            if not compile_res["ok"] and syntax_assist and self.raw_codes:
                if basis_indices and basis_weights:
                    active_codes = [self.raw_codes[idx] for idx in basis_indices]
                    repaired = blend_fractal_codes_ast(active_codes, basis_weights)
                else:
                    repaired = self.raw_codes[int(nearest_idx[0])]
                fallback_res = self.compiler.compile_code(repaired)
                if fallback_res["ok"]:
                    compile_res = fallback_res
                    compiled_code = repaired
                    used_syntax_fallback = True

            return {
                "z": z_arr.tolist(),
                "is_code_mode": True,
                "raw_code": raw_decoded_code,
                "compiled_code": compiled_code,
                "used_syntax_fallback": used_syntax_fallback,
                "compiler": compile_res,
                "nearest": nearest,
            }

        h, w = self.dataset.image_shape
        probs_clipped = np.clip(probs, 0.0, 1.0)
        binary = (probs_clipped >= float(threshold)).astype(int)
        return {
            "z": z_arr.tolist(),
            "is_code_mode": False,
            "probs": probs_clipped.reshape(h, w).tolist(),
            "binary": binary.reshape(h, w).tolist(),
            "nearest": nearest,
        }


# =========================================================================
# UNIFIED WEB UI (2D Bitmap Character Mode vs. N-D Fractal Vocab Code Mode)
# =========================================================================
EXPLORER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TP5 Latent Space Explorer (Images & Fractal Code)</title>
<style>
  :root {
    --bg: #0b0f19; --panel: #161f33; --border: #2e3c56;
    --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8; --accent2: #f43f5e; --ok: #10b981;
  }
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); padding: 16px; }
  .top-bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; background: var(--panel); border: 1px solid var(--border); padding: 12px 16px; border-radius: 10px; margin-bottom: 14px; }
  .loader-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  select, input[type=text], input[type=number] { background: #090d16; color: var(--text); border: 1px solid var(--border); padding: 7px 10px; border-radius: 6px; }
  button.btn { background: var(--accent); color: #090d16; border: none; padding: 7px 14px; border-radius: 6px; font-weight: 700; cursor: pointer; }
  button.btn.secondary { background: #334155; color: var(--text); border: 1px solid var(--border); font-weight: 600; }
  button.btn:hover { opacity: 0.9; }
  .badge { background: #0284c7; padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
  .grid-2 { display: grid; grid-template-columns: 1.15fr 1fr; gap: 16px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
  .card h2 { margin: 0 0 10px 0; font-size: 1rem; color: var(--accent); display: flex; justify-content: space-between; align-items: center; }
  canvas.scatter { width: 100%; height: 480px; background: #070a12; border-radius: 8px; cursor: crosshair; border: 1px solid var(--border); display: block; }
  .gallery-code { max-height: 240px; overflow-y: auto; background: #070a12; border: 1px solid var(--border); border-radius: 8px; padding: 6px; display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 6px; }
  .code-item { padding: 6px 9px; border-radius: 6px; background: var(--panel); border: 1px solid transparent; cursor: pointer; font-size: 0.78rem; font-family: monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .code-item:hover { border-color: var(--muted); }
  .code-item.selected { border-color: var(--accent); background: #0c4a6e; font-weight: 700; }
  .dir-row { display: grid; grid-template-columns: 150px 1fr 72px auto; gap: 8px; align-items: center; background: #090d16; padding: 7px 10px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 6px; }
  input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
  .code-img-split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }
  textarea.code-box { width: 100%; height: 360px; background: #070a12; color: #38bdf8; border: 1px solid var(--border); border-radius: 8px; padding: 10px; font-family: monospace; font-size: 0.8rem; resize: vertical; }
  .fractal-Basic-view { width: 100%; height: 360px; background: #000; border: 1px solid var(--border); border-radius: 8px; display: flex; align-items: center; justify-content: center; overflow: hidden; position: relative; }
  .fractal-Basic-view img { width: 100%; height: 100%; object-fit: contain; }
  canvas.bitmap { width: 180px; aspect-ratio: 5 / 7; image-rendering: pixelated; border: 1px solid var(--border); border-radius: 6px; background: #000; margin: 0 auto; display: block; }
  .mono { font-family: monospace; font-size: 0.8rem; color: var(--muted); }
  .pill { display: inline-block; background: #090d16; border: 1px solid var(--border); padding: 3px 8px; border-radius: 5px; font-size: 0.76rem; margin: 2px; cursor: pointer; }
  .pill:hover { border-color: var(--accent); }
</style>
</head>
<body>

<div class="top-bar">
  <div>
    <strong style="font-size:1.15rem;">TP5 Interactive Latent Space Explorer</strong>
    <div class="mono" id="status-line">Select and load a trained .pkl model checkpoint</div>
  </div>
  <div class="loader-controls">
    <select id="model-select" style="min-width:260px;"></select>
    <button class="btn" onclick="loadSelectedDropdownModel()">Load Selected Model</button>
    <input type="text" id="custom-model-path" placeholder="Or type path to .pkl..." style="width:200px;">
    <button class="btn secondary" onclick="loadCustomPathModel()">Load Path</button>
    <button class="btn secondary" onclick="refreshModelList()">↻ Refresh</button>
    <span class="badge" id="mode-badge">No Model</span>
  </div>
</div>

<div class="grid-2">
  <!-- LEFT PANEL: MODE-SPECIFIC CONTROLS -->
  <div class="card">
    <!-- N-DIMENSIONAL CODE VOCAB EXPLORER (Only shown in Code Mode) -->
    <div id="panel-nd-left" style="display:none;">
      <h2>
        <span>1. Click Dataset Vocab Samples to Load as Latent Directions</span>
        <input type="text" id="filter-input" placeholder="Filter vocab..." oninput="renderGallery()" style="font-size:0.78rem; padding:4px 8px;">
      </h2>
      <div id="gallery" class="gallery-code"></div>

      <div style="display:flex; gap:6px; flex-wrap:wrap; margin:10px 0;">
        <button class="btn secondary" onclick="applyPreset('midpoint')">Midpoint (1/N ... 1/N)</button>
        <button class="btn secondary" onclick="applyPreset('sum')">Sum (1, 1 ... 1)</button>
        <button class="btn secondary" onclick="applyPreset('zero')">Zero (0 ... 0)</button>
        <button class="btn secondary" onclick="clearDirections()">Clear All</button>
      </div>

      <h2>2. Move Along Loaded Directions: z = Σ αᵢ · encode(Sampleᵢ)</h2>
      <div id="sliders-list"></div>
    </div>

    <!-- 2D LATENT PLANE EXPLORER (Only shown in 2D Bitmap Mode) -->
    <div id="panel-2d-left" style="display:none;">
      <h2>
        <span>2D Latent Space (z₁, z₂) — Click or Drag to Generate Characters</span>
        <span class="mono" style="font-weight:normal;">Red = Current z</span>
      </h2>
      <canvas id="scatter2d" class="scatter" width="680" height="480"></canvas>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px;">
        <div>
          <label class="mono">z₁:</label>
          <input type="number" id="z1-input" step="0.05" value="0" onchange="onManual2D()" style="width:100%;">
        </div>
        <div>
          <label class="mono">z₂:</label>
          <input type="number" id="z2-input" step="0.05" value="0" onchange="onManual2D()" style="width:100%;">
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT PANEL: REAL-TIME DECODED OUTPUT -->
  <div class="card">
    <h2>
      <span id="output-title">Real-Time Decoded Output</span>
      <label id="syntax-toggle-wrap" class="mono" style="font-weight:normal; display:none;">
        <input type="checkbox" id="syntax-assist" checked onchange="triggerDecode()">
        Compiler Syntax Assist on Mid-Interpolation
      </label>
    </h2>

    <!-- CODE + COMPILER VIEW (N-Dimensional Code Mode) -->
    <div id="code-output-panel" style="display:none;">
      <div class="code-img-split">
        <div>
          <div class="mono" style="margin-bottom:4px; display:flex; justify-content:space-between;">
            <span>Decoded Code (<= 1000 chars):</span>
            <button class="btn secondary" style="padding:2px 8px; font-size:0.72rem;" onclick="compileEditedCode()">Recompile Edited Text</button>
          </div>
          <textarea id="decoded-code-text" class="code-box" spellcheck="false"></textarea>
        </div>
        <div>
          <div class="mono" style="margin-bottom:4px;">Real-Time Compiler Output (.bmp):</div>
          <div class="fractal-Basic-view">
            <img id="compiler-img" alt="Compiled Fractal Preview" style="display:none;">
            <div id="compiler-status" class="mono" style="padding:14px; text-align:center;">Waiting for decode...</div>
          </div>
        </div>
      </div>
    </div>

    <!-- BITMAP VIEW (2D Character Image Mode) -->
    <div id="bitmap-output-panel" style="display:none;">
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; text-align:center; padding:20px 0;">
        <div>
          <canvas id="bmp-prob" class="bitmap"></canvas>
          <p class="mono">Continuous Probability P(x=1)</p>
        </div>
        <div>
          <canvas id="bmp-bin" class="bitmap"></canvas>
          <p class="mono">Binarized Reconstruction</p>
        </div>
      </div>
    </div>

    <hr style="border-color:var(--border); margin:12px 0;">
    <div class="mono">Latent Vector z: <span id="z-readout">[]</span></div>
    <div style="margin-top:8px;">
      <span class="mono">Nearest Dataset Samples:</span>
      <div id="nearest-list" style="margin-top:4px;"></div>
    </div>
  </div>
</div>

<script>
let STATE = null;
let selectedDirs = []; // Only used in N-D Code mode: [{index, label, z, weight}]
let probe2D = [0, 0];  // Only used in 2D Bitmap mode: [z1, z2]
let bounds = {minX: -3, maxX: 3, minY: -3, maxY: 3};
let decodeTimer = null;

async function refreshModelList() {
  const res = await fetch('/api/state');
  STATE = await res.json();
  populateModelDropdown();
  if (STATE.loaded) setupLoadedState();
}

function populateModelDropdown() {
  const sel = document.getElementById('model-select');
  sel.innerHTML = '';
  (STATE.available_models || []).forEach(p => {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p;
    if (p === STATE.current_model_path) opt.selected = true;
    sel.appendChild(opt);
  });
}

async function loadSelectedDropdownModel() {
  const path = document.getElementById('model-select').value;
  if (path) await loadModelByPath(path);
}

async function loadCustomPathModel() {
  const path = document.getElementById('custom-model-path').value.trim();
  if (path) await loadModelByPath(path);
}

async function loadModelByPath(path) {
  document.getElementById('status-line').textContent = `Loading ${path}...`;
  const res = await fetch('/api/load_model', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path})
  });
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  STATE = data;
  populateModelDropdown();
  setupLoadedState();
}

function setupLoadedState() {
  const isCode = Boolean(STATE.is_code_mode);

  document.getElementById('status-line').textContent =
    `Loaded: ${STATE.current_model_path} | Samples: ${STATE.labels.length} | Input: ${STATE.input_dim}D → Latent: ${STATE.latent_dim}D`;
  document.getElementById('mode-badge').textContent =
    isCode ? `CODE + COMPILER (${STATE.latent_dim}D)` : `BITMAP (${STATE.latent_dim}D)`;

  // Toggle Left & Right Panels strictly by mode
  document.getElementById('panel-nd-left').style.display = isCode ? 'block' : 'none';
  document.getElementById('panel-2d-left').style.display = isCode ? 'none' : 'block';
  document.getElementById('code-output-panel').style.display = isCode ? 'block' : 'none';
  document.getElementById('bitmap-output-panel').style.display = isCode ? 'none' : 'block';
  document.getElementById('syntax-toggle-wrap').style.display = isCode ? 'inline-block' : 'none';

  selectedDirs = [];

  if (isCode) {
    renderGallery();
    if (STATE.labels.length >= 2) {
      toggleDirection(0, 1.0);
      toggleDirection(1, 0.0);
    } else {
      renderSliders();
      triggerDecode();
    }
  } else {
    computeBounds();
    if (STATE.latent_coords && STATE.latent_coords.length > 0) {
      probe2D = [STATE.latent_coords[0][0] ?? 0, STATE.latent_coords[0][1] ?? 0];
    } else {
      probe2D = [0, 0];
    }
    document.getElementById('z1-input').value = probe2D[0].toFixed(3);
    document.getElementById('z2-input').value = probe2D[1].toFixed(3);
    triggerDecode();
  }
}

function computeBounds() {
  const xs = STATE.latent_coords.map(c => c[0] ?? 0);
  const ys = STATE.latent_coords.map(c => c[1] ?? 0);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const px = Math.max(0.5, (maxX - minX) * 0.15);
  const py = Math.max(0.5, (maxY - minY) * 0.15);
  bounds = {
    minX: minX - px, maxX: maxX + px,
    minY: minY - py, maxY: maxY + py
  };
}

function renderGallery() {
  if (!STATE || !STATE.loaded || !STATE.is_code_mode) return;
  const q = (document.getElementById('filter-input').value || '').toLowerCase();
  const gal = document.getElementById('gallery');
  gal.innerHTML = '';
  STATE.labels.forEach((lbl, idx) => {
    if (q && !lbl.toLowerCase().includes(q)) return;
    const item = document.createElement('div');
    const isSel = selectedDirs.some(d => d.index === idx);
    item.className = 'code-item' + (isSel ? ' selected' : '');
    item.textContent = `${idx + 1}. ${lbl}`;
    item.title = lbl;
    item.onclick = () => toggleDirection(idx);
    gal.appendChild(item);
  });
}

function toggleDirection(idx, defaultWeight = null) {
  const pos = selectedDirs.findIndex(d => d.index === idx);
  if (pos >= 0) {
    selectedDirs.splice(pos, 1);
  } else {
    const w = defaultWeight !== null ? defaultWeight : (selectedDirs.length === 0 ? 1.0 : 0.5);
    selectedDirs.push({
      index: idx,
      label: STATE.labels[idx],
      z: STATE.latent_coords[idx],
      weight: w
    });
  }
  renderGallery();
  renderSliders();
  triggerDecode();
}

function renderSliders() {
  const list = document.getElementById('sliders-list');
  if (selectedDirs.length === 0) {
    list.innerHTML = '<div class="mono">Click any vocab code sample above to add its latent vector direction.</div>';
    return;
  }
  list.innerHTML = '';
  selectedDirs.forEach((d, i) => {
    const row = document.createElement('div');
    row.className = 'dir-row';
    row.innerHTML = `
      <div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${d.label}">
        <b>${d.label}</b>
      </div>
      <input type="range" min="-0.5" max="1.5" step="0.01" value="${d.weight}"
             oninput="onSliderInput(${i}, this.value)">
      <input type="number" step="0.05" value="${d.weight.toFixed(2)}"
             onchange="onSliderNumber(${i}, this.value)">
      <div style="display:flex; gap:4px;">
        <button class="btn secondary" style="padding:4px 8px; font-size:0.74rem;" onclick="soloDir(${i})">1,0</button>
        <button class="btn secondary" style="padding:4px 8px; font-size:0.74rem;" onclick="toggleDirection(${d.index})">✕</button>
      </div>
    `;
    list.appendChild(row);
  });
}

function onSliderInput(i, val) {
  selectedDirs[i].weight = parseFloat(val) || 0;
  const numInputs = document.querySelectorAll('#sliders-list input[type=number]');
  if (numInputs[i]) numInputs[i].value = selectedDirs[i].weight.toFixed(2);
  scheduleDecode();
}

function onSliderNumber(i, val) {
  selectedDirs[i].weight = parseFloat(val) || 0;
  renderSliders();
  triggerDecode();
}

function soloDir(target) {
  selectedDirs.forEach((d, i) => { d.weight = (i === target) ? 1.0 : 0.0; });
  renderSliders();
  triggerDecode();
}

function applyPreset(mode) {
  const n = selectedDirs.length;
  if (!n) return;
  selectedDirs.forEach(d => {
    if (mode === 'midpoint') d.weight = +(1.0 / n).toFixed(4);
    else if (mode === 'sum') d.weight = 1.0;
    else d.weight = 0.0;
  });
  renderSliders();
  triggerDecode();
}

function clearDirections() {
  selectedDirs = [];
  renderGallery();
  renderSliders();
  triggerDecode();
}

function computeCurrentLatentZ() {
  if (!STATE || !STATE.loaded) return [0, 0];
  if (!STATE.is_code_mode) return probe2D;
  const dim = STATE.latent_dim;
  const z = new Array(dim).fill(0.0);
  selectedDirs.forEach(d => {
    for (let k = 0; k < dim; k++) z[k] += d.weight * d.z[k];
  });
  return z;
}

function scheduleDecode() {
  if (decodeTimer) clearTimeout(decodeTimer);
  decodeTimer = setTimeout(triggerDecode, 60);
}

async function triggerDecode() {
  if (!STATE || !STATE.loaded) return;
  const z = computeCurrentLatentZ();

  if (!STATE.is_code_mode) {
    drawScatter2D(z);
  }

  const res = await fetch('/api/decode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      z,
      basis_indices: STATE.is_code_mode ? selectedDirs.map(d => d.index) : [],
      basis_weights: STATE.is_code_mode ? selectedDirs.map(d => d.weight) : [],
      syntax_assist: document.getElementById('syntax-assist').checked
    })
  });
  const data = await res.json();
  renderDecodedOutput(data);
}

function snapToSample(idx) {
  if (!STATE || !STATE.loaded || STATE.is_code_mode) return;
  const pt = STATE.latent_coords[idx];
  if (!pt) return;
  probe2D = [pt[0] ?? 0, pt[1] ?? 0];
  document.getElementById('z1-input').value = probe2D[0].toFixed(3);
  document.getElementById('z2-input').value = probe2D[1].toFixed(3);
  triggerDecode();
}

function renderDecodedOutput(data) {
  if (data.error) return;
  document.getElementById('z-readout').textContent =
    `[${data.z.slice(0, 8).map(v => v.toFixed(3)).join(', ')}${data.z.length > 8 ? ', ...' : ''}]`;

  document.getElementById('nearest-list').innerHTML = (data.nearest || []).map(
    n => `<span class="pill" onclick="snapToSample(${n.index})" title="Click to jump to sample"><b>${n.label}</b> (d=${n.distance.toFixed(2)})</span>`
  ).join('');

  if (data.is_code_mode) {
    document.getElementById('decoded-code-text').value = data.compiled_code || data.raw_code || '';
    const img = document.getElementById('compiler-img');
    const st = document.getElementById('compiler-status');
    if (data.compiler && data.compiler.ok && data.compiler.image_b64) {
      img.src = data.compiler.image_b64;
      img.style.display = 'block';
      st.style.display = 'none';
    } else {
      img.style.display = 'none';
      st.style.display = 'block';
      st.textContent = (data.compiler && data.compiler.error) ? data.compiler.error : 'Compile error';
    }
  } else {
    drawBitmap('bmp-prob', data.probs, true);
    drawBitmap('bmp-bin', data.binary, false);
  }
}

async function compileEditedCode() {
  const code = document.getElementById('decoded-code-text').value;
  const res = await fetch('/api/compile_raw', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({code})
  });
  const comp = await res.json();
  const img = document.getElementById('compiler-img');
  const st = document.getElementById('compiler-status');
  if (comp.ok && comp.image_b64) {
    img.src = comp.image_b64;
    img.style.display = 'block';
    st.style.display = 'none';
  } else {
    img.style.display = 'none';
    st.style.display = 'block';
    st.textContent = comp.error || 'Compilation failed';
  }
}

function drawBitmap(canvasId, grid, isProb) {
  const c = document.getElementById(canvasId);
  const rows = grid.length, cols = grid[0].length;
  c.width = cols; c.height = rows;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {
    for (let col = 0; col < cols; col++) {
      const v = Math.max(0, Math.min(1, grid[r][col]));
      const i = (r * cols + col) * 4;
      const bw = isProb ? Math.round(v * 255) : (v >= 0.5 ? 245 : 15);
      img.data[i] = isProb ? Math.round(20 + v * 36) : bw;
      img.data[i + 1] = isProb ? Math.round(30 + v * 160) : bw;
      img.data[i + 2] = isProb ? Math.round(50 + v * 200) : bw;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function worldToCanvas(x, y, canvas) {
  const pad = 36;
  const w = canvas.width - 2 * pad, h = canvas.height - 2 * pad;
  return [
    pad + ((x - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * w,
    canvas.height - pad - ((y - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * h
  ];
}

function canvasToWorld(cx, cy, canvas) {
  const pad = 36;
  const w = canvas.width - 2 * pad, h = canvas.height - 2 * pad;
  return [
    bounds.minX + ((cx - pad) / w) * (bounds.maxX - bounds.minX),
    bounds.minY + ((canvas.height - pad - cy) / h) * (bounds.maxY - bounds.minY)
  ];
}

function drawScatter2D(currentZ) {
  const canvas = document.getElementById('scatter2d');
  if (!canvas || !STATE || !STATE.loaded || STATE.is_code_mode) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw zero axes if within bounds
  const [ox, oy] = worldToCanvas(0, 0, canvas);
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(canvas.width, oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, canvas.height); ctx.stroke();

  // Draw axis labels
  ctx.fillStyle = '#64748b';
  ctx.font = '12px monospace';
  ctx.fillText('z₁', canvas.width - 24, Math.min( Math.max(oy - 6, 16), canvas.height - 8 ));
  ctx.fillText('z₂', Math.min( Math.max(ox + 6, 8), canvas.width - 24 ), 16);

  // Draw dataset character points and their text labels
  ctx.font = 'bold 13px monospace';
  STATE.latent_coords.forEach((pt, idx) => {
    const [cx, cy] = worldToCanvas(pt[0] ?? 0, pt[1] ?? 0, canvas);
    const lbl = STATE.labels[idx] ?? String(idx);

    // Point marker
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();

    // Character label with dark outline for readability
    ctx.strokeStyle = '#070a12';
    ctx.lineWidth = 3;
    ctx.strokeText(lbl, cx + 6, cy - 5);
    ctx.fillStyle = '#f1f5f9';
    ctx.fillText(lbl, cx + 6, cy - 5);
  });

  // Draw active probe point z
  const [px, py] = worldToCanvas(currentZ[0] ?? 0, currentZ[1] ?? 0, canvas);
  ctx.fillStyle = '#f43f5e';
  ctx.beginPath();
  ctx.arc(px, py, 6.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function bindCanvasDrag(canvasId) {
  const c = document.getElementById(canvasId);
  let dragging = false;
  const pick = (e) => {
    if (!STATE || !STATE.loaded || STATE.is_code_mode) return;
    const r = c.getBoundingClientRect();
    const cx = (e.clientX - r.left) * (c.width / r.width);
    const cy = (e.clientY - r.top) * (c.height / r.height);
    probe2D = canvasToWorld(cx, cy, c);
    document.getElementById('z1-input').value = probe2D[0].toFixed(3);
    document.getElementById('z2-input').value = probe2D[1].toFixed(3);
    scheduleDecode();
  };
  c.addEventListener('mousedown', e => { dragging = true; pick(e); });
  window.addEventListener('mousemove', e => { if (dragging) pick(e); });
  window.addEventListener('mouseup', () => { dragging = false; });
}

function onManual2D() {
  probe2D[0] = parseFloat(document.getElementById('z1-input').value) || 0;
  probe2D[1] = parseFloat(document.getElementById('z2-input').value) || 0;
  triggerDecode();
}

bindCanvasDrag('scatter2d');
refreshModelList();
</script>
</body>
</html>
"""


def _make_handler(session: ExplorerSession) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send_json(self, data: dict[str, Any]) -> None:
            raw = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/api/state":
                self._send_json(session.get_state())
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

            try:
                if self.path == "/api/load_model":
                    self._send_json(session.load_checkpoint(payload.get("path", "")))
                elif self.path == "/api/decode":
                    self._send_json(
                        session.decode_vector(
                            z_vec=payload.get("z", [0.0, 0.0]),
                            basis_indices=payload.get("basis_indices"),
                            basis_weights=payload.get("basis_weights"),
                            syntax_assist=bool(payload.get("syntax_assist", True)),
                        )
                    )
                elif self.path == "/api/compile_raw":
                    self._send_json(session.compiler.compile_code(payload.get("code", "")))
                else:
                    self.send_error(404)
            except Exception as exc:
                self._send_json({"error": str(exc)})

    return Handler


def main() -> None:
    """Launches the Latent Space Explorer with zero CLI arguments required."""
    session = ExplorerSession(compiler=FractalCompilerService())
    port = 8765
    server = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(session))
    url = f"http://127.0.0.1:{port}"
    print(f"\n[OK] TP5 Latent Space Explorer running at: {url}")
    print("     Use the top bar in the browser to load any trained .pkl model.")
    print("     Press Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
