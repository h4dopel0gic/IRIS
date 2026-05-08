"""
EAR-Lens v0.3 — IRIS + EAR
Sakin.AI — Safina Ecosystem
Field Architect: Tobias Stevenson

Two instruments. One shell.

IRIS — Internal Representation and Insight System
  Tab 01: Extraction      — six-property activation extraction
  Tab 02: Constellation   — PCA geometry viewer + multi-model overlay
  Tab 03: Alignment       — base vs tuned valence comparison
  Tab 04: Drift Monitor   — constellation distance matrix

EAR  — Embedded-space Audio-induced Reflections
  Tab 05: EAR Control     — model/LoRA swap, timeline, pipeline fire
  Tab 06: Output Viewer   — frame gallery, ffmpeg stitch
"""

import gradio as gr
import json
import os
import sys
import time
import subprocess
import struct
import numpy as np
import requests
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import combinations
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COMFYUI_URL    = "http://localhost:8188"
COMFYUI_OUTPUT = r"E:\pinokio\api\comfy.git\app\output"
COMFYUI_MODELS = r"E:\pinokio\api\comfy.git\app\models\checkpoints"
COMFYUI_LORAS  = r"E:\pinokio\api\comfy.git\app\models\loras"
WORKFLOW_PATH  = "EAR_RENDER_API_v02.json"
OUTPUT_PREFIX  = "EAR_OUT"
TIMELINE_AUDIO = "timeline.json"
TIMELINE_LLM   = "timeline_llm.json"
IRIS_OUTPUT    = Path("output/activations")
SPECIAL_TOKENS = {"<|endoftext|>", "<|BOS|>", "<s>", "</s>", "<|padding|>"}

SUGGESTED_MODELS = [
    "gpt2",
    "gpt2-medium",
    "roneneldan/TinyStories-33M",
    "roneneldan/TinyStories-Instruct-33M",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-70m-deduped",
    "mistralai/Mistral-7B-v0.1",
    "EleutherAI/gpt-neo-1.3B",
    "microsoft/phi-2",
]

MODEL_VRAM_WARNINGS = {
    "gpt2":                       None,
    "gpt2-medium":                None,
    "TinyStories":                None,
    "pythia-70m":                 None,
    "gpt2-large":                 "~3GB VRAM",
    "gpt2-xl":                    "~6GB VRAM",
    "mistralai/Mistral-7B-v0.1":  "~14GB — loads in float16 automatically",
    "meta-llama/Llama-3.1-8B":    "~5GB VRAM (4-bit) — HF auth required",
    "microsoft/phi-2":            "~3GB VRAM",
    "EleutherAI/gpt-neo-1.3B":    "~3GB VRAM",
    "EleutherAI/gpt-j-6b":        "~5GB VRAM",
}

# Global model cache
_loaded_model      = None
_loaded_model_name = None

# ─────────────────────────────────────────────────────────────
# PALETTE & CSS
# ─────────────────────────────────────────────────────────────

IRIS_TEAL  = "#0D7377"
EAR_GOLD   = "#e2b96f"
VIZ_COLOURS = {
    "bg":       "#0F1117",
    "surface":  "#1C1F2B",
    "text":     "#E8E8E8",
    "subtext":  "#8A8A9A",
    "grid":     "#2A2D3E",
    "model_a":  "#4A90D9",
    "model_b":  "#E8A838",
    "positive": "#6DB87A",
    "neutral":  "#9B8AC4",
    "negative": "#C76B8A",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:        #0a0a12;
    --surface:   #12121e;
    --surface2:  #1a1a2e;
    --border:    #2a2a4a;
    --gold:      #e2b96f;
    --teal:      #0D7377;
    --teal-lt:   #14FFEC;
    --blue:      #a8d8ea;
    --green:     #6bcb77;
    --text:      #e8e8e8;
    --muted:     #6b7280;
    --accent:    #0f3460;
    --warn:      #f59e0b;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
}
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
     color: var(--gold) !important; letter-spacing: -1px !important; }
h3, h4 { color: var(--blue) !important; font-family: 'Space Mono', monospace !important; }

/* IRIS tabs — teal */
.iris-tab .tab-nav button.selected {
    background: var(--teal) !important;
    color: var(--teal-lt) !important;
    border-color: var(--teal-lt) !important;
}

.tab-nav button {
    font-family: 'Space Mono', monospace !important;
    background: var(--surface) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
.tab-nav button.selected {
    background: var(--accent) !important;
    color: var(--gold) !important;
    border-color: var(--gold) !important;
}

.gr-button-primary {
    background: var(--gold) !important;
    color: #0a0a12 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    letter-spacing: 1px !important;
}
.btn-teal {
    background: var(--teal) !important;
    color: var(--teal-lt) !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: 1px solid var(--teal-lt) !important;
}
.gr-button-secondary {
    background: var(--surface2) !important;
    color: var(--blue) !important;
    font-family: 'Space Mono', monospace !important;
    border: 1px solid var(--border) !important;
}

label { color: var(--blue) !important; font-size: 11px !important;
        letter-spacing: 1px !important; text-transform: uppercase !important; }

input, textarea, select, .gr-box {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}
.gr-panel, .gr-group {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

.status-ok   { background:var(--surface2); border:1px solid var(--border);
               border-left:3px solid var(--green); padding:8px 12px;
               font-size:11px; color:var(--green); letter-spacing:1px; margin-bottom:12px; }
.status-warn { background:var(--surface2); border:1px solid var(--border);
               border-left:3px solid var(--warn); padding:8px 12px;
               font-size:11px; color:var(--warn); letter-spacing:1px; margin:8px 0; }
.iris-header { background:linear-gradient(90deg,#0D7377,#0a0a12);
               padding:12px 20px; margin-bottom:8px; border-radius:4px; }
.ear-header  { background:linear-gradient(90deg,#3d2b00,#0a0a12);
               padding:12px 20px; margin-bottom:8px; border-radius:4px; }
.divider     { border:none; border-top:1px solid var(--border); margin:12px 0; }
"""

# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────

def scan_folder(path: str, exts: List[str]) -> List[str]:
    p = Path(path)
    if not p.exists():
        return [f"[not found: {path}]"]
    files = []
    for ext in exts:
        files.extend([f.name for f in p.glob(f"*{ext}")])
    return sorted(files) if files else ["[empty]"]

def get_models()  -> List[str]: return scan_folder(COMFYUI_MODELS, [".safetensors", ".ckpt"])
def get_loras()   -> List[str]: return ["[none]"] + scan_folder(COMFYUI_LORAS, [".safetensors", ".ckpt", ".pt"])

def comfyui_alive() -> bool:
    try:
        return requests.get(f"{COMFYUI_URL}/system_stats", timeout=3).status_code == 200
    except Exception:
        return False

def check_transformerlens() -> bool:
    try:
        import transformer_lens
        return True
    except ImportError:
        return False

def get_model_warning(model_id: str) -> str:
    if not model_id:
        return ""
    for key, warn in MODEL_VRAM_WARNINGS.items():
        if key.lower() in model_id.lower():
            return f'<div class="status-ok">✓ {warn or "Lightweight — safe to load"}</div>'
    name = model_id.lower()
    if any(x in name for x in ["70b","34b","13b"]):
        return '<div class="status-warn">⚠ Very large — likely OOM without quantization</div>'
    if any(x in name for x in ["7b","8b","6b"]):
        return '<div class="status-warn">⚠ ~14GB — loads in float16 automatically</div>'
    return '<div class="status-warn">? Unknown size — check model card</div>'

def bos_mask(tokens: list) -> list:
    return [t not in SPECIAL_TOKENS for t in tokens]

def normalise_coords(coords: np.ndarray) -> np.ndarray:
    coords = coords - coords.mean(axis=0)
    std = coords.std(axis=0)
    std[std < 1e-9] = 1.0
    return coords / std

def scan_constellations() -> List[str]:
    IRIS_OUTPUT.mkdir(parents=True, exist_ok=True)
    files = sorted(IRIS_OUTPUT.rglob("constellation_*.json"))
    return [str(f) for f in files] if files else ["[no constellations found — run extraction first]"]

def load_constellation(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# IRIS — EXTRACTION (Tab 01)
# ─────────────────────────────────────────────────────────────

def load_llm_model(model_id: str):
    global _loaded_model, _loaded_model_name
    import transformer_lens
    import torch
    if _loaded_model_name == model_id and _loaded_model is not None:
        return _loaded_model, f"Already loaded: {model_id}"
    if _loaded_model is not None:
        del _loaded_model
        _loaded_model = None
        _loaded_model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    small = {"gpt2", "gpt2-small"}
    dtype = torch.float32 if model_id in small else torch.float16
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_id, device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=dtype, fold_ln=True,
        center_writing_weights=True, center_unembed=True,
    )
    model.eval()
    _loaded_model = model
    _loaded_model_name = model_id
    return model, f"Loaded: {model_id} | layers: {model.cfg.n_layers} | d_model: {model.cfg.d_model}"

def run_iris_extraction(model_id, layer, prompt, seed, progress=gr.Progress()):
    import torch
    from sklearn.decomposition import PCA

    if not check_transformerlens():
        return "TransformerLens not installed.", ""
    if not model_id.strip():
        return "Enter a model ID.", ""

    model_id = model_id.strip()
    layer    = int(layer)
    seed     = int(seed)

    # Fix seed
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    progress(0.1, desc="Loading model...")
    try:
        model, load_msg = load_llm_model(model_id)
    except Exception as e:
        return f"Model load error:\n{e}", ""

    actual_layer = min(layer, model.cfg.n_layers - 1)
    progress(0.3, desc="Forward pass...")
    tokens     = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    resid = cache[f"blocks.{actual_layer}.hook_resid_post"].squeeze()

    progress(0.6, desc="Extracting properties...")

    # 1 — RMS
    rms = resid.norm(dim=-1).cpu().numpy()

    # 2 — Layer delta
    if actual_layer > 0:
        prev  = cache[f"blocks.{actual_layer-1}.hook_resid_post"].squeeze()
        delta = (resid - prev).norm(dim=-1).cpu().numpy()
    else:
        delta = rms.copy()

    # 3 — Attention entropy
    try:
        attn = cache[f"blocks.{actual_layer}.attn.hook_pattern"].squeeze()
        p    = attn + 1e-9
        ent  = -(p * p.log()).sum(dim=-1).mean(dim=0).cpu().numpy()
    except Exception:
        ent  = np.full(len(token_strs), float("nan"))

    # 4 — Centroid and cosine distance
    centroid = resid.mean(dim=0)
    cos_sim  = torch.nn.functional.cosine_similarity(
        resid, centroid.unsqueeze(0).expand_as(resid), dim=-1)
    cos_dist = (1 - cos_sim).cpu().numpy()

    # 5 — PCA
    resid_np = resid.cpu().numpy()
    pca      = PCA(n_components=2)
    coords   = pca.fit_transform(resid_np)
    ev       = pca.explained_variance_ratio_

    # Full layer delta profile
    delta_profile = []
    for l in range(1, model.cfg.n_layers):
        try:
            post = cache[f"blocks.{l}.hook_resid_post"].squeeze()
            prev = cache[f"blocks.{l-1}.hook_resid_post"].squeeze()
            delta_profile.append((post - prev).norm(dim=-1).cpu().numpy().tolist())
        except Exception:
            break

    progress(0.8, desc="Saving...")

    # Build token table
    token_table = []
    for i, tok in enumerate(token_strs):
        token_table.append({
            "index":        i,
            "token":        tok,
            "rms":          float(rms[i]),
            "delta_mag":    float(delta[i]),
            "attn_entropy": float(ent[i]) if not np.isnan(ent[i]) else None,
            "cos_dist":     float(cos_dist[i]),
            "pc1":          float(coords[i, 0]),
            "pc2":          float(coords[i, 1]),
        })

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_").replace(" ", "_")
    run_id     = f"{model_slug}_L{actual_layer}_seed{seed}_{timestamp}"

    IRIS_OUTPUT.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": model_id, "prompt": prompt, "layer": actual_layer,
        "n_layers": model.cfg.n_layers, "d_model": model.cfg.d_model,
        "seq_len": len(token_strs), "seed": seed,
        "pca_explained_var": ev.tolist(),
        "centroid_norm": float(np.linalg.norm(centroid.cpu().numpy())),
        "run_id": run_id, "extractor_version": "v2",
    }

    constellation = {
        "meta": meta,
        "centroid": centroid.cpu().numpy().tolist(),
        "pca_coords": coords.tolist(),
        "token_labels": token_strs,
        "delta_profile": delta_profile,
        "explained_variance": ev.tolist(),
        "tokens": token_table,
    }
    timeline = {"meta": meta, "tokens": token_table}

    c_path = IRIS_OUTPUT / f"constellation_{run_id}.json"
    t_path = IRIS_OUTPUT / f"timeline_{run_id}.json"
    with open(c_path, "w") as f: json.dump(constellation, f, indent=2)
    with open(t_path, "w") as f: json.dump(timeline,      f, indent=2)

    # LLM-compatible timeline for EAR bridge
    ear_timeline = {
        "fps": 2, "source": "llm_activations",
        "layer": actual_layer, "model": model_id,
        "frames": [{
            "i": t["index"], "token": t["token"],
            "rms": t["rms"],
            "cfg": 4.5 + (1 - (t["attn_entropy"] or 0.5)) * 4.0,
            "denoise": 0.30 + t["rms"] / max(r["rms"] for r in token_table) * 0.55,
            "seed": 112233 + t["index"] * 13, "steps": 25, "prompt": "fractal fields",
        } for t in token_table]
    }
    with open(TIMELINE_LLM, "w") as f: json.dump(ear_timeline, f, indent=2)

    # Format output
    lines = [
        f"IRIS Extraction Complete",
        f"{'─'*48}",
        f"Run ID   : {run_id}",
        f"Model    : {model_id}",
        f"Layer    : {actual_layer} / {model.cfg.n_layers - 1}",
        f"Tokens   : {len(token_strs)}",
        f"PC1 var  : {ev[0]:.1%}   PC2 var: {ev[1]:.1%}",
        f"Centroid : {np.linalg.norm(centroid.cpu().numpy()):.3f}",
        f"",
        f"{'Token':<14} {'RMS':>8} {'Delta':>8} {'Entropy':>9} {'CosDist':>9} {'PC1':>8} {'PC2':>8}",
        f"{'─'*70}",
    ]
    for t in token_table:
        ent_str = f"{t['attn_entropy']:>9.4f}" if t['attn_entropy'] is not None else "     nan "
        lines.append(
            f"{t['token']:<14} {t['rms']:>8.3f} {t['delta_mag']:>8.3f} "
            f"{ent_str} {t['cos_dist']:>9.4f} {t['pc1']:>8.3f} {t['pc2']:>8.3f}"
        )
    lines += ["", f"Saved → {c_path.name}", f"EAR timeline → {TIMELINE_LLM}"]

    progress(1.0)
    return "\n".join(lines), str(c_path)


def unload_model():
    global _loaded_model, _loaded_model_name
    import torch
    if _loaded_model is not None:
        del _loaded_model
        _loaded_model = None
        _loaded_model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return "Model unloaded from memory."


# ─────────────────────────────────────────────────────────────
# IRIS — CONSTELLATION VIEWER (Tab 02)
# ─────────────────────────────────────────────────────────────

def plot_single_constellation(path: str, normalise: bool = False) -> Optional[str]:
    data = load_constellation(path)
    if not data:
        return None

    coords = np.array(data["pca_coords"])
    tokens = data["token_labels"]
    mask   = bos_mask(tokens)
    coords = coords[mask]
    tokens = [t.replace("Ġ","").strip() or t for t, m in zip(tokens, mask) if m]

    if normalise:
        coords = normalise_coords(coords)

    rms_vals = [t["rms"] for t, m in zip(data.get("tokens",[]), mask) if m] if data.get("tokens") else None
    sizes    = None
    if rms_vals:
        r = np.array(rms_vals)
        sizes = 80 + 200 * (r - r.min()) / (r.max() - r.min() + 1e-9)

    ev     = data["meta"].get("pca_explained_var", [0, 0])
    model  = data["meta"]["model"]
    prompt = data["meta"]["prompt"]
    layer  = data["meta"]["layer"]

    fig = plt.figure(figsize=(16, 10), facecolor=VIZ_COLOURS["bg"])
    fig.suptitle(
        f"IRIS Constellation — {model}\n\"{prompt}\"  |  Layer {layer}  |  "
        f"PC1={ev[0]:.1%}  PC2={ev[1]:.1%}",
        color=VIZ_COLOURS["text"], fontsize=11, y=0.98
    )
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.96, top=0.92, bottom=0.07)

    # Constellation
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(VIZ_COLOURS["surface"])
    ax.tick_params(colors=VIZ_COLOURS["subtext"], labelsize=8)
    ax.spines[:].set_color(VIZ_COLOURS["grid"])
    ax.grid(color=VIZ_COLOURS["grid"], linewidth=0.5, alpha=0.7)
    ax.set_title("Activation Constellation — PCA Projection (BOS masked)",
                 color=VIZ_COLOURS["text"], fontsize=10)
    ax.plot(coords[:,0], coords[:,1], color=VIZ_COLOURS["model_a"],
            alpha=0.25, linewidth=1.0)
    ax.scatter(coords[:,0], coords[:,1], s=sizes if sizes is not None else 80,
               c=VIZ_COLOURS["model_a"], alpha=0.85,
               edgecolors=VIZ_COLOURS["bg"], linewidths=0.8)
    for i, tok in enumerate(tokens):
        ax.annotate(tok, (coords[i,0], coords[i,1]),
                    textcoords="offset points", xytext=(6,4),
                    fontsize=7.5, color=VIZ_COLOURS["model_a"], alpha=0.9)
    x_lbl = "PC1 (normalised)" if normalise else f"PC1 ({ev[0]:.1%} variance)"
    y_lbl = "PC2 (normalised)" if normalise else f"PC2 ({ev[1]:.1%} variance)"
    ax.set_xlabel(x_lbl, color=VIZ_COLOURS["subtext"], fontsize=8)
    ax.set_ylabel(y_lbl, color=VIZ_COLOURS["subtext"], fontsize=8)

    # Delta heatmap
    ax_h = fig.add_subplot(gs[1, 0])
    if data.get("delta_profile"):
        profile = np.array(data["delta_profile"])
        profile = profile[:, mask[:profile.shape[1]]] if profile.shape[1] > sum(mask) else profile
        im = ax_h.imshow(profile, aspect="auto", cmap="magma", interpolation="nearest")
        ax_h.set_xticks(range(len(tokens)))
        ax_h.set_xticklabels(tokens, rotation=30, ha="right",
                              fontsize=7, color=VIZ_COLOURS["subtext"])
        ax_h.set_ylabel("Layer", color=VIZ_COLOURS["subtext"], fontsize=8)
        ax_h.tick_params(colors=VIZ_COLOURS["subtext"])
        ax_h.set_title("Layer Delta — Where Meaning Crystallises",
                       color=VIZ_COLOURS["text"], fontsize=9)
        ax_h.set_facecolor(VIZ_COLOURS["surface"])
        ax_h.spines[:].set_color(VIZ_COLOURS["grid"])
        cbar = plt.colorbar(im, ax=ax_h, fraction=0.02, pad=0.02)
        cbar.set_label("\u0394 Magnitude", color=VIZ_COLOURS["subtext"], fontsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=VIZ_COLOURS["subtext"])

    # Property bars
    ax_b = fig.add_subplot(gs[1, 1])
    if data.get("tokens"):
        toks_masked = [t for t, m in zip(data["tokens"], bos_mask(
            [t["token"] for t in data["tokens"]])) if m]
        labels_b = [t["token"].replace("Ġ","").strip() or t["token"] for t in toks_masked]
        props    = ["rms", "delta_mag", "cos_dist"]
        plabels  = ["RMS", "\u0394 Mag", "Cos Dist"]
        pcolours = [VIZ_COLOURS["model_a"], VIZ_COLOURS["model_b"], VIZ_COLOURS["positive"]]
        x = np.arange(len(labels_b)); width = 0.25
        for i, (prop, plbl, c) in enumerate(zip(props, plabels, pcolours)):
            vals = np.array([t[prop] for t in toks_masked], dtype=float)
            if vals.max() > vals.min():
                vals = (vals - vals.min()) / (vals.max() - vals.min())
            ax_b.bar(x + i*width, vals, width, label=plbl, color=c, alpha=0.8)
        ax_b.set_xticks(x + width)
        ax_b.set_xticklabels(labels_b, rotation=30, ha="right",
                              fontsize=7, color=VIZ_COLOURS["subtext"])
        ax_b.legend(fontsize=7, labelcolor=VIZ_COLOURS["subtext"],
                    facecolor=VIZ_COLOURS["surface"], edgecolor=VIZ_COLOURS["grid"])
        ax_b.set_facecolor(VIZ_COLOURS["surface"])
        ax_b.tick_params(colors=VIZ_COLOURS["subtext"])
        ax_b.spines[:].set_color(VIZ_COLOURS["grid"])
        ax_b.grid(color=VIZ_COLOURS["grid"], linewidth=0.5, alpha=0.5)
        ax_b.set_title("Per-Token Properties (Normalised)",
                       color=VIZ_COLOURS["text"], fontsize=9)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight",
                facecolor=VIZ_COLOURS["bg"])
    plt.close()
    return tmp.name


def plot_comparison_constellation(path_a: str, path_b: str,
                                   label_a: str, label_b: str,
                                   normalise: bool = True) -> Optional[str]:
    da = load_constellation(path_a)
    db = load_constellation(path_b)
    if not da or not db:
        return None

    fig, ax = plt.subplots(figsize=(14, 9), facecolor=VIZ_COLOURS["bg"])
    ax.set_facecolor(VIZ_COLOURS["surface"])
    ax.tick_params(colors=VIZ_COLOURS["subtext"], labelsize=8)
    ax.spines[:].set_color(VIZ_COLOURS["grid"])
    ax.grid(color=VIZ_COLOURS["grid"], linewidth=0.5, alpha=0.5)
    mode = "Normalised — Shape" if normalise else "Raw — Position & Scale"
    ax.set_title(f"Constellation Comparison ({mode})",
                 color=VIZ_COLOURS["text"], fontsize=10)

    ev_ref = None
    for data, label, colour in [
        (da, label_a, VIZ_COLOURS["model_a"]),
        (db, label_b, VIZ_COLOURS["model_b"]),
    ]:
        coords = np.array(data["pca_coords"])
        tokens = data["token_labels"]
        mask   = bos_mask(tokens)
        coords = coords[mask]
        tokens = [t.replace("Ġ","").strip() or t for t, m in zip(tokens, mask) if m]
        if normalise:
            coords = normalise_coords(coords)
        ax.plot(coords[:,0], coords[:,1], color=colour, alpha=0.2, linewidth=0.8)
        ax.scatter(coords[:,0], coords[:,1], s=80, c=colour, alpha=0.85,
                   edgecolors=VIZ_COLOURS["bg"], linewidths=0.7, label=label)
        for i, tok in enumerate(tokens):
            ax.annotate(tok, (coords[i,0], coords[i,1]),
                        textcoords="offset points", xytext=(5,3),
                        fontsize=7, color=colour, alpha=0.9)
        ev = data["meta"].get("pca_explained_var", [0, 0])
        if ev_ref is None:
            ev_ref = ev

    x_lbl = "PC1 (normalised)" if normalise else f"PC1 ({ev_ref[0]:.1%} var, ref)"
    y_lbl = "PC2 (normalised)" if normalise else f"PC2 ({ev_ref[1]:.1%} var, ref)"
    ax.set_xlabel(x_lbl, color=VIZ_COLOURS["subtext"], fontsize=8)
    ax.set_ylabel(y_lbl, color=VIZ_COLOURS["subtext"], fontsize=8)
    ax.legend(fontsize=9, labelcolor=VIZ_COLOURS["text"],
              facecolor=VIZ_COLOURS["surface"], edgecolor=VIZ_COLOURS["grid"])

    norm_note = "Centred + unit variance · Shape preserved · Scale removed"
    fig.suptitle(
        f"IRIS — Constellation Comparison\n{norm_note if normalise else 'Same prompt · Different models · Different geometry'}",
        color=VIZ_COLOURS["text"], fontsize=11, y=0.97
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight", facecolor=VIZ_COLOURS["bg"])
    plt.close()
    return tmp.name


def viewer_render_single(path, normalise):
    if not path or path.startswith("["):
        return None, "Select a constellation file first."
    img = plot_single_constellation(path, normalise)
    return img, f"Rendered: {Path(path).name}"

def viewer_render_compare(path_a, path_b, label_a, label_b, normalise):
    if not path_a or not path_b or path_a.startswith("[") or path_b.startswith("["):
        return None, "Select two constellation files to compare."
    img = plot_comparison_constellation(path_a, path_b, label_a, label_b, normalise)
    return img, f"Comparison: {Path(path_a).name}  ↔  {Path(path_b).name}"

def refresh_constellation_list():
    files = scan_constellations()
    return gr.Dropdown(choices=files, value=files[0] if files else None)


# ─────────────────────────────────────────────────────────────
# IRIS — ALIGNMENT COMPARISON (Tab 03)
# ─────────────────────────────────────────────────────────────

VALENCE_PROMPTS = {
    "positive": [
        "This is a wonderful day full of hope and possibility",
        "The results exceeded all expectations and everyone celebrated",
        "She felt grateful and joyful walking through the garden",
        "The team succeeded brilliantly and the future looks bright",
        "Love and kindness filled the room as they reunited",
    ],
    "neutral": [
        "In the name of God, the most gracious",
        "The train arrives at the station every morning",
        "Water flows downhill toward the sea",
        "The document was placed on the table",
        "She walked across the room and opened the window",
    ],
    "negative": [
        "The situation deteriorated rapidly and all hope was lost",
        "Failure and disappointment defined the entire experience",
        "The damage was severe and recovery seemed impossible",
        "Fear and despair spread through the crowd",
        "The loss was devastating and the pain overwhelming",
    ],
}

def cosine_dist_np(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten(), b.flatten()
    return float(1 - np.dot(af,bf) / (np.linalg.norm(af)*np.linalg.norm(bf)+1e-9))

def run_alignment_comparison(
    base_model_id, tuned_model_id, layer, seed,
    progress=gr.Progress()
):
    import torch
    from sklearn.decomposition import PCA

    if not base_model_id.strip() or not tuned_model_id.strip():
        return "Enter both model IDs.", None

    layer = int(layer); seed = int(seed)
    valences = ["positive", "neutral", "negative"]
    results  = {}

    total_steps = 2 * 3 * 5 + 2
    step = 0

    for model_key, model_id in [("base", base_model_id.strip()),
                                  ("tuned", tuned_model_id.strip())]:
        results[model_key] = {}
        progress(step/total_steps, desc=f"Loading {model_key}: {model_id}...")

        try:
            model, _ = load_llm_model(model_id)
        except Exception as e:
            return f"Error loading {model_key} model:\n{e}", None

        actual_layer = min(layer, model.cfg.n_layers - 1)

        for valence in valences:
            results[model_key][valence] = []
            for prompt in VALENCE_PROMPTS[valence]:
                step += 1
                progress(step/total_steps, desc=f"{model_key} | {valence} | {prompt[:30]}...")

                import random
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens)

                resid    = cache[f"blocks.{actual_layer}.hook_resid_post"].squeeze()
                resid_np = resid.cpu().numpy()
                pca      = PCA(n_components=2)
                coords   = pca.fit_transform(resid_np)
                token_strs = model.to_str_tokens(prompt)
                mask     = bos_mask(token_strs)
                coords_m = coords[mask]
                results[model_key][valence].append({
                    "coords": coords_m,
                    "centroid": coords_m.mean(axis=0),
                    "tokens": [t for t, m in zip(token_strs, mask) if m],
                })

    # Compute metrics
    progress(0.92, desc="Analysing...")
    comparison = {}
    for valence in valences:
        dists = []
        for br, tr in zip(results["base"][valence], results["tuned"][valence]):
            bc = normalise_coords(br["coords"])
            tc = normalise_coords(tr["coords"])
            min_len = min(len(bc), len(tc))
            dists.append(cosine_dist_np(bc[:min_len], tc[:min_len]))
        comparison[valence] = {
            "mean_dist": float(np.mean(dists)),
            "dists": dists,
        }

    # Valence bias
    def mean_centroid(model_key, valence):
        return np.mean([r["centroid"] for r in results[model_key][valence]], axis=0)

    bc_pos = mean_centroid("base",  "positive"); bc_neg = mean_centroid("base",  "negative")
    tc_pos = mean_centroid("tuned", "positive"); tc_neg = mean_centroid("tuned", "negative")
    bc_neu = mean_centroid("base",  "neutral");  tc_neu = mean_centroid("tuned", "neutral")

    vb = {
        "base_pos_neg":    cosine_dist_np(bc_pos, bc_neg),
        "tuned_pos_neg":   cosine_dist_np(tc_pos, tc_neg),
        "base_pos_neu":    cosine_dist_np(bc_pos, bc_neu),
        "tuned_pos_neu":   cosine_dist_np(tc_pos, tc_neu),
        "base_neg_neu":    cosine_dist_np(bc_neg, bc_neu),
        "tuned_neg_neu":   cosine_dist_np(tc_neg, tc_neu),
    }

    # Build summary text
    max_v = max(valences, key=lambda v: comparison[v]["mean_dist"])
    min_v = min(valences, key=lambda v: comparison[v]["mean_dist"])

    lines = [
        "IRIS Alignment Comparison",
        "─" * 50,
        f"Base  : {base_model_id}",
        f"Tuned : {tuned_model_id}",
        f"Layer : {actual_layer}  |  Seed: {seed}",
        "",
        "Cross-model cosine distance per valence class:",
        f"  POSITIVE : {comparison['positive']['mean_dist']:.6f}",
        f"  NEUTRAL  : {comparison['neutral']['mean_dist']:.6f}",
        f"  NEGATIVE : {comparison['negative']['mean_dist']:.6f}",
        "",
        "Valence Bias Index (within-model distances):",
        f"  Base  — Pos↔Neg : {vb['base_pos_neg']:.6f}",
        f"  Tuned — Pos↔Neg : {vb['tuned_pos_neg']:.6f}",
        f"  Base  — Pos↔Neu : {vb['base_pos_neu']:.6f}",
        f"  Tuned — Pos↔Neu : {vb['tuned_pos_neu']:.6f}",
        f"  Base  — Neg↔Neu : {vb['base_neg_neu']:.6f}",
        f"  Tuned — Neg↔Neu : {vb['tuned_neg_neu']:.6f}",
        "",
        "─" * 50,
        f"KEY FINDING:",
        f"  Greatest divergence: {max_v.upper()} prompts ({comparison[max_v]['mean_dist']:.6f})",
        f"  Least divergence  : {min_v.upper()} prompts ({comparison[min_v]['mean_dist']:.6f})",
    ]

    if vb["tuned_pos_neg"] > vb["base_pos_neg"]:
        lines.append(f"\n  Tuning EXPANDED Pos↔Neg geometric distance ({vb['base_pos_neg']:.4f} → {vb['tuned_pos_neg']:.4f})")
        lines.append("  Model differentiates valence more strongly after tuning.")
    else:
        lines.append(f"\n  Tuning COMPRESSED Pos↔Neg geometric distance ({vb['base_pos_neg']:.4f} → {vb['tuned_pos_neg']:.4f})")
        lines.append("  Model leans toward positive geometry regardless of prompt.")

    # Visualise
    progress(0.96, desc="Rendering...")
    fig = plt.figure(figsize=(20, 13), facecolor=VIZ_COLOURS["bg"])
    fig.suptitle(
        f"IRIS — Alignment Comparison\n{base_model_id} (base) vs {tuned_model_id} (tuned)  |  Layer {actual_layer}",
        color=VIZ_COLOURS["text"], fontsize=11, y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    val_colours = [VIZ_COLOURS["positive"], VIZ_COLOURS["neutral"], VIZ_COLOURS["negative"]]
    model_c     = {"base": VIZ_COLOURS["model_a"], "tuned": VIZ_COLOURS["model_b"]}

    for col, (valence, vcol) in enumerate(zip(valences, val_colours)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(VIZ_COLOURS["surface"])
        ax.tick_params(colors=VIZ_COLOURS["subtext"], labelsize=7)
        ax.spines[:].set_color(VIZ_COLOURS["grid"])
        ax.grid(color=VIZ_COLOURS["grid"], linewidth=0.5, alpha=0.5)
        ax.set_title(f"Valence: {valence.upper()}", color=vcol, fontsize=9, pad=5)

        for mk in ["base", "tuned"]:
            mc = model_c[mk]
            for ci, run in enumerate(results[mk][valence]):
                coords = normalise_coords(run["coords"])
                tokens = run["tokens"]
                alpha  = 0.8 if ci == 0 else 0.3
                ax.plot(coords[:,0], coords[:,1], color=mc, alpha=0.15, linewidth=0.7)
                ax.scatter(coords[:,0], coords[:,1], s=35, c=mc, alpha=alpha,
                           edgecolors=VIZ_COLOURS["bg"], linewidths=0.5,
                           label=mk if ci == 0 else None)
                if ci == 0:
                    for i, tok in enumerate(tokens):
                        ax.annotate(tok, (coords[i,0], coords[i,1]),
                                    textcoords="offset points", xytext=(3,2),
                                    fontsize=6, color=mc, alpha=0.8)
        if col == 0:
            ax.legend(fontsize=7, labelcolor=VIZ_COLOURS["text"],
                      facecolor=VIZ_COLOURS["surface"], edgecolor=VIZ_COLOURS["grid"])
        ax.set_xlabel("PC1 (norm)", color=VIZ_COLOURS["subtext"], fontsize=7)
        ax.set_ylabel("PC2 (norm)", color=VIZ_COLOURS["subtext"], fontsize=7)

    # Distance bars
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.set_facecolor(VIZ_COLOURS["surface"])
    ax_d.tick_params(colors=VIZ_COLOURS["subtext"], labelsize=8)
    ax_d.spines[:].set_color(VIZ_COLOURS["grid"])
    ax_d.grid(color=VIZ_COLOURS["grid"], linewidth=0.5, alpha=0.5, axis='y')
    dists = [comparison[v]["mean_dist"] for v in valences]
    bars  = ax_d.bar(["Positive","Neutral","Negative"], dists,
                     color=val_colours, alpha=0.85,
                     edgecolor=VIZ_COLOURS["bg"], linewidth=0.8)
    ax_d.set_title("Base ↔ Tuned Distance\nper Valence Class",
                   color=VIZ_COLOURS["text"], fontsize=9, pad=5)
    ax_d.set_ylabel("Mean Cosine Distance", color=VIZ_COLOURS["subtext"], fontsize=8)
    for bar, d in zip(bars, dists):
        ax_d.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                  f"{d:.4f}", ha='center', va='bottom',
                  color=VIZ_COLOURS["text"], fontsize=8)

    # Valence bias
    ax_v = fig.add_subplot(gs[1, 1])
    ax_v.set_facecolor(VIZ_COLOURS["surface"])
    ax_v.tick_params(colors=VIZ_COLOURS["subtext"], labelsize=7)
    ax_v.spines[:].set_color(VIZ_COLOURS["grid"])
    ax_v.grid(color=VIZ_COLOURS["grid"], linewidth=0.5, alpha=0.5, axis='y')
    pairs   = ["Pos↔Neg", "Pos↔Neutral", "Neg↔Neutral"]
    base_d  = [vb["base_pos_neg"],  vb["base_pos_neu"],  vb["base_neg_neu"]]
    tuned_d = [vb["tuned_pos_neg"], vb["tuned_pos_neu"], vb["tuned_neg_neu"]]
    x = np.arange(len(pairs)); w = 0.35
    ax_v.bar(x-w/2, base_d,  w, label="Base",  color=VIZ_COLOURS["model_a"], alpha=0.85)
    ax_v.bar(x+w/2, tuned_d, w, label="Tuned", color=VIZ_COLOURS["model_b"], alpha=0.85)
    ax_v.set_xticks(x); ax_v.set_xticklabels(pairs, fontsize=7, color=VIZ_COLOURS["subtext"])
    ax_v.set_title("Valence Bias Index\nWithin-Model Distances",
                   color=VIZ_COLOURS["text"], fontsize=9, pad=5)
    ax_v.set_ylabel("Cosine Distance", color=VIZ_COLOURS["subtext"], fontsize=8)
    ax_v.legend(fontsize=7, labelcolor=VIZ_COLOURS["text"],
                facecolor=VIZ_COLOURS["surface"], edgecolor=VIZ_COLOURS["grid"])

    # Token shift
    ax_s = fig.add_subplot(gs[1, 2])
    ax_s.set_facecolor(VIZ_COLOURS["surface"])
    ax_s.tick_params(colors=VIZ_COLOURS["subtext"], labelsize=8)
    ax_s.spines[:].set_color(VIZ_COLOURS["grid"])
    ax_s.grid(color=VIZ_COLOURS["grid"], linewidth=0.5, alpha=0.5, axis='y')
    mean_shifts = []
    for valence in valences:
        shifts = []
        for br, tr in zip(results["base"][valence], results["tuned"][valence]):
            bc = normalise_coords(br["coords"])
            tc = normalise_coords(tr["coords"])
            ml = min(len(bc), len(tc))
            shifts.append(float(np.linalg.norm(bc[:ml]-tc[:ml], axis=1).mean()))
        mean_shifts.append(np.mean(shifts))
    ax_s.bar(["Positive","Neutral","Negative"], mean_shifts,
             color=val_colours, alpha=0.85,
             edgecolor=VIZ_COLOURS["bg"], linewidth=0.8)
    ax_s.set_title("Mean Per-Token Geometric Shift\n(Normalised Space)",
                   color=VIZ_COLOURS["text"], fontsize=9, pad=5)
    ax_s.set_ylabel("Mean Euclidean Shift", color=VIZ_COLOURS["subtext"], fontsize=8)
    for i, s in enumerate(mean_shifts):
        ax_s.text(i, s+0.01, f"{s:.4f}", ha='center', va='bottom',
                  color=VIZ_COLOURS["text"], fontsize=8)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight", facecolor=VIZ_COLOURS["bg"])
    plt.close()
    progress(1.0)
    return "\n".join(lines), tmp.name


# ─────────────────────────────────────────────────────────────
# IRIS — DRIFT MONITOR (Tab 04)
# ─────────────────────────────────────────────────────────────

def iris_drift_compare(path_a, path_b):
    if not path_a or not path_b or path_a.startswith("[") or path_b.startswith("["):
        return "Select two constellation files."
    da = load_constellation(path_a)
    db = load_constellation(path_b)
    if not da or not db:
        return "Could not load one or both files."

    coords_a = normalise_coords(np.array(da["pca_coords"])[bos_mask(da["token_labels"])])
    coords_b = normalise_coords(np.array(db["pca_coords"])[bos_mask(db["token_labels"])])
    tokens_a = [t.replace("Ġ","").strip() or t
                for t, m in zip(da["token_labels"], bos_mask(da["token_labels"])) if m]

    dist = cosine_dist_np(coords_a, coords_b)
    cn_a = da["meta"]["centroid_norm"]
    cn_b = db["meta"]["centroid_norm"]

    lines = [
        "IRIS Drift Report",
        "─" * 50,
        f"A: {da['meta']['model']}  L{da['meta']['layer']}",
        f"B: {db['meta']['model']}  L{db['meta']['layer']}",
        f"Prompt: {da['meta']['prompt']}",
        "",
        f"Global cosine distance : {dist:.6f}",
        f"Centroid norm A        : {cn_a:.3f}",
        f"Centroid norm B        : {cn_b:.3f}",
        f"Centroid norm delta    : {abs(cn_a - cn_b):.3f}",
        "",
    ]

    if dist < 1e-6:
        lines.append("VERDICT: IDENTICAL — same model, same run")
    elif dist < 0.05:
        lines.append("VERDICT: MINIMAL DRIFT — negligible geometric change")
    elif dist < 0.30:
        lines.append("VERDICT: MODERATE DRIFT — meaningful representational difference")
    else:
        lines.append("VERDICT: SIGNIFICANT DRIFT — substantially different geometry")

    # Per-token shift if tokenisation matches
    min_len = min(len(coords_a), len(coords_b))
    shifts  = np.linalg.norm(coords_a[:min_len] - coords_b[:min_len], axis=1)
    lines += ["", "Per-token shift (normalised space):"]
    lines.append(f"{'Token':<14} {'Shift':>8}  {'Bar'}")
    lines.append("─" * 40)
    max_s = shifts.max() if shifts.max() > 0 else 1.0
    for i, (tok, s) in enumerate(zip(tokens_a[:min_len], shifts)):
        bar = "█" * int(s / max_s * 20)
        lines.append(f"{tok:<14} {s:>8.4f}  {bar}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# EAR — HELPERS (shared with original)
# ─────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_title_map(wf):
    return {node.get("_meta",{}).get("title"): nid
            for nid, node in wf.items() if node.get("_meta",{}).get("title")}

def submit_prompt(wf):
    r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": wf}, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]

def wait_for_completion(pid, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{COMFYUI_URL}/history/{pid}", timeout=10)
        history = r.json()
        if pid in history:
            saved = []
            for node_out in history[pid].get("outputs",{}).values():
                for img in node_out.get("images",[]):
                    saved.append(f"{img.get('subfolder','')}/{img.get('filename','')}")
            return saved
        time.sleep(0.75)
    raise TimeoutError(f"Timed out after {timeout}s")

def load_latent_tensor(path):
    with open(path, "rb") as f:
        jl = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(jl))
        tb   = f.read()
    shape  = meta["latent_tensor"]["shape"]
    tensor = np.frombuffer(tb, dtype=np.float32).reshape(shape)
    try:
        pm = json.loads(meta["__metadata__"]["prompt"])
        model = pm["1"]["inputs"]["ckpt_name"]
    except Exception:
        model = "unknown"
    return tensor, model

def cosine_distance_latent(a, b):
    af, bf = a.flatten(), b.flatten()
    return float(1.0 - np.dot(af,bf) / (np.linalg.norm(af)*np.linalg.norm(bf)+1e-12))

def channel_distances(a, b):
    return [cosine_distance_latent(a[0,c], b[0,c]) for c in range(a.shape[1])]

def get_available_timelines():
    tls = []
    if os.path.exists(TIMELINE_AUDIO): tls.append(TIMELINE_AUDIO)
    if os.path.exists(TIMELINE_LLM):   tls.append(TIMELINE_LLM)
    for f in Path(".").glob("timeline*.json"):
        if str(f) not in tls: tls.append(str(f))
    # Also scan IRIS output for timelines
    for f in IRIS_OUTPUT.rglob("timeline_*.json"):
        if str(f) not in tls: tls.append(str(f))
    return tls if tls else ["[no timelines found]"]

def refresh_timelines():
    tls = get_available_timelines()
    return gr.Dropdown(choices=tls, value=tls[0] if tls else None)

def get_timeline_info(path):
    if not path or not os.path.exists(path): return "Not found."
    try:
        tl = load_json(path)
        src = tl.get("source","audio")
        if src == "llm_activations":
            return f"LLM | {tl.get('model','?')} | Layer {tl.get('layer','?')} | {len(tl.get('frames',[]))} frames"
        return f"Audio | {len(tl.get('frames',[]))} frames | {tl.get('fps','?')} fps"
    except Exception as e:
        return f"Error: {e}"

def refresh_ear():
    models = get_models(); loras = get_loras()
    status = "ComfyUI online" if comfyui_alive() else "ComfyUI offline"
    return (gr.Dropdown(choices=models, value=models[0] if models else None),
            gr.Dropdown(choices=loras,  value=loras[0]  if loras  else None),
            status)

def run_analyser(audio, fps, prompt, steps, seed_base, seed_stride,
                 cfg_min, cfg_max, dn_min, dn_max):
    if audio is None: return "No audio provided.", gr.Dropdown()
    try:
        cmd = ["python","EAR_ANALYSER_v01.py", audio,
               "--fps", str(int(fps)), "--prompt", str(prompt),
               "--steps", str(int(steps)), "--out", TIMELINE_AUDIO]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0: return f"Error:\n{r.stderr}", gr.Dropdown()
        tl = load_json(TIMELINE_AUDIO)
        return f"Generated {len(tl.get('frames',[]))} frames → {TIMELINE_AUDIO}", refresh_timelines()
    except Exception as e:
        return f"Error: {e}", gr.Dropdown()

def run_ear_pipeline(
    model_name, lora_name, lora_strength,
    prompt_pos, prompt_neg,
    fps, steps, cfg_min, cfg_max, dn_min, dn_max,
    seed_base, seed_stride, audio_path, timeline_path,
    progress=gr.Progress()
):
    if not comfyui_alive():   return "ComfyUI offline.", ""
    if not os.path.exists(WORKFLOW_PATH): return f"Workflow not found: {WORKFLOW_PATH}", ""
    active = timeline_path if timeline_path and os.path.exists(timeline_path) else TIMELINE_AUDIO
    if not os.path.exists(active): return f"Timeline not found: {active}", ""
    try:
        wft = load_json(WORKFLOW_PATH); tld = load_json(active)
    except Exception as e:
        return f"Load error: {e}", ""
    frames = tld.get("frames",[]); n = len(frames)
    if n == 0: return "Empty timeline.", ""
    src = tld.get("source","audio"); tm = build_title_map(wft)
    saved = []; log = [f"Starting render — {n} frames | {src}", ""]
    for i, frame in enumerate(frames):
        progress(i/n, desc=f"Frame {i+1}/{n}")
        wf = json.loads(json.dumps(wft))
        for nid, node in wf.items():
            if node.get("class_type") == "CheckpointLoaderSimple":
                node["inputs"]["ckpt_name"] = model_name; break
        if "EAR_LORA" in tm:
            ln = wf[tm["EAR_LORA"]]
            ln["inputs"]["lora_name"]      = lora_name if lora_name != "[none]" else ""
            ln["inputs"]["strength_model"] = float(lora_strength) if lora_name != "[none]" else 0.0
            ln["inputs"]["strength_clip"]  = float(lora_strength) if lora_name != "[none]" else 0.0
        if "EAR_PROMPT_POS" in tm: wf[tm["EAR_PROMPT_POS"]]["inputs"]["text"] = frame.get("prompt", prompt_pos)
        if "EAR_PROMPT_NEG" in tm: wf[tm["EAR_PROMPT_NEG"]]["inputs"]["text"] = prompt_neg
        if "EAR_SAMPLER" in tm:
            s = wf[tm["EAR_SAMPLER"]]["inputs"]
            s["cfg"]     = float(frame.get("cfg",     (cfg_min+cfg_max)/2))
            s["denoise"] = float(frame.get("denoise", (dn_min+dn_max)/2))
            s["seed"]    = int(frame.get("seed",      seed_base + i*seed_stride))
            s["steps"]   = int(frame.get("steps",     steps))
        tag = f"{i:05d}"
        for t in ["EAR_Output","EAR_OUTPUT","Save Image"]:
            if t in tm:
                wf[tm[t]]["inputs"]["filename_prefix"] = f"{OUTPUT_PREFIX}/frame_{tag}"; break
        if "EAR_LATENT_SAVE" in tm:
            wf[tm["EAR_LATENT_SAVE"]]["inputs"]["filename_prefix"] = f"latents/EAR_frame_{tag}"
        try:
            pid = submit_prompt(wf); imgs = wait_for_completion(pid)
            saved.extend(imgs)
            tok = f"'{frame.get('token','')}'" if src == "llm_activations" else ""
            log.append(f"  {i+1:03d} {tok:<14} OK  cfg={frame.get('cfg',0):.3f}")
        except Exception as e:
            log.append(f"  {i+1:03d} ERROR: {e}")
    concat = os.path.join(COMFYUI_OUTPUT, f"{OUTPUT_PREFIX}_concat.txt")
    try:
        with open(concat,"w",encoding="ascii",newline="\n") as f:
            for p in saved: f.write(f"file '{p}'\n")
        log += ["", f"Done. {len(saved)} frames.", f"Concat: {concat}"]
    except Exception as e:
        log.append(f"Concat error: {e}")
    return "\n".join(log), concat if os.path.exists(concat) else ""

def compare_latents(fa, fb):
    if fa is None or fb is None: return "Upload two .latent files.", ""
    try:
        ta, ma = load_latent_tensor(fa.name)
        tb, mb = load_latent_tensor(fb.name)
    except Exception as e:
        return f"Load error: {e}", ""
    cd = cosine_distance_latent(ta, tb)
    l2 = float(np.linalg.norm(ta.flatten()-tb.flatten()))
    chs = channel_distances(ta, tb)
    ch_labels = ["Ch0 Luminance","Ch1 Structure","Ch2 Color","Ch3 Detail"]
    max_ch = ch_labels[np.argmax(chs)]
    lines = ["EAR Drift Report","─"*50,
             f"Model A: {ma}", f"Model B: {mb}", "",
             f"Cosine distance : {cd:.6f}",
             f"L2 distance     : {l2:.4f}", "",
             "Per-channel cosine distance:"]
    for lbl, val in zip(ch_labels, chs):
        bar = "█" * int(val*20)
        flag = " ◄ MAX" if lbl == max_ch else ""
        lines.append(f"  {lbl:<22} {val:.4f}  {bar}{flag}")
    verdict = "SIMILAR" if cd < 0.3 else "MODERATE DRIFT" if cd < 0.6 else "HIGH DRIFT"
    lines += ["", f"Verdict: {verdict}"]
    return "\n".join(lines), f"Cosine: {cd:.4f} | L2: {l2:.1f} | Max: {max_ch}"

def compare_multiple_latents(files):
    if not files or len(files) < 2: return "Upload 2+ .latent files."
    tensors = {}
    for f in files:
        try:
            t, m = load_latent_tensor(f.name)
            tensors[f"{Path(f.name).stem} ({m[:16]})"] = t
        except Exception as e:
            return f"Error: {e}"
    names = list(tensors.keys())
    lines = ["EAR Distance Matrix","─"*60,
             f"{'Pair':<50} {'Cosine':>8} {'L2':>10}", "─"*70]
    for n1, n2 in combinations(names, 2):
        cd = cosine_distance_latent(tensors[n1], tensors[n2])
        l2 = float(np.linalg.norm(tensors[n1].flatten()-tensors[n2].flatten()))
        lines.append(f"{f'{n1[:22]} ↔ {n2[:22]}':<50} {cd:>8.4f} {l2:>10.2f}")
    return "\n".join(lines)

def get_output_frames():
    out = Path(COMFYUI_OUTPUT) / OUTPUT_PREFIX
    if not out.exists(): return [], "Output folder not found."
    frames = sorted(out.glob("*.png"))
    return [str(f) for f in frames], f"{len(frames)} frames in {out}"

def stitch_video(fps, name):
    concat = os.path.join(COMFYUI_OUTPUT, f"{OUTPUT_PREFIX}_concat.txt")
    if not os.path.exists(concat): return "Concat file not found."
    out = os.path.join(COMFYUI_OUTPUT, f"{name}.mp4")
    cmd = ["ffmpeg","-y","-r",str(int(fps)),"-f","concat","-safe","0",
           "-i",concat,"-pix_fmt","yuv420p",out]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return f"Saved: {out}" if r.returncode == 0 else f"Error:\n{r.stderr[-400:]}"
    except FileNotFoundError:
        return "ffmpeg not found."
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

def build_ui():
    tl_choices   = get_available_timelines()
    const_files  = scan_constellations()
    tl_installed = check_transformerlens()

    with gr.Blocks(css=CSS, title="EAR-Lens v0.3") as app:

        # ── MASTHEAD ──────────────────────────────────────────────
        gr.HTML("""
        <div style="padding:24px 0 4px 0; display:flex; align-items:baseline; gap:20px;">
            <div>
                <h1 style="font-size:2rem; margin:0; line-height:1;">EAR-Lens</h1>
                <p style="color:#6b7280; font-family:'Space Mono',monospace; font-size:10px;
                           letter-spacing:3px; text-transform:uppercase; margin:2px 0 0 2px;">
                    Sakin.AI — Safina Ecosystem — v0.3
                </p>
            </div>
            <div style="display:flex; gap:16px; align-items:center;">
                <span style="background:#0D7377; color:#14FFEC; font-family:'Space Mono',monospace;
                             font-size:10px; letter-spacing:2px; padding:3px 10px;
                             border-radius:2px; text-transform:uppercase;">
                    IRIS — Internal Representation and Insight System
                </span>
                <span style="background:#3d2b00; color:#e2b96f; font-family:'Space Mono',monospace;
                             font-size:10px; letter-spacing:2px; padding:3px 10px;
                             border-radius:2px; text-transform:uppercase;">
                    EAR — Embedded-space Audio-induced Reflections
                </span>
            </div>
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════════════════════════
            # IRIS TABS
            # ════════════════════════════════════════════════════

            # ── IRIS TAB 01: EXTRACTION ───────────────────────
            with gr.Tab("IRIS 01 / EXTRACTION"):
                gr.HTML('<div class="iris-header"><span style="color:#14FFEC; font-family:Space Mono; font-size:11px; letter-spacing:3px;">IRIS — EXTRACTION · Six geometric properties from LLM residual stream</span></div>')

                if tl_installed:
                    gr.HTML('<div class="status-ok">● TransformerLens ready</div>')
                else:
                    gr.HTML('<div class="status-warn">⚠ TransformerLens not found — install in ear_lens env</div>')

                with gr.Row():
                    with gr.Column(scale=2):
                        iris_model = gr.Dropdown(
                            choices=SUGGESTED_MODELS, value="gpt2",
                            label="Model (HuggingFace ID)",
                            allow_custom_value=True)
                        iris_model_warn = gr.HTML(value=get_model_warning("gpt2"))
                    with gr.Column(scale=1):
                        iris_layer = gr.Slider(0, 32, value=6, step=1,
                            label="Layer",
                            info="0-3: syntax | 4-8: semantic | 9+: task")
                        iris_seed  = gr.Number(value=42, label="Seed (fixed for reproducibility)")

                iris_prompt = gr.Textbox(
                    label="Prompt",
                    value="In the name of God, the most gracious",
                    lines=2)

                with gr.Row():
                    iris_run_btn   = gr.Button("Extract Activations →",
                                               variant="primary", scale=3)
                    iris_unload_btn = gr.Button("Unload Model",
                                                variant="secondary", scale=1)

                iris_output   = gr.Textbox(label="Extraction output",
                                           lines=20, interactive=False)
                iris_const_path = gr.Textbox(label="Constellation file path",
                                             interactive=False)


            # ── IRIS TAB 02: CONSTELLATION VIEWER ────────────
            with gr.Tab("IRIS 02 / CONSTELLATION"):
                gr.HTML('<div class="iris-header"><span style="color:#14FFEC; font-family:Space Mono; font-size:11px; letter-spacing:3px;">IRIS — CONSTELLATION VIEWER · PCA geometry · Single or comparison</span></div>')

                with gr.Row():
                    refresh_const_btn = gr.Button("↻ Refresh files",
                                                  variant="secondary", scale=1)

                gr.Markdown("#### Single Model View")
                with gr.Row():
                    single_file = gr.Dropdown(
                        choices=const_files,
                        value=const_files[0] if const_files else None,
                        label="Constellation file", scale=3)
                    single_norm = gr.Checkbox(label="Normalise", value=False)
                    single_btn  = gr.Button("Render", variant="primary", scale=1)
                single_status = gr.Textbox(label="", interactive=False, lines=1)
                single_img    = gr.Image(label="Constellation", type="filepath")

                gr.HTML("<hr class='divider'>")

                gr.Markdown("#### Multi-Model Comparison")
                with gr.Row():
                    cmp_file_a  = gr.Dropdown(choices=const_files,
                        value=const_files[0] if const_files else None,
                        label="Model A constellation", scale=2)
                    cmp_label_a = gr.Textbox(value="Model A", label="Label A", scale=1)
                with gr.Row():
                    cmp_file_b  = gr.Dropdown(choices=const_files,
                        value=const_files[1] if len(const_files) > 1 else None,
                        label="Model B constellation", scale=2)
                    cmp_label_b = gr.Textbox(value="Model B", label="Label B", scale=1)
                with gr.Row():
                    cmp_norm = gr.Checkbox(label="Normalise (recommended for cross-model)",
                                           value=True)
                    cmp_btn  = gr.Button("Compare", variant="primary", scale=1)
                cmp_status = gr.Textbox(label="", interactive=False, lines=1)
                cmp_img    = gr.Image(label="Comparison", type="filepath")


            # ── IRIS TAB 03: ALIGNMENT COMPARISON ────────────
            with gr.Tab("IRIS 03 / ALIGNMENT"):
                gr.HTML('<div class="iris-header"><span style="color:#14FFEC; font-family:Space Mono; font-size:11px; letter-spacing:3px;">IRIS — ALIGNMENT COMPARISON · Valence bias · Geometric drift detection</span></div>')

                gr.Markdown(
                    "Compare a base model against an instruction-tuned variant "
                    "across positive, neutral, and negative prompts. "
                    "IRIS detects internal geometric differences that output monitoring misses."
                )

                with gr.Row():
                    with gr.Column():
                        align_base  = gr.Dropdown(
                            choices=SUGGESTED_MODELS,
                            value="roneneldan/TinyStories-33M",
                            label="Base model", allow_custom_value=True)
                    with gr.Column():
                        align_tuned = gr.Dropdown(
                            choices=SUGGESTED_MODELS,
                            value="roneneldan/TinyStories-Instruct-33M",
                            label="Tuned model", allow_custom_value=True)

                with gr.Row():
                    align_layer = gr.Slider(0, 32, value=6, step=1, label="Layer")
                    align_seed  = gr.Number(value=42, label="Seed")

                gr.HTML('<div class="status-warn">Runs 30 extractions (5 prompts × 3 valences × 2 models). Allow 3–5 minutes on GPU.</div>')

                align_btn    = gr.Button("Run Alignment Comparison →",
                                         variant="primary")
                align_output = gr.Textbox(label="Results", lines=25,
                                          interactive=False)
                align_img    = gr.Image(label="Six-panel comparison",
                                        type="filepath")


            # ── IRIS TAB 04: DRIFT MONITOR ────────────────────
            with gr.Tab("IRIS 04 / DRIFT"):
                gr.HTML('<div class="iris-header"><span style="color:#14FFEC; font-family:Space Mono; font-size:11px; letter-spacing:3px;">IRIS — DRIFT MONITOR · Constellation distance · Per-token shift</span></div>')

                gr.Markdown(
                    "Load two constellation files — same prompt, different model versions — "
                    "and measure geometric drift between them."
                )
                with gr.Row():
                    drift_a = gr.Dropdown(choices=const_files,
                        value=const_files[0] if const_files else None,
                        label="Reference constellation (version A)", scale=3)
                    refresh_drift_btn = gr.Button("↻", variant="secondary", scale=1)
                drift_b = gr.Dropdown(choices=const_files,
                    value=const_files[1] if len(const_files) > 1 else None,
                    label="Comparison constellation (version B)")
                drift_btn    = gr.Button("Measure Drift →", variant="primary")
                drift_output = gr.Textbox(label="Drift report",
                                          lines=28, interactive=False)


            # ════════════════════════════════════════════════════
            # EAR TABS
            # ════════════════════════════════════════════════════

            # ── EAR TAB 05: CONTROL ───────────────────────────
            with gr.Tab("EAR 05 / CONTROL"):
                gr.HTML('<div class="ear-header"><span style="color:#e2b96f; font-family:Space Mono; font-size:11px; letter-spacing:3px;">EAR — CONTROL · Model · LoRA · Timeline · Pipeline fire</span></div>')

                with gr.Row():
                    ear_comfy_status = gr.Textbox(
                        label="ComfyUI status",
                        value="Click Refresh", interactive=False, scale=3)
                    ear_refresh_btn  = gr.Button("Refresh", variant="secondary", scale=1)

                gr.HTML("<hr class='divider'>")
                gr.Markdown("#### Timeline Source")
                with gr.Row():
                    ear_timeline_dd = gr.Dropdown(
                        choices=tl_choices,
                        value=tl_choices[0] if tl_choices else None,
                        label="Active timeline",
                        info="Audio timeline or IRIS-generated LLM timeline",
                        scale=3)
                    ear_refresh_tl = gr.Button("↻", variant="secondary", scale=1)
                ear_tl_info = gr.Textbox(
                    label="Timeline info",
                    value=get_timeline_info(tl_choices[0]) if tl_choices else "",
                    interactive=False)

                gr.HTML("<hr class='divider'>")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Model & LoRA")
                        ear_model_dd   = gr.Dropdown(choices=get_models(),
                            label="Checkpoint", value=None)
                        ear_lora_dd    = gr.Dropdown(choices=get_loras(),
                            label="LoRA", value="[none]")
                        ear_lora_str   = gr.Slider(0.0, 1.5, value=1.0,
                            step=0.05, label="LoRA strength")
                    with gr.Column(scale=1):
                        gr.Markdown("#### Prompts")
                        ear_prompt_pos = gr.Textbox(label="Positive",
                            value="fractal fields", lines=2)
                        ear_prompt_neg = gr.Textbox(label="Negative",
                            value="text, watermark, logo, blurry", lines=2)

                gr.HTML("<hr class='divider'>")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Sampler")
                        with gr.Row():
                            ear_cfg_min = gr.Slider(1.0, 10.0, value=4.5,
                                step=0.5, label="CFG min")
                            ear_cfg_max = gr.Slider(1.0, 15.0, value=8.5,
                                step=0.5, label="CFG max")
                        with gr.Row():
                            ear_dn_min  = gr.Slider(0.1, 1.0, value=0.20,
                                step=0.05, label="Denoise min")
                            ear_dn_max  = gr.Slider(0.1, 1.0, value=0.60,
                                step=0.05, label="Denoise max")
                        ear_steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
                    with gr.Column(scale=1):
                        gr.Markdown("#### Timeline / Seed")
                        ear_fps        = gr.Slider(1, 12, value=2, step=1, label="FPS")
                        ear_seed_base  = gr.Number(value=112233, label="Seed base")
                        ear_seed_str   = gr.Number(value=13,     label="Seed stride")

                gr.HTML("<hr class='divider'>")
                gr.Markdown("#### Generate Timeline from Audio (optional)")
                with gr.Row():
                    ear_audio    = gr.Audio(label="Audio file",
                        type="filepath", scale=3)
                    ear_analyse  = gr.Button("Analyse Audio →",
                        variant="secondary", scale=1)
                ear_analyse_out = gr.Textbox(label="Analyser output",
                                              lines=3, interactive=False)

                gr.HTML("<hr class='divider'>")
                ear_render_btn = gr.Button("Run EAR Pipeline →", variant="primary")
                ear_render_log = gr.Textbox(label="Render log",
                                             lines=14, interactive=False)
                ear_concat_out = gr.Textbox(label="Concat path", interactive=False)


            # ── EAR TAB 06: OUTPUT VIEWER ─────────────────────
            with gr.Tab("EAR 06 / OUTPUT"):
                gr.HTML('<div class="ear-header"><span style="color:#e2b96f; font-family:Space Mono; font-size:11px; letter-spacing:3px;">EAR — OUTPUT · Frames · Stitch · Export</span></div>')

                with gr.Row():
                    ear_scan_btn    = gr.Button("Scan output folder",
                                                variant="secondary")
                    ear_frame_count = gr.Textbox(label="Frames",
                                                 interactive=False)
                ear_gallery = gr.Gallery(label="Rendered frames",
                                         columns=4, height=420,
                                         object_fit="contain")

                gr.HTML("<hr class='divider'>")
                gr.Markdown("#### Stitch Video")
                with gr.Row():
                    ear_stitch_fps  = gr.Slider(1, 24, value=2, step=1,
                        label="Output FPS")
                    ear_output_name = gr.Textbox(value="EAR_output",
                        label="Filename")
                    ear_stitch_btn  = gr.Button("Stitch with ffmpeg",
                        variant="primary")
                ear_stitch_out = gr.Textbox(label="ffmpeg output",
                                             lines=4, interactive=False)


        # ════════════════════════════════════════════════════
        # WIRING
        # ════════════════════════════════════════════════════

        # IRIS 01
        iris_model.change(
            lambda m: get_model_warning(m),
            inputs=[iris_model], outputs=[iris_model_warn])
        iris_run_btn.click(
            run_iris_extraction,
            inputs=[iris_model, iris_layer, iris_prompt, iris_seed],
            outputs=[iris_output, iris_const_path])
        iris_unload_btn.click(unload_model, outputs=[iris_output])

        # IRIS 02
        refresh_const_btn.click(
            lambda: (refresh_constellation_list(),
                     refresh_constellation_list(),
                     refresh_constellation_list(),
                     refresh_constellation_list()),
            outputs=[single_file, cmp_file_a, cmp_file_b, drift_a])
        single_btn.click(
            viewer_render_single,
            inputs=[single_file, single_norm],
            outputs=[single_img, single_status])
        cmp_btn.click(
            viewer_render_compare,
            inputs=[cmp_file_a, cmp_file_b, cmp_label_a, cmp_label_b, cmp_norm],
            outputs=[cmp_img, cmp_status])

        # IRIS 03
        align_btn.click(
            run_alignment_comparison,
            inputs=[align_base, align_tuned, align_layer, align_seed],
            outputs=[align_output, align_img])

        # IRIS 04
        refresh_drift_btn.click(
            lambda: (refresh_constellation_list(), refresh_constellation_list()),
            outputs=[drift_a, drift_b])
        drift_btn.click(
            iris_drift_compare,
            inputs=[drift_a, drift_b],
            outputs=[drift_output])

        # EAR 05
        ear_refresh_btn.click(
            refresh_ear,
            outputs=[ear_model_dd, ear_lora_dd, ear_comfy_status])
        ear_refresh_tl.click(
            refresh_timelines, outputs=[ear_timeline_dd])
        ear_timeline_dd.change(
            get_timeline_info,
            inputs=[ear_timeline_dd], outputs=[ear_tl_info])
        ear_analyse.click(
            run_analyser,
            inputs=[ear_audio, ear_fps, ear_prompt_pos, ear_steps,
                    ear_seed_base, ear_seed_str,
                    ear_cfg_min, ear_cfg_max, ear_dn_min, ear_dn_max],
            outputs=[ear_analyse_out, ear_timeline_dd])
        ear_render_btn.click(
            run_ear_pipeline,
            inputs=[ear_model_dd, ear_lora_dd, ear_lora_str,
                    ear_prompt_pos, ear_prompt_neg,
                    ear_fps, ear_steps,
                    ear_cfg_min, ear_cfg_max, ear_dn_min, ear_dn_max,
                    ear_seed_base, ear_seed_str,
                    ear_audio, ear_timeline_dd],
            outputs=[ear_render_log, ear_concat_out])

        # EAR 06
        ear_scan_btn.click(
            get_output_frames,
            outputs=[ear_gallery, ear_frame_count])
        ear_stitch_btn.click(
            stitch_video,
            inputs=[ear_stitch_fps, ear_output_name],
            outputs=[ear_stitch_out])

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
        inbrowser=True,
    )
