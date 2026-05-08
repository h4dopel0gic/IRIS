"""
EAR-Lens — Constellation Visualiser
Sakin.AI / Safina Ecosystem

Reads constellation.json and produces:
  - 2D PCA scatter (the constellation)
  - Layer delta profile heatmap (where meaning crystallises)
  - Property radar per token
  - Multi-model overlay (if multiple constellation files provided)

Usage:
  python ear_constellation_viz.py --input output/activations/constellation.json
  python ear_constellation_viz.py --compare gpt2.json mistral.json --labels GPT-2 Mistral
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


# ---------------------------------------------------------------------------
# Palette — clean, readable on dark and light
# ---------------------------------------------------------------------------

COLOURS = {
    "primary":    "#4A90D9",
    "secondary":  "#E8A838",
    "tertiary":   "#6DB87A",
    "quaternary": "#C76B8A",
    "bg":         "#0F1117",
    "surface":    "#1C1F2B",
    "text":       "#E8E8E8",
    "subtext":    "#8A8A9A",
    "grid":       "#2A2D3E",
}

MODEL_COLOURS = [
    COLOURS["primary"],
    COLOURS["secondary"],
    COLOURS["tertiary"],
    COLOURS["quaternary"],
]


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLOURS["surface"])
    ax.tick_params(colors=COLOURS["subtext"], labelsize=8)
    ax.spines[:].set_color(COLOURS["grid"])
    ax.grid(color=COLOURS["grid"], linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=COLOURS["text"], fontsize=10, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=COLOURS["subtext"], fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=COLOURS["subtext"], fontsize=8)


# ---------------------------------------------------------------------------
# Plot 1 — Constellation (PCA scatter)
# ---------------------------------------------------------------------------

def plot_constellation(ax, constellation: dict, colour: str, label: str = "", alpha: float = 1.0):
    coords = np.array(constellation["pca_coords"])       # [seq, 2]
    tokens = constellation["token_labels"]
    rms_vals = [t["rms"] for t in constellation.get("tokens", [])] if "tokens" in constellation else None

    # Size by RMS if available, else uniform
    sizes = None
    if rms_vals:
        rms_arr = np.array(rms_vals)
        sizes = 80 + 200 * (rms_arr - rms_arr.min()) / (rms_arr.max() - rms_arr.min() + 1e-9)

    # Draw connecting path — sequence order
    ax.plot(coords[:, 0], coords[:, 1],
            color=colour, alpha=0.25 * alpha, linewidth=1.0, zorder=1)

    # Scatter
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    s=sizes if sizes is not None else 100,
                    c=colour, alpha=0.85 * alpha,
                    edgecolors=COLOURS["bg"], linewidths=0.8,
                    zorder=3, label=label)

    # Token labels
    for i, tok in enumerate(tokens):
        tok_clean = tok.replace("Ġ", "").strip() or tok
        ax.annotate(
            tok_clean,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
            color=colour,
            alpha=0.9 * alpha,
            zorder=4,
        )

    # Centroid marker
    if "centroid_projected" in constellation:
        cx, cy = constellation["centroid_projected"]
        ax.scatter([cx], [cy], marker="+", s=120,
                   color=colour, linewidths=1.5, zorder=5, alpha=alpha)

    ev = constellation["meta"].get("pca_explained_var", [0, 0])
    return ev


# ---------------------------------------------------------------------------
# Plot 2 — Layer Delta Profile Heatmap
# ---------------------------------------------------------------------------

def plot_delta_heatmap(ax, constellation: dict, colour: str):
    profile = np.array(constellation["delta_profile"])   # [layers, seq]
    tokens  = constellation["token_labels"]
    tokens_clean = [t.replace("Ġ", "").strip() or t for t in tokens]

    im = ax.imshow(profile, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(tokens_clean)))
    ax.set_xticklabels(tokens_clean, rotation=30, ha="right",
                       fontsize=7.5, color=COLOURS["subtext"])
    ax.set_ylabel("Layer", color=COLOURS["subtext"], fontsize=8)
    ax.tick_params(colors=COLOURS["subtext"])
    ax.set_title("Layer Delta Profile — Where Meaning Crystallises",
                 color=COLOURS["text"], fontsize=10, pad=8)
    ax.set_facecolor(COLOURS["surface"])
    ax.spines[:].set_color(COLOURS["grid"])

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label="Δ Magnitude").ax.yaxis.label.set_color(COLOURS["subtext"])


# ---------------------------------------------------------------------------
# Plot 3 — Per-token property bars
# ---------------------------------------------------------------------------

def plot_property_bars(ax, constellation: dict, colour: str):
    if "tokens" not in constellation:
        ax.set_visible(False)
        return

    tokens = constellation["tokens"]
    labels = [t["token"].replace("Ġ", "").strip() or t["token"] for t in tokens]
    props  = ["rms", "delta_mag", "attn_entropy", "cos_dist"]
    prop_labels = ["RMS", "Δ Mag", "Entropy", "Cos Dist"]
    colours_bar = [COLOURS["primary"], COLOURS["secondary"],
                   COLOURS["tertiary"], COLOURS["quaternary"]]

    x = np.arange(len(labels))
    width = 0.2

    for i, (prop, plabel, c) in enumerate(zip(props, prop_labels, colours_bar)):
        vals = np.array([t[prop] for t in tokens], dtype=float)
        # Normalise for visual comparison
        if vals.max() > vals.min():
            vals_norm = (vals - vals.min()) / (vals.max() - vals.min())
        else:
            vals_norm = np.zeros_like(vals)
        ax.bar(x + i * width, vals_norm, width, label=plabel, color=c, alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=30, ha="right",
                       fontsize=7.5, color=COLOURS["subtext"])
    ax.legend(fontsize=7, labelcolor=COLOURS["subtext"],
              facecolor=COLOURS["surface"], edgecolor=COLOURS["grid"])
    style_ax(ax, title="Per-Token Properties (Normalised)", ylabel="Normalised Value")


# ---------------------------------------------------------------------------
# Single model — full figure
# ---------------------------------------------------------------------------

def visualise_single(constellation_path: Path, output_dir: Path):
    with open(constellation_path) as f:
        data = json.load(f)

    # Inject token data into constellation if separate
    # (timeline_llm_v2.json tokens merged into constellation.json at extraction)

    fig = plt.figure(figsize=(16, 12), facecolor=COLOURS["bg"])
    fig.suptitle(
        f"EAR-Lens Constellation — {data['meta']['model']}\n"
        f""{data['meta']['prompt']}"  |  Layer {data['meta']['layer']}",
        color=COLOURS["text"], fontsize=12, y=0.97
    )

    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35,
                          left=0.07, right=0.96, top=0.92, bottom=0.07)

    # --- Constellation ---
    ax_const = fig.add_subplot(gs[0, :])
    style_ax(ax_const, title="Activation Constellation — PCA Projection")
    ev = plot_constellation(ax_const, data, colour=COLOURS["primary"],
                            label=data["meta"]["model"])
    ax_const.set_xlabel(f"PC1  ({ev[0]:.1%} variance)", color=COLOURS["subtext"], fontsize=8)
    ax_const.set_ylabel(f"PC2  ({ev[1]:.1%} variance)", color=COLOURS["subtext"], fontsize=8)

    # --- Delta heatmap ---
    ax_heat = fig.add_subplot(gs[1, 0])
    plot_delta_heatmap(ax_heat, data, colour=COLOURS["secondary"])

    # --- Property bars ---
    ax_bars = fig.add_subplot(gs[1, 1])

    # Merge token data if available alongside constellation
    timeline_path = constellation_path.parent / "timeline_llm_v2.json"
    if timeline_path.exists() and "tokens" not in data:
        with open(timeline_path) as f:
            timeline = json.load(f)
        data["tokens"] = timeline["tokens"]

    plot_property_bars(ax_bars, data, colour=COLOURS["tertiary"])

    out_path = output_dir / f"constellation_{data['meta']['model'].replace('/', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOURS["bg"])
    print(f"Saved → {out_path}")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Multi-model overlay — constellation comparison
# ---------------------------------------------------------------------------

def visualise_compare(paths: list[Path], labels: list[str], output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=COLOURS["bg"])
    ax.set_facecolor(COLOURS["surface"])
    style_ax(ax, title="Activation Constellation — Multi-Model Comparison")

    ev_ref = None
    for i, (path, label) in enumerate(zip(paths, labels)):
        with open(path) as f:
            data = json.load(f)
        colour = MODEL_COLOURS[i % len(MODEL_COLOURS)]
        ev = plot_constellation(ax, data, colour=colour,
                                label=label, alpha=0.85)
        if i == 0:
            ev_ref = ev

    ax.set_xlabel(f"PC1  ({ev_ref[0]:.1%} variance, ref model)",
                  color=COLOURS["subtext"], fontsize=8)
    ax.set_ylabel(f"PC2  ({ev_ref[1]:.1%} variance, ref model)",
                  color=COLOURS["subtext"], fontsize=8)
    ax.legend(fontsize=9, labelcolor=COLOURS["text"],
              facecolor=COLOURS["surface"], edgecolor=COLOURS["grid"])

    fig.suptitle("EAR-Lens — Constellation Comparison\n"
                 "Same prompt, different activation geometry",
                 color=COLOURS["text"], fontsize=11, y=0.97)

    out_path = output_dir / "constellation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOURS["bg"])
    print(f"Saved → {out_path}")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAR-Lens Constellation Visualiser")
    parser.add_argument("--input",   help="Single constellation.json path")
    parser.add_argument("--compare", nargs="+", help="Multiple constellation.json paths for overlay")
    parser.add_argument("--labels",  nargs="+", help="Labels for --compare mode")
    parser.add_argument("--output",  default="output/activations", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        paths  = [Path(p) for p in args.compare]
        labels = args.labels if args.labels else [p.stem for p in paths]
        visualise_compare(paths, labels, output_dir)
    elif args.input:
        visualise_single(Path(args.input), output_dir)
    else:
        print("Provide --input <file> or --compare <file1> <file2> ...")