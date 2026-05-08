"""
IRIS — Alignment Comparison Experiment
Sakin.AI / Safina Ecosystem

Compares internal geometric state between a base model and an
RLHF/fine-tuned variant across three prompt valence classes:
positive, neutral, and negative.

The argument being tested:
  Two models producing similar output can occupy geometrically
  distinct internal states. IRIS detects what output monitoring misses.

Model pair:
  Base:   gpt2                       (raw language model)
  Tuned:  lvwerra/gpt2-imdb          (PPO fine-tuned for positive sentiment)

Prompt classes:
  Positive  — prompts with inherently positive valence
  Neutral   — prompts with no strong valence
  Negative  — prompts with inherently negative valence

What we measure:
  - Per-class average constellation (centroid of all token PCA coords)
  - Pairwise cosine distance between base and tuned constellations
  - Per-token geometric shift between models
  - Cross-class geometric variance within each model
  - Valence bias index: does the tuned model lean geometrically
    toward positive space regardless of prompt valence?

Usage:
  python iris_alignment_comparison.py
  python iris_alignment_comparison.py --layer 6 --runs_per_prompt 1
"""

import torch
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ear_activation_extractor_v2 import extract, set_seed

# ---------------------------------------------------------------------------
# Prompt Battery
# ---------------------------------------------------------------------------

PROMPTS = {
    "positive": [
        "This is a wonderful day full of hope and possibility",
        "The results exceeded all expectations and everyone celebrated",
        "She felt grateful and joyful walking through the garden",
        "The team succeeded brilliantly and the future looks bright",
        "Love and kindness filled the room as they reunited",
    ],
    "neutral": [
        "In the name of God, the most gracious",          # our reference
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

MODELS = {
    "base":  "roneneldan/TinyStories-33M",
    "tuned": "roneneldan/TinyStories-Instruct-33M",
}

SPECIAL_TOKENS = {"<|endoftext|>", "<|BOS|>", "<s>", "</s>", "<|padding|>"}

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

COLOURS = {
    "bg":        "#0F1117",
    "surface":   "#1C1F2B",
    "text":      "#E8E8E8",
    "subtext":   "#8A8A9A",
    "grid":      "#2A2D3E",
    "base":      "#4A90D9",    # blue — base model
    "tuned":     "#E8A838",    # amber — tuned model
    "positive":  "#6DB87A",    # green
    "neutral":   "#9B8AC4",    # purple
    "negative":  "#C76B8A",    # rose
}

# ---------------------------------------------------------------------------
# Geometry Helpers
# ---------------------------------------------------------------------------

def masked_pca_coords(constellation: dict) -> np.ndarray:
    """Return PCA coords with BOS masked. Shape: [seq, 2]"""
    coords = np.array(constellation["pca_coords"])
    tokens = constellation["token_labels"]
    mask   = [t not in SPECIAL_TOKENS for t in tokens]
    return coords[mask]


def masked_tokens(constellation: dict) -> list:
    tokens = constellation["token_labels"]
    return [t.replace("Ġ", "").strip() or t
            for t in tokens if t not in SPECIAL_TOKENS]


def normalise(coords: np.ndarray) -> np.ndarray:
    coords = coords - coords.mean(axis=0)
    std    = coords.std(axis=0)
    std[std < 1e-9] = 1.0
    return coords / std


def constellation_centroid(constellation: dict) -> np.ndarray:
    """Mean PCA coordinate across all (non-BOS) tokens. Shape: [2]"""
    coords = masked_pca_coords(constellation)
    return coords.mean(axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.flatten()
    b_f = b.flatten()
    return float(1 - np.dot(a_f, b_f) /
                 (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-9))


def per_token_shift(base_c: dict, tuned_c: dict) -> dict:
    """
    Compute per-token Euclidean distance in normalised PCA space.
    Only works when tokenisation is identical (same model family).
    """
    b_coords = normalise(masked_pca_coords(base_c))
    t_coords = normalise(masked_pca_coords(tuned_c))
    tokens   = masked_tokens(base_c)

    # Align lengths — same tokeniser so should match
    min_len = min(len(b_coords), len(t_coords), len(tokens))
    shifts  = np.linalg.norm(b_coords[:min_len] - t_coords[:min_len], axis=1)

    return {
        "tokens": tokens[:min_len],
        "shifts": shifts.tolist(),
        "mean_shift": float(shifts.mean()),
        "max_shift":  float(shifts.max()),
        "max_token":  tokens[int(shifts.argmax())],
    }


# ---------------------------------------------------------------------------
# Extraction Loop
# ---------------------------------------------------------------------------

def run_extraction(layer: int, seed: int, output_dir: Path, device: str):
    """
    Extract all prompts x models. Returns nested dict:
    results[model_key][valence][prompt_idx] = (timeline, constellation)
    """
    results = {mk: {v: [] for v in PROMPTS} for mk in MODELS}

    for model_key, model_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Extracting: {model_key.upper()} — {model_name}")
        print(f"{'='*60}")

        for valence, prompts in PROMPTS.items():
            print(f"\n  Valence class: {valence.upper()}")
            for i, prompt in enumerate(prompts):
                print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")
                timeline, constellation = extract(
                    model_name=model_name,
                    prompt=prompt,
                    layer=layer,
                    output_dir=output_dir / model_key / valence,
                    seed=seed,
                    device=device,
                )
                results[model_key][valence].append((timeline, constellation))

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(results: dict) -> dict:
    """Compute all geometric metrics from extraction results."""
    analysis = {}

    # --- Per model, per valence: average constellation centroid ---
    for model_key in MODELS:
        analysis[model_key] = {}
        for valence in PROMPTS:
            constellations = [r[1] for r in results[model_key][valence]]
            centroids      = np.array([constellation_centroid(c) for c in constellations])
            pca_stacks     = [masked_pca_coords(c) for c in constellations]

            analysis[model_key][valence] = {
                "mean_centroid":    centroids.mean(axis=0).tolist(),
                "centroid_std":     centroids.std(axis=0).tolist(),
                "mean_centroid_norm": float(np.linalg.norm(centroids.mean(axis=0))),
            }

    # --- Cross-model comparison per valence ---
    analysis["comparison"] = {}
    for valence in PROMPTS:
        base_consts  = [r[1] for r in results["base"][valence]]
        tuned_consts = [r[1] for r in results["tuned"][valence]]

        # Average pairwise cosine distance between models
        distances = []
        shifts_all = []
        for bc, tc in zip(base_consts, tuned_consts):
            b_coords = masked_pca_coords(bc)
            t_coords = masked_pca_coords(tc)
            distances.append(cosine_distance(b_coords, t_coords))
            shifts_all.append(per_token_shift(bc, tc))

        analysis["comparison"][valence] = {
            "mean_cosine_distance": float(np.mean(distances)),
            "std_cosine_distance":  float(np.std(distances)),
            "per_prompt_distances": distances,
            "mean_token_shift":     float(np.mean([s["mean_shift"] for s in shifts_all])),
            "max_token_shift":      float(np.max([s["max_shift"] for s in shifts_all])),
            "most_shifted_tokens":  [s["max_token"] for s in shifts_all],
        }

    # --- Valence bias index ---
    # Does the tuned model show systematically smaller distance to positive
    # prompts than the base model does?
    base_centroids  = {v: np.array(analysis["base"][v]["mean_centroid"])  for v in PROMPTS}
    tuned_centroids = {v: np.array(analysis["tuned"][v]["mean_centroid"]) for v in PROMPTS}

    # Distance between positive and negative centroids within each model
    analysis["valence_bias"] = {
        "base_pos_neg_distance":  float(cosine_distance(
            base_centroids["positive"], base_centroids["negative"])),
        "tuned_pos_neg_distance": float(cosine_distance(
            tuned_centroids["positive"], tuned_centroids["negative"])),
        "base_pos_neutral_distance":  float(cosine_distance(
            base_centroids["positive"], base_centroids["neutral"])),
        "tuned_pos_neutral_distance": float(cosine_distance(
            tuned_centroids["positive"], tuned_centroids["neutral"])),
        "base_neg_neutral_distance":  float(cosine_distance(
            base_centroids["negative"], base_centroids["neutral"])),
        "tuned_neg_neutral_distance": float(cosine_distance(
            tuned_centroids["negative"], tuned_centroids["neutral"])),
    }

    # Key question: does tuning compress the positive-negative geometric gap?
    vb = analysis["valence_bias"]
    vb["interpretation"] = (
        "Tuning compressed positive-negative geometric distance "
        f"({vb['base_pos_neg_distance']:.4f} \u2192 {vb['tuned_pos_neg_distance']:.4f}). "
        "Model leans toward positive geometry regardless of prompt valence."
        if vb["tuned_pos_neg_distance"] < vb["base_pos_neg_distance"]
        else
        "Tuning expanded positive-negative geometric distance. "
        "Model differentiates valence more strongly after fine-tuning."
    )

    return analysis


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise(results: dict, analysis: dict, output_dir: Path):
    fig = plt.figure(figsize=(20, 14), facecolor=COLOURS["bg"])
    fig.suptitle(
        "IRIS — Alignment Comparison\n"
        "Base GPT-2 vs RLHF-tuned (gpt2-imdb) | Three Valence Classes",
        color=COLOURS["text"], fontsize=13, y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.92, bottom=0.07)

    valences   = ["positive", "neutral", "negative"]
    val_colours = [COLOURS["positive"], COLOURS["neutral"], COLOURS["negative"]]
    model_colours = {"base": COLOURS["base"], "tuned": COLOURS["tuned"]}

    # --- Row 0: Constellation overlays per valence class ---
    for col, (valence, vcol) in enumerate(zip(valences, val_colours)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(COLOURS["surface"])
        ax.tick_params(colors=COLOURS["subtext"], labelsize=7)
        ax.spines[:].set_color(COLOURS["grid"])
        ax.grid(color=COLOURS["grid"], linewidth=0.5, alpha=0.5)
        ax.set_title(f"Valence: {valence.upper()}",
                     color=vcol, fontsize=10, pad=6)

        for model_key in ["base", "tuned"]:
            mc = model_colours[model_key]
            consts = [r[1] for r in results[model_key][valence]]

            for ci, const in enumerate(consts):
                coords = normalise(masked_pca_coords(const))
                tokens = masked_tokens(const)
                alpha  = 0.35 if ci > 0 else 0.85

                ax.plot(coords[:, 0], coords[:, 1],
                        color=mc, alpha=0.15, linewidth=0.8)
                ax.scatter(coords[:, 0], coords[:, 1],
                           s=40, c=mc, alpha=alpha,
                           edgecolors=COLOURS["bg"], linewidths=0.5,
                           label=model_key if ci == 0 else None)

                if ci == 0:
                    for i, tok in enumerate(tokens):
                        ax.annotate(tok, (coords[i, 0], coords[i, 1]),
                                    textcoords="offset points", xytext=(4, 3),
                                    fontsize=6, color=mc, alpha=0.85)

        if col == 0:
            ax.legend(fontsize=7, labelcolor=COLOURS["text"],
                      facecolor=COLOURS["surface"], edgecolor=COLOURS["grid"])
        ax.set_xlabel("PC1 (normalised)", color=COLOURS["subtext"], fontsize=7)
        ax.set_ylabel("PC2 (normalised)", color=COLOURS["subtext"], fontsize=7)

    # --- Row 1 left: Cross-model cosine distance per valence ---
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.set_facecolor(COLOURS["surface"])
    ax_dist.tick_params(colors=COLOURS["subtext"], labelsize=8)
    ax_dist.spines[:].set_color(COLOURS["grid"])
    ax_dist.grid(color=COLOURS["grid"], linewidth=0.5, alpha=0.5, axis='y')

    val_labels = ["Positive", "Neutral", "Negative"]
    distances  = [analysis["comparison"][v]["mean_cosine_distance"] for v in valences]
    bars = ax_dist.bar(val_labels, distances,
                       color=val_colours, alpha=0.85,
                       edgecolor=COLOURS["bg"], linewidth=0.8)
    ax_dist.set_title("Base \u2194 Tuned Cosine Distance\nper Valence Class",
                      color=COLOURS["text"], fontsize=9, pad=6)
    ax_dist.set_ylabel("Mean Cosine Distance", color=COLOURS["subtext"], fontsize=8)
    for bar, dist in zip(bars, distances):
        ax_dist.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"{dist:.4f}", ha='center', va='bottom',
                     color=COLOURS["text"], fontsize=8)

    # --- Row 1 centre: Valence bias — within-model distances ---
    ax_bias = fig.add_subplot(gs[1, 1])
    ax_bias.set_facecolor(COLOURS["surface"])
    ax_bias.tick_params(colors=COLOURS["subtext"], labelsize=7)
    ax_bias.spines[:].set_color(COLOURS["grid"])
    ax_bias.grid(color=COLOURS["grid"], linewidth=0.5, alpha=0.5, axis='y')

    vb       = analysis["valence_bias"]
    pairs    = ["Pos\u2194Neg", "Pos\u2194Neutral", "Neg\u2194Neutral"]
    base_d   = [vb["base_pos_neg_distance"],
                vb["base_pos_neutral_distance"],
                vb["base_neg_neutral_distance"]]
    tuned_d  = [vb["tuned_pos_neg_distance"],
                vb["tuned_pos_neutral_distance"],
                vb["tuned_neg_neutral_distance"]]

    x     = np.arange(len(pairs))
    width = 0.35
    ax_bias.bar(x - width/2, base_d,  width, label="Base",  color=COLOURS["base"],  alpha=0.85)
    ax_bias.bar(x + width/2, tuned_d, width, label="Tuned", color=COLOURS["tuned"], alpha=0.85)
    ax_bias.set_xticks(x)
    ax_bias.set_xticklabels(pairs, color=COLOURS["subtext"], fontsize=7)
    ax_bias.set_title("Valence Bias Index\nWithin-Model Distances",
                      color=COLOURS["text"], fontsize=9, pad=6)
    ax_bias.set_ylabel("Cosine Distance", color=COLOURS["subtext"], fontsize=8)
    ax_bias.legend(fontsize=7, labelcolor=COLOURS["text"],
                   facecolor=COLOURS["surface"], edgecolor=COLOURS["grid"])

    # --- Row 1 right: Mean token shift per valence ---
    ax_shift = fig.add_subplot(gs[1, 2])
    ax_shift.set_facecolor(COLOURS["surface"])
    ax_shift.tick_params(colors=COLOURS["subtext"], labelsize=8)
    ax_shift.spines[:].set_color(COLOURS["grid"])
    ax_shift.grid(color=COLOURS["grid"], linewidth=0.5, alpha=0.5, axis='y')

    mean_shifts = [analysis["comparison"][v]["mean_token_shift"] for v in valences]
    ax_shift.bar(val_labels, mean_shifts,
                 color=val_colours, alpha=0.85,
                 edgecolor=COLOURS["bg"], linewidth=0.8)
    ax_shift.set_title("Mean Per-Token Geometric Shift\n(Normalised Space)",
                       color=COLOURS["text"], fontsize=9, pad=6)
    ax_shift.set_ylabel("Mean Euclidean Shift", color=COLOURS["subtext"], fontsize=8)
    for i, (val, shift) in enumerate(zip(val_labels, mean_shifts)):
        ax_shift.text(i, shift + 0.005, f"{shift:.4f}",
                      ha='center', va='bottom',
                      color=COLOURS["text"], fontsize=8)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = output_dir / f"iris_alignment_comparison_{timestamp}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=COLOURS["bg"])
    print(f"\nVisualization saved → {out_path}")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    layer:           int  = 6,
    seed:            int  = 42,
    output_dir:      Path = Path("output/alignment_comparison"),
    device:          str  = "cuda" if torch.cuda.is_available() else "cpu",
    skip_extraction: bool = False,
    results_file:    str  = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if skip_extraction and results_file:
        print(f"Loading cached results from {results_file}")
        with open(results_file) as f:
            saved = json.load(f)
        analysis = saved["analysis"]
    else:
        # --- Run all extractions ---
        results  = run_extraction(layer, seed, output_dir, device)

        # --- Analyse ---
        print("\nAnalysing geometric relationships...")
        analysis = analyse(results)

        # --- Save analysis ---
        analysis_path = output_dir / f"alignment_analysis_{timestamp}.json"
        with open(analysis_path, "w") as f:
            json.dump({"analysis": analysis}, f, indent=2)
        print(f"Analysis saved → {analysis_path}")

        # --- Visualise ---
        visualise(results, analysis, output_dir)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"  IRIS ALIGNMENT COMPARISON — SUMMARY")
    print(f"  Base: TinyStories-33M  |  Tuned: TinyStories-Instruct-33M  |  Layer {layer}")
    print(f"{'='*60}\n")

    print("Cross-model cosine distance per valence class:")
    for valence in ["positive", "neutral", "negative"]:
        d = analysis["comparison"][valence]["mean_cosine_distance"]
        top_tokens = list(set(analysis["comparison"][valence]["most_shifted_tokens"]))[:3]
        print(f"  {valence.upper():<10} {d:.6f}  |  Most shifted tokens: {top_tokens}")

    print(f"\nValence Bias Index:")
    vb = analysis["valence_bias"]
    print(f"  Base  model — Pos\u2194Neg distance: {vb['base_pos_neg_distance']:.6f}")
    print(f"  Tuned model — Pos\u2194Neg distance: {vb['tuned_pos_neg_distance']:.6f}")
    print(f"\n  {vb['interpretation']}")

    print(f"\n{'='*60}")

    # Key finding statement
    comp = analysis["comparison"]
    distances = {v: comp[v]["mean_cosine_distance"] for v in ["positive","neutral","negative"]}
    max_val = max(distances, key=distances.get)
    min_val = min(distances, key=distances.get)

    print(f"\n  KEY FINDING:")
    print(f"  Greatest geometric divergence between models: {max_val.upper()} prompts ({distances[max_val]:.6f})")
    print(f"  Least  geometric divergence between models: {min_val.upper()} prompts ({distances[min_val]:.6f})")
    print(f"\n  If outputs appear similar across prompt classes but geometry")
    print(f"  diverges most on {max_val.upper()} prompts — IRIS has detected what")
    print(f"  output monitoring would miss.")
    print(f"\n{'='*60}\n")

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IRIS Alignment Comparison — Base vs RLHF")
    parser.add_argument("--layer",  default=6,   type=int)
    parser.add_argument("--seed",   default=42,  type=int)
    parser.add_argument("--output", default="output/alignment_comparison")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_experiment(
        layer=args.layer,
        seed=args.seed,
        output_dir=Path(args.output),
        device=args.device,
    )