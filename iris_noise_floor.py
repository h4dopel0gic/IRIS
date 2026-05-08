"""
IRIS — Noise Floor Experiment
Sakin.AI / Safina Ecosystem

Runs the same extraction N times with a fixed seed and measures
constellation stability. If the constellation is stable, the drift
monitoring argument holds. If it shifts, we have a calibration problem.

What we measure:
  - Pairwise cosine distance between constellation runs (should be ~0)
  - PC1/PC2 coordinate variance per token across runs
  - RMS variance per token across runs
  - Centroid norm variance across runs

A stable instrument shows near-zero variance. Any variance above
numerical noise is a calibration concern to document.

Usage:
  python iris_noise_floor.py
  python iris_noise_floor.py --runs 5 --model gpt2 --layer 6 --seed 42
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

# Import from the extractor
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ear_activation_extractor_v2 import extract, DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# Noise Floor Runner
# ---------------------------------------------------------------------------

def run_noise_floor(
    model_name: str = "gpt2",
    prompt: str = DEFAULT_PROMPT,
    layer: int = 6,
    n_runs: int = 5,
    seed: int = 42,
    output_dir: Path = Path("output/noise_floor"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  IRIS Noise Floor Experiment")
    print(f"  Model  : {model_name}")
    print(f"  Prompt : {prompt}")
    print(f"  Layer  : {layer}")
    print(f"  Runs   : {n_runs}")
    print(f"  Seed   : {seed} (fixed across all runs)")
    print(f"{'='*60}\n")

    constellations = []
    timelines      = []

    for i in range(n_runs):
        print(f"\n--- Run {i+1} of {n_runs} ---")
        timeline, constellation = extract(
            model_name=model_name,
            prompt=prompt,
            layer=layer,
            output_dir=output_dir,
            seed=seed,
            device=device,
        )
        constellations.append(constellation)
        timelines.append(timeline)

    # --- Extract arrays across runs ---
    tokens     = constellations[0]["token_labels"]
    n_tokens   = len(tokens)

    # Filter BOS
    SPECIAL = {"<|endoftext|>", "<|BOS|>", "<s>", "</s>"}
    mask    = [t not in SPECIAL for t in tokens]
    tokens_clean = [t.replace("Ġ", "").strip() or t for t, m in zip(tokens, mask) if m]

    # PCA coords: [runs, seq, 2] → masked
    pca_all = np.array([c["pca_coords"] for c in constellations])  # [runs, seq, 2]
    pca_all = pca_all[:, mask, :]                                   # [runs, masked_seq, 2]

    # RMS: [runs, masked_seq]
    rms_all = np.array([
        [t["rms"] for t, m in zip(c["tokens"], mask) if m]
        for c in constellations
    ])

    # Centroid norms: [runs]
    centroid_norms = np.array([
        np.linalg.norm(c["centroid"]) for c in constellations
    ])

    # --- Compute stability metrics ---
    pc1_std = pca_all[:, :, 0].std(axis=0)   # [masked_seq]
    pc2_std = pca_all[:, :, 1].std(axis=0)   # [masked_seq]
    rms_std = rms_all.std(axis=0)            # [masked_seq]

    pc1_mean = pca_all[:, :, 0].mean(axis=0)
    pc2_mean = pca_all[:, :, 1].mean(axis=0)

    # Pairwise cosine distances between full constellation vectors
    # Flatten each run's PCA coords to a single vector
    flat_constellations = pca_all.reshape(n_runs, -1)  # [runs, masked_seq*2]
    pairwise_distances  = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            a, b = flat_constellations[i], flat_constellations[j]
            cos_d = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            pairwise_distances.append(cos_d)

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"  IRIS NOISE FLOOR RESULTS")
    print(f"  {n_runs} runs | Fixed seed {seed} | {model_name} L{layer}")
    print(f"{'='*60}\n")

    print(f"Centroid norm across runs:")
    print(f"  Mean: {centroid_norms.mean():.4f}  Std: {centroid_norms.std():.6f}  "
          f"Min: {centroid_norms.min():.4f}  Max: {centroid_norms.max():.4f}")

    print(f"\nPairwise cosine distance between constellation runs:")
    print(f"  Mean: {np.mean(pairwise_distances):.8f}  "
          f"Max:  {np.max(pairwise_distances):.8f}")
    if np.max(pairwise_distances) < 1e-6:
        print(f"  ✓ PERFECTLY STABLE — numerical precision only")
    elif np.max(pairwise_distances) < 1e-4:
        print(f"  ✓ STABLE — variance within acceptable noise floor")
    else:
        print(f"  ⚠ VARIANCE DETECTED — investigate before using as drift baseline")

    print(f"\nPer-token PC1 standard deviation across runs:")
    print(f"{'Token':<14} {'PC1 mean':>10} {'PC1 std':>10} {'PC2 mean':>10} {'PC2 std':>10} {'RMS std':>10}")
    print("-" * 66)
    for i, tok in enumerate(tokens_clean):
        print(f"{tok:<14} {pc1_mean[i]:>10.3f} {pc1_std[i]:>10.6f} "
              f"{pc2_mean[i]:>10.3f} {pc2_std[i]:>10.6f} {rms_std[i]:>10.6f}")

    # --- Save results ---
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_name.replace("/", "_")
    results = {
        "meta": {
            "experiment":    "noise_floor",
            "model":         model_name,
            "prompt":        prompt,
            "layer":         layer,
            "n_runs":        n_runs,
            "seed":          seed,
            "timestamp":     timestamp,
        },
        "centroid_norm": {
            "mean": float(centroid_norms.mean()),
            "std":  float(centroid_norms.std()),
            "min":  float(centroid_norms.min()),
            "max":  float(centroid_norms.max()),
        },
        "pairwise_cosine_distance": {
            "mean": float(np.mean(pairwise_distances)),
            "max":  float(np.max(pairwise_distances)),
            "all":  [float(d) for d in pairwise_distances],
        },
        "per_token": [
            {
                "token":    tok,
                "pc1_mean": float(pc1_mean[i]),
                "pc1_std":  float(pc1_std[i]),
                "pc2_mean": float(pc2_mean[i]),
                "pc2_std":  float(pc2_std[i]),
                "rms_std":  float(rms_std[i]),
            }
            for i, tok in enumerate(tokens_clean)
        ],
        "verdict": (
            "perfectly_stable" if np.max(pairwise_distances) < 1e-6
            else "stable" if np.max(pairwise_distances) < 1e-4
            else "variance_detected"
        )
    }

    out_path = output_dir / f"noise_floor_{model_slug}_L{layer}_seed{seed}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Verdict: {results['verdict'].upper()}")
    print(f"  Results saved → {out_path}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRIS Noise Floor Experiment")
    parser.add_argument("--model",  default="gpt2")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--layer",  default=6,   type=int)
    parser.add_argument("--runs",   default=5,   type=int)
    parser.add_argument("--seed",   default=42,  type=int)
    parser.add_argument("--output", default="output/noise_floor")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_noise_floor(
        model_name=args.model,
        prompt=args.prompt,
        layer=args.layer,
        n_runs=args.runs,
        seed=args.seed,
        output_dir=Path(args.output),
        device=args.device,
    )
