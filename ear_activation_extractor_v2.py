"""
EAR-Lens — Extended Activation Extractor v2
Sakin.AI / Safina Ecosystem
Field Architect: Tobias Stevenson

Extracts six geometric properties from LLM residual stream via TransformerLens.
Designed as the principled input layer for Bridge 4 translator training.

Properties extracted per token:
  1. RMS magnitude          — representational energy / salience
  2. Layer delta magnitude  — where meaning crystallises across layers
  3. Attention entropy      — confidence / focus of attention
  4a. Sequence centroid     — absolute position of constellation in activation space
  4b. Cosine distance       — each token's deviation from the centroid
  5. PCA projection         — relational geometry (PC1, PC2) — the constellation

Output: timeline_llm_v2.json — extended format, backward compatible with EAR pipeline
        constellation.json   — PCA coordinates + centroid for visualisation
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
import transformer_lens
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "mistral-7b",   # will require sufficient VRAM
]

DEFAULT_LAYER = 6       # mid-network — semantics are forming here
DEFAULT_PROMPT = "In the name of God, the most gracious"
DEFAULT_OUTPUT_DIR = Path("output/activations")


# ---------------------------------------------------------------------------
# Property Extractors
# ---------------------------------------------------------------------------

def extract_rms(resid: torch.Tensor) -> np.ndarray:
    """
    Property 1: RMS magnitude per token.
    resid: [seq, d_model]
    Returns: [seq] float array
    """
    return resid.norm(dim=-1).cpu().numpy()


def extract_layer_deltas(cache, layer: int, n_layers: int) -> np.ndarray:
    """
    Property 2: Layer delta magnitude per token across all layers up to `layer`.
    Shows where the model does the most work on each token.
    Returns: [seq] float — magnitude of change AT the specified layer.
             Also returns [n_layers, seq] full profile if needed.
    """
    if layer == 0:
        # No previous layer to diff against
        delta = cache['resid_post', 0]
        return delta.norm(dim=-1).squeeze().cpu().numpy()

    deltas = []
    for l in range(1, layer + 1):
        post = cache['resid_post', l].squeeze()   # [seq, d_model]
        prev = cache['resid_post', l - 1].squeeze()
        delta = (post - prev).norm(dim=-1)         # [seq]
        deltas.append(delta.cpu().numpy())

    delta_profile = np.stack(deltas, axis=0)       # [layer, seq]
    delta_at_layer = deltas[-1]                    # magnitude at chosen layer

    return delta_at_layer, delta_profile


def extract_attention_entropy(cache, layer: int) -> np.ndarray:
    """
    Property 3: Attention entropy per token, averaged across heads.
    Low entropy  = confident, focused attention  → maps to high CFG
    High entropy = uncertain, diffuse attention  → maps to low CFG
    Returns: [seq] float array
    """
    attn = cache['pattern', layer].squeeze()       # [heads, seq, seq]
    p = attn + 1e-9                                # numerical stability
    entropy = -(p * p.log()).sum(dim=-1)            # [heads, seq]
    mean_entropy = entropy.mean(dim=0).cpu().numpy()  # [seq]
    return mean_entropy


def extract_centroid_and_deviation(resid: torch.Tensor):
    """
    Property 4a: Sequence centroid — absolute position of constellation.
    Property 4b: Cosine distance from centroid per token.

    resid: [seq, d_model]
    Returns:
        centroid: [d_model] numpy array — the centre of mass
        cos_dist: [seq] numpy array — each token's distance from centre
    """
    centroid = resid.mean(dim=0)                   # [d_model]

    cos_sim = torch.nn.functional.cosine_similarity(
        resid,
        centroid.unsqueeze(0).expand_as(resid),
        dim=-1
    )                                              # [seq]

    cos_dist = (1 - cos_sim).cpu().numpy()         # distance, not similarity

    return centroid.cpu().numpy(), cos_dist


def extract_pca_projection(resid: torch.Tensor, n_components: int = 2):
    """
    Property 5: PCA projection — the constellation.
    Projects token activations onto top n_components principal axes.
    Preserves relational geometry — distances between tokens meaningful.

    resid: [seq, d_model]
    Returns:
        coords:          [seq, n_components] — 2D coordinates per token
        explained_var:   [n_components] — variance explained by each component
        pca:             fitted PCA object (for cross-model alignment later)
    """
    resid_np = resid.cpu().numpy()                 # [seq, d_model]

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(resid_np)           # [seq, 2]

    return coords, pca.explained_variance_ratio_, pca


# ---------------------------------------------------------------------------
# EAR Parameter Mapping
# ---------------------------------------------------------------------------

def map_to_ear_params(
    rms: np.ndarray,
    attn_entropy: np.ndarray,
    cos_dist: np.ndarray,
    delta_mag: np.ndarray,
) -> list[dict]:
    """
    Maps extracted properties to EAR diffusion parameters.

    Mapping rationale:
      RMS           → denoise strength   (energy = how much to sculpt)
      Attn entropy  → CFG scale          (confidence = guidance strength)
      Cos distance  → latent offset      (deviation = distance from centre)
      Layer delta   → step weight        (work done = resolution needed)

    All values normalised to their natural diffusion ranges.
    """

    def norm(arr, lo, hi):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.full_like(arr, (lo + hi) / 2.0)
        return lo + (arr - mn) / (mx - mn) * (hi - lo)

    denoise   = norm(rms, 0.30, 0.85)          # higher energy → more denoising
    cfg       = norm(1 - attn_entropy, 4.0, 9.0)  # lower entropy → higher CFG
    offset    = norm(cos_dist, 0.0, 1.0)        # deviation from centre
    step_w    = norm(delta_mag, 0.5, 1.5)       # layer work → step weight

    params = []
    for i in range(len(rms)):
        params.append({
            "denoise":    float(denoise[i]),
            "cfg":        float(cfg[i]),
            "offset":     float(offset[i]),
            "step_weight": float(step_w[i]),
        })
    return params


# ---------------------------------------------------------------------------
# Main Extraction Pipeline
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Fix all random sources for reproducible PCA and model inference."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract(
    model_name: str = "gpt2",
    prompt: str = DEFAULT_PROMPT,
    layer: int = DEFAULT_LAYER,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EAR-Lens Extended Extractor v2")
    print(f"  Model  : {model_name}")
    print(f"  Prompt : {prompt}")
    print(f"  Layer  : {layer}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # --- Fix seed for reproducibility ---
    set_seed(seed)
    print(f"Seed: {seed}")

    # --- Load model ---
    # Large models (7B+) need float16 to fit in 16GB VRAM.
    # TransformerLens defaults to float32 — override for anything non-GPT2-small.
    small_models = {"gpt2", "gpt2-small"}
    dtype = torch.float32 if model_name in small_models else torch.float16

    print(f"Loading model via TransformerLens... (dtype={dtype})")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )
    model.eval()

    # --- Tokenise ---
    tokens = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)
    seq_len = tokens.shape[1]
    n_layers = model.cfg.n_layers
    actual_layer = min(layer, n_layers - 1)

    print(f"Tokens ({seq_len}): {token_strs}")
    print(f"Model layers: {n_layers} | Extracting at layer: {actual_layer}\n")

    # --- Run with cache ---
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    resid = cache['resid_post', actual_layer].squeeze()  # [seq, d_model]

    # --- Extract all properties ---
    print("Extracting properties...")

    rms                      = extract_rms(resid)
    delta_at_layer, delta_profile = extract_layer_deltas(cache, actual_layer, n_layers)
    attn_entropy             = extract_attention_entropy(cache, actual_layer)
    centroid, cos_dist       = extract_centroid_and_deviation(resid)
    pca_coords, explained_var, pca_obj = extract_pca_projection(resid)

    # --- Map to EAR params ---
    ear_params = map_to_ear_params(rms, attn_entropy, cos_dist, delta_at_layer)

    # --- Build token table ---
    token_table = []
    for i, tok in enumerate(token_strs):
        token_table.append({
            "index":        i,
            "token":        tok,
            "rms":          float(rms[i]),
            "delta_mag":    float(delta_at_layer[i]),
            "attn_entropy": float(attn_entropy[i]),
            "cos_dist":     float(cos_dist[i]),
            "pc1":          float(pca_coords[i, 0]),
            "pc2":          float(pca_coords[i, 1]),
            "ear":          ear_params[i],
        })

    # --- Print token table ---
    print(f"\n{'Token':<12} {'RMS':>8} {'Delta':>8} {'Entropy':>9} {'CosDist':>9} {'PC1':>8} {'PC2':>8} {'CFG':>6} {'Denoise':>8}")
    print("-" * 85)
    for t in token_table:
        print(
            f"{t['token']:<12} "
            f"{t['rms']:>8.3f} "
            f"{t['delta_mag']:>8.3f} "
            f"{t['attn_entropy']:>9.4f} "
            f"{t['cos_dist']:>9.4f} "
            f"{t['pc1']:>8.3f} "
            f"{t['pc2']:>8.3f} "
            f"{t['ear']['cfg']:>6.3f} "
            f"{t['ear']['denoise']:>8.4f}"
        )

    # --- Build timeline_llm_v2.json (EAR pipeline compatible) ---
    timeline = {
        "meta": {
            "model":            model_name,
            "prompt":           prompt,
            "layer":            actual_layer,
            "n_layers":         n_layers,
            "d_model":          model.cfg.d_model,
            "seq_len":          seq_len,
            "pca_explained_var": explained_var.tolist(),
            "centroid_norm":    float(np.linalg.norm(centroid)),
            "extractor_version": "v2",
            "seed": seed,
        },
        "tokens": token_table,
    }

    # --- Build unique filenames: model_layer_timestamp ---
    from datetime import datetime
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_name.replace("/", "_").replace(" ", "_")
    run_id     = f"{model_slug}_L{actual_layer}_seed{seed}_{timestamp}"

    timeline_path = output_dir / f"timeline_{run_id}.json"
    with open(timeline_path, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"\nTimeline saved → {timeline_path}")

    # --- Build constellation.json (for visualisation + translator training) ---
    constellation = {
        "meta": timeline["meta"],
        "centroid": centroid.tolist(),
        "pca_coords": pca_coords.tolist(),
        "token_labels": token_strs,
        "delta_profile": delta_profile.tolist(),   # full [layer, seq] matrix
        "explained_variance": explained_var.tolist(),
        "tokens": token_table,                     # embed token data for viz
    }

    constellation_path = output_dir / f"constellation_{run_id}.json"
    with open(constellation_path, "w") as f:
        json.dump(constellation, f, indent=2)
    print(f"Constellation saved → {constellation_path}")

    print(f"\n{'='*60}")
    print(f"  Extraction complete.")
    print(f"  Run ID  : {run_id}")
    print(f"  Timeline: {timeline_path.name}")
    print(f"  Constell: {constellation_path.name}")
    print(f"  PCA variance explained: PC1={explained_var[0]:.1%}  PC2={explained_var[1]:.1%}")
    print(f"  Centroid norm: {np.linalg.norm(centroid):.3f}")
    print(f"{'='*60}\n")

    return timeline, constellation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAR-Lens Extended Activation Extractor v2")
    parser.add_argument("--model",  default="gpt2",                          help="TransformerLens model name")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,                  help="Input prompt")
    parser.add_argument("--layer",  default=DEFAULT_LAYER, type=int,         help="Residual stream layer to extract from")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),         help="Output directory")
    parser.add_argument("--seed",   default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    extract(
        model_name=args.model,
        prompt=args.prompt,
        layer=args.layer,
        output_dir=Path(args.output),
        seed=args.seed,
        device=args.device,
    )
