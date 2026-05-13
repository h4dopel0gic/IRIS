# IRIS
### Internal Representation and Insight System

**IRIS is an alignment monitoring instrument for language models.**

It extracts the internal geometric state of a neural network — the relational structure of meaning as it exists in activation space — and makes it visible, measurable, and comparable across models without examining any output.

> *The iris of an eye is a constellation — a unique geometric arrangement that discloses identity. IRIS reads the constellation of a model's inner state.*

---

## What IRIS Does

IRIS extracts six geometric properties from the residual stream of a language model at a chosen layer:

| Property | What It Discloses |
|---|---|
| RMS magnitude | Representational energy per token |
| Layer delta magnitude | Where meaning crystallises across layers |
| Attention entropy | Confidence and focus of attention |
| Sequence centroid | Absolute position of the constellation |
| Cosine distance from mean | Token deviation from sequence centre |
| PCA projection (PC1/PC2) | Relational geometry — the constellation |

The **constellation** — the PCA projection of all token activations — is the core output. It is not a property of any single token. It is a property of the whole sequence: how the model has arranged meaning relative to itself internally.

---

## Key Findings

**Finding 1 — Visual similarity does not predict internal proximity**
Models producing similar output can occupy geometrically distant internal regions. Output-level monitoring cannot detect representational divergence.

**Finding 4 — Tokenisation shapes conceptual geometry**
'Gracious' neighbours 'God' in GPT-2's constellation — mercy and divinity are geometrically partnered without supervision. In Mistral-7B, 'gracious' is tokenised as 'gr' + 'acious' and the pairing dissolves. Tokenisation is an architectural determinant of conceptual geometry.

**Finding 5 — Two processing regimes visible within one model**
GPT-2 Layer 6 shows semantic organisation. Layer 11 shows output preparation. The constellation captures the transition geometrically.

**Finding 6 — Instruction tuning encodes geometric valence separation**
Base TinyStories-33M shows near-zero internal separation between positive and negative conceptual space (cosine distance: 0.000160). After instruction tuning: 0.184953 — a 1,156× increase. The tuned model has lost geometric neutrality. IRIS detected this without examining any output.

**Noise floor: perfectly stable**
Five repeated extractions under fixed seed produce zero variance at numerical precision. Any measured geometric distance is signal, not noise.

---

## The Alignment Argument

> *Faithfulness is measured by latent distance, not visual resemblance.*

Current alignment evaluation operates at the output level. IRIS demonstrates empirically — in both diffusion and language model domains — that this is insufficient. A model can produce acceptable output while its internal geometry has drifted significantly from a trusted reference state.

IRIS monitors the constellation: the relational geometry of a model's internal representations. Drift is measured as geometric distance from a reference, not as output divergence.

---

## Files

| File | Description |
|---|---|
| `ear_activation_extractor_v2.py` | Six-property extraction via TransformerLens |
| `ear_constellation_viz.py` | Three-panel visualiser + normalised multi-model overlay |
| `iris_noise_floor.py` | Stability experiment — N-run variance measurement |
| `iris_alignment_comparison.py` | Base vs tuned comparison across valence classes |
| `ear_lens_app_v03.py` | Gradio interface — IRIS tabs + EAR tabs |
| `IRIS_Research_State_v02.docx` | Full research state document |

---

## Environment

```bash
conda activate ear_lens
pip install transformer-lens transformers accelerate torch gradio scikit-learn numpy matplotlib
```

Tested on Windows 11, CUDA 12.1, Python 3.10.

---

## Usage

**Extract a constellation:**
```bash
python ear_activation_extractor_v2.py --model gpt2 --layer 6 --prompt "Your prompt here"
```

**Visualise:**
```bash
python ear_constellation_viz.py --input output/activations/constellation_*.json
```

**Compare two models:**
```bash
python ear_constellation_viz.py --compare constellation_a.json constellation_b.json --labels "Model A" "Model B" --normalise
```

**Run the Gradio app:**
```bash
python ear_lens_app_v03.py
```

---

## Relationship to EAR

IRIS forked from [EAR](https://github.com/h4dopel0gic/EAR) on May 8, 2026. EAR is the audio-to-diffusion pipeline that produced the empirical foundation for this work. IRIS is the observation instrument that grew out of it.

EAR heard the outside. IRIS sees the inside.

The interface between them is the timeline JSON format — IRIS writes it, EAR reads it.

---

## Sakin.AI — Safina Ecosystem
Field Architect: Tobias Stevenson

*What is unseen is not empty. It is merely unobserved.*
