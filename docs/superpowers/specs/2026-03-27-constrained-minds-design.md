# Constrained Minds — Design Specification
**Date:** 2026-03-27
**Status:** Approved
**Repo:** https://github.com/sub-surface/constrained-minds

---

## 1. Research Claim & Paper Arc

### Central Claim

> *When language models are trained on maximally diverse and constrained corpora, a Universal Sparse Autoencoder reveals a bimodal feature structure: a set of corpus-invariant "platonic" features shared across all models, and a set of corpus-exclusive features that encode the unique conceptual world of each training domain. The geometry of shared features converges predictably; the exclusive features do not.*

This is novel because every prior universality study (Platonic Representation Hypothesis, arXiv:2405.07987; Universal SAE, arXiv:2502.03714; SAE Universal Feature Spaces, arXiv:2410.06981) tests models trained on similar internet-scale data. We test universality across models with **deliberately alien training corpora** — Victorian novels, continental philosophy, synthetic children's stories, pure medical abstracts, distilled reasoning traces.

### Key Literature

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| Platonic Representation Hypothesis (Huh et al., ICML 2024) | 2405.07987 | Foundational convergence claim |
| Universal Sparse Autoencoders (ICML 2025) | 2502.03714 | Primary architectural blueprint |
| SAEs Reveal Universal Feature Spaces (2024) | 2410.06981 | Empirical universality evidence; SVCCA validation method |
| Cross-Architecture Model Diffing with Crosscoders (2025) | 2602.11729 | Pairwise diff method; DFC architecture |
| Towards Monosemanticity (Anthropic, 2023) | — | SAE foundations |
| Scaling Monosemanticity (Anthropic, 2024) | — | Proves SAEs scale; feature steering |
| Linear Representation Hypothesis (Park et al., ICML 2024) | 2311.03658 | Theoretical basis for geometric analysis |
| Geometry of Categorical/Hierarchical Concepts (Park et al., ICLR 2025 oral) | 2406.01506 | Simplex/polytope fitting method |
| Sparse Crosscoders for Model Diffing (Anthropic, 2024) | — | Crosscoder methodology |
| SAEBench (ICML 2025) | 2503.09532 | SAE architecture selection guide |
| Features at Convergence Theorem (2025) | 2507.05644 | Theoretical grounding for universality |
| How Many Features Can a LM Store (2025) | 2602.11246 | Capacity bounds; guides expansion factor |
| Group-SAE: Efficient Training (ACL 2025) | 2410.21508 | 3× speedup via layer grouping |

### Paper Phases → Research Phases

| Phase | Research Question | Paper Section | Key Output |
|-------|-------------------|---------------|------------|
| 1 | Unified model loading & inference | §2 Methods | Working pipeline |
| 2 | Do constrained-corpus models encode concepts differently? | §3 Surface Representations | Figures 1–3 |
| 3 | What features are universal vs. corpus-exclusive? | §4 Universal Feature Discovery | Figures 4–7 |
| 4 | Do per-model SAEs and crosscoders agree? | §5 Cross-Validation | Figures 8–9 |
| 5 | Does universal feature geometry converge? | §6 Geometric Analysis | Figures 10–11 |
| 6 | Are the differences causally load-bearing? | §7 Causal Verification | Figures 12–14 |

---

## 2. Model Zoo

Eight models chosen to maximise corpus diversity across a single interpretable axis: *how constrained and alien is the training data?*

| ID | Model | HF Handle | Size | Corpus | Architecture | Key Pair |
|----|-------|-----------|------|--------|--------------|----------|
| M1 | Mr. Chatterbox | `tventurella/mr_chatterbox_model` | 340M | Victorian books 1837–1899 (28K books, 2.93B tokens) | nanochat (custom) | M1↔M8 (same era) |
| M2 | philosophy-mistral | `Heralax/philosophy-mistral` | 7B | 5 philosophy books: Russell, Nietzsche ×2, Machiavelli, Locke | Mistral 7B fine-tune | M2↔M3 (philosophy pair) |
| M3 | deleuze-qwen | `wisdomfunction/deleuze-qwen-1.5b` | 1.5B | Deleuze's complete works (A Thousand Plateaus, Diff & Rep, Anti-Oedipus, etc.) | DeepSeek-R1-Distill-Qwen LoRA | M2↔M3 |
| M4 | TinyStories-28M | `roneneldan/TinyStories-28M` | 28M | Synthetic children's stories (GPT-4 generated, restricted vocabulary) | GPT-Neo | Constrained-synthetic pole |
| M5 | BioMedLM | `stanford-crfm/BioMedLM` | 2.7B | PubMed abstracts only | GPT-2 style | M5↔M7 (domain vs. general) |
| M6 | Qwen3.5-4B-Opus-Distilled | `Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2` | 4B | 14K+ Claude Opus reasoning traces (reasoning style, not content) | Qwen3.5 SFT+LoRA | M6↔M7 (process vs. content) |
| M7 | pythia-410m | `EleutherAI/pythia-410m-deduped` | 410M | The Pile — modern diverse web | GPT-NeoX | General baseline |
| M8 | gothic-gpt2 | `Dwaraka/PROJECT_GUTENBERG_GOTHIC_FICTION_TEXT_GENERATION_gpt2` | ~125M | Victorian Gothic fiction (Gutenberg) | GPT-2 fine-tune | M1↔M8 |

**Local models** (GGUF in LM Studio, loaded via `llama-cpp-python`):
- `ortegaalfredo/MechaEpstein-8000` — Qwen3 8B, alignment-modified (optional: alignment feature analysis)
- `wazuki/EpsteinFilez-8B` — Qwen3 8B, alignment-modified (optional: same)

**Controlled pairs for targeted analysis:**
- **Same era, different curation:** M1 ↔ M8 → tests whether corpus geometry is curation-sensitive within a period
- **Same domain, different scope:** M2 ↔ M3 → 5 books vs. one author's complete works
- **Same scale, different corpus:** M1 ↔ M7 → closest parameter count match (340M vs 410M); isolates corpus effect at similar capacity
- **Domain science vs. domain humanities:** M5 ↔ M2/M3 → modern constrained vs. historical constrained
- **Content-trained vs. process-trained:** M6 ↔ M7 → is reasoning style encoded differently from world knowledge?

---

## 3. Architecture & Components

### Repository Structure

```
constrained-minds/
├── constrained_minds/
│   ├── models/           # Unified model loading & inference interface
│   ├── activations/      # Extraction, caching, normalisation
│   ├── sae/              # Universal SAE + per-model SAE + crosscoder
│   ├── analysis/         # Logit lens, geometry, patching, register metric
│   └── viz/              # All figure generation (publication-quality)
├── notebooks/
│   ├── 01_model_zoo.ipynb              # Load all models, sanity checks, sample generation
│   ├── 02_surface_representations.ipynb  # Logit lens, nearest-neighbour geometry
│   ├── 03_universal_sae.ipynb          # Train + inspect Universal SAE
│   ├── 04_validation.ipynb             # Per-model SAEs, cross-validation
│   ├── 05_geometry.ipynb               # Simplex/polytope analysis
│   ├── 06_causal.ipynb                 # Activation patching, steering
│   └── 07_figures.ipynb                # Final publication figures
├── scripts/
│   ├── extract_activations.py          # CLI: cache activations for all models
│   ├── train_universal_sae.py          # CLI: train the Universal SAE
│   └── train_per_model_saes.py         # CLI: train per-model SAEs for validation
├── data/
│   ├── probe_sentences/                # Curated seed concept sentences
│   └── activations/                    # Cached activation tensors (gitignored)
├── results/
│   ├── checkpoints/                    # SAE weights (gitignored)
│   └── figures/                        # Saved publication figures (committed)
└── docs/superpowers/specs/             # This document
```

### The Five Layers

**`models/`** — A unified `ModelWrapper` interface normalising across all 8 architectures. Every model exposes identical methods:
- `encode(text: str) → token_ids`
- `forward(token_ids, layer: int) → activations`
- `decode_logits(hidden_state) → logit_distribution`
- `get_layer_count() → int`
- `get_d_model() → int`

Sub-classes: `NanochatWrapper` (handles `.pt` loading for M1), `HuggingFaceWrapper` (M2–M7), `GGUFWrapper` (local LM Studio models via `llama-cpp-python`).

**`activations/`** — Extraction and caching. Runs all models over the shared neutral corpus, saves per-model per-layer activation tensors to disk. Handles:
- Normalisation to `√(d_model)` mean norm (crosscoder convention)
- Per-model projection to shared `d_concept = 1152` space
- Stratified sampling to ensure no model has home-field advantage

**`sae/`** — Three classes:
- `UniversalSAE`: per-model encoders → shared latent (8192 features, TopK k=32) → per-model decoders. Trained by encoding one model and decoding through all simultaneously. Built on `YorkUCVIL/UniversalSAE` / `KempnerInstitute/overcomplete`.
- `PerModelSAE`: standard SAELens-backed SAE per model for cross-validation.
- `Crosscoder`: wraps `oli-clive-griffin/crosscode` for pairwise DFC analysis.

**`analysis/`** — Five modules:
- `logit_lens.py`: layer-by-layer unembedding traces
- `geometry.py`: PCA/UMAP, nearest-neighbour, simplex fitting (Park et al. 2024)
- `patching.py`: activation patching across models at matched layer checkpoints
- `register_metric.py`: domain register score (perplexity ratio between models)
- `autointerp.py`: automated feature labelling via Claude API (claude-haiku-4-5)

**`viz/`** — All 14 figures as pure functions: `(analysis_output) → matplotlib.Figure`. Enforces consistent style (fonts, DPI, colour palette) across the whole paper. No analysis logic.

---

## 4. Data Flow & Training Procedure

### 4.1 Shared Neutral Corpus

~200K tokens total, ~25K per stratum. No stratum favours any zoo member.

| Stratum | Source | Rationale |
|---------|--------|-----------|
| Contemporary news | CC-News 2024 | No model trained on this specific slice |
| Modern fiction | PG-19 held-out | Literary but post-1900 |
| Scientific abstracts (non-medical) | arXiv CS/physics 2024 | Neutral to BioMedLM's domain |
| Philosophical dialogue | Stanford Encyclopedia of Philosophy | Neutral to single-author philosophy models |
| Children's educational text | Wikipedia Simple English | Neutral to TinyStories' synthetic domain |
| Verse / structured poetry | Public domain post-1900 | Covers rhythmic/formal register |

### 4.2 Layer Selection

Activations extracted at **early (~17%), middle (~50%), late (~83%)** of each model's depth:

| Model | Total layers | Early | Middle | Late |
|-------|-------------|-------|--------|------|
| M1 Mr. Chatterbox | 18 | 3 | 9 | 15 |
| M2 philosophy-mistral | 32 | 5 | 16 | 27 |
| M3 deleuze-qwen | 28 | 5 | 14 | 23 |
| M4 TinyStories | 6 | 1 | 3 | 5 |
| M5 BioMedLM | 24 | 4 | 12 | 20 |
| M6 Qwen-Opus-Distilled | 36 | 6 | 18 | 30 |
| M7 pythia-410m | 24 | 4 | 12 | 20 |
| M8 gothic-gpt2 | 12 | 2 | 6 | 10 |

### 4.3 Normalisation & Dimension Alignment

1. Per-model activations normalised to `√(d_model)` mean norm
2. Per-model linear projection: `W_proj_i : d_model_i → d_concept = 1152`
3. Shared concept space dimension set to 1152 (Mr. Chatterbox's d_model — most constrained model sets floor)

### 4.4 Universal SAE Architecture

```
For model i with activations A_i ∈ R^(d_model_i):

  Projection:   H_i = W_proj_i · A_i              # d_model_i → 1152
  Encode:       Z   = TopK(W_enc · H_i + b_enc)   # 1152 → 8192 features, k=32
  Decode_all:   Â_j = W_dec_j · Z + b_dec_j       # 8192 → 1152 → d_model_j  ∀j

  Loss = Σ_j ||A_j - Â_j||²  (all M decoders per step)
```

**Training hyperparameters:**
- Dictionary size: 8192 (expansion factor ~7× relative to d_concept=1152)
- TopK: k=32 active features per token
- Batch: 2048 tokens per step, one model sampled per step
- Optimiser: AdamW — lr=2e-4, β1=0.9, β2=0.999, weight_decay=1e-4
- Steps: ~50K (early stop on R² plateau)
- Hardware: RTX 2060 (6GB) for SAE; all models CPU-offloaded with pre-cached activations

**Feature classification post-training:**
- **Firing Entropy (FE):** `H = -Σ_i p_i log p_i`, normalised ∈ [0,1], where `p_i` = fraction of feature's total fires from model i
  - FE > 0.85 → **universal**
  - FE < 0.20 → **corpus-exclusive**
  - 0.20–0.85 → **partially shared**
- **Co-Fire Proportion (CFP):** fraction of fires where ≥4 models activate simultaneously

### 4.5 Cross-Validation Strategy

1. **Per-model SAEs** (SAELens): train independently per model, measure SVCCA alignment with Universal SAE model-specific features. Target: >0.65 SVCCA score (following arXiv:2410.06981 baseline of 0.70 for same-family models)
2. **Pairwise DFC crosscoders** (oli-clive-griffin/crosscode): for pairs M1↔M7, M2↔M3, M6↔M7 — partitions features into exclusive-A, exclusive-B, shared as independent universality check
3. **Held-out reconstruction:** 20% corpus reserved; R² degradation >10% from in-distribution → overfitting flag

---

## 5. Analysis Methods & Figures

### 5.1 Surface Representations

**Figure 1 — Logit Lens Grid**
8×3 grid (models × layers). Each cell: top-5 predicted tokens with probabilities for fixed probe sentence *"The nature of the mind is..."*. Shows Victorian register lock-in by layer 9 in M1 vs. divergence in M5/M7.
- Function: `viz.logit_lens_grid(traces: dict[model_id, LayerTrace]) → Figure`

**Figure 2 — Concept Neighbourhood Comparison**
Structured table: 40 seed concepts × 8 models. Each cell: top-3 nearest neighbours. Colour-coded by semantic field (theological, scientific, literary, medical, philosophical). Reveals "science" → {Providence, natural philosophy} in M1 vs. {rhizome, multiplicity} in M3 vs. {clinical trial, dosage} in M5.
- Seed concepts include: *science, nature, feeling, character, progress, empire, mind, body, truth, reason, evolution, hysteria, machine, virtue, moral, soul, knowledge, death, love, time, power, order, system, law, will, form, matter, spirit, voice, memory, desire, labour, class, race, God, natural, civil, common, free, human*
- Function: `viz.concept_neighbourhood_table(neighbours: dict[model_id, dict[concept, list[str]]]) → Figure`

**Figure 3 — UMAP Concept Map**
2D UMAP of 40×8=320 concept embeddings. Colour = model, shape = concept. Tight cross-model clustering → Platonic convergence; scatter → corpus-specific encoding.
- Function: `viz.umap_concept_map(embeddings: dict[model_id, dict[concept, ndarray]]) → Figure`

### 5.2 Universal Feature Discovery

**Figure 4 — Feature Entropy Histogram** *(paper headline figure)*
Histogram of FE across all 8192 features. Vertical threshold lines at 0.20 and 0.85. Inset pie: universal/partial/exclusive counts and percentages.
- Function: `viz.entropy_histogram(fe_scores: ndarray) → Figure`

**Figure 5 — Universal Feature Gallery**
12 highest-FE features (FE>0.95). Per feature: auto-generated label, maximally-activating passage from 4 representative models side by side. Demonstrates same feature fires on temporal sequencing, causal connectives, etc. across Victorian prose, philosophy, medical text, children's stories.
- Function: `viz.feature_gallery(features: list[FeatureProfile], mode='universal') → Figure`

**Figure 6 — Exclusive Feature Gallery**
Top-5 exclusive features per model (8 panels). Labels and example activations. Expected content: class markers/Providence (M1), dosage/anatomy (M5), rhizome/deterritorialisation (M3), reasoning scaffolds (M6).
- Function: `viz.feature_gallery(features: list[FeatureProfile], mode='exclusive') → Figure`

**Figure 7 — Corpus Exclusivity Heatmap**
Feature×model heatmap, top 200 features by variance. Hierarchically clustered on both axes. Shows model-specific activation blocks as visual proof of corpus ownership.
- Function: `viz.exclusivity_heatmap(activations: ndarray, model_ids: list[str]) → Figure`

### 5.3 Cross-Validation

**Figure 8 — Validation Alignment Matrix**
8×3 matrix (models × layers). Each cell: SVCCA score between per-model SAE and Universal SAE. Red→green colour gradient. Failing cells flagged for investigation.
- Function: `viz.validation_matrix(svcca_scores: ndarray, model_ids: list[str]) → Figure`

**Figure 9 — Crosscoder Venn Diagrams**
Three Venn diagrams for pairs M1↔M7, M2↔M3, M6↔M7. Circle size ∝ feature count; overlap = shared features; top-5 exclusive feature labels annotated per circle.
- Function: `viz.crosscoder_venns(crosscoder_results: list[CrosscoderResult]) → Figure`

### 5.4 Geometric Analysis

**Figure 10 — Concept Geometry Comparison**
Side-by-side 3D PCA plots for emotional-states simplex (joy, grief, anger, fear, love) in M1, M7, M2. Tests whether simplex vertex positions/angles are consistent across models under rotation. Platonic prediction: same shape, different orientation.
- Function: `viz.simplex_comparison(concept_embeddings: dict[model_id, ndarray], concepts: list[str]) → Figure`

**Figure 11 — Hierarchy Orthogonality Scores**
Bar chart: cosine similarity between parent and child concept directions for 10 hierarchical pairs (e.g. animal→mammal→dog, emotion→grief, knowledge→science) across all 8 models. Park et al. prediction: near zero. Deviations quantify idiosyncratic geometry.
- Function: `viz.orthogonality_bars(scores: dict[model_id, dict[pair, float]]) → Figure`

### 5.5 Causal Verification

**Figure 12 — Patching Effect Heatmap**
2D heatmap: x = layer patched, y = model pair. Cell value = KL divergence (patched vs. unpatched output). Identifies which layers carry domain-specific signal. Expected: middle layers brightest.
- Function: `viz.patching_heatmap(kl_scores: ndarray, model_pairs: list[tuple], layers: list[int]) → Figure`

**Figure 13 — Steering Demonstration**
Before/after text comparison: 4 rows (target models M2, M3, M5, M7), 2 columns (unsteered / steered with M1's top exclusive Victorian feature projected into target via shared latent). Quantitative register score per cell.
- Function: `viz.steering_demo(generations: dict[model_id, SteeringResult]) → Figure`

**Figure 14 — Register Metric Distribution**
Distribution of domain register scores across the shared corpus for all 8 models. Each model shown as a KDE curve. Separation between curves = quantitative measure of corpus distinctiveness.
- Register metric definition: `R(text, M_i) = log P_{M1}(text) - log P_{M_i}(text)` (log-perplexity ratio to Mr. Chatterbox as anchor)
- Function: `viz.register_distributions(scores: dict[model_id, ndarray]) → Figure`

---

## 6. Seed Concept Vocabulary

40 words chosen for historically variable meanings or cultural salience across training corpora. Grouped by expected cross-model divergence:

**High divergence expected** (meanings shifted substantially across eras/domains):
`science, nature, evolution, hysteria, progress, empire, machine, constitution, character, feeling`

**Medium divergence expected** (present in all corpora but differently weighted):
`truth, reason, mind, body, soul, knowledge, death, love, time, power, order, system, law, will, form`

**Low divergence expected** (stable concrete/functional concepts — control group):
`matter, voice, memory, desire, labour, class, race, God, natural, civil, common, free, human`

The control group (low divergence) is critical for calibration: if these concepts also show high scatter in the UMAP, something is wrong with the method.

---

## 7. The Register Metric

A quantitative "domain signature score" for any text passage, defined as the log-perplexity ratio relative to Mr. Chatterbox (M1) as anchor:

```
R(text, M_i) = log P_{M1}(text) − log P_{M_i}(text)
```

- Positive R → text is more surprising to M_i than to M1 (M1's domain)
- Negative R → text is more surprising to M1 than to M_i (M_i's domain)
- R ≈ 0 → text is equally surprising to both (domain-neutral)

This gives every piece of text a continuous multi-dimensional "location" in model-space. Used throughout the paper as the dependent variable in causal experiments.

---

## 8. Negative Results Plan

Explicitly planned for:

- **If cross-model steering fails:** report as a finding — features are not universally transferable; the shared latent space is necessary but not sufficient for steering. Discuss implications for the Platonic Representation Hypothesis.
- **If SVCCA alignment is low (<0.5):** the Universal SAE may be dominated by the larger models (M2, M6). Investigate whether re-weighting the training loss by inverse model size recovers alignment.
- **If FE distribution is unimodal** (no clear universal/exclusive split): this falsifies the bimodal prediction and is itself a publishable result — constrained-corpus models may not exhibit the same universality as general models.
- **If simplex geometry diverges completely:** report that Park et al.'s geometric predictions do not generalise to maximally constrained training regimes. Discuss whether the linear representation hypothesis requires scale/diversity thresholds to hold.

---

## 9. Compute Budget

| Task | Hardware | Estimated time |
|------|----------|---------------|
| Activation extraction (all 8 models, 200K tokens, 3 layers each) | CPU + RTX 2060 | 4–8 hours |
| Universal SAE training (50K steps, batch 2048) | RTX 2060 | 12–24 hours |
| Per-model SAEs × 8 (validation) | RTX 2060 | 8–16 hours |
| Pairwise crosscoders × 3 | RTX 2060 | 6–12 hours |
| All analysis + figure generation | CPU | 2–4 hours |
| **Total** | | **~32–64 hours** |

All training is designed to fit within 6GB VRAM by pre-caching activations and keeping base models CPU-offloaded during SAE training.

---

## 10. Open Questions for Implementation

1. **M1 tokeniser**: nanochat uses a custom 32K BPE tokeniser trained with RustBPE/tiktoken. Must clone `karpathy/nanochat` and use `get_tokenizer()` — cannot use HuggingFace AutoTokenizer.
2. **M2 philosophy-mistral weight format**: 6 training epochs on narrow data may cause memorisation artefacts; activation distributions may be unusual (verify norm statistics before SAE training).
3. **GGUF activation extraction**: `llama-cpp-python` exposes per-layer hidden states via `LogitsProcessor` hooks — needs validation that activations are numerically comparable to PyTorch models.
4. **d_concept floor**: setting d_concept=1152 (M1's d_model) may bottleneck information from larger models (M2: d_model=4096, M6: d_model=2048). May need to trial d_concept=2048 and compare reconstruction R².
5. **TinyStories scale mismatch**: at 28M params M4 may have degenerate activations relative to the others. Consider training a slightly larger TinyStories variant (110M) if M4 reconstruction R² is consistently poor.
