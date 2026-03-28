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

Five models, all trained from scratch, all causal/autoregressive, all genuinely constrained. Selected after explicit review to eliminate fine-tuned and LoRA-adapted models, which would confound corpus-driven feature analysis with inherited general-pretraining representations.

**Design rationale:** The reviewer's central critique was that fine-tuned models (LoRA on Deleuze, Mistral fine-tuned on philosophy books, Qwen distilled on reasoning traces) are not genuinely constrained-corpus models — their features reflect the general-purpose base model far more than the narrow fine-tuning data. All five models here are pretrained from random initialisation on a single corpus. This is the cleanest possible test of the central claim.

| ID | Model | HF Handle | Size | Corpus | Architecture | Constraint type |
|----|-------|-----------|------|--------|--------------|-----------------|
| M1 | Mr. Chatterbox | `tventurella/mr_chatterbox_model` | 340M | Victorian books 1837–1899 (28K books, 2.93B tokens) | nanochat (custom GPT) | Historical/literary |
| M2 | TinyStories-33M | `roneneldan/TinyStories-33M` | 33M | Synthetic children's stories (GPT-4 generated, 3–4 year old vocabulary) | GPT-Neo (4 layers, d=768) | Synthetic/vocabulary-constrained |
| M3 | BioMedLM | `stanford-crfm/BioMedLM` | 2.7B | PubMed abstracts + full text only (~300B tokens) | GPT-2 style | Domain-constrained (science) |
| M4 | CodeParrot | `codeparrot/codeparrot` | 1.5B | Python code only (~26B tokens from GitHub) | GPT-2 (48 layers, d=1600) | Formal/syntactic constraint |
| M5 | Pythia-410m | `EleutherAI/pythia-410m-deduped` | 410M | The Pile — diverse modern web (825 GiB, 22 sources) | GPT-NeoX | General baseline (single control) |

**Corpus axis:** The five models span a deliberate gradient of constraint type:
- **Synthetic + vocabulary-locked** (M2): every word chosen to be known by a 3-year-old; grammar perfectly regular
- **Historical + culturally isolated** (M1): a complete world that ended in 1899; no concept formed after Victoria
- **Domain-locked modern science** (M3): only PubMed; the world is entirely biomedical
- **Formally constrained** (M4): code is not natural language; syntax is executable, semantics are operational
- **General** (M5): the control — everything, deduplicated

**Controlled comparisons:**
- **Same scale, different corpus:** M1 (340M) ↔ M5 (410M) — isolates corpus effect at matched capacity
- **Domain science vs. natural language:** M3 ↔ M1/M2 — modern constrained vs. historical/synthetic constrained
- **Formal vs. natural language:** M4 ↔ any natural language model — code as an extreme register
- **Scale within constraint type:** M2 (33M) is the only sub-100M model; provides a natural test of whether small models develop degenerate features relative to the universal SAE

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

**`models/`** — A unified `ModelWrapper` interface normalising across all 5 architectures. Every model exposes identical methods:
- `encode(text: str) → token_ids`
- `forward(token_ids, layer: int) → activations`
- `decode_logits(hidden_state) → logit_distribution`
- `get_layer_count() → int`
- `get_d_model() → int`

Sub-classes: `NanochatWrapper` (handles `.pt` loading for M1), `HuggingFaceWrapper` (M2–M5).

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
| M2 TinyStories-33M | 4 | 1 | 2 | 3 |
| M3 BioMedLM | 32 | 5 | 16 | 27 |
| M4 CodeParrot | 48 | 8 | 24 | 40 |
| M5 Pythia-410m | 24 | 4 | 12 | 20 |

Note: M2 TinyStories has only 4 layers — early/middle/late checkpoints are at layers 1, 2, 3 rather than the standard 17%/50%/83% targets. This is noted as a limitation in §10.

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

1. **Per-model SAEs** (SAELens): train independently per model, measure SVCCA alignment with Universal SAE model-specific features. Target: >0.65 SVCCA score (note: the arXiv:2410.06981 baseline of 0.70 was for same-family models; our cross-architecture target is deliberately lower and treated as exploratory)
2. **Pairwise DFC crosscoders** (oli-clive-griffin/crosscode): for pairs M1↔M5 (historical vs. general), M2↔M1 (synthetic vs. historical), M4↔M5 (code vs. general) — partitions features into exclusive-A, exclusive-B, shared as independent universality check
3. **Held-out reconstruction:** 20% corpus reserved; R² degradation >10% from in-distribution → overfitting flag

---

## 5. Analysis Methods & Figures

### 5.1 Surface Representations

**Figure 1 — Logit Lens Grid**
5×3 grid (models × layers). Each cell: top-5 predicted tokens with probabilities for fixed probe sentence *"The nature of the mind is..."*. Shows Victorian register lock-in by layer 9 in M1; biomedical vocabulary in M3; syntactic/code-like completions in M4; and the general baseline in M5.
- Function: `viz.logit_lens_grid(traces: dict[model_id, LayerTrace]) → Figure`

**Figure 2 — Concept Neighbourhood Comparison**
Structured table: 38 seed concepts × 5 models. Each cell: top-3 nearest neighbours. Colour-coded by semantic field (theological, scientific, literary, medical, formal/code). Reveals "science" → {Providence, natural philosophy} in M1 vs. {clinical trial, dosage} in M3 vs. {function, module, import} in M4.
- Seed concepts include: *science, nature, feeling, character, progress, empire, mind, body, truth, reason, evolution, hysteria, machine, virtue, moral, soul, knowledge, death, love, time, power, order, system, law, will, form, matter, spirit, voice, memory, desire, labour, class, race, God, natural, civil, common, free, human*
- Function: `viz.concept_neighbourhood_table(neighbours: dict[model_id, dict[concept, list[str]]]) → Figure`

**Figure 3 — UMAP Concept Map**
2D UMAP of 38×5=190 concept embeddings. Colour = model, shape = concept. Tight cross-model clustering → Platonic convergence; scatter → corpus-specific encoding.
- Function: `viz.umap_concept_map(embeddings: dict[model_id, dict[concept, ndarray]]) → Figure`

### 5.2 Universal Feature Discovery

**Figure 4 — Feature Entropy Histogram** *(paper headline figure)*
Histogram of FE across all 8192 features. Vertical threshold lines at 0.20 and 0.85. Inset pie: universal/partial/exclusive counts and percentages.
- Function: `viz.entropy_histogram(fe_scores: ndarray) → Figure`

**Figure 5 — Universal Feature Gallery**
12 highest-FE features (FE>0.95). Per feature: auto-generated label, maximally-activating passage from 4 representative models side by side. Demonstrates same feature fires on temporal sequencing, causal connectives, etc. across Victorian prose, philosophy, medical text, children's stories.
- Function: `viz.feature_gallery(features: list[FeatureProfile], mode='universal') → Figure`

**Figure 6 — Exclusive Feature Gallery**
Top-5 exclusive features per model (5 panels). Labels and example activations. Expected content: class markers/Providence (M1), simplified grammar/narrative scaffolds (M2), dosage/anatomy (M3), code syntax/indentation/scope (M4), diverse general vocabulary (M5).
- Function: `viz.feature_gallery(features: list[FeatureProfile], mode='exclusive') → Figure`

**Figure 7 — Corpus Exclusivity Heatmap**
Feature×model heatmap, top 200 features by variance. Hierarchically clustered on both axes. Shows model-specific activation blocks as visual proof of corpus ownership. With 5 clean from-scratch models, the block structure should be particularly crisp.
- Function: `viz.exclusivity_heatmap(activations: ndarray, model_ids: list[str]) → Figure`

### 5.3 Cross-Validation

**Figure 8 — Validation Alignment Matrix**
5×3 matrix (models × layers). Each cell: SVCCA score between per-model SAE and Universal SAE. Red→green colour gradient. Failing cells flagged for investigation. Expected: M2 (TinyStories, 33M, only 6 layers) may show lower alignment at early layers due to shallow architecture.
- Function: `viz.validation_matrix(svcca_scores: ndarray, model_ids: list[str]) → Figure`

**Figure 9 — Crosscoder Venn Diagrams**
Three Venn diagrams for pairs M1↔M5 (historical vs. general), M2↔M1 (synthetic vs. historical), M4↔M5 (code vs. general). Circle size ∝ feature count; overlap = shared features; top-5 exclusive feature labels annotated per circle.
- Function: `viz.crosscoder_venns(crosscoder_results: list[CrosscoderResult]) → Figure`

### 5.4 Geometric Analysis

**Figure 10 — Concept Geometry Comparison**
Side-by-side 3D PCA plots for emotional-states simplex (joy, grief, anger, fear, love) in M1, M5, M2. Tests whether simplex vertex positions/angles are consistent across models under rotation. Platonic prediction: same shape, different orientation.
- Function: `viz.simplex_comparison(concept_embeddings: dict[model_id, ndarray], concepts: list[str]) → Figure`

**Figure 11 — Hierarchy Orthogonality Scores**
Bar chart: cosine similarity between parent and child concept directions for 10 hierarchical pairs (e.g. animal→mammal→dog, emotion→grief, knowledge→science) across all 5 models. Park et al. prediction: near zero. Deviations quantify idiosyncratic geometry.
- Function: `viz.orthogonality_bars(scores: dict[model_id, dict[pair, float]]) → Figure`

### 5.5 Causal Verification

**Figure 12 — Patching Effect Heatmap**
2D heatmap: x = layer patched, y = model pair. Cell value = KL divergence (patched vs. unpatched output). Identifies which layers carry domain-specific signal. Expected: middle layers brightest.
- Function: `viz.patching_heatmap(kl_scores: ndarray, model_pairs: list[tuple], layers: list[int]) → Figure`

**Figure 13 — Steering Demonstration**
Before/after text comparison: 4 rows (target models M2, M3, M4, M5), 2 columns (unsteered / steered with M1's top exclusive Victorian feature projected into target via shared latent). Quantitative register score per cell. M4 (CodeParrot) is the most interesting target — does steering a code model with a Victorian moral-register feature produce anything interpretable?
- Function: `viz.steering_demo(generations: dict[model_id, SteeringResult]) → Figure`

**Figure 14 — Register Metric Distribution**
Distribution of domain register scores across the shared corpus for all 5 models. Each model shown as a KDE curve. Separation between curves = quantitative measure of corpus distinctiveness. With 5 clean from-scratch models the separation is expected to be cleaner than in designs with fine-tuned models.
- Register metric definition: `R(text, M_i) = log P_{M1}(text) - log P_{M_i}(text)` (log-perplexity ratio to Mr. Chatterbox as anchor)
- Function: `viz.register_distributions(scores: dict[model_id, ndarray]) → Figure`

---

## 6. Seed Concept Vocabulary

38 words chosen for historically variable meanings or cultural salience across training corpora. Grouped by expected cross-model divergence:

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
| Activation extraction (5 models, 200K tokens, 3 layers each) | CPU + RTX 2060 | 3–6 hours |
| Universal SAE training (50K steps, batch 2048) | RTX 2060 | 12–24 hours |
| Per-model SAEs × 5 (validation) | RTX 2060 | 5–10 hours |
| Pairwise crosscoders × 3 | RTX 2060 | 6–12 hours |
| All analysis + figure generation | CPU | 2–4 hours |
| **Total** | | **~28–56 hours** |

All training is designed to fit within 6GB VRAM by pre-caching activations and keeping base models CPU-offloaded during SAE training. The largest model (BioMedLM, 2.7B) will be slow on CPU for extraction — plan for this step to dominate the extraction budget.

---

## 10. Open Questions for Implementation

1. **M1 tokeniser**: nanochat uses a custom 32K BPE tokeniser trained with RustBPE/tiktoken. Must clone `karpathy/nanochat` and use `get_tokenizer()` — cannot use HuggingFace AutoTokenizer. Verify tokeniser is available in the repo before starting.
2. **d_concept floor**: setting d_concept=1152 (M1's d_model) may bottleneck information from BioMedLM (d_model=2560) and CodeParrot (d_model=1600). Trial d_concept=2048 and compare reconstruction R² before committing.
3. **M2 TinyStories shallow architecture**: at 4 layers, the early/middle/late checkpoint scheme collapses to layers 1/2/3. The "early" and "late" representations may not be meaningfully distinct. Monitor whether M2's per-layer activations show sufficient variance to be useful. If not, use only the final layer for M2.
4. **CodeParrot tokenisation of natural language**: CodeParrot's tokeniser is trained on Python code. When run over the shared neutral corpus (natural language), many tokens will be rare or fragmented. Verify that the shared corpus produces meaningful activations rather than degenerate out-of-distribution behaviour. May need a code-adjacent corpus stratum (e.g. code comments, README files) for M4's activation extraction.
5. **Token boundary alignment across models**: five different tokenisers will produce different token segmentations of the same text. For concept neighbourhood analysis (Figure 2), the activation for a word like "science" must be extracted from the correct token position in each model's tokenisation. Implement a robust token-finding utility that handles subword splits.
