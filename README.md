# Constrained Minds

**Cross-Corpus Universal Feature Analysis via Sparse Autoencoders**

A mechanistic interpretability study asking: when language models are trained on maximally diverse and constrained corpora — Victorian novels, continental philosophy, synthetic children's stories, pure medical abstracts, distilled reasoning traces — what concepts are universal across all of them, and what remains irreducibly their own?

## The Model Zoo

All five models are trained **from scratch** on a single corpus — no fine-tuning, no LoRA, no inherited general pretraining.

| ID | Model | Size | Corpus |
|----|-------|------|--------|
| M1 | Mr. Chatterbox (tventurella) | 340M | Victorian books 1837–1899 |
| M2 | TinyStories-33M (roneneldan) | 33M | Synthetic children's stories (GPT-4, restricted vocab) |
| M3 | BioMedLM (stanford-crfm) | 2.7B | PubMed abstracts only |
| M4 | CodeParrot (codeparrot) | 1.5B | Python code only |
| M5 | pythia-410m-deduped (EleutherAI) | 410M | The Pile — general baseline (single control) |

## Method

We train a **Universal Sparse Autoencoder** (following arXiv:2502.03714) jointly across all models, discovering features that are:
- **Universal** — shared across all corpora, consistent with the Platonic Representation Hypothesis
- **Corpus-exclusive** — present only in models trained on specific domains

Cross-validation is performed with per-model SAEs (via SAELens) and pairwise crosscoders (arXiv:2602.11729).

## Research Phases

1. Surface representations — logit lens, nearest-neighbour geometry
2. Universal SAE training — shared vs. exclusive feature discovery
3. Cross-validation — per-model SAEs + crosscoders
4. Geometric analysis — simplex/polytope structure (Park et al. 2024)
5. Causal verification — activation patching, cross-model steering

## Setup

```bash
pip install -e ".[dev]"
```

See `notebooks/01_model_zoo.ipynb` to get started.
