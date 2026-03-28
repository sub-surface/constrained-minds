# Constrained Minds

**Cross-Corpus Universal Feature Analysis via Sparse Autoencoders**

A mechanistic interpretability study asking: when language models are trained on maximally diverse and constrained corpora — Victorian novels, continental philosophy, synthetic children's stories, pure medical abstracts, distilled reasoning traces — what concepts are universal across all of them, and what remains irreducibly their own?

## The Model Zoo

| ID | Model | Corpus |
|----|-------|--------|
| M1 | Mr. Chatterbox (tventurella) | Victorian books 1837–1899 |
| M2 | philosophy-mistral (Heralax) | 5 philosophy books (Nietzsche, Russell, Machiavelli, Locke) |
| M3 | deleuze-qwen-1.5b (wisdomfunction) | Deleuze's complete works |
| M4 | TinyStories-28M (roneneldan) | Synthetic children's stories |
| M5 | BioMedLM (stanford-crfm) | PubMed abstracts only |
| M6 | Qwen3.5-4B-Opus-Distilled-v2 (Jackrong) | Claude Opus reasoning traces |
| M7 | pythia-410m-deduped (EleutherAI) | The Pile (general baseline) |
| M8 | gothic-gpt2 (Dwaraka) | Victorian Gothic fiction |

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
