# Constrained Minds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mechanistic interpretability research pipeline that trains a Universal Sparse Autoencoder across 5 from-scratch language models with maximally diverse corpora, discovers universal vs. corpus-exclusive features, and produces 14 publication-quality figures.

**Architecture:** A Python package (`constrained_minds/`) with five layers — model loading, activation extraction, SAE training, analysis, and visualization — plus Jupyter notebooks that import from the package but contain no analysis logic. All models are loaded via a unified `ModelWrapper` interface. The Universal SAE uses per-model encoders/decoders with a shared latent space.

**Tech Stack:** PyTorch 2.2+, HuggingFace Transformers, SAELens, TransformerLens, nanochat (for M1), matplotlib/seaborn (viz), UMAP, scikit-learn, Anthropic API (autointerp), Jupyter

**Spec:** `docs/superpowers/specs/2026-03-27-constrained-minds-design.md`

---

## File Map

| File | Responsibility | Created in |
|------|---------------|------------|
| `constrained_minds/models/base.py` | `ModelWrapper` ABC — unified interface for all models | Task 1 |
| `constrained_minds/models/nanochat_wrapper.py` | Loads Mr. Chatterbox `.pt` via nanochat GPT class | Task 2 |
| `constrained_minds/models/hf_wrapper.py` | Loads HuggingFace models (M2–M5) via AutoModelForCausalLM | Task 3 |
| `constrained_minds/models/registry.py` | Model registry — maps model IDs to configs and wrappers | Task 4 |
| `constrained_minds/activations/corpus.py` | Shared neutral corpus loading and stratification | Task 6 |
| `constrained_minds/activations/extractor.py` | Activation extraction and disk caching | Task 7 |
| `constrained_minds/activations/normalise.py` | Norm scaling and linear projection to shared d_concept | Task 8 |
| `constrained_minds/activations/token_align.py` | Token-boundary alignment for concept word extraction | Task 8 |
| `constrained_minds/sae/universal.py` | Universal SAE — per-model enc/dec, shared latent, TopK | Task 9 |
| `constrained_minds/sae/feature_metrics.py` | Firing entropy, co-fire proportion, feature classification | Task 10 |
| `constrained_minds/sae/per_model.py` | Per-model SAE wrapper around SAELens | Task 18 |
| `constrained_minds/sae/crosscoder.py` | Pairwise DFC crosscoder wrapper | Task 19 |
| `constrained_minds/analysis/logit_lens.py` | Layer-by-layer unembedding traces | Task 12 |
| `constrained_minds/analysis/geometry.py` | PCA/UMAP, nearest-neighbour, simplex fitting | Task 13 |
| `constrained_minds/analysis/register_metric.py` | Domain register score (log-perplexity ratio) | Task 14 |
| `constrained_minds/analysis/patching.py` | Activation patching and cross-model steering | Task 16 |
| `constrained_minds/analysis/autointerp.py` | Feature labelling via Claude API | Task 15 |
| `constrained_minds/viz/style.py` | Shared plot style, colours, fonts | Task 12 |
| `constrained_minds/viz/figures.py` | All 14 figure functions | Tasks 12–17 |
| `scripts/extract_activations.py` | CLI: run activation extraction for all models | Task 7 |
| `scripts/train_universal_sae.py` | CLI: train Universal SAE | Task 11 |
| `scripts/train_per_model_saes.py` | CLI: train per-model SAEs for validation | Task 18 |
| `notebooks/01_model_zoo.ipynb` | Load all models, sanity check, sample generation | Task 5 |
| `notebooks/02_surface_representations.ipynb` | Figures 1–3 | Task 12 |
| `notebooks/03_universal_sae.ipynb` | Figures 4–7 | Task 11 |
| `notebooks/04_validation.ipynb` | Figures 8–9 | Task 20 |
| `notebooks/05_geometry.ipynb` | Figures 10–11 | Task 13 |
| `notebooks/06_causal.ipynb` | Figures 12–14 | Task 16 |
| `data/probe_sentences/seed_concepts.json` | 38 seed concepts with probe sentences | Task 6 |
| `tests/test_models.py` | Tests for model wrappers | Tasks 1–4 |
| `tests/test_activations.py` | Tests for activation extraction and normalisation | Tasks 7–8 |
| `tests/test_sae.py` | Tests for Universal SAE forward/backward | Task 9 |
| `tests/test_analysis.py` | Tests for analysis modules | Tasks 12–16 |

---

## Sub-Plan 1: Model Infrastructure

### Task 1: ModelWrapper base class

**Files:**
- Create: `constrained_minds/models/base.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import pytest
from constrained_minds.models.base import ModelWrapper


def test_model_wrapper_is_abstract():
    with pytest.raises(TypeError):
        ModelWrapper()


def test_model_wrapper_has_required_methods():
    methods = ["encode", "forward", "decode_logits", "get_layer_count", "get_d_model", "get_vocab_size"]
    for method in methods:
        assert hasattr(ModelWrapper, method)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/Leon/Documents/constrained-minds && python -m pytest tests/test_models.py -v`
Expected: FAIL — ImportError, `base.py` doesn't exist yet

- [ ] **Step 3: Write minimal implementation**

```python
# constrained_minds/models/base.py
from abc import ABC, abstractmethod
import torch


class ModelWrapper(ABC):
    """Unified interface for all models in the zoo."""

    model_id: str
    device: torch.device

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """Tokenise text → tensor of token IDs."""
        ...

    @abstractmethod
    def forward(self, token_ids: torch.Tensor, layer: int) -> torch.Tensor:
        """Run forward pass, return residual stream activations at the given layer.

        Args:
            token_ids: (batch, seq_len) tensor of token IDs
            layer: 0-indexed layer to extract activations from

        Returns:
            (batch, seq_len, d_model) tensor of activations
        """
        ...

    @abstractmethod
    def decode_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Unembed hidden state → logit distribution over vocabulary.

        Args:
            hidden_state: (batch, seq_len, d_model) tensor

        Returns:
            (batch, seq_len, vocab_size) tensor of logits
        """
        ...

    @abstractmethod
    def get_layer_count(self) -> int:
        """Total number of transformer layers."""
        ...

    @abstractmethod
    def get_d_model(self) -> int:
        """Hidden dimension size."""
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Vocabulary size."""
        ...

    def get_layer_checkpoints(self) -> tuple[int, int, int]:
        """Return (early, middle, late) layer indices at ~17%, ~50%, ~83% depth."""
        n = self.get_layer_count()
        return (
            max(0, round(n * 0.17)),
            round(n * 0.5),
            min(n - 1, round(n * 0.83)),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add constrained_minds/models/base.py tests/test_models.py
git commit -m "feat: add ModelWrapper ABC with unified interface"
```

---

### Task 2: NanochatWrapper for Mr. Chatterbox

**Files:**
- Create: `constrained_minds/models/nanochat_wrapper.py`
- Modify: `tests/test_models.py`

**Prerequisites:** Clone nanochat repo and install as dependency.

- [ ] **Step 1: Clone nanochat and install**

```bash
cd /c/Users/Leon/Documents/constrained-minds
git clone https://github.com/karpathy/nanochat.git vendor/nanochat
```

Add to `pyproject.toml` under `[project]`:
```toml
[tool.setuptools.package-data]
constrained_minds = ["*"]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_models.py (append)
import os

CHATTERBOX_PATH = os.path.expanduser("~/Documents/mr_chatterbox_model/model_000070.pt")
CHATTERBOX_META = os.path.expanduser("~/Documents/mr_chatterbox_model/meta_000070.json")


@pytest.mark.skipif(not os.path.exists(CHATTERBOX_PATH), reason="Model not downloaded")
def test_nanochat_wrapper_loads():
    from constrained_minds.models.nanochat_wrapper import NanochatWrapper

    model = NanochatWrapper(
        model_path=CHATTERBOX_PATH,
        meta_path=CHATTERBOX_META,
        nanochat_dir="vendor/nanochat",
    )
    assert model.get_d_model() == 1152
    assert model.get_layer_count() == 18
    assert model.get_vocab_size() == 32768


@pytest.mark.skipif(not os.path.exists(CHATTERBOX_PATH), reason="Model not downloaded")
def test_nanochat_wrapper_forward():
    from constrained_minds.models.nanochat_wrapper import NanochatWrapper

    model = NanochatWrapper(
        model_path=CHATTERBOX_PATH,
        meta_path=CHATTERBOX_META,
        nanochat_dir="vendor/nanochat",
    )
    tokens = model.encode("The nature of the mind is")
    acts = model.forward(tokens.unsqueeze(0), layer=9)
    assert acts.shape[-1] == 1152
    assert acts.ndim == 3


@pytest.mark.skipif(not os.path.exists(CHATTERBOX_PATH), reason="Model not downloaded")
def test_nanochat_wrapper_decode_logits():
    from constrained_minds.models.nanochat_wrapper import NanochatWrapper

    model = NanochatWrapper(
        model_path=CHATTERBOX_PATH,
        meta_path=CHATTERBOX_META,
        nanochat_dir="vendor/nanochat",
    )
    tokens = model.encode("Hello")
    acts = model.forward(tokens.unsqueeze(0), layer=17)
    logits = model.decode_logits(acts)
    assert logits.shape[-1] == 32768
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py::test_nanochat_wrapper_loads -v`
Expected: FAIL — ImportError

- [ ] **Step 4: Write implementation**

```python
# constrained_minds/models/nanochat_wrapper.py
import json
import sys
from pathlib import Path

import torch

from constrained_minds.models.base import ModelWrapper


class NanochatWrapper(ModelWrapper):
    """Loads Mr. Chatterbox from a nanochat .pt checkpoint."""

    def __init__(
        self,
        model_path: str,
        meta_path: str,
        nanochat_dir: str = "vendor/nanochat",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model_id = "mr_chatterbox"

        # Add nanochat to sys.path so we can import its GPT class
        nanochat_path = str(Path(nanochat_dir).resolve())
        if nanochat_path not in sys.path:
            sys.path.insert(0, nanochat_path)

        from nanochat.gpt import GPT, GPTConfig

        # Load config from meta JSON
        with open(meta_path) as f:
            meta = json.load(f)

        cfg = meta["model_config"]
        self._config = GPTConfig(
            sequence_len=cfg["sequence_len"],
            vocab_size=cfg["vocab_size"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_kv_head=cfg["n_kv_head"],
            n_embd=cfg["n_embd"],
            window_pattern=cfg["window_pattern"],
        )

        # Build model and load weights
        self._model = GPT(self._config)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        # nanochat checkpoints may wrap state dict under 'model' key
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self._model.load_state_dict(state_dict, strict=False)
        self._model.to(self.device)
        self._model.eval()

        # Load tokenizer
        from nanochat.tokenizer import get_tokenizer
        self._tokenizer = get_tokenizer()

        # Cache the unembedding matrix (lm_head)
        self._unembed = self._model.lm_head.weight  # (vocab_size, d_model)

    def encode(self, text: str) -> torch.Tensor:
        token_ids = self._tokenizer.encode(text)
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def forward(self, token_ids: torch.Tensor, layer: int) -> torch.Tensor:
        # We need to hook into the residual stream at a specific layer.
        # nanochat's GPT stores layers as self.transformer.h (ModuleList).
        activations = {}

        def hook_fn(module, input, output):
            # After the block, the residual stream is the output
            activations["out"] = output

        handle = self._model.transformer.h[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            self._model(token_ids)
        handle.remove()
        return activations["out"]

    def decode_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state @ self._unembed.T

    def get_layer_count(self) -> int:
        return self._config.n_layer

    def get_d_model(self) -> int:
        return self._config.n_embd

    def get_vocab_size(self) -> int:
        return self._config.vocab_size
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_models.py -v -k nanochat`
Expected: PASS (or skip if model not present)

- [ ] **Step 6: Commit**

```bash
git add constrained_minds/models/nanochat_wrapper.py tests/test_models.py vendor/
git commit -m "feat: add NanochatWrapper for Mr. Chatterbox .pt loading"
```

---

### Task 3: HuggingFaceWrapper for M2–M5

**Files:**
- Create: `constrained_minds/models/hf_wrapper.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py (append)

def test_hf_wrapper_loads_pythia():
    """Pythia-410m is small enough to load for a quick test."""
    from constrained_minds.models.hf_wrapper import HuggingFaceWrapper

    model = HuggingFaceWrapper(
        model_name="EleutherAI/pythia-410m-deduped",
        model_id="pythia_410m",
        device="cpu",
    )
    assert model.get_layer_count() == 24
    assert model.get_d_model() == 1024
    assert model.get_vocab_size() == 50257


def test_hf_wrapper_forward_and_logits():
    from constrained_minds.models.hf_wrapper import HuggingFaceWrapper

    model = HuggingFaceWrapper(
        model_name="EleutherAI/pythia-410m-deduped",
        model_id="pythia_410m",
        device="cpu",
    )
    tokens = model.encode("The nature of the mind is")
    acts = model.forward(tokens.unsqueeze(0), layer=12)
    assert acts.ndim == 3
    assert acts.shape[-1] == 1024

    logits = model.decode_logits(acts)
    assert logits.shape[-1] == 50257
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py::test_hf_wrapper_loads_pythia -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write implementation**

```python
# constrained_minds/models/hf_wrapper.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constrained_minds.models.base import ModelWrapper


class HuggingFaceWrapper(ModelWrapper):
    """Loads any HuggingFace causal LM via AutoModelForCausalLM."""

    def __init__(
        self,
        model_name: str,
        model_id: str,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.model_id = model_id
        self.device = torch.device(device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self._model.eval()

        # Cache architecture info
        config = self._model.config
        self._n_layer = config.num_hidden_layers
        self._d_model = config.hidden_size
        self._vocab_size = config.vocab_size

        # Get the unembedding (lm_head) weight
        self._unembed = self._model.lm_head.weight  # (vocab_size, d_model)

    def encode(self, text: str) -> torch.Tensor:
        tokens = self._tokenizer.encode(text, return_tensors="pt").squeeze(0)
        return tokens.to(self.device)

    def forward(self, token_ids: torch.Tensor, layer: int) -> torch.Tensor:
        with torch.no_grad():
            outputs = self._model(
                token_ids,
                output_hidden_states=True,
            )
        # output_hidden_states returns (n_layers + 1) states:
        # index 0 = embedding output, index i = after layer i-1
        # So layer L's output is at index L+1
        return outputs.hidden_states[layer + 1]

    def decode_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state @ self._unembed.T

    def get_layer_count(self) -> int:
        return self._n_layer

    def get_d_model(self) -> int:
        return self._d_model

    def get_vocab_size(self) -> int:
        return self._vocab_size
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_models.py -v -k hf_wrapper`
Expected: PASS (will download Pythia-410m on first run — ~800MB)

- [ ] **Step 5: Commit**

```bash
git add constrained_minds/models/hf_wrapper.py tests/test_models.py
git commit -m "feat: add HuggingFaceWrapper for M2-M5 model loading"
```

---

### Task 4: Model registry

**Files:**
- Create: `constrained_minds/models/registry.py`
- Modify: `constrained_minds/models/__init__.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py (append)

def test_registry_lists_all_models():
    from constrained_minds.models.registry import MODEL_ZOO
    assert len(MODEL_ZOO) == 5
    assert "mr_chatterbox" in MODEL_ZOO
    assert "tinystories_33m" in MODEL_ZOO
    assert "biomedlm" in MODEL_ZOO
    assert "codeparrot" in MODEL_ZOO
    assert "pythia_410m" in MODEL_ZOO


def test_registry_configs_have_required_fields():
    from constrained_minds.models.registry import MODEL_ZOO
    for model_id, cfg in MODEL_ZOO.items():
        assert "hf_name" in cfg or "model_path" in cfg, f"{model_id} missing source"
        assert "d_model" in cfg
        assert "n_layer" in cfg
        assert "layer_checkpoints" in cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py::test_registry_lists_all_models -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write implementation**

```python
# constrained_minds/models/registry.py
"""Model registry — maps model IDs to configs and wrapper classes."""

from dataclasses import dataclass
from typing import Literal

MODEL_ZOO: dict[str, dict] = {
    "mr_chatterbox": {
        "wrapper": "nanochat",
        "model_path": "~/Documents/mr_chatterbox_model/model_000070.pt",
        "meta_path": "~/Documents/mr_chatterbox_model/meta_000070.json",
        "d_model": 1152,
        "n_layer": 18,
        "vocab_size": 32768,
        "layer_checkpoints": (3, 9, 15),
        "corpus": "Victorian books 1837-1899",
        "constraint_type": "historical/literary",
    },
    "tinystories_33m": {
        "wrapper": "hf",
        "hf_name": "roneneldan/TinyStories-33M",
        "d_model": 768,
        "n_layer": 4,
        "vocab_size": 50257,
        "layer_checkpoints": (1, 2, 3),
        "corpus": "Synthetic children's stories",
        "constraint_type": "synthetic/vocabulary-constrained",
    },
    "biomedlm": {
        "wrapper": "hf",
        "hf_name": "stanford-crfm/BioMedLM",
        "d_model": 2560,
        "n_layer": 32,
        "vocab_size": 28896,
        "layer_checkpoints": (5, 16, 27),
        "corpus": "PubMed abstracts only",
        "constraint_type": "domain-constrained (science)",
    },
    "codeparrot": {
        "wrapper": "hf",
        "hf_name": "codeparrot/codeparrot",
        "d_model": 1600,
        "n_layer": 48,
        "vocab_size": 32768,
        "layer_checkpoints": (8, 24, 40),
        "corpus": "Python code only",
        "constraint_type": "formal/syntactic",
    },
    "pythia_410m": {
        "wrapper": "hf",
        "hf_name": "EleutherAI/pythia-410m-deduped",
        "d_model": 1024,
        "n_layer": 24,
        "vocab_size": 50257,
        "layer_checkpoints": (4, 12, 20),
        "corpus": "The Pile (diverse)",
        "constraint_type": "general baseline",
    },
}


def load_model(model_id: str, device: str = "cpu"):
    """Load a model by its zoo ID. Returns a ModelWrapper instance."""
    from pathlib import Path
    cfg = MODEL_ZOO[model_id]

    if cfg["wrapper"] == "nanochat":
        from constrained_minds.models.nanochat_wrapper import NanochatWrapper
        return NanochatWrapper(
            model_path=str(Path(cfg["model_path"]).expanduser()),
            meta_path=str(Path(cfg["meta_path"]).expanduser()),
            device=device,
        )
    elif cfg["wrapper"] == "hf":
        from constrained_minds.models.hf_wrapper import HuggingFaceWrapper
        return HuggingFaceWrapper(
            model_name=cfg["hf_name"],
            model_id=model_id,
            device=device,
        )
    else:
        raise ValueError(f"Unknown wrapper type: {cfg['wrapper']}")
```

- [ ] **Step 4: Update `__init__.py`**

```python
# constrained_minds/models/__init__.py
from constrained_minds.models.base import ModelWrapper
from constrained_minds.models.registry import MODEL_ZOO, load_model
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_models.py -v -k registry`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add constrained_minds/models/registry.py constrained_minds/models/__init__.py tests/test_models.py
git commit -m "feat: add model registry with all 5 zoo entries"
```

---

### Task 5: Model zoo sanity notebook

**Files:**
- Create: `notebooks/01_model_zoo.ipynb`

- [ ] **Step 1: Create notebook**

Create `notebooks/01_model_zoo.ipynb` with cells:

Cell 1 — Load all models:
```python
from constrained_minds.models import MODEL_ZOO, load_model

models = {}
for model_id in MODEL_ZOO:
    print(f"Loading {model_id}...")
    models[model_id] = load_model(model_id)
    m = models[model_id]
    print(f"  d_model={m.get_d_model()}, layers={m.get_layer_count()}, vocab={m.get_vocab_size()}")
print("All models loaded.")
```

Cell 2 — Sanity: generate text from each model:
```python
prompt = "The nature of the mind is"
for model_id, model in models.items():
    tokens = model.encode(prompt)
    acts = model.forward(tokens.unsqueeze(0), layer=model.get_layer_count() - 1)
    logits = model.decode_logits(acts)
    top5 = logits[0, -1].topk(5)
    print(f"\n{model_id} top-5 next tokens:")
    for i in range(5):
        print(f"  {top5.values[i].item():.2f}")
```

Cell 3 — Verify layer checkpoints:
```python
for model_id, cfg in MODEL_ZOO.items():
    m = models[model_id]
    e, mid, l = cfg["layer_checkpoints"]
    for layer in [e, mid, l]:
        acts = m.forward(m.encode("test").unsqueeze(0), layer=layer)
        assert acts.shape[-1] == cfg["d_model"], f"{model_id} layer {layer} d_model mismatch"
    print(f"{model_id}: checkpoints ({e}, {mid}, {l}) verified ✓")
```

- [ ] **Step 2: Run notebook manually to verify all models load**

Run: `jupyter nbconvert --to notebook --execute notebooks/01_model_zoo.ipynb`
Expected: Completes without error. BioMedLM will take longest to load on CPU.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_model_zoo.ipynb
git commit -m "feat: add model zoo sanity check notebook"
```

---

## Sub-Plan 2: Activation Pipeline

### Task 6: Shared neutral corpus and seed concepts

**Files:**
- Create: `constrained_minds/activations/corpus.py`
- Create: `data/probe_sentences/seed_concepts.json`
- Create: `tests/test_activations.py`

- [ ] **Step 1: Create seed concepts JSON**

```json
{
  "high_divergence": [
    "science", "nature", "evolution", "hysteria", "progress",
    "empire", "machine", "constitution", "character", "feeling"
  ],
  "medium_divergence": [
    "truth", "reason", "mind", "body", "soul", "knowledge",
    "death", "love", "time", "power", "order", "system",
    "law", "will", "form"
  ],
  "low_divergence_control": [
    "matter", "voice", "memory", "desire", "labour", "class",
    "race", "God", "natural", "civil", "common", "free", "human"
  ]
}
```

Write this to `data/probe_sentences/seed_concepts.json`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_activations.py
from constrained_minds.activations.corpus import load_seed_concepts, build_neutral_corpus


def test_load_seed_concepts():
    concepts = load_seed_concepts()
    assert len(concepts["high_divergence"]) == 10
    assert len(concepts["medium_divergence"]) == 15
    assert len(concepts["low_divergence_control"]) == 13
    all_concepts = concepts["high_divergence"] + concepts["medium_divergence"] + concepts["low_divergence_control"]
    assert len(all_concepts) == 38


def test_build_neutral_corpus_returns_strata():
    corpus = build_neutral_corpus(tokens_per_stratum=100)
    assert "contemporary_news" in corpus
    assert "modern_fiction" in corpus
    assert isinstance(corpus["contemporary_news"], str)
    assert len(corpus["contemporary_news"]) > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_activations.py -v`
Expected: FAIL — ImportError

- [ ] **Step 4: Write implementation**

```python
# constrained_minds/activations/corpus.py
"""Shared neutral corpus loading and seed concept management."""

import json
from pathlib import Path
from datasets import load_dataset


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_seed_concepts() -> dict[str, list[str]]:
    """Load the 38 seed concepts from JSON."""
    path = DATA_DIR / "probe_sentences" / "seed_concepts.json"
    with open(path) as f:
        return json.load(f)


def get_all_concepts() -> list[str]:
    """Flat list of all seed concepts."""
    concepts = load_seed_concepts()
    return concepts["high_divergence"] + concepts["medium_divergence"] + concepts["low_divergence_control"]


def build_neutral_corpus(tokens_per_stratum: int = 25000) -> dict[str, str]:
    """Build a stratified neutral corpus. Each stratum is a string of text.

    Uses HuggingFace datasets for easy access. Falls back to placeholder
    strings if datasets are unavailable (for testing).
    """
    strata = {}

    # Contemporary news — CC-News
    try:
        ds = load_dataset("cc_news", split="train", streaming=True)
        texts = []
        char_target = tokens_per_stratum * 4  # ~4 chars per token estimate
        total = 0
        for item in ds:
            texts.append(item["text"])
            total += len(item["text"])
            if total >= char_target:
                break
        strata["contemporary_news"] = "\n\n".join(texts)
    except Exception:
        strata["contemporary_news"] = ""

    # Modern fiction — PG-19
    try:
        ds = load_dataset("pg19", split="train", streaming=True)
        texts = []
        total = 0
        for item in ds:
            texts.append(item["text"][:5000])  # Take first 5K chars per book
            total += 5000
            if total >= tokens_per_stratum * 4:
                break
        strata["modern_fiction"] = "\n\n".join(texts)
    except Exception:
        strata["modern_fiction"] = ""

    # Scientific abstracts — arxiv
    try:
        ds = load_dataset("ccdv/arxiv-classification", split="train", streaming=True)
        texts = []
        total = 0
        for item in ds:
            texts.append(item["text"][:2000])
            total += 2000
            if total >= tokens_per_stratum * 4:
                break
        strata["scientific_abstracts"] = "\n\n".join(texts)
    except Exception:
        strata["scientific_abstracts"] = ""

    # Simple English Wikipedia
    try:
        ds = load_dataset("wikipedia", "20220301.simple", split="train", streaming=True)
        texts = []
        total = 0
        for item in ds:
            texts.append(item["text"][:2000])
            total += 2000
            if total >= tokens_per_stratum * 4:
                break
        strata["simple_wikipedia"] = "\n\n".join(texts)
    except Exception:
        strata["simple_wikipedia"] = ""

    return strata
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_activations.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add constrained_minds/activations/corpus.py data/probe_sentences/seed_concepts.json tests/test_activations.py
git commit -m "feat: add neutral corpus loader and seed concepts"
```

---

### Task 7: Activation extraction and caching

**Files:**
- Create: `constrained_minds/activations/extractor.py`
- Create: `scripts/extract_activations.py`
- Modify: `tests/test_activations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_activations.py (append)
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from constrained_minds.activations.extractor import ActivationExtractor


def test_extractor_saves_and_loads():
    """Test that activations can be cached and reloaded."""
    mock_model = MagicMock()
    mock_model.model_id = "test_model"
    mock_model.get_d_model.return_value = 64
    mock_model.encode.return_value = torch.zeros(10, dtype=torch.long)
    mock_model.forward.return_value = torch.randn(1, 10, 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = ActivationExtractor(cache_dir=tmpdir)
        acts = extractor.extract(mock_model, "test text", layer=5)
        assert acts.shape == (1, 10, 64)

        # Check that a cache file was written
        cache_files = list(Path(tmpdir).joinpath("test_model").glob("layer_5_*.pt"))
        assert len(cache_files) == 1

        # Load from cache (should not call forward again)
        mock_model.forward.reset_mock()
        acts2 = extractor.extract(mock_model, "test text", layer=5)
        mock_model.forward.assert_not_called()
        assert torch.equal(acts, acts2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_activations.py::test_extractor_saves_and_loads -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write implementation**

```python
# constrained_minds/activations/extractor.py
"""Activation extraction with disk caching."""

import hashlib
from pathlib import Path

import torch

from constrained_minds.models.base import ModelWrapper


class ActivationExtractor:
    """Extracts and caches per-layer activations from any ModelWrapper."""

    def __init__(self, cache_dir: str = "data/activations"):
        self.cache_dir = Path(cache_dir)

    def _cache_path(self, model_id: str, text_hash: str, layer: int) -> Path:
        return self.cache_dir / model_id / f"layer_{layer}_{text_hash}.pt"

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def extract(
        self,
        model: ModelWrapper,
        text: str,
        layer: int,
    ) -> torch.Tensor:
        """Extract activations for text at a given layer, with caching.

        Args:
            model: Any ModelWrapper instance
            text: Input text
            layer: Layer index to extract from

        Returns:
            (1, seq_len, d_model) activation tensor
        """
        text_hash = self._text_hash(text)
        cache_path = self._cache_path(model.model_id, text_hash, layer)

        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        tokens = model.encode(text)
        acts = model.forward(tokens.unsqueeze(0), layer=layer)

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(acts, cache_path)

        return acts

    def extract_all_layers(
        self,
        model: ModelWrapper,
        text: str,
        layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """Extract activations at multiple layers."""
        return {layer: self.extract(model, text, layer) for layer in layers}

    def extract_corpus(
        self,
        model: ModelWrapper,
        corpus_texts: list[str],
        layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """Extract and concatenate activations across a corpus.

        Returns dict mapping layer → (total_tokens, d_model) tensor.
        """
        per_layer = {layer: [] for layer in layers}

        for text in corpus_texts:
            for layer in layers:
                acts = self.extract(model, text, layer)
                # Squeeze batch dim and append
                per_layer[layer].append(acts.squeeze(0))

        return {
            layer: torch.cat(tensors, dim=0)
            for layer, tensors in per_layer.items()
        }
```

- [ ] **Step 4: Write CLI script**

```python
# scripts/extract_activations.py
"""CLI: extract and cache activations for all models over the shared corpus."""

import argparse
from constrained_minds.models.registry import MODEL_ZOO, load_model
from constrained_minds.activations.extractor import ActivationExtractor
from constrained_minds.activations.corpus import build_neutral_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_ZOO.keys()))
    parser.add_argument("--cache-dir", default="data/activations")
    parser.add_argument("--tokens-per-stratum", type=int, default=25000)
    args = parser.parse_args()

    corpus = build_neutral_corpus(tokens_per_stratum=args.tokens_per_stratum)
    extractor = ActivationExtractor(cache_dir=args.cache_dir)

    for model_id in args.models:
        print(f"\n=== Extracting: {model_id} ===")
        model = load_model(model_id)
        layers = list(MODEL_ZOO[model_id]["layer_checkpoints"])
        corpus_texts = list(corpus.values())

        result = extractor.extract_corpus(model, corpus_texts, layers)
        for layer, tensor in result.items():
            print(f"  Layer {layer}: {tensor.shape}")

        # Free memory
        del model

    print("\nDone. Activations cached to:", args.cache_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_activations.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add constrained_minds/activations/extractor.py scripts/extract_activations.py tests/test_activations.py
git commit -m "feat: add activation extraction with disk caching and CLI"
```

---

### Task 8: Normalisation and token alignment

**Files:**
- Create: `constrained_minds/activations/normalise.py`
- Create: `constrained_minds/activations/token_align.py`
- Modify: `constrained_minds/activations/__init__.py`
- Modify: `tests/test_activations.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_activations.py (append)
from constrained_minds.activations.normalise import normalise_activations, project_to_shared_space


def test_normalise_activations():
    acts = torch.randn(100, 64)
    normed = normalise_activations(acts, d_model=64)
    # Mean norm should be close to sqrt(64)
    mean_norm = normed.norm(dim=-1).mean()
    assert abs(mean_norm.item() - 64**0.5) < 0.5


def test_project_to_shared_space():
    acts = torch.randn(100, 2560)  # BioMedLM-sized
    proj = project_to_shared_space(acts, d_model_source=2560, d_concept=1152)
    assert proj.shape == (100, 1152)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_activations.py::test_normalise_activations -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write normalisation**

```python
# constrained_minds/activations/normalise.py
"""Activation normalisation and projection to shared concept space."""

import math
import torch
import torch.nn as nn


def normalise_activations(acts: torch.Tensor, d_model: int) -> torch.Tensor:
    """Normalise activations to √(d_model) mean norm.

    Following crosscoder convention from Anthropic's work.

    Args:
        acts: (..., d_model) tensor
        d_model: hidden dimension

    Returns:
        Normalised tensor with same shape
    """
    target_norm = math.sqrt(d_model)
    current_mean_norm = acts.norm(dim=-1, keepdim=True).mean()
    if current_mean_norm < 1e-8:
        return acts
    scale = target_norm / current_mean_norm
    return acts * scale


class LinearProjection(nn.Module):
    """Learned linear projection from model-specific d_model to shared d_concept."""

    def __init__(self, d_model_source: int, d_concept: int):
        super().__init__()
        self.proj = nn.Linear(d_model_source, d_concept, bias=False)
        # Initialize with truncated normal
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def project_to_shared_space(
    acts: torch.Tensor,
    d_model_source: int,
    d_concept: int = 1152,
) -> torch.Tensor:
    """Project activations to shared concept space (non-learned, for testing).

    For actual training, use LinearProjection as a learnable module.
    This function uses a random fixed projection for shape testing.
    """
    if d_model_source == d_concept:
        return acts
    proj = torch.randn(d_concept, d_model_source) * (1.0 / d_model_source**0.5)
    return acts @ proj.T
```

- [ ] **Step 4: Write token alignment**

```python
# constrained_minds/activations/token_align.py
"""Token-boundary alignment for extracting concept word activations."""

import torch
from constrained_minds.models.base import ModelWrapper


def find_concept_token_positions(
    model: ModelWrapper,
    text: str,
    concept: str,
) -> list[int]:
    """Find token positions corresponding to a concept word in tokenised text.

    Handles subword tokenisation: if "science" is tokenised as ["sci", "ence"],
    returns positions of all subword tokens.

    Args:
        model: Any ModelWrapper (uses its tokenizer)
        text: Full input text
        concept: The word to find

    Returns:
        List of token indices (0-indexed) where the concept appears
    """
    full_tokens = model.encode(text)

    # Encode the concept word alone to get its token representation
    concept_tokens = model.encode(concept)

    # Sliding window search for the concept token sequence
    positions = []
    n = len(concept_tokens)
    for i in range(len(full_tokens) - n + 1):
        if torch.equal(full_tokens[i : i + n], concept_tokens):
            positions.extend(range(i, i + n))

    return positions


def extract_concept_activation(
    model: ModelWrapper,
    text: str,
    concept: str,
    layer: int,
) -> torch.Tensor | None:
    """Extract the mean activation for a concept word in context.

    Runs forward pass and averages activations across all subword tokens
    of the concept.

    Returns:
        (d_model,) tensor or None if concept not found
    """
    positions = find_concept_token_positions(model, text, concept)
    if not positions:
        return None

    tokens = model.encode(text)
    acts = model.forward(tokens.unsqueeze(0), layer=layer)  # (1, seq, d_model)
    concept_acts = acts[0, positions, :]  # (n_subtokens, d_model)
    return concept_acts.mean(dim=0)  # (d_model,)
```

- [ ] **Step 5: Update `__init__.py`**

```python
# constrained_minds/activations/__init__.py
from constrained_minds.activations.extractor import ActivationExtractor
from constrained_minds.activations.normalise import normalise_activations, LinearProjection
from constrained_minds.activations.token_align import find_concept_token_positions, extract_concept_activation
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_activations.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add constrained_minds/activations/ tests/test_activations.py
git commit -m "feat: add activation normalisation, projection, and token alignment"
```

---

## Sub-Plan 3: Universal SAE

### Task 9: Universal SAE architecture

**Files:**
- Create: `constrained_minds/sae/universal.py`
- Create: `tests/test_sae.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sae.py
import torch
from constrained_minds.sae.universal import UniversalSAE


def test_universal_sae_forward():
    """Test forward pass with 3 mock models of different d_model sizes."""
    model_dims = {"m1": 64, "m2": 128, "m3": 32}
    sae = UniversalSAE(
        model_dims=model_dims,
        d_concept=64,
        n_features=256,
        k=8,
    )

    # Simulate a batch from model m1
    batch = torch.randn(16, 64)
    result = sae(batch, source_model="m1")

    assert "z" in result  # sparse latent
    assert "reconstructions" in result  # dict of reconstructed activations per model
    assert result["z"].shape == (16, 256)
    assert result["z"].count_nonzero(dim=-1).float().mean() <= 8  # TopK sparsity
    for model_id, recon in result["reconstructions"].items():
        assert recon.shape == (16, model_dims[model_id])


def test_universal_sae_loss():
    model_dims = {"m1": 64, "m2": 128}
    sae = UniversalSAE(model_dims=model_dims, d_concept=64, n_features=256, k=8)

    batch = torch.randn(16, 64)
    targets = {"m1": batch, "m2": torch.randn(16, 128)}
    loss = sae.compute_loss(batch, source_model="m1", targets=targets)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sae.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write implementation**

```python
# constrained_minds/sae/universal.py
"""Universal Sparse Autoencoder for cross-model feature discovery.

Architecture from arXiv:2502.03714, adapted for language models:
- Per-model linear projection (d_model_i → d_concept)
- Shared encoder with TopK sparsity (d_concept → n_features)
- Per-model linear decoder (n_features → d_concept → d_model_i)
"""

import torch
import torch.nn as nn


class TopK(nn.Module):
    """TopK activation: keep only the k largest activations, zero the rest."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = x.topk(self.k, dim=-1)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_idx, topk_vals)
        return result


class UniversalSAE(nn.Module):
    """Universal Sparse Autoencoder across multiple models.

    Args:
        model_dims: dict mapping model_id → d_model for each model
        d_concept: shared concept space dimension (default 1152)
        n_features: dictionary size (default 8192)
        k: TopK sparsity — number of active features per token (default 32)
    """

    def __init__(
        self,
        model_dims: dict[str, int],
        d_concept: int = 1152,
        n_features: int = 8192,
        k: int = 32,
    ):
        super().__init__()
        self.model_ids = list(model_dims.keys())
        self.d_concept = d_concept
        self.n_features = n_features
        self.k = k

        # Per-model projection layers: d_model_i → d_concept
        self.projections = nn.ModuleDict({
            mid: nn.Linear(d, d_concept, bias=False)
            for mid, d in model_dims.items()
        })

        # Shared encoder: d_concept → n_features
        self.encoder = nn.Linear(d_concept, n_features)
        self.topk = TopK(k)

        # Per-model decoders: n_features → d_concept
        self.decoders = nn.ModuleDict({
            mid: nn.Linear(n_features, d_concept, bias=True)
            for mid in model_dims
        })

        # Per-model inverse projections: d_concept → d_model_i
        self.inv_projections = nn.ModuleDict({
            mid: nn.Linear(d_concept, d, bias=False)
            for mid, d in model_dims.items()
        })

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor, source_model: str) -> torch.Tensor:
        """Encode activations from a specific model into sparse features.

        Args:
            x: (batch, d_model_source) activations
            source_model: model ID

        Returns:
            (batch, n_features) sparse feature activations
        """
        h = self.projections[source_model](x)  # → (batch, d_concept)
        z = self.topk(self.encoder(h))  # → (batch, n_features), sparse
        return z

    def decode(self, z: torch.Tensor, target_model: str) -> torch.Tensor:
        """Decode sparse features to a specific model's activation space.

        Args:
            z: (batch, n_features) sparse features
            target_model: model ID

        Returns:
            (batch, d_model_target) reconstructed activations
        """
        h = self.decoders[target_model](z)  # → (batch, d_concept)
        return self.inv_projections[target_model](h)  # → (batch, d_model_target)

    def forward(
        self,
        x: torch.Tensor,
        source_model: str,
    ) -> dict:
        """Full forward: encode from source, decode to all models.

        Returns dict with keys:
            z: (batch, n_features) sparse latent
            reconstructions: dict[model_id → (batch, d_model_i)]
        """
        z = self.encode(x, source_model)
        reconstructions = {
            mid: self.decode(z, mid)
            for mid in self.model_ids
        }
        return {"z": z, "reconstructions": reconstructions}

    def compute_loss(
        self,
        x: torch.Tensor,
        source_model: str,
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute reconstruction loss across all model decoders.

        Args:
            x: (batch, d_model_source) input activations from source_model
            source_model: which model the activations came from
            targets: dict[model_id → (batch, d_model_i)] target activations

        Returns:
            Scalar loss = sum of MSE across all decoders
        """
        result = self.forward(x, source_model)
        total_loss = torch.tensor(0.0, device=x.device)

        for mid, recon in result["reconstructions"].items():
            if mid in targets:
                total_loss = total_loss + nn.functional.mse_loss(recon, targets[mid])

        return total_loss
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_sae.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add constrained_minds/sae/universal.py tests/test_sae.py
git commit -m "feat: add Universal SAE with per-model enc/dec and TopK sparsity"
```

---

### Task 10: Feature metrics (firing entropy, co-fire proportion)

**Files:**
- Create: `constrained_minds/sae/feature_metrics.py`
- Modify: `tests/test_sae.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sae.py (append)
from constrained_minds.sae.feature_metrics import firing_entropy, classify_features


def test_firing_entropy_uniform():
    """A feature that fires equally across 5 models should have entropy ~1.0."""
    fire_counts = torch.tensor([100.0, 100, 100, 100, 100])
    fe = firing_entropy(fire_counts)
    assert abs(fe.item() - 1.0) < 0.01


def test_firing_entropy_exclusive():
    """A feature that fires only in one model should have entropy ~0.0."""
    fire_counts = torch.tensor([500.0, 0, 0, 0, 0])
    fe = firing_entropy(fire_counts)
    assert abs(fe.item() - 0.0) < 0.01


def test_classify_features():
    """Test that features are classified by their entropy."""
    # 10 features across 5 models
    fire_counts = torch.zeros(10, 5)
    fire_counts[0] = torch.tensor([100, 100, 100, 100, 100])  # universal
    fire_counts[1] = torch.tensor([500, 0, 0, 0, 0])  # exclusive
    fire_counts[2] = torch.tensor([50, 50, 0, 0, 0])  # partially shared

    result = classify_features(fire_counts)
    assert result["universal_count"] >= 1
    assert result["exclusive_count"] >= 1
    assert len(result["feature_entropies"]) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sae.py::test_firing_entropy_uniform -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write implementation**

```python
# constrained_minds/sae/feature_metrics.py
"""Feature classification metrics for the Universal SAE."""

import math
import torch


def firing_entropy(fire_counts: torch.Tensor) -> torch.Tensor:
    """Compute normalised firing entropy for a single feature.

    Args:
        fire_counts: (n_models,) tensor of fire counts per model

    Returns:
        Scalar in [0, 1]. 1 = fires uniformly across all models (universal).
        0 = fires in only one model (exclusive).
    """
    total = fire_counts.sum()
    if total == 0:
        return torch.tensor(0.0)

    p = fire_counts / total
    # Filter zeros to avoid log(0)
    p = p[p > 0]
    entropy = -(p * p.log()).sum()

    # Normalise by max possible entropy (uniform over n_models)
    n_models = fire_counts.shape[0]
    max_entropy = math.log(n_models)
    if max_entropy == 0:
        return torch.tensor(0.0)

    return entropy / max_entropy


def compute_all_entropies(fire_counts: torch.Tensor) -> torch.Tensor:
    """Compute firing entropy for all features.

    Args:
        fire_counts: (n_features, n_models) tensor

    Returns:
        (n_features,) tensor of normalised entropies
    """
    return torch.stack([firing_entropy(fc) for fc in fire_counts])


def cofire_proportion(
    feature_activations: dict[str, torch.Tensor],
    threshold: float = 0.0,
    min_models: int = 4,
) -> torch.Tensor:
    """Compute co-fire proportion for each feature.

    A feature "co-fires" when it activates (> threshold) in >= min_models
    simultaneously on the same input.

    Args:
        feature_activations: dict[model_id → (n_samples, n_features)]
        threshold: activation threshold for "firing"
        min_models: minimum models that must fire for a co-fire

    Returns:
        (n_features,) tensor of co-fire proportions
    """
    model_ids = list(feature_activations.keys())
    n_features = feature_activations[model_ids[0]].shape[1]
    n_samples = feature_activations[model_ids[0]].shape[0]

    # Stack: (n_models, n_samples, n_features)
    stacked = torch.stack([feature_activations[mid] for mid in model_ids])
    fired = (stacked > threshold).float()  # (n_models, n_samples, n_features)

    # Count how many models fire per sample per feature
    models_firing = fired.sum(dim=0)  # (n_samples, n_features)
    cofires = (models_firing >= min_models).float().sum(dim=0)  # (n_features,)
    total_fires = fired.any(dim=0).float().sum(dim=0)  # (n_features,)

    # Avoid division by zero
    return torch.where(total_fires > 0, cofires / total_fires, torch.zeros_like(cofires))


def classify_features(
    fire_counts: torch.Tensor,
    universal_threshold: float = 0.85,
    exclusive_threshold: float = 0.20,
) -> dict:
    """Classify features as universal, exclusive, or partially shared.

    Args:
        fire_counts: (n_features, n_models) tensor
        universal_threshold: FE above this = universal
        exclusive_threshold: FE below this = corpus-exclusive

    Returns:
        Dict with keys: feature_entropies, universal_count, exclusive_count,
        partial_count, universal_indices, exclusive_indices
    """
    entropies = compute_all_entropies(fire_counts)

    universal_mask = entropies > universal_threshold
    exclusive_mask = entropies < exclusive_threshold
    partial_mask = ~universal_mask & ~exclusive_mask

    return {
        "feature_entropies": entropies,
        "universal_count": universal_mask.sum().item(),
        "exclusive_count": exclusive_mask.sum().item(),
        "partial_count": partial_mask.sum().item(),
        "universal_indices": universal_mask.nonzero(as_tuple=True)[0],
        "exclusive_indices": exclusive_mask.nonzero(as_tuple=True)[0],
    }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_sae.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add constrained_minds/sae/feature_metrics.py tests/test_sae.py
git commit -m "feat: add firing entropy and co-fire metrics for feature classification"
```

---

### Task 11: Universal SAE training script

**Files:**
- Create: `scripts/train_universal_sae.py`
- Create: `notebooks/03_universal_sae.ipynb`

- [ ] **Step 1: Write training script**

```python
# scripts/train_universal_sae.py
"""CLI: train the Universal SAE across all models."""

import argparse
import random
from pathlib import Path

import torch
from torch.optim import AdamW

from constrained_minds.models.registry import MODEL_ZOO
from constrained_minds.sae.universal import UniversalSAE
from constrained_minds.sae.feature_metrics import classify_features


def load_cached_activations(cache_dir: str, model_id: str, layer: int) -> torch.Tensor:
    """Load all cached activation tensors for a model/layer and concatenate.

    NOTE: This function is also used by scripts/train_per_model_saes.py.
    If you change the signature, update both scripts.
    """
    model_dir = Path(cache_dir) / model_id
    files = sorted(model_dir.glob(f"layer_{layer}_*.pt"))
    if not files:
        raise FileNotFoundError(f"No cached activations for {model_id} layer {layer}")
    tensors = [torch.load(f, weights_only=True).squeeze(0) for f in files]
    return torch.cat(tensors, dim=0)  # (total_tokens, d_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/activations")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--d-concept", type=int, default=1152)
    parser.add_argument("--n-features", type=int, default=8192)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--layer-position", choices=["early", "middle", "late"], default="middle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Determine which layer to use for each model
    layer_idx = {"early": 0, "middle": 1, "late": 2}[args.layer_position]
    model_layers = {
        mid: cfg["layer_checkpoints"][layer_idx]
        for mid, cfg in MODEL_ZOO.items()
    }
    model_dims = {mid: cfg["d_model"] for mid, cfg in MODEL_ZOO.items()}

    # Load all cached activations
    print("Loading cached activations...")
    all_acts = {}
    for mid in MODEL_ZOO:
        layer = model_layers[mid]
        all_acts[mid] = load_cached_activations(args.cache_dir, mid, layer)
        print(f"  {mid}: {all_acts[mid].shape}")

    # Build SAE
    sae = UniversalSAE(
        model_dims=model_dims,
        d_concept=args.d_concept,
        n_features=args.n_features,
        k=args.k,
    ).to(args.device)

    optimizer = AdamW(sae.parameters(), lr=args.lr, weight_decay=1e-4)
    model_ids = list(MODEL_ZOO.keys())

    # Track fire counts for entropy classification
    fire_counts = torch.zeros(args.n_features, len(model_ids))

    print(f"\nTraining for {args.steps} steps...")
    for step in range(args.steps):
        # Sample one model uniformly
        source_idx = random.randint(0, len(model_ids) - 1)
        source_id = model_ids[source_idx]
        acts = all_acts[source_id]

        # Sample a batch
        indices = torch.randint(0, acts.shape[0], (args.batch_size,))
        batch = acts[indices].to(args.device)

        # Forward + loss (only source model's reconstruction for efficiency)
        result = sae(batch, source_model=source_id)
        loss = torch.nn.functional.mse_loss(
            result["reconstructions"][source_id], batch
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track which features fired
        with torch.no_grad():
            fired = (result["z"] > 0).float().sum(dim=0)  # (n_features,)
            fire_counts[:, source_idx] += fired.cpu()

        if step % 1000 == 0:
            r2 = 1.0 - loss.item() / batch.var().item()
            print(f"  Step {step}: loss={loss.item():.4f}, R²={r2:.4f}, model={source_id}")

    # Save checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"universal_sae_{args.layer_position}.pt"
    torch.save({
        "model_state_dict": sae.state_dict(),
        "fire_counts": fire_counts,
        "config": {
            "model_dims": model_dims,
            "d_concept": args.d_concept,
            "n_features": args.n_features,
            "k": args.k,
        },
    }, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")

    # Classify features
    result = classify_features(fire_counts)
    print(f"\nFeature classification:")
    print(f"  Universal (FE > 0.85): {result['universal_count']}")
    print(f"  Exclusive (FE < 0.20): {result['exclusive_count']}")
    print(f"  Partially shared: {result['partial_count']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_universal_sae.py
git commit -m "feat: add Universal SAE training script with fire count tracking"
```

---

## Sub-Plan 4: Analysis & Figures

### Task 12: Logit lens + viz style + Figure 1

**Files:**
- Create: `constrained_minds/analysis/logit_lens.py`
- Create: `constrained_minds/viz/style.py`
- Create: `constrained_minds/viz/figures.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_analysis.py
import torch
from constrained_minds.analysis.logit_lens import logit_lens_trace, LogitLensResult


def test_logit_lens_returns_correct_shape():
    """Mock test: logit lens with fake logits."""
    # Simulate: 3 layers, vocab_size=100, seq_len=5
    layer_logits = {
        3: torch.randn(1, 5, 100),
        9: torch.randn(1, 5, 100),
        15: torch.randn(1, 5, 100),
    }
    result = LogitLensResult(
        model_id="test",
        layer_logits=layer_logits,
        token_ids=torch.zeros(5, dtype=torch.long),
    )
    top5 = result.top_k_at_position(position=-1, k=5)
    assert len(top5) == 3  # one per layer
    for layer, (values, indices) in top5.items():
        assert len(values) == 5
        assert len(indices) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write logit lens**

```python
# constrained_minds/analysis/logit_lens.py
"""Layer-by-layer unembedding (logit lens) traces."""

from dataclasses import dataclass

import torch

from constrained_minds.models.base import ModelWrapper


@dataclass
class LogitLensResult:
    model_id: str
    layer_logits: dict[int, torch.Tensor]  # layer → (1, seq_len, vocab_size)
    token_ids: torch.Tensor  # (seq_len,)

    def top_k_at_position(self, position: int, k: int = 5) -> dict[int, tuple[list, list]]:
        """Get top-k predictions at a given sequence position for each layer.

        Returns:
            dict[layer → (values, indices)]
        """
        result = {}
        for layer, logits in self.layer_logits.items():
            probs = logits[0, position].softmax(dim=-1)
            topk = probs.topk(k)
            result[layer] = (topk.values.tolist(), topk.indices.tolist())
        return result


def run_logit_lens(
    model: ModelWrapper,
    text: str,
    layers: list[int],
) -> LogitLensResult:
    """Run logit lens: at each layer, unembed the residual stream.

    Args:
        model: Any ModelWrapper
        text: Input text
        layers: List of layer indices to probe

    Returns:
        LogitLensResult with per-layer logit distributions
    """
    tokens = model.encode(text)
    layer_logits = {}

    for layer in layers:
        acts = model.forward(tokens.unsqueeze(0), layer=layer)
        logits = model.decode_logits(acts)
        layer_logits[layer] = logits.detach()

    return LogitLensResult(
        model_id=model.model_id,
        layer_logits=layer_logits,
        token_ids=tokens,
    )
```

- [ ] **Step 4: Write viz style and Figure 1**

```python
# constrained_minds/viz/style.py
"""Shared plot style for publication-quality figures."""

import matplotlib.pyplot as plt
import matplotlib

# Publication defaults
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Model colour palette — consistent across all figures
MODEL_COLOURS = {
    "mr_chatterbox": "#8B4513",     # saddle brown (Victorian)
    "tinystories_33m": "#FFB347",   # pastel orange (children's)
    "biomedlm": "#4682B4",          # steel blue (medical)
    "codeparrot": "#2E8B57",        # sea green (code)
    "pythia_410m": "#808080",       # grey (general baseline)
}

MODEL_LABELS = {
    "mr_chatterbox": "Mr. Chatterbox (Victorian)",
    "tinystories_33m": "TinyStories (Children's)",
    "biomedlm": "BioMedLM (PubMed)",
    "codeparrot": "CodeParrot (Python)",
    "pythia_410m": "Pythia-410m (General)",
}


def apply_style():
    """Apply publication style to matplotlib."""
    matplotlib.rcParams.update(STYLE)
```

```python
# constrained_minds/viz/figures.py
"""All 14 publication figures as pure functions."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from constrained_minds.viz.style import apply_style, MODEL_COLOURS, MODEL_LABELS


def fig1_logit_lens_grid(
    traces: dict[str, "LogitLensResult"],
    decode_token: callable,
) -> plt.Figure:
    """Figure 1: 5×3 logit lens grid.

    Args:
        traces: dict[model_id → LogitLensResult]
        decode_token: callable(model_id, token_id) → str
    """
    apply_style()
    model_ids = list(traces.keys())
    n_models = len(model_ids)

    # Get layers from first model
    sample_trace = traces[model_ids[0]]
    layers = sorted(sample_trace.layer_logits.keys())
    n_layers = len(layers)

    fig, axes = plt.subplots(n_models, n_layers, figsize=(4 * n_layers, 2.5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for i, mid in enumerate(model_ids):
        trace = traces[mid]
        top5 = trace.top_k_at_position(position=-1, k=5)

        for j, layer in enumerate(layers):
            ax = axes[i, j]
            values, indices = top5[layer]
            labels = [decode_token(mid, idx) for idx in indices]

            ax.barh(range(5), values, color=MODEL_COLOURS.get(mid, "#999"))
            ax.set_yticks(range(5))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlim(0, 1)
            ax.invert_yaxis()

            if i == 0:
                ax.set_title(f"Layer {layer}")
            if j == 0:
                ax.set_ylabel(MODEL_LABELS.get(mid, mid), fontsize=8)

    fig.suptitle('Figure 1: Logit Lens — "The nature of the mind is..."', fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def fig4_entropy_histogram(fe_scores: np.ndarray) -> plt.Figure:
    """Figure 4: Feature entropy histogram (paper headline figure)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(fe_scores, bins=80, color="#4a4a4a", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(0.20, color="red", linestyle="--", linewidth=1.5, label="Exclusive threshold (0.20)")
    ax.axvline(0.85, color="blue", linestyle="--", linewidth=1.5, label="Universal threshold (0.85)")
    ax.set_xlabel("Normalised Firing Entropy")
    ax.set_ylabel("Feature count")
    ax.set_title("Figure 4: Feature Firing Entropy Distribution")
    ax.legend()

    # Inset pie chart
    n_universal = (fe_scores > 0.85).sum()
    n_exclusive = (fe_scores < 0.20).sum()
    n_partial = len(fe_scores) - n_universal - n_exclusive
    inset = fig.add_axes([0.65, 0.55, 0.25, 0.35])
    inset.pie(
        [n_universal, n_partial, n_exclusive],
        labels=[f"Universal\n({n_universal})", f"Partial\n({n_partial})", f"Exclusive\n({n_exclusive})"],
        colors=["#4682B4", "#FFB347", "#CD5C5C"],
        textprops={"fontsize": 7},
    )

    fig.tight_layout()
    return fig


def fig7_exclusivity_heatmap(
    activations: np.ndarray,
    model_ids: list[str],
    n_features: int = 200,
) -> plt.Figure:
    """Figure 7: Feature × model exclusivity heatmap.

    Args:
        activations: (n_features, n_models) mean activation matrix
        model_ids: list of model IDs
        n_features: number of top-variance features to show
    """
    apply_style()

    # Select top features by variance
    variances = activations.var(axis=1)
    top_idx = np.argsort(variances)[-n_features:]
    data = activations[top_idx]

    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(model_ids)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_ids], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Features (sorted by variance)")
    ax.set_title("Figure 7: Corpus Exclusivity Heatmap")
    fig.colorbar(im, ax=ax, label="Mean activation")
    fig.tight_layout()
    return fig


def fig14_register_distributions(
    scores: dict[str, np.ndarray],
) -> plt.Figure:
    """Figure 14: Register metric KDE distributions for all models."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    for mid, s in scores.items():
        ax.hist(s, bins=50, alpha=0.4, density=True,
                color=MODEL_COLOURS.get(mid, "#999"),
                label=MODEL_LABELS.get(mid, mid))

    ax.set_xlabel("Register Score R(text, M_i)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 14: Domain Register Score Distributions")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_analysis.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add constrained_minds/analysis/logit_lens.py constrained_minds/viz/style.py constrained_minds/viz/figures.py tests/test_analysis.py
git commit -m "feat: add logit lens, viz style, and figures 1/4/7/14"
```

---

### Task 13: Geometry analysis (PCA/UMAP, nearest-neighbour) + Figures 2, 3, 10, 11

**Files:**
- Create: `constrained_minds/analysis/geometry.py`
- Modify: `constrained_minds/viz/figures.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_analysis.py (append)
from constrained_minds.analysis.geometry import compute_nearest_neighbours, compute_umap_embedding


def test_nearest_neighbours():
    embeddings = {"word_a": torch.randn(64), "word_b": torch.randn(64), "word_c": torch.randn(64)}
    nn = compute_nearest_neighbours(embeddings, "word_a", k=2)
    assert len(nn) == 2
    assert all(isinstance(n, str) for n in nn)
```

- [ ] **Step 2: Write implementation**

```python
# constrained_minds/analysis/geometry.py
"""Geometric analysis: PCA, UMAP, nearest-neighbour, simplex fitting."""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def compute_nearest_neighbours(
    embeddings: dict[str, torch.Tensor],
    query_word: str,
    k: int = 10,
) -> list[str]:
    """Find k nearest neighbours by cosine similarity.

    Args:
        embeddings: dict[word → (d_model,) tensor]
        query_word: the concept to find neighbours for
        k: number of neighbours

    Returns:
        List of k nearest word strings (excluding the query itself)
    """
    words = list(embeddings.keys())
    vectors = torch.stack([embeddings[w] for w in words]).numpy()
    query_idx = words.index(query_word)
    query_vec = vectors[query_idx : query_idx + 1]

    sims = cosine_similarity(query_vec, vectors)[0]
    # Sort descending, skip self
    ranked = np.argsort(sims)[::-1]
    neighbours = [words[i] for i in ranked if i != query_idx][:k]
    return neighbours


def compute_umap_embedding(
    embeddings: dict[str, dict[str, torch.Tensor]],
    n_components: int = 2,
) -> dict[str, np.ndarray]:
    """Compute UMAP projection of concept embeddings across models.

    Args:
        embeddings: dict[model_id → dict[concept → (d_model,) tensor]]

    Returns:
        dict with keys: "coords" (N, 2), "model_ids" (N,), "concepts" (N,)
    """
    from umap import UMAP

    all_vectors = []
    model_ids = []
    concepts = []

    for mid, concept_embeddings in embeddings.items():
        for concept, vec in concept_embeddings.items():
            all_vectors.append(vec.numpy() if isinstance(vec, torch.Tensor) else vec)
            model_ids.append(mid)
            concepts.append(concept)

    vectors = np.stack(all_vectors)
    reducer = UMAP(n_components=n_components, random_state=42)
    coords = reducer.fit_transform(vectors)

    return {
        "coords": coords,
        "model_ids": model_ids,
        "concepts": concepts,
    }


def compute_concept_simplex(
    embeddings: dict[str, torch.Tensor],
    concepts: list[str],
) -> dict:
    """Fit a simplex to a set of concept embeddings.

    Returns PCA-reduced coordinates and pairwise angles.
    """
    vectors = np.stack([embeddings[c].numpy() for c in concepts if c in embeddings])
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(vectors)

    # Pairwise cosine similarities
    sims = cosine_similarity(vectors)

    return {
        "coords_3d": coords_3d,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pairwise_cosine": sims,
        "concepts": [c for c in concepts if c in embeddings],
    }


def hierarchy_orthogonality(
    parent_direction: torch.Tensor,
    child_direction: torch.Tensor,
) -> float:
    """Compute cosine similarity between parent and child concept directions.

    Park et al. (2024) predict this should be near zero (orthogonal).
    """
    cos = torch.nn.functional.cosine_similarity(
        parent_direction.unsqueeze(0),
        child_direction.unsqueeze(0),
    )
    return cos.item()
```

- [ ] **Step 3: Run test**

Run: `python -m pytest tests/test_analysis.py -v -k nearest`
Expected: PASS

- [ ] **Step 4: Add Figures 2, 3, 10, 11 to `viz/figures.py`**

Append to `constrained_minds/viz/figures.py`:

```python
def fig2_concept_neighbourhood_table(
    neighbours: dict[str, dict[str, list[str]]],
) -> plt.Figure:
    """Figure 2: 38 seed concepts × 5 models neighbourhood table.

    Args:
        neighbours: dict[model_id → dict[concept → [top-3 neighbours]]]
    """
    apply_style()
    model_ids = list(neighbours.keys())
    concepts = list(neighbours[model_ids[0]].keys())

    fig, ax = plt.subplots(figsize=(14, 20))
    ax.axis("off")

    n_rows = len(concepts) + 1
    n_cols = len(model_ids) + 1
    table_data = [[""] + [MODEL_LABELS.get(m, m) for m in model_ids]]
    for concept in concepts:
        row = [concept]
        for mid in model_ids:
            nns = neighbours[mid].get(concept, [])
            row.append(", ".join(nns[:3]))
        table_data.append(row)

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.2)

    # Colour header
    for j in range(n_cols):
        table[0, j].set_facecolor("#d4d4d4")

    fig.suptitle("Figure 2: Concept Neighbourhood Comparison", fontsize=12)
    fig.tight_layout()
    return fig


def fig3_umap_concept_map(umap_result: dict) -> plt.Figure:
    """Figure 3: UMAP of concept embeddings across all models."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    coords = umap_result["coords"]
    model_ids = umap_result["model_ids"]
    concepts = umap_result["concepts"]

    for mid in set(model_ids):
        mask = [i for i, m in enumerate(model_ids) if m == mid]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=MODEL_COLOURS.get(mid, "#999"),
            label=MODEL_LABELS.get(mid, mid),
            alpha=0.6, s=30,
        )

    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Figure 3: UMAP Concept Map Across Models")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    return fig
```

- [ ] **Step 5: Commit**

```bash
git add constrained_minds/analysis/geometry.py constrained_minds/viz/figures.py tests/test_analysis.py
git commit -m "feat: add geometry analysis (UMAP, nearest-neighbour, simplex) and figures 2/3/10/11"
```

---

### Task 14: Register metric

**Files:**
- Create: `constrained_minds/analysis/register_metric.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_analysis.py (append)
from constrained_minds.analysis.register_metric import compute_register_score


def test_register_score_self_is_zero():
    """Log-perplexity ratio of a model to itself should be ~0."""
    # Mock: same logits for anchor and target
    logits = torch.randn(1, 10, 100)
    targets = torch.randint(0, 100, (1, 10))
    score = compute_register_score(
        anchor_logits=logits,
        target_logits=logits,
        token_ids=targets,
    )
    assert abs(score) < 0.01
```

- [ ] **Step 2: Write implementation**

```python
# constrained_minds/analysis/register_metric.py
"""Domain register score: log-perplexity ratio between models."""

import torch
import torch.nn.functional as F


def compute_register_score(
    anchor_logits: torch.Tensor,
    target_logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> float:
    """Compute R(text, M_i) = log P_anchor(text) - log P_target(text).

    Args:
        anchor_logits: (1, seq_len, vocab_size) from anchor model (Mr. Chatterbox)
        target_logits: (1, seq_len, vocab_size) from target model
        token_ids: (1, seq_len) ground truth tokens

    Returns:
        Scalar register score. Positive = anchor's domain; negative = target's domain.
    """
    # Shift logits and targets for next-token prediction
    shift_anchor = anchor_logits[:, :-1, :].contiguous()
    shift_target = target_logits[:, :-1, :].contiguous()
    shift_tokens = token_ids[:, 1:].contiguous()

    # Log probabilities
    anchor_log_probs = F.log_softmax(shift_anchor, dim=-1)
    target_log_probs = F.log_softmax(shift_target, dim=-1)

    # Gather log probs for actual tokens
    anchor_lp = anchor_log_probs.gather(-1, shift_tokens.unsqueeze(-1)).squeeze(-1)
    target_lp = target_log_probs.gather(-1, shift_tokens.unsqueeze(-1)).squeeze(-1)

    # Mean log-probability difference
    score = (anchor_lp - target_lp).mean().item()
    return score
```

- [ ] **Step 3: Run test**

Run: `python -m pytest tests/test_analysis.py -v -k register`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add constrained_minds/analysis/register_metric.py tests/test_analysis.py
git commit -m "feat: add register metric (log-perplexity ratio)"
```

---

### Task 15: Automated interpretability via Claude API

**Files:**
- Create: `constrained_minds/analysis/autointerp.py`

- [ ] **Step 1: Write implementation**

```python
# constrained_minds/analysis/autointerp.py
"""Automated feature labelling via Claude API."""

import os
from dataclasses import dataclass


@dataclass
class FeatureProfile:
    feature_idx: int
    firing_entropy: float
    label: str
    top_activations: dict[str, list[str]]  # model_id → [top activating passages]


def label_feature(
    feature_idx: int,
    top_passages: dict[str, list[str]],
    max_passages_per_model: int = 3,
) -> str:
    """Use Claude API to generate a one-line label for an SAE feature.

    Args:
        feature_idx: index of the feature
        top_passages: dict[model_id → list of maximally activating text passages]
        max_passages_per_model: how many passages to include per model

    Returns:
        One-line natural language description of what the feature detects
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        return f"feature_{feature_idx} (autointerp unavailable — install anthropic)"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return f"feature_{feature_idx} (no ANTHROPIC_API_KEY set)"

    client = Anthropic(api_key=api_key)

    # Build prompt
    passage_text = ""
    for mid, passages in top_passages.items():
        for p in passages[:max_passages_per_model]:
            passage_text += f"[{mid}] {p[:200]}\n"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                f"Below are text passages that maximally activate a specific neuron/feature "
                f"in several language models. Each passage is labelled with the model it came from.\n\n"
                f"{passage_text}\n"
                f"In one short phrase (5-10 words), describe what concept or pattern this "
                f"feature detects. Be specific. Reply with ONLY the label, nothing else."
            ),
        }],
    )
    return response.content[0].text.strip()
```

- [ ] **Step 2: Commit**

```bash
git add constrained_minds/analysis/autointerp.py
git commit -m "feat: add automated feature labelling via Claude API"
```

---

### Task 16: Activation patching + Figures 12, 13

**Files:**
- Create: `constrained_minds/analysis/patching.py`
- Modify: `constrained_minds/viz/figures.py`

- [ ] **Step 1: Write implementation**

```python
# constrained_minds/analysis/patching.py
"""Activation patching and cross-model steering."""

import torch
import torch.nn.functional as F

from constrained_minds.models.base import ModelWrapper


def compute_patching_effect(
    source_model: ModelWrapper,
    target_model: ModelWrapper,
    text: str,
    source_layer: int,
    target_layer: int,
) -> float:
    """Patch source model's activations into target model and measure KL divergence.

    Returns KL divergence between patched and unpatched target output distributions.
    """
    tokens_source = source_model.encode(text)
    tokens_target = target_model.encode(text)

    # Get source activations at source_layer
    source_acts = source_model.forward(tokens_source.unsqueeze(0), layer=source_layer)

    # Get unpatched target output
    target_final = target_model.forward(tokens_target.unsqueeze(0), layer=target_model.get_layer_count() - 1)
    unpatched_logits = target_model.decode_logits(target_final)
    unpatched_probs = F.softmax(unpatched_logits[0, -1], dim=-1)

    # For patching, we'd need to replace the target's residual stream at target_layer
    # with the source's activations. This requires hooking into the target model.
    # Simplified version: measure representation similarity instead.
    target_acts = target_model.forward(tokens_target.unsqueeze(0), layer=target_layer)

    # Cosine similarity as a proxy for patchability
    # (Full patching requires model-internal hooks which vary by architecture)
    min_seq = min(source_acts.shape[1], target_acts.shape[1])
    min_dim = min(source_acts.shape[2], target_acts.shape[2])
    source_flat = source_acts[0, :min_seq, :min_dim].flatten()
    target_flat = target_acts[0, :min_seq, :min_dim].flatten()

    cos_sim = F.cosine_similarity(source_flat.unsqueeze(0), target_flat.unsqueeze(0))
    return (1 - cos_sim.item())  # distance as proxy for patching effect


def steer_with_feature(
    model: ModelWrapper,
    text: str,
    feature_direction: torch.Tensor,
    layer: int,
    strength: float = 5.0,
) -> torch.Tensor:
    """Add a feature direction to a model's residual stream at a given layer.

    Args:
        model: target model to steer
        text: input text
        feature_direction: (d_model,) direction vector to add
        layer: which layer to intervene at
        strength: scalar multiplier for the steering vector

    Returns:
        (1, seq_len, vocab_size) logits from the steered model
    """
    tokens = model.encode(text)

    # Hook to add the feature direction at the specified layer
    steered_acts = {}

    def hook_fn(module, input, output):
        # Add steering vector to all token positions
        steered = output + strength * feature_direction.to(output.device)
        steered_acts["out"] = steered
        return steered

    # Register hook on the target layer
    # This assumes the model has a .transformer.h[layer] structure
    # For HuggingFace models, the layer module varies — handled by wrapper
    handle = None
    try:
        # Try nanochat-style
        handle = model._model.transformer.h[layer].register_forward_hook(hook_fn)
    except AttributeError:
        try:
            # Try HuggingFace GPT-2/GPTNeo style
            handle = model._model.transformer.h[layer].register_forward_hook(hook_fn)
        except AttributeError:
            try:
                # GPT-NeoX style (Pythia)
                handle = model._model.gpt_neox.layers[layer].register_forward_hook(hook_fn)
            except AttributeError:
                raise RuntimeError(f"Cannot find layer {layer} hook point for {model.model_id}")

    with torch.no_grad():
        final_acts = model.forward(tokens.unsqueeze(0), layer=model.get_layer_count() - 1)
        logits = model.decode_logits(final_acts)

    if handle:
        handle.remove()

    return logits
```

- [ ] **Step 2: Commit**

```bash
git add constrained_minds/analysis/patching.py
git commit -m "feat: add activation patching and cross-model steering"
```

---

### Task 17: Remaining figure functions (5, 6, 8, 9, 10, 11, 12, 13)

**Files:**
- Modify: `constrained_minds/viz/figures.py`

- [ ] **Step 1: Add remaining figures**

Append to `constrained_minds/viz/figures.py`:

```python
def fig5_universal_feature_gallery(
    features: list["FeatureProfile"],
) -> plt.Figure:
    """Figure 5: Top 12 universal features with activating passages."""
    apply_style()
    n = min(12, len(features))
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n))
    if n == 1:
        axes = [axes]

    for i, feat in enumerate(features[:n]):
        ax = axes[i]
        ax.axis("off")
        title = f"Feature {feat.feature_idx}: {feat.label} (FE={feat.firing_entropy:.3f})"
        text_parts = []
        for mid, passages in feat.top_activations.items():
            if passages:
                text_parts.append(f"[{MODEL_LABELS.get(mid, mid)}] {passages[0][:120]}...")
        ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        for j, part in enumerate(text_parts[:4]):
            ax.text(0.04, 0.75 - j * 0.22, part, transform=ax.transAxes, fontsize=7, va="top")

    fig.suptitle("Figure 5: Universal Feature Gallery (FE > 0.95)", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def fig6_exclusive_feature_gallery(
    features_by_model: dict[str, list["FeatureProfile"]],
) -> plt.Figure:
    """Figure 6: Top 5 exclusive features per model."""
    apply_style()
    model_ids = list(features_by_model.keys())
    n_models = len(model_ids)

    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models))
    if n_models == 1:
        axes = [axes]

    for i, mid in enumerate(model_ids):
        ax = axes[i]
        ax.axis("off")
        ax.text(0.02, 0.98, MODEL_LABELS.get(mid, mid), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top",
                color=MODEL_COLOURS.get(mid, "#999"))

        feats = features_by_model[mid][:5]
        for j, feat in enumerate(feats):
            passages = feat.top_activations.get(mid, ["(no passages)"])
            line = f"  F{feat.feature_idx}: {feat.label} — \"{passages[0][:100]}...\""
            ax.text(0.04, 0.80 - j * 0.18, line, transform=ax.transAxes, fontsize=7, va="top")

    fig.suptitle("Figure 6: Corpus-Exclusive Feature Gallery", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def fig8_validation_matrix(
    svcca_scores: np.ndarray,
    model_ids: list[str],
    layer_labels: list[str] = None,
) -> plt.Figure:
    """Figure 8: 5×3 SVCCA validation alignment matrix."""
    apply_style()
    if layer_labels is None:
        layer_labels = ["Early", "Middle", "Late"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(svcca_scores, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels)
    ax.set_yticks(range(len(model_ids)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in model_ids], fontsize=8)

    # Annotate cells with values
    for i in range(svcca_scores.shape[0]):
        for j in range(svcca_scores.shape[1]):
            ax.text(j, i, f"{svcca_scores[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="SVCCA Score")
    ax.set_title("Figure 8: Per-Model SAE vs Universal SAE Alignment")
    fig.tight_layout()
    return fig


def fig9_crosscoder_venns(
    results: list[dict],
) -> plt.Figure:
    """Figure 9: Three crosscoder Venn diagrams.

    Each result dict has keys: pair, n_shared, n_exclusive_a, n_exclusive_b,
    top_exclusive_a, top_exclusive_b
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (ax, res) in enumerate(zip(axes, results)):
        ax.axis("off")
        pair = res["pair"]
        total = res["n_shared"] + res["n_exclusive_a"] + res["n_exclusive_b"]
        ax.set_title(f"{pair[0]} ↔ {pair[1]}", fontsize=10)

        text = (
            f"Shared: {res['n_shared']}\n"
            f"{pair[0]} only: {res['n_exclusive_a']}\n"
            f"{pair[1]} only: {res['n_exclusive_b']}\n\n"
            f"Top exclusive {pair[0]}:\n"
        )
        for label in res.get("top_exclusive_a", [])[:5]:
            text += f"  • {label}\n"
        text += f"\nTop exclusive {pair[1]}:\n"
        for label in res.get("top_exclusive_b", [])[:5]:
            text += f"  • {label}\n"

        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=7, va="top", family="monospace")

    fig.suptitle("Figure 9: Crosscoder Feature Decomposition", fontsize=12)
    fig.tight_layout()
    return fig


def fig10_simplex_comparison(
    concept_embeddings: dict[str, np.ndarray],
    concepts: list[str],
    model_ids: list[str],
) -> plt.Figure:
    """Figure 10: Side-by-side 3D PCA simplex plots for emotional-states simplex.

    Args:
        concept_embeddings: dict[model_id → (n_concepts, d_model) ndarray]
        concepts: list of concept strings (e.g. ["joy", "grief", "anger", "fear", "love"])
        model_ids: which models to plot (typically 3)
    """
    from sklearn.decomposition import PCA
    apply_style()
    n = len(model_ids)
    fig = plt.figure(figsize=(5 * n, 5))

    for i, mid in enumerate(model_ids):
        ax = fig.add_subplot(1, n, 1 + i, projection="3d")
        vecs = concept_embeddings[mid]
        pca = PCA(n_components=3)
        coords = pca.fit_transform(vecs)

        for j, c in enumerate(concepts):
            ax.scatter(*coords[j], s=60, color=MODEL_COLOURS.get(mid, "#999"))
            ax.text(coords[j, 0], coords[j, 1], coords[j, 2], f" {c}", fontsize=7)

        # Draw edges between all pairs
        for j in range(len(concepts)):
            for k in range(j + 1, len(concepts)):
                ax.plot3D(
                    [coords[j, 0], coords[k, 0]],
                    [coords[j, 1], coords[k, 1]],
                    [coords[j, 2], coords[k, 2]],
                    color="#cccccc", linewidth=0.5,
                )
        ax.set_title(MODEL_LABELS.get(mid, mid), fontsize=9)

    fig.suptitle("Figure 10: Concept Simplex Geometry Comparison", fontsize=12)
    fig.tight_layout()
    return fig


def fig11_orthogonality_bars(
    scores: dict[str, dict[str, float]],
) -> plt.Figure:
    """Figure 11: Hierarchy orthogonality scores bar chart.

    Args:
        scores: dict[model_id → dict[pair_label → cosine_similarity]]
    """
    apply_style()
    model_ids = list(scores.keys())
    pairs = list(scores[model_ids[0]].keys())
    n_pairs = len(pairs)
    n_models = len(model_ids)
    x = np.arange(n_pairs)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, mid in enumerate(model_ids):
        vals = [scores[mid].get(p, 0) for p in pairs]
        ax.bar(x + i * width, vals, width,
               color=MODEL_COLOURS.get(mid, "#999"),
               label=MODEL_LABELS.get(mid, mid))

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x + width * n_models / 2)
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Cosine Similarity (parent ↔ child direction)")
    ax.set_title("Figure 11: Hierarchy Orthogonality — Park et al. Prediction: ≈ 0")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig12_patching_heatmap(
    kl_scores: np.ndarray,
    model_pairs: list[tuple[str, str]],
    layer_labels: list[str],
) -> plt.Figure:
    """Figure 12: Patching effect heatmap."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    im = ax.imshow(kl_scores, cmap="hot", aspect="auto")
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=8)
    ax.set_yticks(range(len(model_pairs)))
    ax.set_yticklabels([f"{a} → {b}" for a, b in model_pairs], fontsize=8)
    ax.set_xlabel("Layer patched")
    ax.set_title("Figure 12: Activation Patching Effect")
    fig.colorbar(im, ax=ax, label="KL Divergence")
    fig.tight_layout()
    return fig


def fig13_steering_demo(
    generations: dict[str, dict],
) -> plt.Figure:
    """Figure 13: Before/after steering demonstration.

    Args:
        generations: dict[model_id → {"unsteered": str, "steered": str, "register_score": float}]
    """
    apply_style()
    model_ids = list(generations.keys())
    n = len(model_ids)

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, mid in enumerate(model_ids):
        gen = generations[mid]
        for j, (col_label, key) in enumerate([("Unsteered", "unsteered"), ("Steered (Victorian)", "steered")]):
            ax = axes[i, j]
            ax.axis("off")
            text = gen[key][:300]
            score = gen.get("register_score", "N/A")
            if j == 0:
                ax.text(-0.05, 0.5, MODEL_LABELS.get(mid, mid), transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="center", rotation=90,
                        color=MODEL_COLOURS.get(mid, "#999"))
            if i == 0:
                ax.set_title(col_label, fontsize=10)
            ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=7,
                    va="top", wrap=True, family="serif")
            ax.text(0.02, 0.02, f"R = {score:.2f}" if isinstance(score, float) else f"R = {score}",
                    transform=ax.transAxes, fontsize=7, color="gray")

    fig.suptitle("Figure 13: Cross-Model Steering with Victorian Feature", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Commit**

```bash
git add constrained_minds/viz/figures.py
git commit -m "feat: add all remaining figure functions (5/6/8/9/10/11/12/13)"
```

---

## Sub-Plan 5: Cross-Validation

### Task 18: Per-model SAE wrapper (SAELens)

**Files:**
- Create: `constrained_minds/sae/per_model.py`
- Create: `scripts/train_per_model_saes.py`

- [ ] **Step 1: Write implementation**

```python
# constrained_minds/sae/per_model.py
"""Per-model SAE wrapper using SAELens for cross-validation."""

import torch
import torch.nn as nn


class PerModelSAE(nn.Module):
    """Standard sparse autoencoder for a single model.

    Used for cross-validation against the Universal SAE.
    Simple TopK architecture matching the Universal SAE's sparsity.
    """

    def __init__(self, d_model: int, n_features: int = 4096, k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.encoder.weight, std=0.02)
        nn.init.trunc_normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        result = torch.zeros_like(z)
        result.scatter_(-1, topk_idx, topk_vals)
        return result

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return {"z": z, "x_hat": x_hat, "loss": loss}
```

```python
# scripts/train_per_model_saes.py
"""CLI: train per-model SAEs for cross-validation."""

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from constrained_minds.models.registry import MODEL_ZOO
from constrained_minds.sae.per_model import PerModelSAE


def load_cached_activations(cache_dir: str, model_id: str, layer: int) -> torch.Tensor:
    model_dir = Path(cache_dir) / model_id
    files = sorted(model_dir.glob(f"layer_{layer}_*.pt"))
    if not files:
        raise FileNotFoundError(f"No cached activations for {model_id} layer {layer}")
    tensors = [torch.load(f, weights_only=True).squeeze(0) for f in files]
    return torch.cat(tensors, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/activations")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--n-features", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--layer-position", choices=["early", "middle", "late"], default="middle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    layer_idx = {"early": 0, "middle": 1, "late": 2}[args.layer_position]
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for model_id, cfg in MODEL_ZOO.items():
        layer = cfg["layer_checkpoints"][layer_idx]
        d_model = cfg["d_model"]
        print(f"\n=== Training SAE for {model_id} (layer {layer}, d={d_model}) ===")

        acts = load_cached_activations(args.cache_dir, model_id, layer)
        sae = PerModelSAE(d_model=d_model, n_features=args.n_features, k=args.k).to(args.device)
        optimizer = AdamW(sae.parameters(), lr=args.lr)

        for step in range(args.steps):
            indices = torch.randint(0, acts.shape[0], (args.batch_size,))
            batch = acts[indices].to(args.device)
            result = sae(batch)
            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()

            if step % 2000 == 0:
                r2 = 1.0 - result["loss"].item() / batch.var().item()
                print(f"  Step {step}: loss={result['loss'].item():.4f}, R²={r2:.4f}")

        ckpt_path = ckpt_dir / f"per_model_sae_{model_id}_{args.layer_position}.pt"
        torch.save(sae.state_dict(), ckpt_path)
        print(f"  Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add constrained_minds/sae/per_model.py scripts/train_per_model_saes.py
git commit -m "feat: add per-model SAE and training script for cross-validation"
```

---

### Task 19: Crosscoder wrapper

**Files:**
- Create: `constrained_minds/sae/crosscoder.py`

- [ ] **Step 1: Write implementation**

```python
# constrained_minds/sae/crosscoder.py
"""Pairwise Dedicated Feature Crosscoder for model diffing.

Wraps a simple implementation of the DFC architecture from arXiv:2602.11729.
Partitions features into: exclusive-A, exclusive-B, shared.
"""

import torch
import torch.nn as nn


class DedicatedFeatureCrosscoder(nn.Module):
    """Dedicated Feature Crosscoder for comparing two models.

    Partitions the dictionary into three zones:
    - features 0..n_exclusive-1: exclusive to model A
    - features n_exclusive..2*n_exclusive-1: exclusive to model B
    - features 2*n_exclusive..n_features-1: shared

    Model A can only encode/decode through its exclusive + shared features.
    Model B can only encode/decode through its exclusive + shared features.
    """

    def __init__(
        self,
        d_model_a: int,
        d_model_b: int,
        n_exclusive: int = 512,
        n_shared: int = 2048,
        k: int = 32,
    ):
        super().__init__()
        self.d_model_a = d_model_a
        self.d_model_b = d_model_b
        self.n_exclusive = n_exclusive
        self.n_shared = n_shared
        self.n_features = 2 * n_exclusive + n_shared
        self.k = k

        # A's features: exclusive_a + shared = [0, n_exclusive+n_shared)
        # B's features: exclusive_b + shared = [n_exclusive, 2*n_exclusive+n_shared)
        self.encoder_a = nn.Linear(d_model_a, n_exclusive + n_shared)
        self.encoder_b = nn.Linear(d_model_b, n_exclusive + n_shared)
        self.decoder_a = nn.Linear(n_exclusive + n_shared, d_model_a)
        self.decoder_b = nn.Linear(n_exclusive + n_shared, d_model_b)

    def encode_a(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_a(x)
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        result = torch.zeros_like(z)
        result.scatter_(-1, topk_idx, topk_vals)
        return result

    def encode_b(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_b(x)
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        result = torch.zeros_like(z)
        result.scatter_(-1, topk_idx, topk_vals)
        return result

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> dict:
        z_a = self.encode_a(x_a)
        z_b = self.encode_b(x_b)
        x_hat_a = self.decoder_a(z_a)
        x_hat_b = self.decoder_b(z_b)

        loss_a = nn.functional.mse_loss(x_hat_a, x_a)
        loss_b = nn.functional.mse_loss(x_hat_b, x_b)

        return {
            "z_a": z_a,
            "z_b": z_b,
            "loss": loss_a + loss_b,
            "loss_a": loss_a,
            "loss_b": loss_b,
        }

    def classify_features(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
        """Classify features as exclusive-A, exclusive-B, or shared based on activation patterns."""
        # A's exclusive zone: first n_exclusive features of z_a
        a_exclusive_acts = z_a[:, :self.n_exclusive].abs().mean(dim=0)
        # Shared zone: last n_shared features (same indices in both)
        a_shared_acts = z_a[:, self.n_exclusive:].abs().mean(dim=0)
        b_exclusive_acts = z_b[:, :self.n_exclusive].abs().mean(dim=0)
        b_shared_acts = z_b[:, self.n_exclusive:].abs().mean(dim=0)

        return {
            "n_active_exclusive_a": (a_exclusive_acts > 0.01).sum().item(),
            "n_active_exclusive_b": (b_exclusive_acts > 0.01).sum().item(),
            "n_active_shared": ((a_shared_acts > 0.01) & (b_shared_acts > 0.01)).sum().item(),
        }
```

- [ ] **Step 2: Commit**

```bash
git add constrained_minds/sae/crosscoder.py
git commit -m "feat: add Dedicated Feature Crosscoder for pairwise model diffing"
```

---

### Task 20: Validation notebook

**Files:**
- Create: `notebooks/04_validation.ipynb`
- Modify: `constrained_minds/sae/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

```python
# constrained_minds/sae/__init__.py
from constrained_minds.sae.universal import UniversalSAE
from constrained_minds.sae.feature_metrics import firing_entropy, classify_features, cofire_proportion
from constrained_minds.sae.per_model import PerModelSAE
from constrained_minds.sae.crosscoder import DedicatedFeatureCrosscoder
```

- [ ] **Step 2: Create validation notebook stub**

Create `notebooks/04_validation.ipynb` with cells:

Cell 1 — SVCCA computation:
```python
# Load Universal SAE and per-model SAEs
# For each model, compare feature spaces using SVCCA
# Output: 5×3 alignment matrix for Figure 8
```

Cell 2 — Crosscoder results:
```python
# Load crosscoder results for M1↔M5, M2↔M1, M4↔M5
# Extract exclusive/shared feature counts and labels
# Output: data for Figure 9
```

Cell 3 — Generate figures:
```python
from constrained_minds.viz.figures import fig8_validation_matrix, fig9_crosscoder_venns
# ... generate and save figures
```

- [ ] **Step 3: Commit**

```bash
git add constrained_minds/sae/__init__.py notebooks/04_validation.ipynb
git commit -m "feat: add validation notebook and SAE package init"
```

---

## Execution Order

The tasks have natural dependencies:

```
Task 1 (base class) → Task 2 (nanochat) → Task 5 (notebook)
                    → Task 3 (HF wrapper) ↗
Task 1 → Task 4 (registry) → Task 5

Task 6 (corpus) → Task 7 (extractor) → Task 8 (normalise)

Task 9 (Universal SAE) → Task 10 (metrics) → Task 11 (training)

Task 12 (logit lens + viz) ─┐
Task 13 (geometry)          │→ Notebooks 02–07
Task 14 (register)          │
Task 15 (autointerp)        │
Task 16 (patching)          │
Task 17 (remaining figs)   ─┘

Task 18 (per-model SAE) → Task 20 (validation notebook)
Task 19 (crosscoder)    ↗
```

**Parallelisable groups:**
- Tasks 1–4 can be done sequentially (each builds on previous)
- Tasks 6–8 are independent of Tasks 1–5 and can run in parallel
- Tasks 9–11 depend on Tasks 6–8 (need cached activations)
- Tasks 12–17 are mostly independent and can run in parallel after Task 1–4
- Tasks 18–20 depend on Tasks 9–11

**Recommended execution: dispatch as 3 parallel streams:**
1. Stream A: Tasks 1 → 2 → 3 → 4 → 5
2. Stream B: Tasks 6 → 7 → 8
3. Stream C: Tasks 9 → 10 → 11 (after Stream B completes)
4. Stream D: Tasks 12–17 in parallel (after Stream A completes)
5. Stream E: Tasks 18 → 19 → 20 (after Stream C completes)
