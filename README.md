<h1 align="center">DiffRetriever</h1>

<p align="center"><b>Parallel Representative Tokens for Retrieval with Diffusion Language Models</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.07210"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.07210-b31b1b.svg?logo=arxiv"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2.6" src="https://img.shields.io/badge/PyTorch-2.6+cu126-EE4C2C.svg?logo=pytorch&logoColor=white"></a>
  <a href="https://huggingface.co/Dream-org/Dream-v0-Instruct-7B"><img alt="HF Models" src="https://img.shields.io/badge/🤗_Models-Dream%20%7C%20LLaDA-yellow"></a>
  <a href="https://github.com/ielab/diffretriever/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/ielab/diffretriever?style=social"></a>
</p>

<p align="center">
  <a href="#-quick-start-30-seconds">Quick Start</a> ·
  <a href="#-setup">Setup</a> ·
  <a href="#-backbones">Backbones</a> ·
  <a href="#-reproducing-the-paper">Reproducing</a> ·
  <a href="#-repository-layout">Layout</a> ·
  <a href="#-citation">Citation</a>
</p>

---

> **TL;DR** — DiffRetriever appends `K` `[MASK]` tokens to a `PromptReps`-style prompt and reads all `K` hidden states **and** next-token logits in a single bidirectional forward pass. That gives multi-vector dense + sparse retrieval at the encoding cost of a single token, where the autoregressive equivalent costs `K` sequential forward passes. Works **zero-shot** on Dream and LLaDA, and reaches SOTA after fine-tuning on MS MARCO.

![Architecture overview](assets/architecture_animated.svg)

<br />

<p align="center">
  <img src="assets/teaser_latency.png" alt="Teaser: BEIR-7 NDCG@10 vs. encoding + search latency" width="60%" />
</p>

<p align="center"><sub><em>BEIR-7 NDCG@10 vs. encoding + search latency (ms/query, 100K-document MS&nbsp;MARCO sample). Left: zero-shot (PromptReps at K&le;20). Right: fine-tuned (K=4). Dashed lines link single-token (open) and multi-token (filled) variants. <strong>DiffRetriever</strong> gains from multi-token at near single-token cost in both panels; <strong>PromptReps</strong> pays &asymp;15&times; the latency at zero-shot and &asymp;3&times; at fine-tuning, with no consistent gain. Fine-tuned DiffRetriever (Dream, (K<sub>q</sub>,&nbsp;K<sub>p</sub>)=(4,&nbsp;16)) is the strongest BEIR-7 retriever in our comparison.</em></sub></p>

<br />

<p align="center">
  <img src="assets/latency_combined.png" alt="Latency scaling: encoding vs input length, search vs index size" width="80%" />
</p>

<p align="center"><sub><em>Latency scaling on synthetic inputs and indices (single H100, same attention implementation across backbones). <strong>Top row:</strong> encoding latency vs.&nbsp;input sequence length. <strong>Bottom row:</strong> search latency vs.&nbsp;index size (log scale). <strong>Left column:</strong> PromptReps on autoregressive backbones (Qwen2.5, LLaMA3). <strong>Right column:</strong> DiffRetriever on diffusion backbones (Dream, LLaDA). Open markers = single-token, filled = multi-token (AR uses K=4, the fine-tuned cap; diffusion uses the train-selected (K<sub>q</sub>*, K<sub>p</sub>*)). DiffRetriever&#39;s multi-token encoding stays close to its single-token cost, while AR multi-token remains 2&ndash;3&times; AR single-token across the entire input range.</em></sub></p>

> **Models on Hugging Face:** trained checkpoints for DiffRetriever (Dream, LLaDA) and the re-trained baselines (PromptReps, DiffEmbed, RepLLaMA) will be released on the Hugging Face Hub **soon**. They are not available yet — this README will be updated with the model URLs when the release lands.

---

## 🚀 Quick Start (30 seconds)

**Zero-shot retrieval with Dream-7B, end-to-end in pure Python — no SLURM, no scripts, no data download.**

After [setting up the env](#-setup) (`pip install -r requirements.txt`), paste this whole block into a file (`demo.py`) at the repo root and run it. The model auto-downloads from HuggingFace on first run (~14 GB).

```python
"""DiffRetriever zero-shot demo — Dream-7B, K=4 representative tokens, single denoising step."""
import sys
sys.path.insert(0, "src")  # repo-local import

import torch
import torch.nn.functional as F
from models.dream_retriever import DreamRetriever

# 1. Load the encoder (zero-shot — no fine-tuning required)
model = DreamRetriever(
    model_name="Dream-org/Dream-v0-Instruct-7B",
    max_length=512,
    n_gen_tokens=4,                       # K = number of [MASK] tokens appended
    num_denoise_steps=1,                  # 1-step is enough for zero-shot
    query_prompt="prompts/default/query_prompt_few.yaml",
    passage_prompt="prompts/default/passage_prompt_few.yaml",
)
model.eval()

# 2. Tiny demo corpus
queries = [
    "what causes the seasons on earth?",
    "best way to learn guitar at home",
]
passages = [
    "The tilt of Earth's axis relative to its orbital plane causes seasonal variation in sunlight.",
    "Pick a beginner-friendly acoustic guitar and practice 15 minutes daily with online tutorials.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "Plate tectonics describes how Earth's lithosphere is divided into moving plates.",
    "Online video lessons are an efficient way to learn an instrument at your own pace.",
]

# 3. Encode — one forward pass returns repr_hidden ([N, K, H]) and sparse activations
with torch.inference_mode():
    q = model.encode(queries,  is_query=True,  encoding_mode="promptreps", encode_type="all_steps")
    p = model.encode(passages, is_query=False, encoding_mode="promptreps", encode_type="all_steps")

# 4. ColBERT MaxSim scoring on the K-vector outputs
q_vec = F.normalize(q["repr_hidden"].float(), dim=-1)         # [Q, K_q, H]
p_vec = F.normalize(p["repr_hidden"].float(), dim=-1)         # [P, K_p, H]
sim   = torch.einsum("qkh,pdh->qkpd", q_vec, p_vec)           # [Q, K_q, P, K_p]
scores = sim.max(dim=-1).values.clamp(min=0).sum(dim=1)       # [Q, P]

# 5. Top-3 hits per query
for i, query in enumerate(queries):
    top = scores[i].topk(3)
    print(f"\nQ: {query}")
    for s, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {s:.3f}  {passages[idx]}")
```

Expected output (scores will vary slightly across GPUs):
```
Q: what causes the seasons on earth?
  3.21  The tilt of Earth's axis relative to its orbital plane causes seasonal variation in sunlight.
  2.18  Plate tectonics describes how Earth's lithosphere is divided into moving plates.
  1.92  Photosynthesis converts carbon dioxide and water into glucose using sunlight.

Q: best way to learn guitar at home
  3.05  Pick a beginner-friendly acoustic guitar and practice 15 minutes daily with online tutorials.
  2.41  Online video lessons are an efficient way to learn an instrument at your own pace.
  1.55  Photosynthesis converts carbon dioxide and water into glucose using sunlight.
```

That's the whole pipeline — append `K` `[MASK]`s, one forward pass, MaxSim. To swap in **LLaDA**, replace `DreamRetriever` with `LLaDA2Retriever` (`from models.llada_retriever import LLaDA2Retriever`). To run **sparse** retrieval, use the `sparse_indices` / `sparse_values` keys also returned by `encode(...)`. To run **fusion**, blend dense + sparse with min-max normalization. The full sweep is wrapped in `scripts/run_encode.sh` + `scripts/run_eval.sh`.

---

## 🧠 What's in this repo

```
src/
├── models/                       Retrievers (zero-shot + trainable)
│   ├── trainable_diff_retriever.py    DiffRetriever (Dream / LLaDA)
│   ├── trainable_ar_retriever.py      PromptReps (autoregressive)
│   ├── diffembed_retriever.py         DiffEmbed baseline
│   ├── repllama_retriever.py          RepLLaMA baseline
│   ├── baseline_retriever.py          Zero-shot PromptReps
│   ├── dream_retriever.py             Dream backbone wrapper
│   ├── llada_retriever.py             LLaDA backbone wrapper
│   ├── bottleneck_retriever.py        Bottleneck / Semantic-Hub variant (ablation)
│   ├── block_schedule.py              Multi-step denoising schedule
│   ├── backbone_adapters.py           HF model loading / LoRA wiring
│   └── sparse_utils.py                Sparse score helpers
└── evaluation/
    └── evaluator.py              Per-query scoring + metric aggregation

scripts/
├── train_retriever.py            Train DiffRetriever
├── train_ar_retriever.py         Train PromptReps
├── train_diffembed.py            Train DiffEmbed
├── train_repllama.py             Train RepLLaMA
├── encode_promptreps.py          Encode queries / passages
├── evaluate_sweep.py             Evaluate over a (K_q, K_p) sweep
├── eval_trec.py                  Compute MRR / NDCG with pytrec-eval
├── prepare_msmarco.py            MS MARCO data prep
├── preprocess_msmarco_aug.py     Augmented triples prep
├── shard_io.py                   Sharded encoding I/O
├── download_data.sh              Fetch MS MARCO + TREC DL + BEIR-7 + NLTK data
├── run_train.sh                  Portable launcher: training
├── run_encode.sh                 Portable launcher: encoding
└── run_eval.sh                   Portable launcher: evaluation

configs/
├── ds_zero2.json                 DeepSpeed ZeRO-2 config
├── ds_zero3.json                 DeepSpeed ZeRO-3 config
├── naming.sh                     Backbone / config naming helpers
└── dataset_config.sh             Dataset path helpers

prompts/
└── default                       Representative-token prompts
```

Note: this repo bundles only what's needed to reproduce the paper. Internal analysis / plot scripts and benchmark drivers are kept in the research repository and are not redistributed here.

---

## 📦 Setup

We use conda. The pinned `requirements.txt` is a freeze of the env used during development on a single H100 node (CUDA 12.6, Linux x86_64, Python 3.10).

```bash
# 1. Create env
conda create -n diffretriever python=3.10 -y
conda activate diffretriever

# 2. Install pinned dependencies (covers training + encoding + eval)
pip install -r requirements.txt

# 3. Download the datasets and the small NLTK corpora (stopwords + punkt)
bash scripts/download_data.sh             # MS MARCO + TREC DL19/DL20 + BEIR-7 + nltk
# or selectively:
# bash scripts/download_data.sh --msmarco
# bash scripts/download_data.sh --beir
```

`requirements.txt` is exhaustive — it covers training (DeepSpeed, accelerate, peft) as well as encoding and evaluation. Training uses HuggingFace `Trainer` directly with the retriever classes under `src/models/`; there is no separate "training extras" file.

**Optional but strongly recommended for speed: flash-attention 2.** It is not pinned in `requirements.txt` because the prebuilt wheel is platform-specific. Install the matching wheel for your CUDA / torch / cxx11abi from the [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases), or:

```bash
pip install flash-attn --no-build-isolation
```

Core versions in the freeze:
- `torch==2.6.0+cu126`, `transformers==4.54.0` (Dream / LLaDA require this exact range)
- `accelerate==1.12.0`, `peft==0.18.1`, `deepspeed==0.18.8`
- `pytrec-eval-terrier==0.5.6` for retrieval metrics

---

## 🤗 Backbones

The four backbones used in the paper:

| Backbone | HF id | Family |
|---|---|---|
| LLaMA3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | Autoregressive |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | Autoregressive |
| Dream-v0-Instruct-7B | `Dream-org/Dream-v0-Instruct-7B` | Diffusion |
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | Diffusion |

`src/models/backbone_adapters.py` handles the HF loading + tokenizer setup for all four.

---

## 🧪 Reproducing the paper

### Data

```bash
bash scripts/download_data.sh             # MS MARCO + TREC DL 2019/2020 + BEIR-7 + NLTK
python scripts/prepare_msmarco.py          # Optional: HF-cached MSMARCO splits
python scripts/preprocess_msmarco_aug.py   # Pre-tokenize Tevatron/msmarco-passage-aug
```

All workflow scripts are minimal portable launchers — open them, edit the variables at the top for your setup, and run. They wrap `scripts/*.py` with the canonical arguments used in the paper.

### Zero-shot retrieval

The portable launchers wrap the underlying Python scripts with the canonical paper arguments — open them and edit the variables at the top for your local paths.

```bash
# Encode queries and passages (zero-shot DiffRetriever / PromptReps)
MODEL_TYPE=dream K=4 PROMPT_VARIANT=few \
    bash scripts/run_encode.sh

# Score the encoded representations
RESULTS_DIR=results/dream_few_K4/msmarco \
QRELS=data/msmarco/qrels.dev.tsv \
    bash scripts/run_eval.sh
```

Or invoke the underlying scripts directly (this is what the launchers call):

```bash
# 1. Encode queries — one shard at a time (--input_file is required)
python scripts/encode_promptreps.py \
    --model_type dream \
    --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
    --input_file  data/msmarco/queries.dev.jsonl \
    --output_dir  results/dream_few_K4/msmarco/queries \
    --is_query \
    --query_prompt   prompts/default/query_prompt_few.yaml \
    --passage_prompt prompts/default/passage_prompt_few.yaml \
    --n_gen_tokens 4 --num_denoise_steps 1 \
    --encode_type all_steps --sparse_topk 256 \
    --shard_id 0 --num_shards 1

# 2. Encode corpus the same way (--input_file data/msmarco/corpus.jsonl,
#    drop --is_query, set --num_shards to fan out across the cluster)

# 3. Score — produces summary.json + {mode}.json + {mode}.trec per mode
python scripts/evaluate_sweep.py \
    --query_dir  results/dream_few_K4/msmarco/queries \
    --corpus_dir results/dream_few_K4/msmarco/corpus \
    --qrels      data/msmarco/qrels.dev.tsv \
    --output_dir results/dream_few_K4/msmarco/eval
```

For the `(K_q, K_p)` sweep over `{1, 2, 4, 8, 16}²`, loop the encode call over the grid (this is what the paper uses to pick `(K_q*, K_p*)` on MS MARCO train). The paper reports `(4, 16)` for Dream and `(4, 4)` for LLaDA.

### Fine-tuning

```bash
# DiffRetriever — Dream / LLaDA backbones
MODEL_TYPE=dream MODEL_NAME=Dream-org/Dream-v0-Instruct-7B \
K_Q=4 K_P=16 \
    bash scripts/run_train.sh

# PromptReps and the re-trained baselines call the matching Python scripts:
#   python scripts/train_ar_retriever.py ...   # PromptReps (AR)
#   python scripts/train_diffembed.py ...      # DiffEmbed
#   python scripts/train_repllama.py ...       # RepLLaMA
```

All training uses LoRA (`r=16, α=64`) + DeepSpeed ZeRO-2, InfoNCE with `τ=0.01`, 1 positive + 15 hard negatives, global batch 128, on the Tevatron MS MARCO augmented triples. Diffusion backbones train at the train-selected `(K_q*, K_p*)`; AR backbones train at `K=4`.

### Evaluation

```bash
# Sweep all score modes against encoded representations (5 modes in one pass:
# single_dense, multi_dense, sparse_max, fusion_single_sparse_max, fusion_multi_sparse_max)
python scripts/evaluate_sweep.py \
    --query_dir  <encoded queries dir> \
    --corpus_dir <encoded corpus dir> \
    --qrels      <qrels.tsv> \
    --output_dir <output dir>

# Or score a single TREC run file with pytrec-eval (positional args: qrels first, run second)
python scripts/eval_trec.py <qrels> <runfile> --metrics mrr_cut_10 ndcg_cut_10
```

---

## 📁 Repository layout

See [What's in this repo](#-whats-in-this-repo) above for the full tree.

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026diffretriever,
  title={DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models},
  author={Wang, Shuai and Yin, Yu and Zhuang, Shengyao and Koopman, Bevan and Zuccon, Guido},
  journal={arXiv preprint arXiv:2605.07210},
  year={2026}
}
```

---

## 📄 License

MIT — see [`LICENSE`](LICENSE).
