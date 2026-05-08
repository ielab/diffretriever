# DiffRetriever

Code for **DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models**.

DiffRetriever is a representative-token retriever for diffusion language models (e.g., Dream, LLaDA). It appends `K` masked positions to a `PromptReps`-style prompt and reads all `K` hidden states and next-token logits in a single bidirectional forward pass — giving multi-vector retrieval at the encoding cost of a single token, where the autoregressive equivalent costs `K` sequential forward passes.

> **Models on Hugging Face:** trained checkpoints for DiffRetriever (Dream, LLaDA) and the re-trained baselines (PromptReps, DiffEmbed, RepLLaMA) will be released on the Hugging Face Hub **soon**. They are not available yet — this README will be updated with the model URLs when the release lands.

---

## What's in this repo

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
├── download_data.sh              Fetch MS MARCO + TREC DL + BEIR-7
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

Note: this repo bundles only what is needed to reproduce the paper. Internal analysis/plot scripts and benchmark drivers are kept in the research repository and are not redistributed here.

---

## Setup

We use conda for the environment (this is what we used during development).

```bash
# 1. Create env
conda create -n diffretriever python=3.11 -y
conda activate diffretriever

# 2. Runtime dependencies (encoding + evaluation)
pip install -r requirements.txt

# 3. (Optional) training dependencies — only needed to retrain
pip install -r requirements-train.txt

# 4. (Optional) Pyserini for the BM25 baseline
pip install pyserini
```

The runtime install is enough to load a released checkpoint, encode queries/passages, and reproduce the effectiveness numbers in the paper. Training uses HuggingFace `Trainer` directly with the retriever classes under `src/models/`; the `requirements-train.txt` extras (DeepSpeed) are only needed if you want to retrain from scratch.

Core versions:
- `torch >= 2.6`
- `transformers 4.54.x` (Dream / LLaDA require this version range)
- `accelerate`, `peft` (runtime) and `deepspeed` (training only)
- `pytrec-eval-terrier` for retrieval metrics

---

## Backbones

The four backbones used in the paper:

| Backbone | HF id | Family |
|---|---|---|
| LLaMA3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | Autoregressive |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | Autoregressive |
| Dream-v0-Instruct-7B | `Dream-org/Dream-v0-Instruct-7B` | Diffusion |
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | Diffusion |

`src/models/backbone_adapters.py` handles the HF loading + tokenizer setup for all four.

---

## Reproducing the paper

### Data

```bash
bash scripts/download_data.sh           # MS MARCO + TREC DL 2019/2020 + BEIR-7
python scripts/prepare_msmarco.py
python scripts/preprocess_msmarco_aug.py # Tevatron augmented triples
```

All workflow scripts are minimal portable launchers — open them, edit the variables at the top for your setup, and run. They wrap `scripts/*.py` with the canonical arguments used in the paper.

### Zero-shot retrieval

```bash
# Encode queries and passages (zero-shot DiffRetriever / PromptReps)
MODEL_TYPE=dream K=4 PROMPT_VARIANT=few \
    bash scripts/run_encode.sh

# Score the encoded representations
RESULTS_DIR=results/dream_few_K4/msmarco \
QRELS=data/msmarco/qrels.dev.tsv \
    bash scripts/run_eval.sh
```

For the (K_q, K_p) sweep over `{1, 2, 4, 8, 16}^2`, loop `run_encode.sh` over the grid (this is what the paper uses to pick `(K_q*, K_p*)` on MS MARCO train). The paper reports `(4, 16)` for Dream and `(4, 4)` for LLaDA.

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

All training uses LoRA (r=16, α=64) + DeepSpeed ZeRO-2, InfoNCE with τ=0.01, 1 positive + 15 hard negatives, global batch 128, on the Tevatron MS MARCO augmented triples. Diffusion backbones train at the train-selected `(K_q*, K_p*)`; AR backbones train at `K=4`.

### Evaluation

```bash
# Sweep all score modes over a results directory
python scripts/evaluate_sweep.py --results_dir <dir> --qrels <qrels>

# Or score a single run with pytrec-eval
python scripts/eval_trec.py --run <runfile> --qrels <qrels>
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026diffretriever,
  title   = {DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models},
  author  = {Wang, Shuai and Yin, Yu and Zhuang, Shengyao and Koopman, Bevan and Zuccon, Guido},
  year    = {2026},
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).