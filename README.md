# PTEB: Paraphrasing Text Embedding Benchmark

<p align="center">
  <strong>Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.06730">📄 Paper</a> •
  <a href="https://github.com/ThisIsManuel/pteb">💻 Code</a> •
  <a href="#citation">📖 Citation</a>
</p>

---

## Overview

PTEB is a dynamic benchmarking framework for evaluating sentence embedding models by generating stochastic paraphrases at evaluation time using LLMs. Unlike static benchmarks, PTEB creates semantically equivalent but textually distinct test variants that stress-test embeddings for robustness to lexical variation.

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/ThisIsManuel/pteb.git
cd pteb
uv sync
```

### Using pip

```bash
git clone https://github.com/ThisIsManuel/pteb.git
cd pteb
pip install -e .
```

### Ollama Model Download
```bash
ollama pull gemma3:27b
```
You may use gemma3:27b or any other [ollama model](https://www.ollama.com/models).

### Requirements
- Python ≥3.10
- [Ollama](https://ollama.com/) for paraphrase generation
- CUDA-capable GPU recommended (tested on RTX 5090)

## Quick Start

### Discovery Commands

```bash
# List all available task types
uv run pteb get-tasks

# List all datasets for a specific task
uv run pteb get-datasets sts

# Show metadata for a specific dataset
uv run pteb get-metadata rte3
```

### Minimal Example

```bash
# Evaluate on a single dataset with defaults
uv run pteb --datasets biosses --emb_model all-mpnet-base-v2
```

## Usage

### Basic Commands

```bash
# Single dataset evaluation
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2

# Multiple datasets (task types auto-detected)
uv run pteb --datasets stsbenchmark sts12 sts13 --emb_model all-mpnet-base-v2
```

### Overwriting the default parameters

```bash
# Multiple paraphrase runs per sentence
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2 --n_runs 3

# Custom generative model
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2 \
  --gen_model gpt-oss:20b

# Adjust generation parameters
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2 \
  --temperature 0.7 --top_p 0.95

# Custom paraphrase prompt
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2 \
  --paraphrase_prompt "Rewrite this sentence differently:"

# Don't save paraphrases to disk
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2 \
  --save_paraphrases False
```

### Language Filtering

```bash
# Filter to English only
uv run pteb --datasets sts17-crosslingual --emb_model all-mpnet-base-v2 \
  --languages EN

# Multiple languages (ISO 639-1 or 639-3 codes)
uv run pteb --datasets sts17-crosslingual --emb_model paraphrase-multilingual-mpnet-base-v2 \
  --languages EN FR DE

# 3-letter codes also work
uv run pteb --datasets sts17-crosslingual --emb_model paraphrase-multilingual-mpnet-base-v2 \
  --languages ENG FRA DEU
```

### Memory & Performance

```bash
# Downsample data for quick testing (use 10% of the data)
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2 \
  --downsample 0.1

# Reduce batch sizes for low-memory GPUs
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2 \
  --emb_batch_size 512 --encoding_chunk_size 512

# Use float16 for lower memory usage
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2 \
  --st_dtype torch.float16

# Adjust clustering batch size
uv run pteb --datasets twentynewsgroups-clustering --emb_model all-mpnet-base-v2 \
  --clustering_batch_size 256

# Enable L2 normalization (recommended for some models)
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2 \
  --normalize_embeddings
```

### Full Example

```bash
# Complete evaluation with all options
uv run pteb \
  --datasets stsbenchmark sts12 biosses \
  --emb_model all-mpnet-base-v2 \
  --gen_model gemma3:27b \
  --n_runs 3 \
  --temperature 0.8 \
  --top_p 0.9 \
  --seed_paraphrasing 1337 \
  --seed_general 1337 \
  --emb_batch_size 4096 \
  --st_dtype torch.bfloat16
```

## Command-Line Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--datasets` | Yes | - | Dataset names (space-separated, case-insensitive) |
| `--emb_model` | Yes | - | Embedding model name |
| `--gen_model` | No | gemma3:27b | Generative model for paraphrasing |
| `--n_runs` | No | 1 | Number of paraphrase runs per sentence |
| `--temperature` | No | 0.8 | Sampling temperature (0.0-2.0) |
| `--top_p` | No | 0.9 | Nucleus sampling probability (0.0-1.0) |
| `--paraphrase_prompt` | No | None | Custom prompt for paraphrasing |
| `--save_paraphrases` | No | True | Save paraphrases to disk |
| `--seed_paraphrasing` | No | random | Seed for paraphrase generation |
| `--seed_general` | No | random | Seed for anything other than paraphrase generation |
| `--emb_batch_size` | No | 4096 | Embedding batch size |
| `--encoding_chunk_size` | No | 4096 | Max sentences per encoding chunk |
| `--clustering_batch_size` | No | 500 | Mini-Batch K-Means batch size |
| `--normalize_embeddings` | No | False | Enable L2 normalization |
| `--st_dtype` | No | torch.bfloat16 | Embedding data type |
| `--languages` | No | None | Language filter codes (ISO 639-1/639-3) |
| `--downsample` | No | None | Fraction of data to use (0.0-1.0) for quick testing |

## Supported Tasks

### 1. Semantic Textual Similarity (STS)

**Datasets**: STSBenchmark, STS12-17, SICK-R, BIOSSES, STS22

**Metric**: Spearman correlation

**Example**:
```bash
uv run pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2
```

### 2. Pair Classification

**Datasets**: TwitterSemEval2015, RTE3

**Metric**: Average Precision

**Example**:
```bash
uv run pteb --datasets twittersemeval2015 --emb_model all-mpnet-base-v2
```

### 3. Clustering

**Datasets**: TwentyNewsgroups, ArXivHierarchy, MedrxivHierarchy, StackExchange, BiorxivHierarchy

**Metric**: V-measure

**Example**:
```bash
uv run pteb --datasets twentynewsgroups-clustering --emb_model all-mpnet-base-v2
```

### 4. Classification

**Datasets**: Banking77, AmazonCounterfactual, EmotionClassification, ToxicConversations

**Metric**: Accuracy

**Example**:
```bash
uv run pteb --datasets banking77classification --emb_model all-mpnet-base-v2
```

### 5. Retrieval

**Datasets**: ArguAna, AskUbuntuDupQuestions, CQADupstack, NFCorpus, SCIDOCS

**Metric**: nDCG@10

**Example**:
```bash
uv run pteb --datasets arguana --emb_model all-mpnet-base-v2
```

### 6. Reranking

**Datasets**: AskUbuntuDupQuestions, MindSmallReranking, StackOverflowDupQuestions

**Metric**: MAP

**Example**:
```bash
uv run pteb --datasets askubuntudupquestions-reranking --emb_model all-mpnet-base-v2
```

### 7. Summarization

**Datasets**: SummEval

**Metric**: Spearman correlation

**Example**:
```bash
uv run pteb --datasets summeval --emb_model all-mpnet-base-v2
```

## Adding New Datasets

Create a metadata class in the appropriate task file (e.g., `clustering_datasets.py`):

```python
class MyNewDatasetMetadata(BaseDatasetMetadata):
    name = "mynewdataset-clustering"
    alternate_names: list[str] = ["mynewdataset"] # optionally add alternative names or leave list empty
    hf_dataset_name = "mteb/mynewdataset-clustering"
    default_splits = ["test"]
    languages = None 
    is_multilingual = False
    task_type = "clustering"
    description = "Description of the dataset"
```

Then add the class name to the imports and to `ALL_DATASET_METADATA_CLASSES` in `__init__.py`.

## Citation

If you use PTEB, please cite our paper:

```bibtex
@misc{frank2026ptebrobusttextembedding,
      title={PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs}, 
      author={Manuel Frank and Haithem Afli},
      year={2026},
      eprint={2510.06730},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.06730}, 
}
```

---

Coding has been assisted by Claude Code.