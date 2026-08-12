# LLM Finetuning & DatasetForge Project

A clean learning and dataset pipeline repo for building an LLM, ingesting training data, and preparing fine-tuning datasets.

## Repository layout

Root folders:

- `dataset_forge/` — dataset factory package and CLI
- `data/raw/` — raw source data files for experiments
- `outputs/` — generated dataset exports
- `week1/`, `week2/`, `week3/` — learning modules for model building, training, and fine-tuning
- `requirements.txt` — project dependencies
- `README.md` — this guide

### Week folders

Week 1: Build an LLM
- `week1/01_tokenizer/`
- `week1/02_embeddings/`
- `week1/03_positional_encoding/`
- `week1/04_self_attention/`
- `week1/05_multi_head_attention/`
- `week1/06_feed_forward/`
- `week1/07_transformer_block/`
- `week1/08_mini_gpt/`

Week 2: Train
- `week2/09_dataset/`
- `week2/10_dataloader/`
- `week2/11_loss/`
- `week2/12_backprop/`
- `week2/13_optimizer/`
- `week2/14_train_loop/`

Week 3: Fine-tuning
- `week3/15_lora/`
- `week3/16_peft/`
- `week3/17_qwen_finetuning/`
- `week3/18_evaluation/`
- `week3/19_deployment/`

## What is included

### DatasetForge

`dataset_forge` is the dataset factory package for ingesting and preparing training data from multiple sources.

Supported sources today:

- Kaggle datasets via `kaggle`
- Hugging Face datasets via `datasets`
- GitHub repositories via clone and file extraction
- PDF documents via `pdfplumber`

### Key example files

- `week1/01_tokenizer/tokenizer_demo.py` — tokenizer download and tokenization demo
- `week1/02_embeddings/embeddings_demo.py` — embedding layer demo
- `week1/03_positional_encoding/explore_model.py` — data/model exploration
- `week1/08_mini_gpt/inference.py` — generation example
- `week2/14_train_loop/train.py` — training loop placeholder

## Setup and install

1. Open a terminal and navigate to the repo root:

```bash
cd llm-finetune
```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Confirm the repo structure:

```bash
tree -L 2
```

## Optional: Kaggle connector setup

If you want to ingest Kaggle datasets, configure your Kaggle API token first.

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

> Download `kaggle.json` from your Kaggle account settings and save it to `~/.kaggle/kaggle.json`.

## DatasetForge CLI commands

### Check the DatasetForge version

```bash
python -m dataset_forge version
```

### Run the full ingestion pipeline

This command extracts source data, cleans text, deduplicates, and exports JSONL output.

#### Kaggle source

```bash
python -m dataset_forge ingest \
  --source kaggle \
  --identifier zillow/zecon \
  --output outputs/datasetforge/kaggle \
  --format jsonl
```

#### Hugging Face source

```bash
python -m dataset_forge ingest \
  --source huggingface \
  --identifier databricks/databricks-dolly-15k \
  --output outputs/datasetforge/hf \
  --format jsonl
```

#### GitHub source

```bash
python -m dataset_forge ingest \
  --source github \
  --identifier https://github.com/huggingface/transformers \
  --output outputs/datasetforge/github \
  --format jsonl
```

#### PDF source

```bash
python -m dataset_forge ingest \
  --source pdf \
  --identifier data/raw/my_document.pdf \
  --output outputs/datasetforge/pdf \
  --format jsonl
```

### Validate a generated dataset file

```bash
python -m dataset_forge validate outputs/datasetforge/kaggle/train.jsonl
```

### Run a small training job

```bash
python week2/14_train_loop/train.py \
  --dataset outputs/datasetforge/kaggle/train.jsonl \
  --model gpt2 \
  --output_dir outputs/training \
  --epochs 1 \
  --batch_size 2
```

## Clear and rerun workflow

To delete old generated dataset outputs and start fresh:

```bash
rm -rf outputs/datasetforge/*
```

Then rerun ingestion with the source you need.

## Data flow summary

1. `dataset_forge/cli.py` parses CLI arguments.
2. `dataset_forge/pipeline.py` selects the correct connector and runs extraction, cleaning, and deduplication.
3. `dataset_forge/export.py` writes the dataset to the selected format.
4. `data/raw/` is for source files you bring into the repo.
5. `outputs/` stores the generated training dataset files.

## Notes

- Use `data/raw/` for dataset source files and `outputs/` for generated outputs.
- Update or extend connectors in `dataset_forge/connectors/`.
- Use `dataset_forge/pipeline.py` to add additional cleaning, tokenization, or filtering.
- The week folders are learning modules and can be expanded with actual model-building code.

Happy building — this repo is designed to grow from a learning project into a usable dataset factory and fine-tuning workflow.
