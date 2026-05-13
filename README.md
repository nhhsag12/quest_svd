# QUEST-SVD

Research codebase for document retrieval/compression experiments on MMDocIR and ViDoRe.

This repository turns notebook workflows into modular scripts for reproducible experiments, ablation studies, and standalone baseline runs.

## What Is Included

- Models: ColSmol and ColQwen query encoders
- Datasets: MMDocIR Page, MMDocIR Layout, ViDoRe loader support
- Core methods:
	- traditional MaxSim
	- QUEST-SVD (SVD importance + cluster pooling)
	- hierarchical Ward pooling
	- attention pruning
	- spherical k-means pooling
	- random pruning
	- document pooling baselines (pool1d, pool2d)
- Metrics:
	- set-based retrieval metrics (page-level)
	- layout area-aware metrics (layout-level)

## Repository Layout

```
quest_svd/
	configs/
		mmdocir_page.example.env
	src/
		datasets/
			common.py
			mmdocir_page.py
			mmdocir_layout.py
			vidore.py
		experiments/
			run_mmdocir_page.py
			run_mmdocir_layout.py
			notebook-*.ipynb
		methodology/
			config.py
			data_loading.py
			retrieval.py
			methods.py
			reporting.py
			efficiency.py
			cluster_pooling/
			svd/
		metrics/
			common.py
			layout.py
			store.py
		model/
			colsmol.py
			colqwen.py
		other method/
			common.py
			traditional_method.py
			hierarchical_method.py
			attention_pruning_method.py
			spherical_kmeans_method.py
			random_pruning_method.py
			pooling_method.py
			doc_pooling.py
	pyproject.toml
	requirements.txt
```

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for full runs

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional editable install:

```powershell
pip install -e .
```

## Quick Start

Set import path from repository root:

```powershell
$env:PYTHONPATH="src"
```

Run a dry check (config/path sanity, no heavy compute):

```powershell
python src/experiments/run_mmdocir_page.py `
	--page-pkl-dir "<PAGE_PKL_DIR>" `
	--colsmol-base "<BASE_MODEL_PATH_OR_ID>" `
	--colsmol-lora "<LORA_PATH_OR_ID>" `
	--annotations-path "<ANNOTATIONS_JSONL>" `
	--pages-parquet "<PAGES_PARQUET>" `
	--dry-run
```

## Run Main Experiments

### MMDocIR Page

```powershell
python src/experiments/run_mmdocir_page.py `
	--page-pkl-dir "<PAGE_PKL_DIR>" `
	--colsmol-base "<BASE_MODEL_PATH_OR_ID>" `
	--colsmol-lora "<LORA_PATH_OR_ID>" `
	--annotations-path "<ANNOTATIONS_JSONL>" `
	--pages-parquet "<PAGES_PARQUET>" `
	--output-dir "outputs/page" `
	--model-family colsmol `
	--methods traditional ours hierarchical attention kmeans random `
	--device cuda
```

Use ColQwen instead:

```powershell
python src/experiments/run_mmdocir_page.py `
	--page-pkl-dir "<PAGE_PKL_DIR>" `
	--colsmol-base "<BASE_MODEL_PATH_OR_ID>" `
	--colsmol-lora "<LORA_PATH_OR_ID>" `
	--annotations-path "<ANNOTATIONS_JSONL>" `
	--pages-parquet "<PAGES_PARQUET>" `
	--model-family colqwen
```

### MMDocIR Layout (Area-Aware Recall)

```powershell
python src/experiments/run_mmdocir_layout.py `
	--layout-pkl-dir "<LAYOUT_PKL_DIR>" `
	--colsmol-base "<BASE_MODEL_PATH_OR_ID>" `
	--colsmol-lora "<LORA_PATH_OR_ID>" `
	--annotations-path "<ANNOTATIONS_JSONL>" `
	--layouts-parquet "<LAYOUTS_PARQUET>" `
	--output-dir "outputs/layout" `
	--layout-iou-threshold 0.5 `
	--methods traditional ours hierarchical attention kmeans random
```

## Run Baselines Independently

All scripts below are under src/other method.

```powershell
$env:PYTHONPATH="src"

python "src/other method/traditional_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<BASE>" --colsmol-lora "<LORA>" --annotations-path "<ANN>" --pages-parquet "<PARQUET>"
python "src/other method/hierarchical_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<BASE>" --colsmol-lora "<LORA>" --annotations-path "<ANN>" --pages-parquet "<PARQUET>"
python "src/other method/attention_pruning_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<BASE>" --colsmol-lora "<LORA>" --annotations-path "<ANN>" --pages-parquet "<PARQUET>"
python "src/other method/spherical_kmeans_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<BASE>" --colsmol-lora "<LORA>" --annotations-path "<ANN>" --pages-parquet "<PARQUET>" --kmeans-iters 10
python "src/other method/random_pruning_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<BASE>" --colsmol-lora "<LORA>" --annotations-path "<ANN>" --pages-parquet "<PARQUET>" --n-random-seeds 1
```

Document pooling baselines (1D/2D):

```powershell
python "src/other method/doc_pooling.py" `
	--page-pkl-dir "<PAGE_PKL_DIR>" `
	--colsmol-base "<BASE>" `
	--colsmol-lora "<LORA>" `
	--annotations-path "<ANN>" `
	--pages-parquet "<PARQUET>" `
	--pooling pool1d pool2d
```

## Output Artifacts

Each run writes CSV summaries to --output-dir, for example:

- traditional summary/domain breakdown
- ours ablation summaries
- attention/hierarchical/kmeans/random summaries
- doc pooling summaries for pool1d and pool2d

The exact file prefix depends on the script (for example, traditional, ours_ablation, doc_pooling).

## Reproducibility Checklist

- Use the same tokenizer/query prompt format as configured in code.
- Keep data splits and annotation files fixed across methods.
- Report both summary-level and domain-level metrics.
- Run with deterministic seeds where applicable (for random pruning variants).

## Notes

- The folder name src/other method contains a space. Keep it quoted in shell commands.
- If CUDA is not available, set --device cpu (runtime will be slower).