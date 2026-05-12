# QUEST-SVD

Codebase for MMDocIR page-level retrieval experiments with:
- Traditional MaxSim baseline
- QUEST-SVD (SVD importance + cluster pool)
- Hierarchical Ward token pooling
- Attention-score token pruning
- Spherical KMeans token pooling
- Random token pruning baseline

The project is refactored from the original notebook workflow into modular Python code for reproducibility and cleaner research artifacts.

## Project Structure

```
quest_svd/
	configs/
		mmdocir_page.example.env
	src/
		experiments/
			notebook-methodology-colsmol-mmdocir-page-ver2.ipynb
			run_mmdocir_page.py
		other method/
			common.py
			traditional_method.py
			hierarchical_method.py
			attention_pruning_method.py
			spherical_kmeans_method.py
			random_pruning_method.py
		methodology/
			config.py
			data_loading.py
			retrieval.py
			metrics.py
			efficiency.py
			reporting.py
			methods.py
			cluster_pooling/
				spherical_kmeans.py
				ward.py
			svd/
				attention_importance.py
		model/
			colsmol.py
	pyproject.toml
	requirements.txt
```

## Environment

Python 3.10+ is recommended.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Or editable package install:

```powershell
pip install -e .
```

## Run Experiments

From repository root:

```powershell
$env:PYTHONPATH="src"
python src/experiments/run_mmdocir_page.py `
	--page-pkl-dir "<PAGE_PKL_DIR>" `
	--colsmol-base "<COLSMOL_BASE>" `
	--colsmol-lora "<COLSMOL_LORA>" `
	--annotations-path "<ANNOTATIONS_PATH>" `
	--pages-parquet "<PAGES_PARQUET>" `
	--output-dir "outputs" `
	--methods traditional ours hierarchical attention kmeans random `
	--device cuda
```

Quick config sanity check (no heavy compute):

```powershell
$env:PYTHONPATH="src"
python src/experiments/run_mmdocir_page.py `
	--page-pkl-dir "<PAGE_PKL_DIR>" `
	--colsmol-base "<COLSMOL_BASE>" `
	--colsmol-lora "<COLSMOL_LORA>" `
	--annotations-path "<ANNOTATIONS_PATH>" `
	--pages-parquet "<PAGES_PARQUET>" `
	--dry-run
```

## Outputs

Per-method summary and domain-level CSVs are written to `--output-dir`, for example:
- `traditional_summary.csv`
- `traditional_domain.csv`
- `ours_ablation_summary.csv`
- `attention_pruning_summary.csv`
- `spherical_kmeans_summary.csv`
- `random_pruning_summary.csv`

## Run Baselines Individually (Other Method)

You can run each baseline as an independent script from `src/other method`:

```powershell
$env:PYTHONPATH="src"
python "src/other method/traditional_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<COLSMOL_BASE>" --colsmol-lora "<COLSMOL_LORA>" --annotations-path "<ANNOTATIONS_PATH>" --pages-parquet "<PAGES_PARQUET>"
python "src/other method/hierarchical_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<COLSMOL_BASE>" --colsmol-lora "<COLSMOL_LORA>" --annotations-path "<ANNOTATIONS_PATH>" --pages-parquet "<PAGES_PARQUET>"
python "src/other method/attention_pruning_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<COLSMOL_BASE>" --colsmol-lora "<COLSMOL_LORA>" --annotations-path "<ANNOTATIONS_PATH>" --pages-parquet "<PAGES_PARQUET>"
python "src/other method/spherical_kmeans_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<COLSMOL_BASE>" --colsmol-lora "<COLSMOL_LORA>" --annotations-path "<ANNOTATIONS_PATH>" --pages-parquet "<PAGES_PARQUET>" --kmeans-iters 10
python "src/other method/random_pruning_method.py" --page-pkl-dir "<PAGE_PKL_DIR>" --colsmol-base "<COLSMOL_BASE>" --colsmol-lora "<COLSMOL_LORA>" --annotations-path "<ANNOTATIONS_PATH>" --pages-parquet "<PAGES_PARQUET>" --n-random-seeds 1
```

## Reproducibility Notes

- Query tokenization is aligned with the notebook: `"Query: {question}" + "<pad>" * 10`.
- Evaluation uses set-based recall and nDCG with per-document page candidate pools.
- The code is organized so each component (data/model/method/evaluation) can be independently unit-tested and benchmarked.