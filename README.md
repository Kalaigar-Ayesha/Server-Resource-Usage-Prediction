# Server Resource Usage Prediction

## Problem Statement
The goal of this project is to predict future server CPU utilization based on historical server metrics. By accurately forecasting CPU usage, infrastructure teams can proactively scale resources, optimize costs, and prevent performance degradation.

## Dataset
A synthetic dataset is generated to simulate realistic server behavior. It includes daily usage patterns, workload variations, correlations between metrics, and random noise.

## Features & Target
- Target: `future_cpu_usage` (CPU utilization 1 hour in the future).
- Features: CPU, memory, disk, network traffic, request counts, response time, and extracted temporal features.

## ML Pipeline Architecture
This project utilizes a cleanly separated pipeline:
1. **Preprocessing (`src/preprocess.py`)**: Cleans data, performs time-based splitting, and scales numeric features.
2. **Training (`src/train.py`)**: Trains a `RandomForestRegressor`.
3. **Evaluation (`src/evaluate.py`)**: Computes metrics (MAE, RMSE, R²) and generates visualizations.

*(Anti-leakage: Uses strict chronological splitting and isolated scaling).*

---

## MLOps / DVC Workflow

This project uses **DVC (Data Version Control)** to manage the ML pipeline. DVC tracks large datasets, models, and execution graphs in a way that Git cannot.

### Why DVC?
Git is designed for tracking small text files (source code), not multi-gigabyte datasets or binary `.pkl` model files. If you commit large files to Git, the repository becomes bloated, slow, and GitHub will reject files over 100MB.
Instead, **Git tracks a tiny metadata file (`dvc.lock`)**, while **DVC tracks the actual large files** and stores them in a remote storage (like S3, Google Drive, or a local network drive). When you checkout a specific Git branch, `dvc pull` restores the exact dataset and model associated with that branch's code.

### Pipeline Stages (`dvc.yaml`)
Our DVC pipeline is defined in `dvc.yaml` and executes as a DAG (Directed Acyclic Graph):
`generate_data` → `preprocess` → `train` → `evaluate`

Because the pipeline tracks dependencies, DVC knows exactly which steps need to be re-run if you change a file. (e.g., If you edit `train.py`, `dvc repro` will skip generating data and preprocessing, and jump straight to training!).

### How to reproduce this project from scratch:

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Pull DVC Data (if cloning on a new machine):**
   ```bash
   dvc pull
   ```
3. **Reproduce the Pipeline:**
   ```bash
   dvc repro
   ```
4. **Compare Metrics (if you made code changes):**
   ```bash
   dvc metrics diff
   ```
