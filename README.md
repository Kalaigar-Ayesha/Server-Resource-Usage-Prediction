# Server Resource Usage Prediction

## Problem Statement
The goal of this project is to predict future server CPU utilization based on historical server metrics. By accurately forecasting CPU usage, infrastructure teams can proactively scale resources, optimize costs, and prevent performance degradation.

## Dataset
A synthetic dataset is generated to simulate realistic server behavior. It includes daily usage patterns, workload variations, correlations between metrics, and random noise.

## Features
- `timestamp`: Date and time of the measurement.
- `cpu_usage`: Current CPU utilization percentage (0-100%).
- `memory_usage`: Current memory utilization percentage (0-100%).
- `disk_usage`: Disk I/O or space usage metric.
- `network_in` & `network_out`: Incoming/Outgoing network traffic.
- `request_count`: Number of requests processed.
- `response_time`: Average response time of requests.
- `hour` & `day_of_week`: Extracted temporal features.

## Target
- `future_cpu_usage`: CPU utilization 1 hour (12 steps of 5 minutes) in the future.

## ML Pipeline Architecture
This project utilizes a cleanly separated pipeline for data processing, training, and evaluation:

1. **Preprocessing (`src/preprocess.py`)**: Cleans data, extracts features, performs time-based splitting, and scales numeric features.
2. **Training (`src/train.py`)**: Trains a `RandomForestRegressor` on the processed data.
3. **Evaluation (`src/evaluate.py`)**: Computes MAE, RMSE, R² and generates visualizations.

### Preventing Data Leakage
In time-series forecasting, data leakage occurs when future information is accidentally used to predict the past. We strictly prevent this via two mechanisms:
1. **Chronological Splitting**: We do **not** use random shuffling. The dataset is strictly divided into chronologically sequential blocks: Train (70%) → Validation (15%) → Test (15%). This mimics a real-world scenario where a model trained on past data predicts unseen future data.
2. **Isolated Scaling**: The `StandardScaler` is fitted **only** on the training set. The validation and test sets are transformed using this learned scale, ensuring no future statistical distributions leak into the training phase.

## How to Run the Workflow

**1. Generate Raw Data**
*(Make sure your environment is activated and dependencies are installed via `pip install -r requirements.txt`)*
```bash
python src/generate_data.py
```

**2. Preprocess the Data**
Creates chronological splits and scales features. Output is saved to `data/processed/`.
```bash
python src/preprocess.py
```

**3. Train the Model**
Trains the RandomForest model and saves it to `models/server_cpu_model.pkl`.
```bash
python src/train.py
```

**4. Evaluate the Model**
Evaluates against the test set and generates `metrics.json` and a prediction plot in `metrics/`.
```bash
python src/evaluate.py
```
