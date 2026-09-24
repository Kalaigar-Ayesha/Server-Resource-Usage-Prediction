# Server Resource Usage Prediction

## Problem Statement
The goal of this project is to predict future server CPU utilization based on historical server metrics. By accurately forecasting CPU usage, infrastructure teams can proactively scale resources, optimize costs, and prevent performance degradation.

## Dataset
Since real server metrics are not currently available, a synthetic dataset is generated to simulate realistic server behavior. It includes daily usage patterns (e.g., peak hours during the day), workload variations, correlations between metrics (like request count and CPU usage), and random noise.

## Features
The dataset contains the following features (collected at 5-minute intervals):
- `timestamp`: The date and time of the measurement.
- `cpu_usage`: Current CPU utilization percentage (0-100%).
- `memory_usage`: Current memory utilization percentage (0-100%).
- `disk_usage`: Disk I/O or space usage metric.
- `network_in`: Incoming network traffic (e.g., Mbps).
- `network_out`: Outgoing network traffic (e.g., Mbps).
- `request_count`: Number of requests processed in the time window.
- `response_time`: Average response time of requests (ms).

## Target
- `future_cpu_usage`: CPU utilization 1 hour (12 steps of 5 minutes) in the future.

## ML Approach
This initial version uses a traditional machine learning approach. A `RandomForestRegressor` is employed as a baseline model to capture non-linear relationships and interactions between the features without requiring complex deep learning architectures.

## How to Run the Project

1. **Install Dependencies**
   Ensure you have Python 3.11 installed. Create a virtual environment and install the requirements:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data**
   Run the data generation script to create the synthetic dataset:
   ```bash
   python src/generate_data.py
   ```
   This will save a CSV file in `data/raw/server_metrics.csv`.

3. **Exploratory Data Analysis (EDA)**
   You can run the EDA script to visualize the data patterns and feature correlations:
   ```bash
   python notebooks/01_exploratory_data_analysis.py
   ```
   This will output some summary statistics and generate a few PNG plots in your root directory.

4. **Train the Model**
   Run the training script to train a `RandomForestRegressor` on the generated data:
   ```bash
   python src/train.py
   ```
   This will save the trained model to `models/rf_model.joblib` and evaluation metrics to `metrics/metrics.json`.
