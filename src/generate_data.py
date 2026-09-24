import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(num_days, interval_minutes, output_path):
    logger.info(f"Generating synthetic data for {num_days} days...")
    
    start_date = datetime(2023, 1, 1)
    num_samples = int(num_days * 24 * 60 / interval_minutes)
    timestamps = [start_date + timedelta(minutes=i*interval_minutes) for i in range(num_samples)]
    
    intervals_per_day = int(24 * 60 / interval_minutes)
    time_of_day_effect = np.sin(np.linspace(0, 2 * np.pi, intervals_per_day))
    time_of_day_effect = np.tile(time_of_day_effect, num_days)
    time_of_day_effect = np.roll(time_of_day_effect, int(intervals_per_day / 4))
    
    base_requests = 1000
    request_variation = time_of_day_effect * 800 + np.random.normal(0, 100, num_samples)
    request_count = np.clip(base_requests + request_variation, 10, None)
    
    cpu_base = 20
    cpu_usage = cpu_base + (request_count / 2000) * 50 + np.random.normal(0, 5, num_samples)
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    memory_base = 40
    memory_usage = memory_base + (request_count / 3000) * 30 + np.random.normal(0, 3, num_samples)
    memory_usage = np.clip(memory_usage, 0, 100)
    
    network_in = request_count * 0.5 + np.random.normal(0, 50, num_samples)
    network_in = np.clip(network_in, 0, None)
    
    network_out = request_count * 2.5 + np.random.normal(0, 200, num_samples)
    network_out = np.clip(network_out, 0, None)
    
    response_time = 50 + (cpu_usage / 100) * 150 + np.random.normal(0, 10, num_samples)
    response_time = np.clip(response_time, 10, None)
    
    disk_usage = np.linspace(30, 70, num_samples) + np.random.normal(0, 1, num_samples)
    disk_usage = np.clip(disk_usage, 0, 100)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'disk_usage': disk_usage,
        'network_in': network_in,
        'network_out': network_out,
        'request_count': request_count,
        'response_time': response_time
    })
    
    steps_ahead = int(60 / interval_minutes)
    df['future_cpu_usage'] = df['cpu_usage'].shift(-steps_ahead)
    df = df.dropna()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to {output_path} | Shape: {df.shape}")

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    np.random.seed(params['base']['random_state'])
    generate_synthetic_data(
        num_days=params['data_generation']['num_days'],
        interval_minutes=params['data_generation']['interval_minutes'],
        output_path=params['data_generation']['output_path']
    )
