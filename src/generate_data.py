import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_days=30, interval_minutes=5, output_path='data/raw/server_metrics.csv'):
    """
    Generates synthetic server resource usage data.
    """
    print(f"Generating synthetic data for {num_days} days...")
    
    # Time generation
    start_date = datetime(2023, 1, 1)
    num_samples = int(num_days * 24 * 60 / interval_minutes)
    timestamps = [start_date + timedelta(minutes=i*interval_minutes) for i in range(num_samples)]
    
    # Feature generation with daily patterns
    # A day has 24 * 60 / 5 = 288 intervals
    intervals_per_day = int(24 * 60 / interval_minutes)
    
    # Base daily pattern (e.g., sinusoidal to simulate day/night workload)
    time_of_day_effect = np.sin(np.linspace(0, 2 * np.pi, intervals_per_day))
    time_of_day_effect = np.tile(time_of_day_effect, num_days)
    # Shift so peak is around midday
    time_of_day_effect = np.roll(time_of_day_effect, int(intervals_per_day / 4))
    
    # Request count drives CPU and memory
    base_requests = 1000
    request_variation = time_of_day_effect * 800 + np.random.normal(0, 100, num_samples)
    request_count = np.clip(base_requests + request_variation, 10, None)
    
    # CPU usage is highly correlated with requests
    cpu_base = 20
    cpu_usage = cpu_base + (request_count / 2000) * 50 + np.random.normal(0, 5, num_samples)
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    # Memory usage has a baseline and increases slightly with requests
    memory_base = 40
    memory_usage = memory_base + (request_count / 3000) * 30 + np.random.normal(0, 3, num_samples)
    memory_usage = np.clip(memory_usage, 0, 100)
    
    # Network traffic
    network_in = request_count * 0.5 + np.random.normal(0, 50, num_samples)
    network_in = np.clip(network_in, 0, None)
    
    network_out = request_count * 2.5 + np.random.normal(0, 200, num_samples)
    network_out = np.clip(network_out, 0, None)
    
    # Response time increases slightly under load
    response_time = 50 + (cpu_usage / 100) * 150 + np.random.normal(0, 10, num_samples)
    response_time = np.clip(response_time, 10, None)
    
    # Disk usage (slowly growing with some spikes)
    disk_usage = np.linspace(30, 70, num_samples) + np.random.normal(0, 1, num_samples)
    disk_usage = np.clip(disk_usage, 0, 100)
    
    # Create DataFrame
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
    
    # Create target column: CPU usage 1 hour in the future (12 steps of 5 mins)
    steps_ahead = int(60 / interval_minutes)
    df['future_cpu_usage'] = df['cpu_usage'].shift(-steps_ahead)
    
    # Drop rows with NaN target (the last hour of data)
    df = df.dropna()
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    generate_synthetic_data()
