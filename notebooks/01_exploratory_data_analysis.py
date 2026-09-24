import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for plots
plt.style.use('ggplot')

def run_eda(data_path='data/raw/server_metrics.csv'):
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run 'python src/generate_data.py' first.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("\n--- Basic Information ---")
    print(df.info())
    
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    # 1. Plot CPU Usage over time (first 3 days)
    plt.figure(figsize=(15, 5))
    subset = df.head(int(3 * 24 * 60 / 5)) # First 3 days
    plt.plot(subset['timestamp'], subset['cpu_usage'], label='CPU Usage')
    plt.title('CPU Usage Over Time (First 3 Days)')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cpu_usage_time_series.png')
    print("\nSaved plot: cpu_usage_time_series.png")
    
    # 2. Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr = df.drop('timestamp', axis=1).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("Saved plot: correlation_matrix.png")
    
    # 3. Scatter plot: Request Count vs CPU Usage
    plt.figure(figsize=(8, 6))
    plt.scatter(df['request_count'], df['cpu_usage'], alpha=0.5)
    plt.title('Request Count vs CPU Usage')
    plt.xlabel('Request Count')
    plt.ylabel('CPU Usage (%)')
    plt.tight_layout()
    plt.savefig('requests_vs_cpu.png')
    print("Saved plot: requests_vs_cpu.png")
    
    print("\nEDA complete. Plots have been saved to the current directory.")

if __name__ == "__main__":
    # If run from the root of the project
    data_path = 'data/raw/server_metrics.csv'
    if not os.path.exists(data_path):
        # Fallback if run from the notebooks directory
        data_path = '../data/raw/server_metrics.csv'
        
    run_eda(data_path)
