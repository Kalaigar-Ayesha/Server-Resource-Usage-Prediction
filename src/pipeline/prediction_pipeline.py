import os
import joblib
import pandas as pd
from datetime import datetime

class PredictPipeline:
    def __init__(self):
        self.model_path = 'models/server_cpu_model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
    def predict(self, features):
        df = pd.DataFrame([features])
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df = df.drop(columns=['timestamp'])
            
        expected_cols = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_in', 
                         'network_out', 'request_count', 'response_time', 'hour', 'day_of_week']
        df = df[expected_cols]
        
        scaled_features = self.scaler.transform(df)
        prediction = self.model.predict(scaled_features)
        return prediction[0]

class CustomData:
    def __init__(self, cpu_usage, memory_usage, disk_usage, network_in, 
                 network_out, request_count, response_time, timestamp=None):
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.disk_usage = disk_usage
        self.network_in = network_in
        self.network_out = network_out
        self.request_count = request_count
        self.response_time = response_time
        self.timestamp = timestamp if timestamp else datetime.now().isoformat()
        
    def get_data_as_dict(self):
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "network_in": self.network_in,
            "network_out": self.network_out,
            "request_count": self.request_count,
            "response_time": self.response_time,
            "timestamp": self.timestamp
        }
