import pandas as pd
import numpy as np
from typing import Tuple

def generate_mock_data(num_requests: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates mock inference logs and demographic datasets.
    """
    np.random.seed(42)
    
    # Inference Logs
    request_ids = [f"REQ_{i:04d}" for i in range(num_requests)]
    priorities = ["urgent", "standard", "low"]
    
    # Introduce some artificial bias: higher chance of urgent
    predicted_priority = np.random.choice(priorities, num_requests, p=[0.15, 0.6, 0.25])
    confidence_scores = np.random.beta(a=5, b=2, size=num_requests)
    
    inference_df = pd.DataFrame({
        "request_id": request_ids,
        "input_feature_1": np.random.normal(0, 1, num_requests),
        "input_feature_2": np.random.exponential(1, num_requests),
        "predicted_priority": predicted_priority,
        "confidence_score": confidence_scores,
        "timestamp": pd.date_range(start="2023-10-01", periods=num_requests, freq="min")
    })
    
    # Demographic Dataset
    genders = ["Male", "Female", "Non-binary", "Unknown"]
    locations = ["Urban", "Suburban", "Rural"]
    
    # Introduce a missing demographic percentage
    demographic_df = pd.DataFrame({
        "request_id": request_ids,
        "gender": np.random.choice(genders, num_requests, p=[0.48, 0.48, 0.02, 0.02]),
        "location": np.random.choice(locations, num_requests, p=[0.5, 0.3, 0.2])
    })
    
    # Randomly drop some demographic rows to simulate missing data
    drop_indices = np.random.choice(demographic_df.index, size=int(num_requests * 0.05), replace=False)
    demographic_df = demographic_df.drop(drop_indices)
    
    return inference_df, demographic_df

def load_and_join_data(inference_df: pd.DataFrame, demographic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins inference logs with demographic data.
    """
    return pd.merge(inference_df, demographic_df, on="request_id", how="left")
