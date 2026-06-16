import pandas as pd
from typing import Dict

def compute_demographic_missingness(df: pd.DataFrame, demographic_columns: list[str]) -> Dict[str, float]:
    """
    Computes the percentage of missing values for demographic columns.
    """
    missingness = {}
    total = len(df)
    if total == 0:
        return missingness
        
    for col in demographic_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            missingness[col] = missing_count / total
            
    return missingness

def compute_feature_completeness(df: pd.DataFrame, features: list[str]) -> Dict[str, float]:
    """
    Computes the completion rate (1 - missing rate) for input features.
    """
    completeness = {}
    total = len(df)
    if total == 0:
        return completeness
        
    for col in features:
        if col in df.columns:
            valid_count = df[col].notnull().sum()
            completeness[col] = valid_count / total
            
    return completeness
