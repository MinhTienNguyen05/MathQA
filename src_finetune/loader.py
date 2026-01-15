import pandas as pd
from datasets import load_dataset
from configs.config import Config

def drop_outliers(df, column, lower_quantile=0.01, upper_quantile=0.99):
    """Lọc bỏ các giá trị ngoại lai dựa trên phân vị."""
    q_low = df[column].quantile(lower_quantile)
    q_high = df[column].quantile(upper_quantile)
    df_filtered = df[(df[column] >= q_low) & (df[column] <= q_high)]
    return df_filtered

def load_and_clean_data(dataset_name=Config.DATASET_NAME):
    print(f"Loading dataset: {dataset_name}...")
    ds = load_dataset(dataset_name)
    df = ds["train"].to_pandas()
    
    print(f"Original size: {len(df)}")
    
    # 1. Handle missing values 
    df = df[df["query_vi"].apply(lambda x: len(str(x)) > 0)]
    df = df[df["response_vi"].apply(lambda x: len(str(x)) > 0)]
    
    # 2. Tính độ dài để lọc ngoại lai
    df["query_len"] = df["query_vi"].apply(len)
    df["response_len"] = df["response_vi"].apply(len)
    
    # 3. Drop outliers
    print("Dropping outliers...")
    df = drop_outliers(df, "response_len", lower_quantile=Config.LOWER_QUANTILE, upper_quantile=Config.UPPER_QUANTILE)
    df = drop_outliers(df, "query_len", lower_quantile=Config.LOWER_QUANTILE, upper_quantile=Config.UPPER_QUANTILE)
    
    # 4. Drop duplicates
    print("Dropping duplicates...")
    df.drop_duplicates(subset=["query_vi", "response_vi"], inplace=True)
    
    print(f"Final size after cleaning: {len(df)}")
    return df