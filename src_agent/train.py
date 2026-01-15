import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import pickle

from config import Config
from agent import ToolUseAgent
from tools import TOOLS_SCHEMA

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = Config.CUDA_ALLOC_CONF
    
    print(f"Loading model: {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME, 
        device_map="auto", 
        torch_dtype="auto"
    )
    
    print(f"Loading data from {Config.DATA_PATH}...")
    with open(Config.DATA_PATH, "rb") as f:
        text_data = pickle.load(f)
        
    inputs = tokenizer(text_data, truncation=True, padding=False, max_length=Config.MAX_NEW_TOKENS)
    ds = Dataset.from_dict(inputs)
    split_ds = ds.train_test_split(test_size=0.2, shuffle=True)
    
    agent = ToolUseAgent(model, tokenizer, tools_metadata=TOOLS_SCHEMA)
    print("Starting training...")
    agent.train(split_ds['train'], split_ds['test'], Config)
    print("Training completed!")

if __name__ == "__main__":
    main()