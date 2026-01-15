# configs/config.py
import os

class Config:
    MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
    DATASET_NAME = "5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated"
    OUTPUT_DIR = "./model_weights_finetuned"
    LOGGING_DIR = "./logs"
    
    MAX_LENGTH = 1024        
    TEST_SIZE = 0.2         
    BATCH_SIZE = 4          
    PER_DEVICE_EVAL_BATCH_SIZE = 4
    
    LOWER_QUANTILE = 0.05
    UPPER_QUANTILE = 0.95
    
    EPOCHS = 2
    LEARNING_RATE = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 16
    WEIGHT_DECAY = 0.05
    MAX_GRAD_NORM = 0.4
    WARMUP_RATIO = 0.03
    
    FP16 = False
    BF16 = True              
    GROUP_BY_LENGTH = True  
    
    LOGGING_STEPS = 10
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    SAVE_TOTAL_LIMIT = 3     
    SAVE_STRATEGY = "steps"
    EVAL_STRATEGY = "steps"
    
    NUM_PROC = 4             
    SEED = 42