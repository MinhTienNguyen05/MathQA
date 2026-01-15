import os

class Config:
    BASE_DIR = "src_agent"
    DATA_PATH = os.path.join(BASE_DIR, "data/agent_data.pkl")
    OUTPUT_DIR = os.path.join(BASE_DIR, "agent_model_weights/checkpoint-318")
    LOGGING_DIR = os.path.join(BASE_DIR, "logs")

    MODEL_NAME = "piikerpham/Vietnamese-Qwen2.5-math-1.5B"

    EPOCHS = 3
    BATCH_SIZE = 2
    PER_DEVICE_EVAL_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 16
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.05
    MAX_GRAD_NORM = 0.4
    WARMUP_RATIO = 0.03
    MAX_NEW_TOKENS = 2048

    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 100
    SAVE_STRATEGY = "steps"
    SAVE_STEPS = 100
    SAVE_TOTAL_LIMIT = 2
    LOAD_BEST_MODEL_AT_END = True

    FP16 = False
    BF16 = True
    CUDA_ALLOC_CONF = "expandable_segments:True"

    USE_LORA = True
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    INFERENCE_CONFIG = {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.05
    }