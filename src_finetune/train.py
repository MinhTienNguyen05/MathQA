# train.py
import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from configs.config import Config
from data.loader import load_and_clean_data
from data.processor import prepare_data

# Fix lỗi phân mảnh bộ nhớ GPU thường gặp
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    # 1. Load Data
    df = load_and_clean_data()
    
    # 2. Load Model & Tokenizer
    print(f"Loading model: {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.bfloat16 if Config.BF16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Cấu hình model cho training
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # 3. Preprocess Data
    train_dataset, eval_dataset = prepare_data(df, tokenizer)
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        logging_dir=Config.LOGGING_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        max_grad_norm=Config.MAX_GRAD_NORM,
        warmup_ratio=Config.WARMUP_RATIO,
        fp16=Config.FP16,
        bf16=Config.BF16,
        logging_steps=Config.LOGGING_STEPS,
        eval_strategy=Config.EVAL_STRATEGY,
        eval_steps=Config.EVAL_STEPS,
        save_strategy=Config.SAVE_STRATEGY,
        save_steps=Config.SAVE_STEPS,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        group_by_length=Config.GROUP_BY_LENGTH,
        report_to="none", 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {Config.OUTPUT_DIR}...")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)

if __name__ == "__main__":
    main()