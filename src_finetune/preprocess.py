# data/processor.py
from datasets import Dataset
from configs.config import Config

def prepare_data(df, tokenizer):
    """
    Chuyển đổi DataFrame thành Dataset, định dạng theo Chat Template và Tokenize.
    """
    ds = Dataset.from_pandas(df)
    
    def format_and_tokenize(examples):
        messages_batch = [
            [
                {"role": "system", "content": "Hãy suy nghĩ từng bước và trả lời câu hỏi"},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
            for query, response in zip(examples["query_vi"], examples["response_vi"])
        ]
        
        formatted_texts = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=False
        )
        
        tokenized = tokenizer(
            formatted_texts,
            padding=False,        
            truncation=True,
            max_length=Config.MAX_LENGTH,
        )
        return tokenized

    ds = ds.shuffle(seed=Config.SEED)
    splits = ds.train_test_split(test_size=Config.TEST_SIZE, seed=Config.SEED)
    ds_train = splits["train"]
    ds_eval = splits["test"]
    
    print("Tokenizing training dataset...")
    ds_train = ds_train.map(
        format_and_tokenize,
        batched=True,
        num_proc=Config.NUM_PROC,
        remove_columns=ds.column_names,
        desc="Tokenizing train"
    )
    
    print("Tokenizing evaluation dataset...")
    ds_eval = ds_eval.map(
        format_and_tokenize,
        batched=True,
        num_proc=Config.NUM_PROC,
        remove_columns=ds.column_names,
        desc="Tokenizing eval"
    )
    
    return ds_train, ds_eval