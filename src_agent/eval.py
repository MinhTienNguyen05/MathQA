import os
import torch
import pandas as pd
import gc
import time
import numpy as np
import logging
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
from sklearn.model_selection import train_test_split

from config import Config
from agent import ToolUseAgent
from tools import TOOLS_SCHEMA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eval_run.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HF_TOKEN = ""

EVAL_CONFIG = {
    "dataset_name": "5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated",
    "split": "train",
    "num_samples": None,

    "eval_target": "hf_hub",

    "hf_model_id": "Qwen/Qwen2.5-Math-1.5B",
    "judge_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "output_file": "eval_results_base.csv"
}

try:
    login(token=HF_TOKEN)
    logger.info("Đã đăng nhập Hugging Face thành công.")
except Exception as e:
    logger.warning(f"Không thể đăng nhập Hugging Face: {e}")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def contains_reasoning_steps(text):
    indicators = ['vì', 'do đó', 'nên', 'vậy', 'step', 'bước', 'first', 'trước tiên', 'then', 'sau đó', '1.', '2.', 'đầu tiên']
    text_lower = str(text).lower()
    return any(ind in text_lower for ind in indicators)

def extract_answer_llm(text, judge_model, judge_tokenizer):
    if not text or pd.isna(text): return None
    question = f'''Given the solution below, extract the final numerical answer... Solution: {str(text)[:2000]}'''
    messages = [{"role": "user", "content": question}]
    try:
        input_text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        input_text = question
    inputs = judge_tokenizer(input_text, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        outputs = judge_model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=judge_tokenizer.eos_token_id)
    return judge_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def to_number(text):
    try:
        if isinstance(text, (int, float)):
            return float(text)
        text = str(text)
        clean = text.replace(',', '').replace('$', '').replace(' ', '').strip()
        if clean.endswith('.'): clean = clean[:-1]
        try:
            return float(clean)
        except:
            pass
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if matches:
            return float(matches[-1])
        return None
    except:
        return None

def generate_responses():
    logger.info("\nGIAI ĐOẠN 1: SINH CÂU TRẢ LỜI (GENERATION)...")

    logger.info(f"Loading dataset: {EVAL_CONFIG['dataset_name']}...")
    try:
        ds_full = load_dataset(EVAL_CONFIG['dataset_name'], split=EVAL_CONFIG['split']).to_pandas()
    except Exception as e:
        logger.error(f"Lỗi load dataset: {e}")
        return None

    logger.info("Splitting dataset (Unseen Test Set)...")
    _, eval_ds_unseen = train_test_split(ds_full, test_size=0.1, random_state=42, shuffle=True)

    if EVAL_CONFIG["num_samples"] is None:
        data = eval_ds_unseen.copy()
    else:
        data = eval_ds_unseen.head(EVAL_CONFIG["num_samples"]).copy()

    logger.info(f" Đã trích xuất {len(data)} mẫu để đánh giá.")

    # Xác định tên cột câu hỏi và câu trả lời
    q_col = 'query_vi' if 'query_vi' in data.columns else 'question'
    a_col = 'response_vi' if 'response_vi' in data.columns else 'answer'

    # Lưu lại tên cột câu hỏi vào dataframe để dùng ở giai đoạn sau
    data['question_content'] = data[q_col]

    target_mode = EVAL_CONFIG.get("eval_target", "local")

    if target_mode == "local":
        local_path = Config.OUTPUT_DIR
        logger.info(f"🛠️ MODE: Evaluating LOCAL CHECKPOINT at: {local_path}")
        is_lora = os.path.exists(os.path.join(local_path, "adapter_config.json"))

        if is_lora:
            logger.info("Phát hiện LoRA Adapter. Đang load Base Model + Adapter...")
            tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, token=HF_TOKEN)
            base_model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16 if Config.BF16 else torch.float16, trust_remote_code=True, token=HF_TOKEN
            )
            model = PeftModel.from_pretrained(base_model, local_path)
        else:
            logger.info("Không thấy Adapter config. Đây là FULL FINE-TUNED MODEL.")
            logger.info(f"Đang load trực tiếp từ thư mục: {local_path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
            except:
                logger.warning("Không load được tokenizer local, dùng tokenizer gốc.")
                tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, token=HF_TOKEN)

            model = AutoModelForCausalLM.from_pretrained(
                local_path, device_map="auto", torch_dtype=torch.bfloat16 if Config.BF16 else torch.float16, trust_remote_code=True
            )
    else:
        hf_id = EVAL_CONFIG["hf_model_id"]
        logger.info(f"☁️ MODE: Evaluating HF HUB MODEL ({hf_id})")
        tokenizer = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(hf_id, device_map="auto", trust_remote_code=True, token=HF_TOKEN)

    agent = ToolUseAgent(model, tokenizer, tools_metadata=TOOLS_SCHEMA)

    predictions = []
    timings = []

    logger.info("Running Inference...")
    for _, row in tqdm(data.iterrows(), total=len(data)):
        start_time = time.time()
        try:
            _, final_answer = agent.inference(row[q_col])
        except Exception as e:
            logger.error(f"Error: {e}")
            final_answer = f"Error: {str(e)}"
        timings.append(time.time() - start_time)
        predictions.append(final_answer)

    data['pred'] = predictions
    data['time'] = timings
    data['truth'] = data[a_col] if a_col else ""

    del model, tokenizer, agent
    clear_gpu_memory()
    logger.info("Xong giai đoạn 1.")
    return data

def evaluate_metrics(data):
    if data is None or len(data) == 0: return

    logger.info("\nGIAI ĐOẠN 2: CHẤM ĐIỂM VỚI JUDGE MODEL...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(EVAL_CONFIG['judge_model'], token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(EVAL_CONFIG['judge_model'], device_map="auto", torch_dtype=torch.float16, token=HF_TOKEN)
    except Exception as e:
        logger.error(f"Lỗi load Judge Model: {e}")
        return

    logger.info("Extracting numerical answers...")
    pred_extracted_list = []
    truth_extracted_list = []

    for text in tqdm(data['pred'], desc="Extr. Preds"):
        pred_extracted_list.append(extract_answer_llm(text, model, tokenizer))
    for text in tqdm(data['truth'], desc="Extr. Truths"):
        truth_extracted_list.append(extract_answer_llm(text, model, tokenizer))

    exact_matches, numerical_matches, reasoning_flags = [], [], []
    numeric_count = 0
    results_detail = []
    data = data.reset_index(drop=True)

    for i in range(len(data)):
        p_ex, t_ex = pred_extracted_list[i], truth_extracted_list[i]

        has_reasoning = contains_reasoning_steps(data.iloc[i]['pred'])
        reasoning_flags.append(has_reasoning)

        try: em = str(p_ex).lower().strip() == str(t_ex).lower().strip()
        except: em = False
        exact_matches.append(em)

        p_num, t_num = to_number(p_ex), to_number(t_ex)
        is_num_correct = False
        if p_num is not None and t_num is not None:
            numeric_count += 1
            if abs(p_num - t_num) <= 1e-4: is_num_correct = True

        final_correct = is_num_correct if (p_num is not None and t_num is not None) else em
        numerical_matches.append(final_correct)

        results_detail.append({
            "question": data.iloc[i]['question_content'],
            "pred_raw": data.iloc[i]['pred'],
            "truth_raw": data.iloc[i]['truth'],
            "pred_extracted": p_ex,
            "truth_extracted": t_ex,
            "is_correct": final_correct,
            "has_reasoning": has_reasoning
        })

    total = len(data)
    metrics = {
        "Target Model": EVAL_CONFIG.get("eval_target"),
        "Dataset Samples": total,
        "Numerical Accuracy (%)": (sum(numerical_matches) / numeric_count * 100) if numeric_count else 0,
        "Exact Match (%)": (sum(exact_matches) / total * 100) if total else 0,
        "Reasoning Rate (%)": (sum(reasoning_flags) / total * 100) if total else 0,
        "Avg Time (s)": data['time'].mean(),
    }

    logger.info("\n" + "="*40)
    logger.info("KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
    logger.info("="*40)

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"{k}: {v:.2f}")
        else:
            logger.info(f"{k}: {v}")

    logger.info("="*40)

    pd.DataFrame(results_detail).to_csv(EVAL_CONFIG['output_file'], index=False)
    logger.info(f"Chi tiết đã lưu tại: {EVAL_CONFIG['output_file']}")

    try:
        excel_file = EVAL_CONFIG['output_file'].replace(".csv", ".xlsx")
        pd.DataFrame(results_detail).to_excel(excel_file, index=False)
        logger.info(f"Đã lưu bản Excel tại: {excel_file}")
    except:
        pass

def main():
    try:
        data_df = generate_responses()
        evaluate_metrics(data_df)
    except Exception as e:
        logger.critical(f"FATAL ERROR: {e}", exc_info=True)

if __name__ == "__main__":
    main()