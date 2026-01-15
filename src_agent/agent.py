import torch
import json
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from utils import parse_tool_call_from_text
from tools import WikipediaRetriever, evaluate, solve_equation, convert_units
from peft import LoraConfig, get_peft_model, TaskType

DEFAULT_SYSTEM_PROMPT = """Bạn là một chuyên gia toán học và lập trình siêu việt. Nhiệm vụ của bạn là giải quyết các bài toán phức tạp bằng cách sử dụng công cụ (Tools) một cách chính xác.

QUY TẮC BẤT DI BẤT DỊCH (BẮT BUỘC TUÂN THỦ):
1. 🚫 KHÔNG BAO GIỜ DỪNG LẠI khi chỉ mới nêu kế hoạch (Ví dụ: "Tôi sẽ tính...", "Đầu tiên..."). 
2. ⚡ HÀNH ĐỘNG NGAY: Ngay sau khi suy nghĩ, bạn PHẢI viết code gọi tool (định dạng JSON) hoặc đưa ra phép tính ngay lập tức.
3. 🛠 SỬ DỤNG TOOL: Với các phép tính phức tạp (số lớn, phương trình, căn bậc), MẮT BUỘC phải gọi tool.
4. 🏁 KẾT LUẬN: Câu trả lời cuối cùng phải ngắn gọn và chứa đáp án số học chính xác (Ví dụ: "Đáp án là: 10").

Định dạng gọi Tool:
<tool_call>
{"name": "tên_hàm", "arguments": {"arg1": "giá_trị"}}
</tool_call>
"""

class ToolUseAgent:
    def __init__(self, model, tokenizer, tools_metadata=None, system_prompt=None, generation_cfg=None):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools_metadata or []
        
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        
        self.generation_cfg = generation_cfg or {
            "max_new_tokens": 1024,   
            "do_sample": False,      
            "temperature": 0.0,      
            "repetition_penalty": 1.05 
        }

    def invoke_tool(self, tool_name, args) -> str:
        """Gọi hàm tương ứng dựa trên tên tool."""
        normalized = tool_name.replace("_", "").lower()
        
        try:
            if normalized in ("wikipediaretriever", "wikipediasearch"):
                return str(WikipediaRetriever(**args))
            elif normalized in ("evaluate", "calculator", "calculate"):
                return str(evaluate(**args))
            elif normalized in ("solveequation", "solve"):
                return str(solve_equation(**args))
            elif normalized in ("convertunits", "unitconverter"):
                return str(convert_units(**args))
            else:
                return f"Error: Tool `{tool_name}` not found."
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def call_llm(self, conversations: list):
        """Sinh văn bản từ LLM."""
        prompt_text = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.generation_cfg.get("max_new_tokens", 1024),
            "do_sample": self.generation_cfg.get("do_sample", False),
            "temperature": self.generation_cfg.get("temperature", 0.0),
            "repetition_penalty": self.generation_cfg.get("repetition_penalty", 1.05),
            "pad_token_id": self.tokenizer.eos_token_id
        }

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)
            
        generated = outputs[0, inputs["input_ids"].shape[-1] :].cpu().numpy()
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def inference(self, question: str):
        """Vòng lặp ReAct: Suy luận -> Gọi Tool -> Nhận kết quả -> Trả lời."""
        
        full_system_prompt = self.system_prompt
        if self.tools:
            tools_desc = json.dumps(self.tools, ensure_ascii=False, indent=2)
            full_system_prompt += f"\n\nDanh sách công cụ khả dụng:\n{tools_desc}"

        conversations = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": question},
        ]

        for _ in range(10):
            llm_response = self.call_llm(conversations)
            
            conversations.append({"role": "assistant", "content": llm_response})
            
            tool_call = parse_tool_call_from_text(llm_response)
            
            if tool_call:
                name = tool_call.get("name")
                args = tool_call.get("arguments", {})

                tool_res = self.invoke_tool(name, args)
                
                conversations.append({"role": "tool", "content": tool_res})
            else:
                break
                
        return conversations, conversations[-1]["content"]

    def train(self, train_dataset, eval_dataset, cfg):
        """Thiết lập Trainer và chạy huấn luyện."""
        self.model.config.use_cache = False 
        
        if getattr(cfg, "USE_LORA", False):
            print("🟢 Setting up LoRA configuration...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=cfg.LORA_R, 
                lora_alpha=cfg.LORA_ALPHA, 
                lora_dropout=cfg.LORA_DROPOUT,
                target_modules=cfg.LORA_TARGET_MODULES
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        training_args = TrainingArguments(
            output_dir=cfg.OUTPUT_DIR,
            num_train_epochs=cfg.EPOCHS,
            per_device_train_batch_size=cfg.BATCH_SIZE,
            per_device_eval_batch_size=getattr(cfg, 'PER_DEVICE_EVAL_BATCH_SIZE', 4),
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=cfg.LEARNING_RATE,
            logging_dir=cfg.LOGGING_DIR,
            
            # Cấu hình phần cứng (FP16/BF16)
            fp16=cfg.FP16,
            bf16=cfg.BF16,
            
            eval_strategy=cfg.EVAL_STRATEGY,
            eval_steps=cfg.EVAL_STEPS,
            save_strategy=cfg.SAVE_STRATEGY,
            save_steps=cfg.SAVE_STEPS,
            save_total_limit=cfg.SAVE_TOTAL_LIMIT,
            load_best_model_at_end=cfg.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model="eval_loss",
            
            warmup_ratio=cfg.WARMUP_RATIO,
            weight_decay=cfg.WEIGHT_DECAY,
            max_grad_norm=cfg.MAX_GRAD_NORM,
            
            report_to="none",
            logging_steps=10,
            remove_unused_columns=True,
            dataloader_num_workers=2,
            
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        trainer.train()
        print(f"Saving model to {cfg.OUTPUT_DIR}...")
        trainer.save_model(cfg.OUTPUT_DIR)