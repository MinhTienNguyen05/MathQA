import json
import gradio as gr
import torch
import os
import gc
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer

from vision import VisionModule
from agent import ToolUseAgent
from tools import TOOLS_SCHEMA

# --- Cấu hình ---
MODEL_OPTIONS = {
    "Qwen Agent": "src_agent/agent_model_weights/checkpoint-318",
    "Vietnamse Qwen 2.5 Math (1.5B)": "piikerpham/Vietnamese-Qwen2.5-math-1.5B",
    "Qwen 2.5 Math (1.5B)": "Qwen/Qwen2.5-Math-1.5B-Instruct"
    # "Qwen 2.5 Math (7B)": "Qwen/Qwen2.5-Math-7B-Instruct", # Bản 7B có thể quá nặng nếu muốn kiểm tra thì mới thêm vào
}

# --- Biến Toàn cục ---
current_model = None
current_tokenizer = None
current_agent = None
loaded_model_name = ""
vision_module = VisionModule()

# --- Xác định thiết bị (Device) phù hợp ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Đã phát hiện GPU Apple (MPS). Model sẽ được tăng tốc.")
elif torch.cuda.is_available():
    # Giữ lại để code vẫn chạy được trên máy có card NVIDIA
    device = torch.device("cuda")
    print("Đã phát hiện GPU NVIDIA (CUDA).")
else:
    device = torch.device("cpu")
    print("Không phát hiện GPU tương thích, đang sử dụng CPU. Tốc độ sẽ chậm.")


def clean_memory():
    """Hàm dọn dẹp bộ nhớ, tương thích với nhiều nền tảng."""
    global current_model, current_tokenizer, current_agent

    del current_model
    del current_tokenizer
    del current_agent

    current_model = None
    current_tokenizer = None
    current_agent = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


    gc.collect()
    print("Đã dọn dẹp bộ nhớ.")


def load_model_pipeline(model_key):
    """Hàm load model và tokenizer, đã được tối ưu cho Mac."""
    global current_model, current_tokenizer, current_agent, loaded_model_name, device

    if loaded_model_name == model_key and current_agent is not None:
        return f"Model '{model_key}' đã sẵn sàng!"

    print(f"Đang chuyển đổi sang model: {model_key}...")

    # Dọn dẹp model cũ trước khi load model mới
    if current_model is not None or current_agent is not None:
        clean_memory()

    model_path = MODEL_OPTIONS[model_key]
    try:
        print(f"Đang tải model từ: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


        print("Loading model với torch_dtype=torch.float16 để tối ưu bộ nhớ trên Mac.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        model.to(device)

        current_model = model
        current_tokenizer = tokenizer
        current_agent = ToolUseAgent(model, tokenizer, tools_metadata=TOOLS_SCHEMA)
        loaded_model_name = model_key

        print(f"Load thành công: {model_key} trên thiết bị {device}")
        return f"Đã chuyển sang: {model_key}"

    except Exception as e:
        print(f"Lỗi load model: {e}")
        traceback.print_exc()
        return f"Lỗi: {str(e)}"


def solve_math_problem(model_select, question, image_path, show_reasoning, temperature, max_tokens):
    global current_agent, loaded_model_name, vision_module, current_model

    reasoning_display = ""
    full_question = question

    # --- Xử lý ảnh (nếu có) ---
    if image_path is not None:
        if current_model is not None:
            print("Phát hiện ảnh, tạm thời unload Math Model để giải phóng bộ nhớ cho Vision Model...")
            clean_memory()

        reasoning_display += "###Xử lý Hình ảnh (Vintern-1B)\n"
        try:
            extracted_text = vision_module.extract_text_from_image(image_path)
            reasoning_display += f"> **Nội dung trích xuất:**\n{extracted_text}\n\n---\n"
            full_question = f"{extracted_text}\n\n{question}"
        except Exception as e:
            reasoning_display += f"> Lỗi đọc ảnh: {str(e)}\n\n---\n"


    if current_agent is None or loaded_model_name != model_select:
        status = load_model_pipeline(model_select)
        if "Lỗi" in status:
            return status, reasoning_display

    if not current_agent:
        return "Lỗi: Không thể khởi tạo Agent.", reasoning_display
    if not full_question.strip():
        return "Vui lòng nhập câu hỏi hoặc upload ảnh.", reasoning_display

    current_agent.generation_cfg = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
    }

    try:
        print(f"Agent đang suy luận với model: {loaded_model_name} trên thiết bị {current_model.device}")
        conversations, final_answer = current_agent.inference(full_question)


        if show_reasoning:
            step_count = 1
            for msg in conversations:
                role = msg['role']
                content = str(msg['content'])
                if role == 'assistant':
                    if "<tool_call>" in content:
                        parts = content.split("<tool_call>")
                        thought = parts[0].strip()
                        tool_code = parts[1].replace("</tool_call>", "").strip()
                        reasoning_display += f"### Bước {step_count}: Suy luận\n"
                        if thought: reasoning_display += f"{thought}\n\n"
                        reasoning_display += f"**⚡ Hành động:**\n```json\n{tool_code}\n```\n\n"
                        step_count += 1
                    else:
                        if content.strip() != final_answer.strip():
                            reasoning_display += f"###  Bước {step_count}: Suy luận\n{content}\n\n"
                            step_count += 1
                elif role == 'tool':
                    clean_res = content.replace("<tool_response>", "").replace("</tool_response>", "").strip()
                    reasoning_display += f"### 🔧 Kết quả Công cụ\n> {clean_res}\n\n---\n"

        if not final_answer:
            final_answer = conversations[-1]['content']

        return final_answer, reasoning_display

    except Exception as e:
        traceback.print_exc()
        return f"Lỗi hệ thống: {str(e)}", reasoning_display

# --- GRADIO UI ---
css = """
#reasoning_box { background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; max-height: 500px; overflow-y: auto; }
#status_box { font-weight: bold; color: #2e7d32; }
"""
with gr.Blocks(title="Math Agent + Vintern Vision", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# Hệ thống Giải Toán Đa Phương Thức (Vintern + Qwen)")
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 1. Cấu hình Model")
                model_selector = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value="Vietnamse Qwen 2.5 Math (1.5B)",
                    label="Math Agent Model",
                    interactive=True
                )
                load_status = gr.Textbox(label="Trạng thái", value="Khởi động...", elem_id="status_box", interactive=False)
            with gr.Group():
                gr.Markdown("### 2. Nhập Đề Bài")
                image_input = gr.Image(type="filepath", label="Upload ảnh bài toán")
                question_input = gr.Textbox(lines=3, placeholder="Nhập thêm yêu cầu (VD: Giải chi tiết bài toán trên)...", label="Câu hỏi bổ sung")
            with gr.Accordion("Cấu hình nâng cao", open=False):
                temperature = gr.Slider(0.0, 1.0, 0.5, label="Temperature")
                max_tokens = gr.Slider(128, 2048, 1024, label="Max Tokens")
                show_reasoning = gr.Checkbox(True, label="Hiện suy luận")
            solve_btn = gr.Button("GIẢI BÀI NGAY", variant="primary", size="lg")
        with gr.Column(scale=5):
            gr.Markdown("### Kết quả cuối cùng")
            answer_output = gr.Textbox(label="", interactive=False, lines=3)
            gr.Markdown("### Quá trình suy luận (Vision -> Thought -> Tools)")
            reasoning_output = gr.Markdown(elem_id="reasoning_box")

    model_selector.change(fn=load_model_pipeline, inputs=[model_selector], outputs=[load_status])
    solve_btn.click(fn=solve_math_problem, inputs=[model_selector, question_input, image_input, show_reasoning, temperature, max_tokens], outputs=[answer_output, reasoning_output])

    # Tự động load model mặc định khi khởi động app
    demo.load(fn=load_model_pipeline, inputs=[model_selector], outputs=[load_status])

if __name__ == "__main__":
    demo.launch(share=True)