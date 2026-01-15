import torch
import gc
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
        for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class VisionModule:
    def __init__(self, model_path="5CD-AI/Vintern-1B-v3_5"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"VisionModule được khởi tạo để chạy trên thiết bị: {self.device}")

    def load_model(self):
        """Load model Vintern vào bộ nhớ (GPU nếu có)"""
        if self.model is None:
            print(f"👁️ Đang load Vision Model (Vintern-1B): {self.model_path}...")
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False, # Giữ nguyên cấu hình của bạn
            ).eval()

            # <<< FIX 2: Thay .cuda() bằng .to(self.device) >>>
            self.model.to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )
            print(f"Vision Model đã sẵn sàng trên thiết bị {self.device}!")

    def unload_model(self):
        """Giải phóng bộ nhớ một cách an toàn"""
        if self.model is not None:
            print("Đang giải phóng Vision Model...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            gc.collect()

    def process_image(self, image_input, input_size=448, max_num=12):
        """Hàm này giữ nguyên, nó chỉ tạo tensor trên CPU"""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError("Input phải là đường dẫn file (str) hoặc PIL Image")

        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def extract_text_from_image(self, image_input):
        """
        Input: filepath (str) hoặc PIL Image
        Output: String nội dung
        """
        try:
            self.load_model()

            pixel_values = self.process_image(image_input, max_num=6).to(torch.bfloat16).to(self.device)

            generation_config = dict(
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
                repetition_penalty=2.5
            )

            question = '<image>\nTrích xuất nguyên văn nội dung chữ chính trong ảnh. Không thêm, không bớt, không suy luận hay diễn giải bất kỳ từ nào.'

            # Đảm bảo model và dữ liệu đã ở trên cùng thiết bị trước khi chat
            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=True
            )

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Lỗi đọc ảnh: {str(e)}"
        finally:
            self.unload_model()