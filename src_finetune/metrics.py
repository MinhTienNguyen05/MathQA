# utils/metrics.py
import re

def extract_answer(text):
    """
    Trích xuất câu trả lời từ text. 
    Thường tìm pattern \\boxed{...} hoặc số cuối cùng.
    """
    if not text: 
        return ""
    
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1] 
    
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1]
        
    return ""

def exact_match(pred, label):
    """So khớp chính xác chuỗi."""
    if pred is None or label is None:
        return False
    return pred.strip() == label.strip()

def numerical_accuracy(pred, label, tolerance=1e-4):
    """So sánh số học với sai số cho phép."""
    try:
        p = float(re.sub(r"[^\d.-]", "", pred))
        l = float(re.sub(r"[^\d.-]", "", label))
        return abs(p - l) < tolerance
    except:
        return exact_match(pred, label)