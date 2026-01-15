import re
import json
import logging

logger = logging.getLogger(__name__)

def sanitize_expression(expr: str) -> str:
    """
    Làm sạch biểu thức toán học sinh ra bởi LLM trước khi đưa vào eval/sympy.
    Khắc phục lỗi cú pháp thường gặp.
    """
    if not isinstance(expr, str):
        return str(expr)
        
    expr = expr.replace("$", "")
    
    open_count = expr.count('(')
    close_count = expr.count(')')
    
    if open_count > close_count:
        expr += ')' * (open_count - close_count)
    elif close_count > open_count:
        expr = expr[:-(close_count - open_count)]
        
    return expr.strip()

def parse_tool_call_from_text(text: str):
    """
    Parse <tool_call>{...}</tool_call> một cách an toàn hơn.
    """
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    if not m:
        return None
    
    json_str = m.group(1)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            import json_repair
            return json_repair.loads(json_str)
        except ImportError:
            pass
            
        try:
            json_str_fixed = re.sub(r'(?<!")(\b\w+\b)(?!")(?=\s*:)', r'"\1"', json_str)
            return json.loads(json_str_fixed)
        except Exception as e:
            logger.error(f"Failed to parse tool_call JSON: {e} | Content: {json_str}")
            return None

def extract_answer(response):
    """Trích xuất câu trả lời cuối cùng."""
    try:
        answer = response.split("\nAnswer:")[1].strip()
    except Exception:
        answer = response
    return answer