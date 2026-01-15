import math
import logging
import importlib.util as importlib_util
import wikipedia
from typing import Optional, Dict, Any
from utils import sanitize_expression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
wikipedia.set_lang("vi")

def WikipediaRetriever(query: str, k: int = 5) -> str:
    """Tìm kiếm Wikipedia tiếng Việt."""
    try:
        page = wikipedia.page(query, auto_suggest=True)
        return page.summary[:2000]
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Nhiều kết quả: {', '.join(e.options[:5])}..."
    except wikipedia.exceptions.PageError:
        return f"Không tìm thấy bài viết cho: {query}"
    except Exception as e:
        return f"Lỗi Wikipedia: {str(e)}"

def evaluate(expression: str, variables: Optional[Dict[str, float]] = None, precision: int = 10) -> Dict[str, Any]:
    """
    Tính toán biểu thức toán học an toàn.
    Đã tích hợp sanitize_expression để sửa lỗi cú pháp từ LLM.
    """
    try:
        variables = variables or {}
        expression = sanitize_expression(expression)
        expression = expression.replace("^", "**")

        if importlib_util.find_spec("sympy") is not None:
            import sympy as sp
            local_dict = {
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp,
                "pi": sp.pi, "e": sp.E, "ln": sp.log
            }
            # Thêm biến vào local_dict
            for var in variables.keys():
                if var not in local_dict:
                    local_dict[var] = sp.Symbol(var)

            expr = sp.sympify(expression, locals=local_dict)
            if variables:
                expr = expr.subs(variables)

            result = float(sp.N(expr, precision))
            return {"result": round(result, precision)}
        else:
            allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            safe_env = {"__builtins__": None, **allowed, **variables}
            return {"result": round(float(eval(expression, safe_env, {})), precision)}

    except Exception as e:
        logger.error(f"Error evaluating '{expression}': {e}")
        return {"error": f"Lỗi tính toán: {str(e)}"}

def solve_equation(equation: str, variable: str = 'x') -> Dict[str, Any]:
    """Giải phương trình dùng SymPy."""
    try:
        import sympy
        equation = equation.replace("$", "") 
        
        if '=' in equation:
            left, right = equation.split('=')
            equation = f"({left}) - ({right})"
            
        x = sympy.Symbol(variable)
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(equation, transformations=transformations)
        
        solutions = sympy.solve(expr, x)
        real_solutions = [float(sol) for sol in solutions if sol.is_real]
        
        return {"solutions": real_solutions} if real_solutions else {"error": "Không tìm thấy nghiệm thực"}
    except Exception as e:
        return {"error": f"Lỗi giải phương trình: {str(e)}"}

def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Chuyển đổi đơn vị cơ bản."""
    length_units = {"m": 1.0, "cm": 0.01, "km": 1000.0, "ft": 0.3048, "mi": 1609.34}
    weight_units = {"kg": 1.0, "g": 0.001, "lb": 0.453592}
    
    systems = [length_units, weight_units]
    for sys in systems:
        if from_unit in sys and to_unit in sys:
            return value * sys[from_unit] / sys[to_unit]
            
    raise ValueError("Đơn vị không tương thích hoặc không hỗ trợ")

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "WikipediaRetriever",
            "description": "Tìm kiếm thông tin, kiến thức xã hội trên Wikipedia tiếng Việt.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Từ khóa tìm kiếm"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate",
            "description": "Máy tính bỏ túi. Dùng để thực hiện các phép tính số học (+, -, *, /, ^, sqrt, sin, cos...).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Biểu thức toán học, ví dụ: '12 * 8 + 5' hoặc 'sqrt(25)'"},
                    "precision": {"type": "integer", "default": 10}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": "Giải phương trình đại số tìm nghiệm x.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "Phương trình cần giải, ví dụ: '2*x + 5 = 15' hoặc 'x^2 - 4 = 0'"},
                    "variable": {"type": "string", "description": "Tên biến cần tìm, mặc định là 'x'"}
                },
                "required": ["equation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Chuyển đổi đơn vị đo lường (độ dài, khối lượng).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Giá trị số cần đổi"},
                    "from_unit": {"type": "string", "description": "Đơn vị gốc (m, cm, km, kg, g, lb...)"},
                    "to_unit": {"type": "string", "description": "Đơn vị đích"}
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    }
]