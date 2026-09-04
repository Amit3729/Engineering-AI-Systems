import ast
import operator
from typing import Any

_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}


def calculator(expression: str) -> str:
    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Only basic arithmetic is supported")
    return str(evaluate(ast.parse(expression, mode="eval").body))


def run_tools(question: str) -> tuple[list[dict[str, Any]], str]:
    import re
    match = re.search(r"(?:calculate|what is)\s+([0-9.+*/()\- ]+)[?]?$", question.lower())
    if not match:
        return [], ""
    expression = match.group(1).strip()
    try:
        result = calculator(expression)
    except (ValueError, SyntaxError, ZeroDivisionError):
        return [], ""
    return [{"name": "calculator", "arguments": {"expression": expression}, "result": result}], f"Calculator result: {result}"
