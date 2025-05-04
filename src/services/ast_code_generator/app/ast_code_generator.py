import logging
import uuid
from datetime import time

from src.services.knowledge_base.db_adapter import DatabaseAdapter
from src.services.shared.constants.base_component import BaseComponent
from src.services.shared.logging.logger import get_logger
from src.services.shared.constants.models import SynthesisResult



from typing import Dict, Any, List


def _generate_function_stub(synthesis_result):
    pass

def _optimize_ast(ast: Dict[str, Any], level: int) -> Dict[str, Any]:
    """Apply optimizations to the AST."""
    # In a real implementation, this would apply various optimizations
    # based on the optimization level:
    #  - Level 1: Basic optimizations (constant folding, dead code elimination)
    #  - Level 2: Intermediate optimizations (loop optimization, strength reduction)
    #  - Level 3: Advanced optimizations (vectorization, inlining)

    # For demonstration, we'll return the original AST
    return ast


def _apply_style(code: str, style_guide: str) -> str:
    """Apply style formatting to the code."""
    # In a real implementation, this would use tools like black, autopep8, etc.
    # For demonstration, we'll just ensure consistent indentation

    lines = code.splitlines()
    formatted_lines = []

    for line in lines:
        if line.strip():
            # Ensure consistent indentation (four spaces)
            indent_level = 0
            for char in line:
                if char == ' ':
                    indent_level += 1
                else:
                    break

            # Normalize indentation to be a multiple of 4 spaces
            normalized_indent = (indent_level // 4) * 4
            formatted_line = ' ' * normalized_indent + line.strip()
            formatted_lines.append(formatted_line)
        else:
            # Keep empty lines
            formatted_lines.append("")

    return "\n".join(formatted_lines) + "\n"



def _add_comments(code: str, ast: Dict[str, Any]) -> str:
    """Add helpful comments to the code."""
    function_name = ast.get("name", "generated_function")
    parameters = ast.get("parameters", [])

    # Create a function docstring
    docstring = f'    """Generated function: {function_name}.\n'

    # Add parameter descriptions
    if parameters:
        docstring += "\n    Parameters:\n"
        for param in parameters:
            docstring += f"    {param}: Description of {param}\n"

    docstring += '    """\n'

    # Insert the docstring after the function definition line
    lines = code.splitlines()
    result = []

    for i, line in enumerate(lines):
        result.append(line)
        if i == 0 and line.strip().startswith("def"):
            result.append(docstring)

    return "\n".join(result) + "\n"


class CodeGenerator:
    """Generates final code from internal program representation."""

    def __init__(self, format_style="pep8"):
        self.format_style = format_style

    def generate(self, synthesis_result):
        """Generate code from synthesis result."""
        # Implementation would convert internal AST to source code
        # This is a placeholder implementation
        if not synthesis_result or not synthesis_result.program_ast:
            return "# No program was synthesized"

        return "# Generated code\ndef placeholder_function():\n    pass"

    def generate_best_effort(self, synthesis_result):
        """Generate best-effort code even if verification failed."""
        # Implementation would generate code with warnings about unverified parts
        # This is a placeholder implementation
        code = self.generate(synthesis_result)
        return code + "\n\n# WARNING: This code was not fully verified and may contain errors"


class ASTCodeGenerator(BaseComponent):
    """Generates code from program ASTs."""

    def __init__(self, **params):
        """Initialize the AST code generator."""
        super().__init__(**params)
        self.optimization_level = self.get_param("optimization_level", 1)
        self.style_guide = self.get_param("style_guide", "pep8")
        self.include_comments = self.get_param("include_comments", True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.code = self.get_param("ast_result")

    def generate(self, synthesis_result:SynthesisResult) -> str:
        """
        Generate code from a synthesis result.

        Args:
            synthesis_result: The s oh I seeynthesis result with program AST

        Returns:
            The generated code as a string
            :param synthesis_result:
        """
        self.logger.info(f"Generating code with optimization level {self.optimization_level}")

        # Get the AST from the synthesis result
        ast = synthesis_result.program_ast

        # Generate code from the AST
        if not ast:
            raise ValueError("Synthesis result contains no program AST")

        # Apply optimizations if needed
        if self.optimization_level > 0:
            ast = _optimize_ast(ast, self.optimization_level)

        # Generate the code
        code = _add_comments(self.code, ast)

        self.logger.info(f"Code generation completed, {len(code.splitlines())} lines generated")

        return code

    def generate_best_effort(self, synthesis_result: SynthesisResult) -> str | None:
        """
        Generate best-effort code from a failed synthesis result.

        Args:
            synthesis_result: The synthesis result

        Returns:
            The generated code as a string
        """
        self.logger.info("Generating best-effort code for failed synthesis")

        # Get the AST from the synthesis result
        ast = synthesis_result.program_ast

        if not ast:
            # If no AST, generate a simple function stub
            return _generate_function_stub(synthesis_result)

        # Generate partial code from the AST
        code = self._generate_from_ast(ast)

        # Add error handling and safety checks
        code = self._add_safety_measures(code, ast)

        # Add comments indicating this is a best-effort solution
        code = f"""# WARNING: This is a best-effort implementation that could not be fully verified.
# Use with caution and additional testing.

{code}"""

        self.logger.info(f"Best-effort code generation completed, {len(code.splitlines())} lines generated")

        return code

    def _generate_from_ast(self, ast: Dict[str, Any]) -> str:
        """Generate code from the AST."""
        # In a real implementation, this would be a complete AST-to-code generator
        # For demonstration, we'll implement a simple translator for function ASTs

        if ast.get("type") != "function":
            raise ValueError(f"Unsupported AST type: {ast.get('type')}")

        # Get function details
        function_name = ast.get("name", "generated_function")
        parameters = ast.get("parameters", [])
        body = ast.get("body", [])

        # Generate function signature
        param_str = ", ".join(parameters)
        code = f"def {function_name}({param_str}):\n"

        # Generate function body
        if not body:
            # Empty body, add a placeholder implementation
            if len(parameters) > 0:
                # Simple implementation that uses the first parameter
                code += f"    return {parameters[0]}\n"
            else:
                # No parameters, return a constant
                code += "    return 0\n"
        else:
            # Generate code for body statements
            body_code = self._generate_body(body)
            code += body_code

        return code

    def _generate_body(self, body: List[Dict[str, Any]]) -> str:
        """Generate code for the function body."""
        # In a real implementation, this would handle different statement types
        # For demonstration, we'll generate a simple implementation

        if not body:
            return "    pass\n"

        code_lines = []

        for statement in body:
            statement_type = statement.get("type", "unknown")

            if statement_type == "return":
                value_expr = self._generate_expression(statement.get("value", {"type": "literal", "value": 0}))
                code_lines.append(f"    return {value_expr}")
            elif statement_type == "assignment":
                target = statement.get("target", "result")
                value_expr = self._generate_expression(statement.get("value", {"type": "literal", "value": 0}))
                code_lines.append(f"    {target} = {value_expr}")
            elif statement_type == "if":
                condition = self._generate_expression(statement.get("condition", {"type": "literal", "value": True}))
                then_body = self._generate_body(statement.get("then_body", []))
                else_body = self._generate_body(statement.get("else_body", []))
                code_lines.append(f"    if {condition}:")
                code_lines.extend([f"    {line}" for line in then_body.splitlines()])
                if else_body.strip():
                    code_lines.append("    else:")
                    code_lines.extend([f"    {line}" for line in else_body.splitlines()])
            elif statement_type == "loop":
                iterator = statement.get("iterator", "i")
                iterable = self._generate_expression(statement.get("iterable", {"type": "literal", "value": "range(10)"}))
                loop_body = self._generate_body(statement.get("body", []))
                code_lines.append(f"    for {iterator} in {iterable}:")
                code_lines.extend([f"    {line}" for line in loop_body.splitlines()])
            else:
                # Unknown statement type, add a comment
                code_lines.append(f"    # Unknown statement type: {statement_type}")

        # Ensure there's at least one statement
        if not code_lines:
            code_lines.append("    pass")

        return "\n".join(code_lines) + "\n"

    def _generate_expression(self, expr: Dict[str, Any]) -> str:
        """Generate code for an expression."""
        expr_type = expr.get("type", "unknown")

        if expr_type == "literal":
            value = expr.get("value", 0)
            if isinstance(value, str):
                return f'"{value}"'
            elif isinstance(value, bool):
                return str(value).lower()
            else:
                return str(value)
        elif expr_type == "variable":
            return expr.get("name", "x")
        elif expr_type == "binary_op":
            left = self._generate_expression(expr.get("left", {"type": "literal", "value": 0}))
            right = self._generate_expression(expr.get("right", {"type": "literal", "value": 0}))
            operator = expr.get("operator", "+")
            return f"({left} {operator} {right})"
        elif expr_type == "function_call":
            func_name = expr.get("function", "func")
            args = [self._generate_expression(arg) for arg in expr.get("arguments", [])]
            args_str = ", ".join(args)
            return f"{func_name}({args_str})"
        else:
            # Unknown expression type, return a placeholder
            return "None"

    def __init__(self, database_adapter: DatabaseAdapter = None, **params):
        """Initialize the AST code generator."""
        super().__init__(**params)
        self.optimization_level = self.get_param("optimization_level", 1)
        self.style_guide = self.get_param("style_guide", "pep8")
        self.include_comments = self.get_param("include_comments", True)
        self.logger = get_logger(self.__class__.__name__)
        self.database_adapter = database_adapter

    async def generate(self, synthesis_result: SynthesisResult) -> str:
        """Generate code from a synthesis result."""
        start_time = time.time()

        # Get the AST from the synthesis result
        ast = synthesis_result.program_ast

        # Generate code from the AST
        if not ast:
            raise ValueError("Synthesis result contains no program AST")

        # Try to find similar ASTs in the database
        if self.database_adapter:
            try:
                # Generate a unique key for this AST
                ast_hash = self.database_adapter._hash_ast(ast)

                # Check if we have an exact match in Redis cache
                cached_result = await self.database_adapter.get_from_cache(f"ast:{ast_hash}")
                if cached_result:
                    self.logger.info("Using exact AST match from cache")
                    return cached_result

                # Try to find similar ASTs
                similar_results = await self.database_adapter.find_similar_ast(ast, limit=1)

                if similar_results and similar_results[0]["score"] > 0.95:
                    # Found a very similar AST, use its code
                    self.logger.info(f"Using similar AST with similarity score {similar_results[0]['score']}")
                    return similar_results[0]["generated_code"]
            except Exception as e:
                self.logger.error(f"Error searching for similar ASTs: {str(e)}")

        # Apply optimizations if needed
        if self.optimization_level > 0:
            ast = _optimize_ast(ast, self.optimization_level)

        # Generate the code
        code = self._generate_from_ast(ast)

        # Apply style if configured
        if self.style_guide:
            code = _apply_style(code, self.style_guide)

        # Add comments if configured
        if self.include_comments:
            code = _add_comments(code, ast)

        # Store in cache and vector database
        if self.database_adapter:
            try:
                # Store in Redis cache for exact matches
                ast_hash = self.database_adapter._hash_ast(ast)
                await self.database_adapter.store_in_cache(f"ast:{ast_hash}", code, ttl=3600)

                # Store in vector database for similarity search
                key = f"ast_{uuid.uuid4()}"
                await self.database_adapter.store_ast(key, ast, code)
            except Exception as e:
                self.logger.error(f"Error storing AST in database: {str(e)}")

        # Log performance metrics
        execution_time = time.time() - start_time
        self.logger.performance(
            "Code generation completed",
            extra={
                "structured_data": {
                    "metric": "code_generation_time",
                    "value": execution_time,
                    "tags": {
                        "optimization_level": self.optimization_level,
                        "lines_generated": len(code.splitlines())
                    }
                }
            }
        )

        return code

    @staticmethod
    def _add_safety_measures(code: str, ast: Dict[str, Any]) -> str:
        """Add error handling and safety measures to the code."""
        # In a real implementation, this would add exception handling,
        # input validation, and other safety measures

        function_name = ast.get("name", "generated_function")
        parameters = ast.get("parameters", [])

        # Start with the original function signature
        param_str = ", ".join(parameters)
        result = [f"def {function_name}({param_str}):", '    """Best-effort implementation.\n',
                  '    WARNING: This function was generated without complete verification.\n', '    """\n']

        # Add a docstring noting this is a best-effort implementation

        # Add input validation for each parameter
        for param in parameters:
            result.append(f"    # Validate {param}")
            result.append(f"    if {param} is None:")
            result.append(f"        raise ValueError(f\"{param} cannot be None\")")

        # Add try-except block around the original body
        result.append("    try:")

        # Include the original function body, indented
        body_lines = code.splitlines()[1:]  # Skip the function signature
        for line in body_lines:
            if line.strip():
                result.append(f"    {line}")
            else:
                result.append("")

        # Add exception handling
        result.append("    except Exception as e:")
        result.append("        # Log the error and return a safe default value")
        result.append("        print(f\"Error in {function_name}: {e}\")")
        result.append("        return None")

        return "\n".join(result) + "\n"


