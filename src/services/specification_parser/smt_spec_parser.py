"""SMT-based specification parser implementation with advanced parsing capabilities."""

from dataclasses import dataclass
import logging
import re
import sys
from typing import Any, Dict, List, Optional

from program_synthesis_system.src.shared import BaseComponent
from program_synthesis_system.src.shared import FormalSpecification
import z3


def _parse_to_ast(specification):
    """Parse specification to abstract syntax tree."""
    # Placeholder implementation
    return {"type": "specification", "content": specification}


def _extract_constraints(ast):
    """Extract constraints from AST."""
    # Placeholder implementation
    return [{"type": "constraint", "content": "placeholder"}]


def _extract_examples(specification, context):
    """Extract examples from specification and context."""
    # Placeholder implementation
    examples = context.get("examples", []) if context else []
    return examples


class SMTSpecificationParser(BaseComponent):
    """Parses specifications into SMT constraints with improved reliability and accuracy."""

    @dataclass
    class Token:
        """Token representation for parser."""

        type: str
        value: str
        position: int

    def __init__(self, **params):
        """Initialize the SMT specification parser with configurable parameters."""
        super().__init__(**params)
        self.smt_solver = self.get_param("smt_solver", "z3")
        self.type_system = self.get_param("type_system", "simple")

        # Configure logging with fallback to console handler
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        self.logger.setLevel(logging.DEBUG)

        # Add a console handler if no handlers exist
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Advanced type system with more supported types
        self.supported_types = {
            "int": z3.Int,  # Integer type
            "float": z3.Real,  # Floating point number
            "bool": z3.Bool,  # Boolean type
            "str": z3.String,  # String type (if z3 has string theory support)
            "array": None,  # Array type
            "list": None,  # List type
            "set": z3.Set,  # Set type
            "enum": z3.Datatype,  # Enumeration type
            "tuple": None,  # Tuple type
            "dict": None,  # Dictionary/Map type
            "function": None,  # Function type
            "object": None,  # Object type
        }

        # Add regex patterns for improved parsing
        self._param_pattern = re.compile(
            r"(?:parameter|param|argument|arg|input|takes|accepts)\s+(\w+)", re.IGNORECASE
        )
        self._type_patterns = {
            "int": re.compile(
                r"(?:integer|int)\s+(\w+)|(\w+)\s+(?:is|as|:)\s+(?:an?\s+)?(?:integer|int)",
                re.IGNORECASE,
            ),
            "float": re.compile(
                r"(?:float|decimal|double)\s+(\w+)|(\w+)\s+(?:is|as|:)\s+(?:a\s+)?(?:float|decimal|double)",
                re.IGNORECASE,
            ),
            "bool": re.compile(
                r"(?:boolean|bool)\s+(\w+)|(\w+)\s+(?:is|as|:)\s+(?:a\s+)?(?:boolean|bool)",
                re.IGNORECASE,
            ),
            "str": re.compile(
                r"(?:string|text|str)\s+(\w+)|(\w+)\s+(?:is|as|:)\s+(?:a\s+)?(?:string|text|str)",
                re.IGNORECASE,
            ),
            "array": re.compile(
                r"(?:array|list|sequence)\s+(?:of\s+)?(\w+)\s+(\w+)|(\w+)\s+(?:is|as|:)\s+(?:an?\s+)?(?:array|list|sequence)\s+(?:of\s+)?(\w+)",
                re.IGNORECASE,
            ),
            "set": re.compile(
                r"(?:set)\s+(?:of\s+)?(\w+)\s+(\w+)|(\w+)\s+(?:is|as|:)\s+(?:a\s+)?(?:set)\s+(?:of\s+)?(\w+)",
                re.IGNORECASE,
            ),
            "enum": re.compile(
                r"(?:enum|enumeration)\s+(\w+)|(\w+)\s+(?:is|as|:)\s+(?:an?\s+)?(?:enum|enumeration)",
                re.IGNORECASE,
            ),
        }

        # Pattern for logical operators
        self._logical_op_pattern = re.compile(r"\b(and|or|not|&&|\|\|)\b", re.IGNORECASE)

        # Enhanced comparison pattern with more operators and support for variable names with underscores
        self._comparison_pattern = re.compile(
            r'([\w_]+)\s*(>|<|==|=|!=|<=|>=|contains|in|not\s+in|is\s+not)\s*([\w_]+|\d+(?:\.\d+)?|"[^"]*"|\'[^\']*\')',
            re.IGNORECASE,
        )

        # Enhanced patterns for ranges and collections
        self._range_pattern = re.compile(
            r"([\w_]+)\s+(?:in range|between)\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        self._collection_pattern = re.compile(
            r"([\w_]+)\s+(?:in|is in|is one of|belongs to|member of)\s+\[(.*?)\]", re.IGNORECASE
        )

        # Key-value pattern for tokenization
        self._key_value_pattern = re.compile(
            r'([\w_]+)\s*[:=]\s*([\w_]+|\d+(?:\.\d+)?|"[^"]*"|\'[^\']*\')', re.IGNORECASE
        )

        # Example patterns
        self._example_pattern = re.compile(
            r"(?:example|instance|test case).*?(?:input|in)(?:\s*:)?\s*(.*?)(?:output|out|result)(?:\s*:)?\s*(.*)",
            re.IGNORECASE | re.DOTALL,
        )
        self._example_kv_pattern = re.compile(
            r'([\w_]+)\s*[:=]\s*([\w_]+|\d+(?:\.\d+)?|"[^"]*"|\'[^\']*\')', re.IGNORECASE
        )

        # Type inference patterns
        self._result_type_pattern = re.compile(
            r"return(?:s|ing)?\s+(?:a|an)?\s+(int(?:eger)?|float|double|decimal|bool(?:ean)?|str(?:ing)?|text|array|list|set|enum(?:eration)?|dict(?:ionary)?|map)",
            re.IGNORECASE,
        )

    def parse(
        self, specification: str, context: Optional[Dict[str, Any]] = None
    ) -> FormalSpecification:
        """
        Parse the specification into a formal model with improved accuracy.

        Args:
            specification: The specification text
            context: Additional context for parsing

        Returns:
            A formal specification object
        """
        if context is None:
            context = {}

        self.logger.info(
            f"Parsing specification with {self.smt_solver} and {self.type_system} type system"
        )

        # Extract parameters and their types first for use in constraint generation
        parameter_names = self._extract_parameter_names(specification)
        types = self._infer_types(specification, context, parameter_names)

        # Now extract the formal patterns with parameter knowledge
        ast = self._create_ast(specification, context, parameter_names, types)
        constraints = _extract_constraints(specification, context, parameter_names, types)
        examples = _extract_examples(specification, context, parameter_names, types)

        self.logger.info(
            f"Parsed specification with {len(constraints)} constraints and {len(examples)} examples"
        )
        self.logger.debug(f"Parameter names: {parameter_names}")
        self.logger.debug(f"Inferred types: {types}")

        return FormalSpecification(ast=ast, constraints=constraints, types=types, examples=examples)

    def _extract_parameter_names(self, specification: str) -> List[str]:
        """Extract parameter names from the specification with improved accuracy."""
        # Try to find explicit parameter declarations
        matches = self._param_pattern.findall(specification)
        parameters = [
            match for match in matches if match and match.isalnum() and not match.isdigit()
        ]

        def _create_ast(
            self,
            specification: str,
            context: Dict[str, Any],
            parameter_names: List[str],
            types: Dict[str, str],
        ) -> Dict[str, Any]:
            """Create an abstract syntax tree from the specification."""
            # Extract function name from context if available
            function_name = context.get("function_name", "generated_function")

            # Create function AST with proper typing
            parameters = []
            for param in parameter_names:
                parameters.append({"name": param, "type": types.get(param, "int")})

            return {
                "type": "function",
                "name": function_name,
                "parameters": parameters,
                "return_type": types.get("result", "int"),
                "body": [],  # Empty body to be filled by synthesis
            }

        # Look for variables used in constraints
        for comparison_match in self._comparison_pattern.finditer(specification):
            var_name = comparison_match.group(1)
            if var_name.isalnum() and not var_name.isdigit() and var_name not in parameters:
                parameters.append(var_name)

        # Look for variables in range constraints
        for range_match in self._range_pattern.finditer(specification):
            var_name = range_match.group(1)
            if var_name.isalnum() and not var_name.isdigit() and var_name not in parameters:
                parameters.append(var_name)

        # If no parameters found, use common names as fallback
        if not parameters:
            parameters = ["x"]  # Default to just x if none found
            self.logger.warning(
                "No parameters detected in specification, using default parameter 'x'"
            )

        return parameters

    def _infer_types(
        self, specification: str, context: Optional[Dict[str, Any]], parameter_names: List[str]
    ) -> Dict[str, str]:
        """Infer types from the specification with improved pattern matching."""
        types = {}
        self.logger.debug(f"Inferring types for {len(parameter_names)} parameters")

        # Start with parameter types - look for explicit type declarations
        for param in parameter_names:
            param_type = "int"  # Default type

            # Check each type pattern for this parameter
            for type_name, pattern in self._type_patterns.items():
                matches = pattern.finditer(specification)
                for match in matches:
                    groups = match.groups()
                    # Check if the parameter name is in any of the capture groups
                    if param in groups:
                        param_type = type_name
                        self.logger.debug(
                            f"Identified {param} as {type_name} based on pattern match"
                        )
                        break

            types[param] = param_type

        # Look for explicit type annotations in key-value format
        type_annotations = re.findall(r"(\w+)\s*:\s*(\w+)", specification)
        for var_name, var_type in type_annotations:
            if var_name in parameter_names:
                # Normalize the type name
                var_type = var_type.lower()
                if var_type in ("integer", "int"):
                    types[var_name] = "int"
                elif var_type in ("float", "double", "decimal", "real"):
                    types[var_name] = "float"
                elif var_type in ("boolean", "bool"):
                    types[var_name] = "bool"
                elif var_type in ("string", "str", "text"):
                    types[var_name] = "str"
                elif var_type in ("array", "list", "sequence"):
                    types[var_name] = "array"
                elif var_type in ("set"):
                    types[var_name] = "set"
                self.logger.debug(
                    f"Identified {var_name} as {types[var_name]} based on type annotation"
                )

        # Infer types from constraints
        # Look for comparison to specific value types
        bool_params = set()
        string_params = set()
        numeric_params = set()

        for match in self._comparison_pattern.finditer(specification):
            left_var, operator, right_val = match.groups()

            if left_var in parameter_names:
                # Check if compared to a string literal
                if (right_val.startswith('"') and right_val.endswith('"')) or (
                    right_val.startswith("'") and right_val.endswith("'")
                ):
                    string_params.add(left_var)

                # Check if compared to a boolean value
                elif right_val.lower() in ("true", "false"):
                    bool_params.add(left_var)

                # Check if compared to a numeric value
                elif re.match(r"-?\d+(?:\.\d+)?", right_val):
                    numeric_params.add(left_var)
                    # Check if float
                    if "." in right_val:
                        if types.get(left_var) != "str":  # Don't override string type
                            types[left_var] = "float"

        # Apply inferences where they don't conflict with explicit annotations
        for param in bool_params:
            if param in parameter_names and types.get(param) == "int":
                types[param] = "bool"

        for param in string_params:
            if param in parameter_names:
                types[param] = "str"

        # Infer return type
        result_match = self._result_type_pattern.search(specification)
        if result_match:
            result_type_text = result_match.group(1).lower()
            if "int" in result_type_text or "integer" in result_type_text:
                types["result"] = "int"
            elif (
                "float" in result_type_text
                or "double" in result_type_text
                or "decimal" in result_type_text
                or "real" in result_type_text
            ):
                types["result"] = "float"
            elif "bool" in result_type_text or "boolean" in result_type_text:
                types["result"] = "bool"
            elif (
                "str" in result_type_text
                or "string" in result_type_text
                or "text" in result_type_text
            ):
                types["result"] = "str"
            elif "array" in result_type_text or "list" in result_type_text:
                types["result"] = "array"
            elif "set" in result_type_text:
                types["result"] = "set"
            elif "dict" in result_type_text or "map" in result_type_text:
                types["result"] = "dict"
            else:
                types["result"] = "int"  # Default
        else:
            # Try to infer from context or fall back to default
            types["result"] = "int"

        # See if we can further infer result type from examples
        # Check output examples for consistent types
        example_pattern = re.compile(r"(?:output|out|result)(?:\s*:)?\s*([^,;\n]+)", re.IGNORECASE)
        example_outputs = example_pattern.findall(specification)

        bool_outputs = 0
        float_outputs = 0
        string_outputs = 0

        for output in example_outputs:
            output = output.strip()
            if output.lower() in ("true", "false"):
                bool_outputs += 1
            elif re.match(r"-?\d+\.\d+", output):
                float_outputs += 1
            elif (output.startswith('"') and output.endswith('"')) or (
                output.startswith("'") and output.endswith("'")
            ):
                string_outputs += 1

        # If we have a consistent pattern in examples, use that
        max_count = max(bool_outputs, float_outputs, string_outputs, 0)
        if max_count > 0:
            if bool_outputs == max_count:
                types["result"] = "bool"
            elif float_outputs == max_count:
                types["result"] = "float"
            elif string_outputs == max_count:
                types["result"] = "str"

        # Override with any types provided in context
        if context and "types" in context:
            types.update(context["types"])

        self.logger.info(f"Inferred types: {types}")
        return types

    def _infer_types(self, ast, context):
        """Infer types from AST and context."""
        # Placeholder implementation
        return {"result": "any"}

    def _extract_constraints(
        self,
        specification: str,
        context: Dict[str, Any],
        parameter_names: List[str],
        types: Dict[str, str],
    ) -> List[Any]:
        """Extract constraints from the specification with improved reliability."""
        constraints = []
        z3_vars = {}

        # Create Z3 variables based on parameter names and types
        for param_name in parameter_names:
            param_type = types.get(param_name, "int")
            if param_type == "int":
                z3_vars[param_name] = z3.Int(param_name)
            elif param_type == "float":
                z3_vars[param_name] = z3.Real(param_name)
            elif param_type == "bool":
                z3_vars[param_name] = z3.Bool(param_name)
            else:  # Default to Int for other types (including string for now)
                z3_vars[param_name] = z3.Int(param_name)

        # Find comparison constraints (x > 5, etc.)
        for match in self._comparison_pattern.finditer(specification):
            left_var, operator, right_val = match.groups()

            if left_var in z3_vars:
                constraint = self._create_comparison_constraint(
                    z3_vars[left_var], operator, right_val, types.get(left_var, "int")
                )
                if constraint is not None:
                    constraints.append(constraint)

        # Find range constraints (x between 1 and 10)
        for match in self._range_pattern.finditer(specification):
            var_name, min_val, max_val = match.groups()

            if var_name in z3_vars:
                var_type = types.get(var_name, "int")
                min_val_conv = self._convert_value(min_val, var_type)
                max_val_conv = self._convert_value(max_val, var_type)

                var = z3_vars.get(var_name)
                if var is not None:
                    constraint = z3.And(var >= min_val_conv, var <= max_val_conv)
                    if constraint is not None:
                        constraints.append(constraint)

        # Add any constraints provided in the context
        if "constraints" in context:
            for constraint in context["constraints"]:
                if isinstance(constraint, str):
                    parsed_constraint = self._parse_constraint_string(constraint, z3_vars, types)
                    if parsed_constraint is not None:
                        constraints.append(parsed_constraint)
                else:
                    constraints.append(constraint)

        # If no constraints were found, add a simple placeholder constraint for each parameter
        if not constraints and parameter_names:
            for param in parameter_names:
                if param in z3_vars:
                    constraints.append(z3_vars[param] >= 0)  # Safe default

        return constraints

    def _create_comparison_constraint(
        self, z3_var: Any, operator: str, right_val: str, var_type: str
    ) -> Optional[Any]:
        """Create a comparison constraint based on operator and variable type."""
        # Convert the right value based on the variable type
        try:
            right_val_conv = None

            if right_val.isalpha():  # Another variable reference
                right_z3 = z3.Int(right_val)  # Default to Int for now
                right_val_conv = right_z3
            else:
                right_val_conv = self._convert_value(right_val, var_type)

            # Create the constraint based on the operator
            result = None
            if right_val_conv is not None:
                if operator == ">":
                    result = z3_var > right_val_conv
                elif operator == "<":
                    result = z3_var < right_val_conv
                elif operator == ">=":
                    result = z3_var >= right_val_conv
                elif operator == "<=":
                    result = z3_var <= right_val_conv
                elif operator == "==" or operator == "=":
                    result = z3_var == right_val_conv
                elif operator == "!=" or operator == "<>":
                    result = z3_var != right_val_conv

            return result
        except (ValueError, TypeError):
            self.logger.warning(
                f"Could not create comparison constraint for {z3_var} {operator} {right_val}"
            )
            return None

    def _convert_value(self, value_str: str, type_str: str) -> Any:
        """Convert a string value to the appropriate type with enhanced type support."""
        if not value_str:
            return self._get_default_value(type_str)

        try:
            # Handle simple types first
            if type_str == "int":
                if value_str.lower() in ("true", "yes"):
                    return 1
                elif value_str.lower() in ("false", "no"):
                    return 0
                return int(float(value_str))  # Handle cases like "5.0" gracefully

            elif type_str == "float":
                if value_str.lower() in ("true", "yes"):
                    return 1.0
                elif value_str.lower() in ("false", "no"):
                    return 0.0
                return float(value_str)

            elif type_str == "bool":
                if value_str.lower() in ("true", "yes", "1", "t", "y"):
                    return True
                else:
                    return False

            elif type_str == "str":
                return str(value_str)

            # Handle collection types
            elif type_str.startswith("array") or type_str.startswith("list"):
                # Check for array notation [1, 2, 3]
                if value_str.startswith("[") and value_str.endswith("]"):
                    elements = [e.strip() for e in value_str[1:-1].split(",")]

                    # Try to determine element type from the collection type
                    element_type = "int"  # Default
                    element_type_match = re.search(r"(?:array|list)\s+of\s+(\w+)", type_str)
                    if element_type_match:
                        element_type = element_type_match.group(1)

                    return [self._convert_value(e, element_type) for e in elements]
                else:
                    # Try to split by commas or spaces
                    elements = [e.strip() for e in re.split(r"[,\s]+", value_str) if e.strip()]
                    return [int(e) if e.isdigit() else e for e in elements]

            elif type_str.startswith("set"):
                # Similar to array/list but with unique elements
                if value_str.startswith("{") and value_str.endswith("}"):
                    elements = [e.strip() for e in value_str[1:-1].split(",")]
                else:
                    elements = [e.strip() for e in re.split(r"[,\s]+", value_str) if e.strip()]

                # Try to determine element type
                element_type = "int"  # Default
                element_type_match = re.search(r"set\s+of\s+(\w+)", type_str)
                if element_type_match:
                    element_type = element_type_match.group(1)

                return set(self._convert_value(e, element_type) for e in elements)

            elif type_str.startswith("dict") or type_str.startswith("map"):
                # Basic dictionary parsing
                if value_str.startswith("{") and value_str.endswith("}"):
                    pairs_str = value_str[1:-1]
                    pairs = [p.strip() for p in pairs_str.split(",")]
                    result = {}

                    for pair in pairs:
                        if ":" in pair:
                            k, v = [p.strip() for p in pair.split(":", 1)]
                            result[k] = v

                    return result

            # Default to int for unknown types
            return int(float(value_str)) if value_str.replace(".", "", 1).isdigit() else value_str

        except (ValueError, TypeError) as e:
            self.logger.warning(
                f"Conversion error: {str(e)} - Cannot convert '{value_str}' to {type_str}"
            )
            return self._get_default_value(type_str)

    def _parse_constraint_string(
        self, constraint_str: str, z3_vars: Dict[str, Any], types: Dict[str, str]
    ) -> Optional[Any]:
        """Parse a constraint from a string representation with improved reliability."""
        constraint_str = constraint_str.strip()

        # Try to match the constraint against patterns
        comparison_match = self._comparison_pattern.match(constraint_str)
        if comparison_match:
            left_var, operator, right_val = comparison_match.groups()

            if left_var in z3_vars:
                return self._create_comparison_constraint(
                    z3_vars[left_var], operator, right_val, types.get(left_var, "int")
                )

        range_match = self._range_pattern.match(constraint_str)
        if range_match:
            var_name, min_val, max_val = range_match.groups()

            result = None
        if var_name in z3_vars:
            var_type = types.get(var_name, "int")
            min_val_conv = self._convert_value(min_val, var_type)
            max_val_conv = self._convert_value(max_val, var_type)

            var = z3_vars.get(var_name)
            if var is not None:
                result = z3.And(var >= min_val_conv, var <= max_val_conv)
        return result

        self.logger.warning(f"Could not parse constraint string: {constraint_str}")
        return None

    def _extract_examples(
        self,
        specification: str,
        context: Dict[str, Any],
        parameter_names: List[str],
        types: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Extract examples from the specification with improved parsing."""
        examples = []
        self.logger.debug(f"Extracting examples from specification")

        # Detect example sections - look for blocks of text that appear to be examples
        example_sections = []
        lines = specification.split("\n")

        # First pass: identify candidate example sections
        current_section = []
        in_example_section = False

        for i, line in enumerate(lines):
            if re.search(
                r"\b(example|instance|test case|sample input|test input)\b", line, re.IGNORECASE
            ):
                if current_section and in_example_section:
                    example_sections.append("\n".join(current_section))
                current_section = [line]
                in_example_section = True
            elif in_example_section:
                # Check if this line continues the example or starts a new section
                if line.strip() and not line.startswith("#") and not line.startswith("//"):
                    current_section.append(line)
                elif not line.strip() and len(current_section) > 1:
                    # Empty line might indicate end of example
                    example_sections.append("\n".join(current_section))
                    current_section = []
                    in_example_section = False

        # Don't forget the last section
        if current_section and in_example_section:
            example_sections.append("\n".join(current_section))

        # Now process each example section
        for section in example_sections:
            # Try different parsing approaches for each section

            # Approach 1: Look for input/output pattern
            match = self._example_pattern.search(section)
            if match:
                input_text, output_text = match.groups()
                example = self._parse_example_text(input_text, output_text, parameter_names, types)
                if example:
                    examples.append(example)
                    continue

            # Approach 2: Look for key-value pairs
            input_values = {}
            output_value = None

            # Extract key-value pairs
            kv_pairs = self._key_value_pattern.findall(section)
            for key, value in kv_pairs:
                key = key.strip()
                value = value.strip().strip("\"'")

                if key in parameter_names:
                    param_type = types.get(key, "int")
                    input_values[key] = self._convert_value(value, param_type)
                elif key.lower() in ("output", "result", "return", "returns"):
                    result_type = types.get("result", "int")
                    output_value = self._convert_value(value, result_type)

            # If we found input values and an output value, add as example
            if input_values and output_value is not None:
                # Fill in any missing parameters with defaults
                for param in parameter_names:
                    if param not in input_values:
                        param_type = types.get(param, "int")
                        input_values[param] = self._get_default_value(param_type)

                examples.append({"input": input_values, "output": output_value})
                continue

            # Approach 3: Look for a structured format like:
            # For input x=3, y=4, the output is 7
            input_pattern = r"(?:for|when|with|given)?\s*(?:input|in)?\s*(?:of|is|as|:)?\s*(.*?)(?:,|;)?\s*(?:the|returns|gives|produces)?\s*(?:output|out|result)\s*(?:is|of|as|:)?\s*(\S+)"
            struct_match = re.search(input_pattern, section, re.IGNORECASE | re.DOTALL)
            if struct_match:
                input_text, output_text = struct_match.groups()
                example = self._parse_example_text(input_text, output_text, parameter_names, types)
                if example:
                    examples.append(example)
                    continue

        # Add any examples provided in the context
        if "examples" in context:
            examples.extend(context["examples"])

        self.logger.info(f"Extracted {len(examples)} examples from specification")
        return examples

    def _get_default_value(self, type_str: str) -> Any:
        """Get a reasonable default value for a given type."""
        if type_str == "int":
            return 0
        elif type_str == "float":
            return 0.0
        elif type_str == "bool":
            return False
        elif type_str == "str":
            return ""
        elif type_str.startswith("array") or type_str.startswith("list"):
            return []
        elif type_str.startswith("set"):
            return set()
        elif type_str.startswith("dict") or type_str.startswith("map"):
            return {}
        else:
            return 0  # Fallback default

    def _parse_example_text(
        self, input_text: str, output_text: str, parameter_names: List[str], types: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Parse example input and output text into structured data with improved detection."""
        if not input_text or not output_text:
            return None

        self.logger.debug(f"Parsing example - Input: '{input_text}', Output: '{output_text}'")
        input_values = {}

        # Approach 1: Try to extract key-value pairs like "x=3, y=4"
        kv_pairs = self._example_kv_pattern.findall(input_text)
        for key, value in kv_pairs:
            if key in parameter_names:
                param_type = types.get(key, "int")
                try:
                    input_values[key] = self._convert_value(value, param_type)
                except ValueError:
                    self.logger.warning(
                        f"Could not convert '{value}' to {param_type} for parameter {key}"
                    )

        # Approach 2: Try to extract values in order
        if not input_values and parameter_names:
            # Look for sequences of values - numbers, quoted strings, true/false
            value_pattern = r'(\d+(?:\.\d+)?|"[^"]*"|\'[^\']*\'|true|false)'
            values = re.findall(value_pattern, input_text, re.IGNORECASE)

            for i, param in enumerate(parameter_names):
                if i < len(values) and param not in input_values:
                    value = values[i].strip("\"'")
                    param_type = types.get(param, "int")
                    try:
                        input_values[param] = self._convert_value(value, param_type)
                    except ValueError:
                        self.logger.warning(
                            f"Could not convert '{value}' to {param_type} for parameter {param}"
                        )

        # If we couldn't extract values for some parameters, use defaults
        for param in parameter_names:
            if param not in input_values:
                param_type = types.get(param, "int")
                input_values[param] = self._get_default_value(param_type)

        # Extract output value
        output_value = None
        result_type = types.get("result", "int")

        # Try to extract a value from the output text
        output_match = re.search(r'(\d+(?:\.\d+)?|"[^"]*"|\'[^\']*\'|true|false)', output_text)
        if output_match:
            try:
                output_value = self._convert_value(output_match.group(1).strip("\"'"), result_type)
            except ValueError:
                self.logger.warning(
                    f"Could not convert output '{output_match.group(1)}' to {result_type}"
                )
                output_value = self._get_default_value(result_type)
        else:
            # If no clear output value, try using the whole output text
            try:
                output_value = self._convert_value(output_text.strip(), result_type)
            except ValueError:
                self.logger.warning(f"Could not convert output '{output_text}' to {result_type}")
                output_value = self._get_default_value(result_type)

        if output_value is None:
            output_value = self._get_default_value(result_type)

        return {"input": input_values, "output": output_value}("\"'")
        param_type = types.get(param, "int")
        try:
            input_values[param] = self._convert_value(value, param_type)
        except ValueError:
            self.logger.warning(
                f"Could not convert '{value}' to {param_type} for parameter {param}"
            )

    # Approach 3: Look for comma or space-separated values if no other structure found
    # Approach 3: Look for comma or space-separated values if no other structure found


if not input_values and parameter_names:
    # Split by commas or spaces
    values = [v.strip() for v in re.split(r"[,\s]+", input_text) if v.strip()]

    # Filter out non-value tokens
    values = [v for v in values if re.match(r'^(\d+(?:\.\d+)?|"[^"]*"|\'[^\']*\'|true|false)$', v)]

    for i, param in enumerate(parameter_names):
        if i < len(values) and param not in input_values:
            value = values[i].strip("\"'")
            param_type = types.get(param, "int")
            try:
                input_values[param] = self._convert_value(value, param_type)
            except ValueError:
                self.logger.warning(
                    f"Could not convert '{value}' to {param_type} for parameter {param}"
                )
