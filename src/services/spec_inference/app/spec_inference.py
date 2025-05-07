for pattern in return_patterns:
    match = re.search(pattern, specification, re.IGNORECASE)
    if match:
        return_type = match.group(1).lower()
        inferred_types["result"] = self._normalize_type(return_type)
        break

    # Default types for parameters without inference
for param in parameters:
    param_name = param.get("name", "")
    if param_name not in inferred_types:
        # Use default type based on parameter name
        if param_name.startswith(('n', 'i', 'j', 'k', 'count')):
            inferred_types[param_name] = "int"
        elif param_name.startswith(('x', 'y', 'z', 'val', 'rate')):
            inferred_types[param_name] = "float"
        elif param_name.startswith(('s', 'str', 'text', 'name')):
            inferred_types[param_name] = "str"
        elif param_name.startswith(('is_', 'has_', 'should_', 'flag')):
            inferred_types[param_name] = "bool"
        elif param_name.startswith(('arr', 'list', 'items', 'elements')):
            inferred_types[param_name] = "List[Any]"
        elif param_name.startswith(('dict', 'map', 'table')):
            inferred_types[param_name] = "Dict[str, Any]"
        else:
            inferred_types[param_name] = "Any"

    # Default return type if not inferred
if "result" not in inferred_types:
    if domain == "math":
        inferred_types["result"] = "int"
    elif domain == "string":
        inferred_types["result"] = "str"
    elif domain == "collections":
        inferred_types["result"] = "List[Any]"
    else:
        inferred_types["result"] = "Any"

return inferred_types

def _normalize_type(self, type_name: str) -> str:
    """Normalize type names to standard Python types."""
    type_mapping = {
        "int": "int",
        "integer": "int",
        "long": "int",
        "number": "int",
        "float": "float",
        "double": "float",
        "decimal": "float",
        "real": "float",
        "bool": "bool",
        "boolean": "bool",
        "flag": "bool",
        "str": "str",
        "string": "str",
        "text": "str",
        "char": "str",
        "character": "str",
        "list": "List[Any]",
        "array": "List[Any]",
        "sequence": "List[Any]",
        "dict": "Dict[str, Any]",
        "dictionary": "Dict[str, Any]",
        "map": "Dict[str, Any]",
        "set": "Set[Any]",
        "function": "Callable",
        "callback": "Callable",
        "object": "Any",
        "any": "Any",
        "void": "None",
        "none": "None"
    }

    return type_mapping.get(type_name.lower(), "Any")

def _infer_constraints(self, specification: str, parameters: List[Dict[str, Any]],
                       types: Dict[str, str], domain: str) -> List[str]:
    """Infer constraints from the specification."""
    inferred_constraints = []

    # Look for explicit constraints in the specification
    for param in parameters:
        param_name = param.get("name", "")
        param_type = types.get(param_name, "Any")

        # Add type-specific constraints
        if param_type == "int":
            self._add_int_constraints(specification, param_name, inferred_constraints)
        elif param_type == "float":
            self._add_float_constraints(specification, param_name, inferred_constraints)
        elif param_type == "str":
            self._add_string_constraints(specification, param_name, inferred_constraints)
        elif param_type.startswith("List") or param_type.startswith("Set"):
            self._add_collection_constraints(specification, param_name, inferred_constraints)

    # Use domain knowledge for additional constraints
    if domain in self.domain_knowledge["constraints"]:
        for pattern in self.domain_knowledge["constraints"][domain]:
            matches = re.finditer(pattern, specification, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    constraint = self._format_constraint_from_match(match, domain)
                    if constraint and constraint not in inferred_constraints:
                        inferred_constraints.append(constraint)

    # Add relationship constraints between parameters
    self._add_relationship_constraints(specification, parameters, inferred_constraints)

    return inferred_constraints

def _add_int_constraints(self, specification: str, param_name: str, constraints: List[str]):
    """Add integer-specific constraints."""
    patterns = [
        (rf"{param_name}\s+(?:must be|should be|is)\s+positive", f"{param_name} > 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+negative", f"{param_name} < 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+non-negative", f"{param_name} >= 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+non-positive", f"{param_name} <= 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+between\s+(\d+)\s+and\s+(\d+)",
         lambda m: f"{param_name} >= {m.group(1)} and {param_name} <= {m.group(2)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:greater|more|larger)\s+than\s+(\d+)",
         lambda m: f"{param_name} > {m.group(1)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:less|smaller)\s+than\s+(\d+)",
         lambda m: f"{param_name} < {m.group(1)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:at least|no less than)\s+(\d+)",
         lambda m: f"{param_name} >= {m.group(1)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:at most|no more than)\s+(\d+)",
         lambda m: f"{param_name} <= {m.group(1)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:divisible by|a multiple of)\s+(\d+)",
         lambda m: f"{param_name} % {m.group(1)} == 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+even", f"{param_name} % 2 == 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+odd", f"{param_name} % 2 == 1")
    ]

    for pattern, constraint_format in patterns:
        match = re.search(pattern, specification, re.IGNORECASE)
        if match:
            if callable(constraint_format):
                constraint = constraint_format(match)
            else:
                constraint = constraint_format

            if constraint not in constraints:
                constraints.append(constraint)

def _add_float_constraints(self, specification: str, param_name: str, constraints: List[str]):
    """Add float-specific constraints."""
    patterns = [
        (rf"{param_name}\s+(?:must be|should be|is)\s+positive", f"{param_name} > 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+negative", f"{param_name} < 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+non-negative", f"{param_name} >= 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+non-positive", f"{param_name} <= 0"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+between\s+([-+]?\d*\.?\d+)\s+and\s+([-+]?\d*\.?\d+)",
         lambda m: f"{param_name} >= {m.group(1)} and {param_name} <= {m.group(2)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:greater|more|larger)\s+than\s+([-+]?\d*\.?\d+)",
         lambda m: f"{param_name} > {m.group(1)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:less|smaller)\s+than\s+([-+]?\d*\.?\d+)",
         lambda m: f"{param_name} < {m.group(1)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:at least|no less than)\s+([-+]?\d*\.?\d+)",
         lambda m: f"{param_name} >= {m.group(1)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:at most|no more than)\s+([-+]?\d*\.?\d+)",
         lambda m: f"{param_name} <= {m.group(1)}")
    ]

    for pattern, constraint_format in patterns:
        match = re.search(pattern, specification, re.IGNORECASE)
        if match:
            if callable(constraint_format):
                constraint = constraint_format(match)
            else:
                constraint = constraint_format

            if constraint not in constraints:
                constraints.append(constraint)

def _add_string_constraints(self, specification: str, param_name: str, constraints: List[str]):
    """Add string-specific constraints."""
    patterns = [
        (rf"{param_name}\s+(?:must|should|has to)\s+contain(?:s)?\s+['\"](.*?)['\"]",
         lambda m: f"'{m.group(1)}' in {param_name}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+uppercase", f"{param_name}.isupper()"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+lowercase", f"{param_name}.islower()"),
        (rf"{param_name}\s+(?:must|should|has to)\s+(?:match|follow)\s+(?:pattern|format|regex)\s+['\"](.*?)['\"]",
         lambda m: f"re.match(r'{m.group(1)}', {param_name})"),
        (rf"{param_name}\s+(?:must have|should have|has)\s+(?:length|size)\s+(?:of|equal to)?\s+(\d+)",
         lambda m: f"len({param_name}) == {m.group(1)}"),
        (rf"{param_name}\s+(?:must have|should have|has)\s+(?:maximum|max)\s+(?:length|size)\s+(?:of)?\s+(\d+)",
         lambda m: f"len({param_name}) <= {m.group(1)}"),
        (rf"{param_name}\s+(?:must have|should have|has)\s+(?:minimum|min)\s+(?:length|size)\s+(?:of)?\s+(\d+)",
         lambda m: f"len({param_name}) >= {m.group(1)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:not empty|non-empty)", f"len({param_name}) > 0"),
        (rf"{param_name}\s+(?:must|should)\s+start(?:s)?\s+with\s+['\"](.*?)['\"]",
         lambda m: f"{param_name}.startswith('{m.group(1)}')"),
        (rf"{param_name}\s+(?:must|should)\s+end(?:s)?\s+with\s+['\"](.*?)['\"]",
         lambda m: f"{param_name}.endswith('{m.group(1)}')")
    ]

    for pattern, constraint_format in patterns:
        match = re.search(pattern, specification, re.IGNORECASE)
        if match:
            if callable(constraint_format):
                constraint = constraint_format(match)
            else:
                constraint = constraint_format

            if constraint not in constraints:
                constraints.append(constraint)

def _add_collection_constraints(self, specification: str, param_name: str, constraints: List[str]):
    """Add collection-specific constraints."""
    patterns = [
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:sorted|ordered)", f"all({param_name}[i] <= {param_name}[i+1] for i in range(len({param_name})-1))"),
        (rf"{param_name}\s+(?:must have|should have|has)\s+(?:length|size)\s+(?:of|equal to)?\s+(\d+)",
         lambda m: f"len({param_name}) == {m.group(1)}"),
        (rf"{param_name}\s+(?:must be|should be|is)\s+(?:not empty|non-empty)", f"len({param_name}) > 0"),
        (rf"{param_name}\s+(?:must have|should have|has)\s+(?:maximum|max)\s+(?:length|size)\s+(?:of)?\s+(\d+)",
         lambda m: f"len({param_name}) <= {m.group(1)}"),
        (rf"{param_name}\s+(?:must have|should have|has)\s+(?:minimum|min)\s+(?:length|size)\s+(?:of)?\s+(\d+)",
         lambda m: f"len({param_name}) >= {m.group(1)}"),
        (rf"{param_name}\s+(?:must contain|should contain|contains)\s+only\s+unique\s+elements", f"len({param_name}) == len(set({param_name}))"),
        (rf"{param_name}\s+(?:must contain|should contain|contains)\s+(\w+)",
         lambda m: f"{m.group(1)} in {param_name}")
    ]

    for pattern, constraint_format in patterns:
        match = re.search(pattern, specification, re.IGNORECASE)
        if match:
            if callable(constraint_format):
                constraint = constraint_format(match)
            else:
                constraint = constraint_format

            if constraint not in constraints:
                constraints.append(constraint)

def _add_relationship_constraints(self, specification: str, parameters: List[Dict[str, Any]], constraints: List[str]):
    """Add constraints between parameters."""
    if len(parameters) < 2:
        return

    param_names = [p.get("name", "") for p in parameters]

    # Look for relationships between parameters
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            param1 = param_names[i]
            param2 = param_names[j]

            # Check common relationship patterns
            patterns = [
                (rf"{param1}\s+(?:must be|should be|is)\s+(?:greater|more|larger)\s+than\s+{param2}", f"{param1} > {param2}"),
                (rf"{param1}\s+(?:must be|should be|is)\s+(?:less|smaller)\s+than\s+{param2}", f"{param1} < {param2}"),
                (rf"{param1}\s+(?:must be|should be|is)\s+(?:equal to|the same as)\s+{param2}", f"{param1} == {param2}"),
                (rf"{param1}\s+(?:must be|should be|is)\s+(?:not equal to|different from)\s+{param2}", f"{param1} != {param2}"),
                (rf"{param1}\s+(?:must be|should be|is)\s+(?:at least|no less than)\s+{param2}", f"{param1} >= {param2}"),
                (rf"{param1}\s+(?:must be|should be|is)\s+(?:at most|no more than)\s+{param2}", f"{param1} <= {param2}")
            ]

            for pattern, constraint in patterns:
                if re.search(pattern, specification, re.IGNORECASE):
                    if constraint not in constraints:
                        constraints.append(constraint)

            # Check in the reverse direction as well
            patterns = [
                (rf"{param2}\s+(?:must be|should be|is)\s+(?:greater|more|larger)\s+than\s+{param1}", f"{param2} > {param1}"),
                (rf"{param2}\s+(?:must be|should be|is)\s+(?:less|smaller)\s+than\s+{param1}", f"{param2} < {param1}"),
                (rf"{param2}\s+(?:must be|should be|is)\s+(?:equal to|the same as)\s+{param1}", f"{param2} == {param1}"),
                (rf"{param2}\s+(?:must be|should be|is)\s+(?:not equal to|different from)\s+{param1}", f"{param2} != {param1}"),
                (rf"{param2}\s+(?:must be|should be|is)\s+(?:at least|no less than)\s+{param1}", f"{param2} >= {param1}"),
                (rf"{param2}\s+(?:must be|should be|is)\s+(?:at most|no more than)\s+{param1}", f"{param2} <= {param1}")
            ]

            for pattern, constraint in patterns:
                if re.search(pattern, specification, re.IGNORECASE):
                    if constraint not in constraints:
                        constraints.append(constraint)

def _format_constraint_from_match(self, match: re.Match, domain: str) -> Optional[str]:
    """Format a constraint from a regex match."""
    if not match.groups():
        return None

    # Extract groups
    groups = match.groups()

    # Format based on domain and pattern
    if domain == "math":
        if len(groups) == 1:
            # Single parameter constraint
            param = groups[0]
            if "positive" in match.group(0):
                return f"{param} > 0"
            elif "negative" in match.group(0):
                return f"{param} < 0"
            elif "non-negative" in match.group(0):
                return f"{param} >= 0"
            elif "non-positive" in match.group(0):
                return f"{param} <= 0"
            elif "even" in match.group(0):
                return f"{param} % 2 == 0"
            elif "odd" in match.group(0):
                return f"{param} % 2 == 1"
        elif len(groups) == 3:
            # Range constraint
            param, min_val, max_val = groups
            return f"{param} >= {min_val} and {param} <= {max_val}"
        elif len(groups) == 2:
            # Divisibility constraint
            param, divisor = groups
            return f"{param} % {divisor} == 0"

    elif domain == "string":
        if len(groups) == 2:
            # String contains constraint
            param, substring = groups
            return f"'{substring}' in {param}"
        elif len(groups) == 1:
            # Single parameter constraint
            param = groups[0]
            if "uppercase" in match.group(0):
                return f"{param}.isupper()"
            elif "lowercase" in match.group(0):
                return f"{param}.islower()"

    # Default: return the raw match as a comment
    return f"# {match.group(0)}"

def _infer_examples(self, specification: str, function_name: str, parameters: List[Dict[str, Any]],
                    types: Dict[str, str], domain: str) -> List[Dict[str, Any]]:
    """Infer examples based on the specification."""
    examples = []

    # First, extract explicit examples from the specification
    explicit_examples = self._extract_explicit_examples(specification, parameters)
    examples.extend(explicit_examples)

    # If we don't have enough examples, generate some based on the function name and domain
    if len(examples) < self.max_examples:
        # Check if we have examples for this function name in our knowledge base
        if function_name in self.domain_knowledge["examples"]:
            known_examples = self.domain_knowledge["examples"][function_name]
            for known_example in known_examples:
                # Adapt example to match our parameter names
                adapted_example = self._adapt_example(known_example, parameters, types)
                if adapted_example and not self._is_duplicate_example(adapted_example, examples):
                    examples.append(adapted_example)
                    if len(examples) >= self.max_examples:
                        break

        # If still not enough examples, generate some based on types
        if len(examples) < self.max_examples:
            generated_examples = self._generate_examples(parameters, types, domain)
            for gen_example in generated_examples:
                if not self._is_duplicate_example(gen_example, examples):
                    examples.append(gen_example)
                    if len(examples) >= self.max_examples:
                        break

    return examples

def _extract_explicit_examples(self, specification: str, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract explicit examples from the specification."""
    examples = []

    # Regular expression to find example sections
    example_patterns = [
        r"(?:example|instance|e\.g\.|for example)[^\n.]*?(?:input|in)[^\n.]*?:?\s*(.*?)\s*(?:output|out|result|return)[^\n.]*?:?\s*(.*?)(?:\n|$)",
        r"(?:when|if)[^\n.]*?(?:input|in)[^\n.]*?:?\s*(.*?)\s*(?:output|out|result|return)[^\n.]*?:?\s*(.*?)(?:\n|$)",
        r"for\s+input\s+(.*?),\s*(?:the\s+)?(?:output|result)\s+(?:is|should be|will be)\s+(.*?)(?:\.|$)"
    ]

    for pattern in example_patterns:
        matches = re.finditer(pattern, specification, re.IGNORECASE | re.DOTALL)
        for match in matches:
            input_text = match.group(1).strip()
            output_text = match.group(2).strip()

            example = self._parse_example(input_text, output_text, parameters)
            if example and not self._is_duplicate_example(example, examples):
                examples.append(example)

    return examples

def _parse_example(self, input_text: str, output_text: str, parameters: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse input/output text into an example."""
    if not input_text or not output_text:
        return None

    # Parse input
    input_values = {}

    # Check for key-value format
    kv_pattern = r'(\w+)\s*[:=]\s*([^,]+)'
    kv_matches = re.finditer(kv_pattern, input_text)

    for match in kv_matches:
        key = match.group(1).strip()
        value_str = match.group(2).strip()

        # Find a matching parameter
        param_match = None
        for param in parameters:
            if param.get("name", "") == key:
                param_match = param
                break

        if param_match:
            # Convert value based on parameter type
            value = self._parse_value(value_str, param_match.get("type", "Any"))
            input_values[key] = value

    # If no key-value pairs found, try to match positional values
    if not input_values and parameters:
        # Split by commas or spaces
        value_strs = re.split(r'[,\s]+', input_text)

        for i, value_str in enumerate(value_strs):
            if i < len(parameters):
                param = parameters[i]
                key = param.get("name", "")
                value = self._parse_value(value_str, param.get("type", "Any"))
                input_values[key] = value

    # Parse output
    output_value = self._parse_output(output_text)

    # Create example
    if input_values:
        return {
            "input": input_values,
            "output": output_value
        }

    return None

def _parse_value(self, value_str: str, type_hint: str) -> Any:
    """Parse a string value based on type hint."""
    value_str = value_str.strip()

    # Remove quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or \
            (value_str.startswith("'") and value_str.endswith("'")):
        value_str = value_str[1:-1]

    # Convert based on type
    if type_hint == "int":
        try:
            return int(value_str)
        except ValueError:
            return 0
    elif type_hint == "float":
        try:
            return float(value_str)
        except ValueError:
            return 0.0
    elif type_hint == "bool":
        lower_str = value_str.lower()
        if lower_str in ["true", "yes", "1", "t", "y"]:
            return True
        elif lower_str in ["false", "no", "0", "f", "n"]:
            return False
        else:
            return False
    elif type_hint.startswith("List") or type_hint.startswith("Set"):
        # Try to parse as a list
        if value_str.startswith("[") and value_str.endswith("]"):
            items = value_str[1:-1].split(",")
            return [item.strip() for item in items]
        else:
            return [value_str]
    elif type_hint.startswith("Dict"):
        # Try to parse as a dictionary
        if value_str.startswith("{") and value_str.endswith("}"):
            try:
                return eval(value_str)  # Dangerous in production, but simple for demonstration
            except:
                return {}
        else:
            return {}
    else:
        # Default to string
        return value_str

def _parse_output(self, output_str: str) -> Any:
    """Parse the output string into an appropriate value."""
    output_str = output_str.strip()

    # Try to infer type from format

    # First check if it's a boolean
    if output_str.lower() in ["true", "yes", "1", "t"]:
        return True
    elif output_str.lower() in ["false", "no", "0", "f"]:
        return False

    # Check if it's an integer
    try:
        return int(output_str)
    except ValueError:
        pass

    # Check if it's a float
    try:
        return float(output_str)
    except ValueError:
        pass

    # Check if it's a list
    if output_str.startswith("[") and output_str.endswith("]"):
        try:
            items = output_str[1:-1].split(",")
            return [item.strip() for item in items]
        except:
            pass

    # Check if it's a dictionary
    if output_str.startswith("{") and output_str.endswith("}"):
        try:
            return eval(output_str)  # Dangerous in production, but simple for demonstration
        except:
            pass

    # Default to string, with quotes removed if present
    if (output_str.startswith('"') and output_str.endswith('"')) or \
            (output_str.startswith("'") and output_str.endswith("'")):
        return output_str[1:-1]

    return output_str

def _adapt_example(self, known_example: Dict[str, Any], parameters: List[Dict[str, Any]],
                   types: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Adapt a known example to match our parameter names."""
    if not known_example or "input" not in known_example or "output" not in known_example:
        return None

    # Create mappings between known example parameters and our parameters
    param_mapping = {}
    known_inputs = list(known_example["input"].keys())

    if len(known_inputs) == len(parameters):
        # Map parameters by position
        for i, known_param in enumerate(known_inputs):
            our_param = param_mapping[known_param]
            adapted_inputs[our_param] = known_value

    # Return adapted example
    return {
        "input": adapted_inputs,
        "output": known_example["output"]
    }

def _generate_examples(self, parameters: List[Dict[str, Any]], types: Dict[str, str], domain: str) -> List[Dict[str, Any]]:
    """Generate examples based on parameter types and domain."""
    examples = []

    # Generate a few examples with different values
    for i in range(self.max_examples):
        input_values = {}

        # Generate input values for each parameter
        for param in parameters:
            param_name = param.get("name", "")
            param_type = types.get(param_name, "Any")

            # Generate value based on type
            if param_type == "int":
                if domain == "math" and param_name in ["n", "num", "count"]:
                    # Common small integers for mathematical functions
                    value = [5, 10, 3][i % 3]
                else:
                    # Random integer
                    value = i * 10 + 5  # Simple deterministic pattern
            elif param_type == "float":
                # Simple float values
                value = (i + 1) * 2.5
            elif param_type == "bool":
                # Alternate boolean values
                value = (i % 2) == 0
            elif param_type == "str":
                # Simple string values
                if domain == "string":
                    value = ["hello", "world", "test"][i % 3]
                else:
                    value = f"example{i+1}"
            elif param_type.startswith("List"):
                # Simple list values
                value = list(range(i+1, i+4))
            elif param_type.startswith("Dict"):
                # Simple dictionary values
                value = {f"key{j}": j for j in range(1, i+2)}
            else:
                # Default value
                value = f"value{i+1}"

            input_values[param_name] = value

        # Determine a plausible output based on domain and input values
        output_value = self._generate_output_value(input_values, types.get("result", "Any"), domain)

        examples.append({
            "input": input_values,
            "output": output_value
        })

    return examples

def _generate_output_value(self, input_values: Dict[str, Any], output_type: str, domain: str) -> Any:
    """Generate a plausible output value based on inputs and domain."""
    # Simple output generation based on domain
    if domain == "math":
        if output_type == "int":
            # Sum of integer inputs
            return sum(v for v in input_values.values() if isinstance(v, int))
        elif output_type == "float":
            # Sum of numeric inputs
            return sum(v for v in input_values.values() if isinstance(v, (int, float)))
        elif output_type == "bool":
            # Check if sum is positive
            sum_val = sum(v for v in input_values.values() if isinstance(v, (int, float)))
            return sum_val > 0

    elif domain == "string":
        if output_type == "str":
            # Concatenate string inputs
            string_values = [str(v) for v in input_values.values() if isinstance(v, str)]
            if string_values:
                return ''.join(string_values)
            else:
                return ""
        elif output_type == "int":
            # Length of first string input
            for v in input_values.values():
                if isinstance(v, str):
                    return len(v)
            return 0
        elif output_type == "bool":
            # Check if any string input is empty
            string_values = [v for v in input_values.values() if isinstance(v, str)]
            return all(len(s) > 0 for s in string_values) if string_values else False

    elif domain == "collections":
        if output_type == "int":
            # Length of first list input
            for v in input_values.values():
                if isinstance(v, list):
                    return len(v)
            return 0
        elif output_type.startswith("List"):
            # Return first list input
            for v in input_values.values():
                if isinstance(v, list):
                    return v
            return []

    # Default outputs by type
    if output_type == "int":
        return 42
    elif output_type == "float":
        return 3.14
    elif output_type == "bool":
        return True
    elif output_type == "str":
        return "result"
    elif output_type.startswith("List"):
        return [1, 2, 3]
    elif output_type.startswith("Dict"):
        return {"key": "value"}
    else:
        return None

def _is_duplicate_example(self, example: Dict[str, Any], existing_examples: List[Dict[str, Any]]) -> bool:
    """Check if an example is a duplicate of an existing one."""
    for existing in existing_examples:
        # Compare inputs
        if example["input"] == existing["input"]:
            return True

    return False

def _infer_function_signature(self, specification: str, function_name: str, parameters: List[Dict[str, Any]],
                              types: Dict[str, str], domain: str) -> Dict[str, Any]:
    """Infer function signature based on the specification."""
    # First, check if we have a known signature for this function name
    if domain in self.domain_knowledge["signatures"] and function_name in self.domain_knowledge["signatures"][domain]:
        known_signature = self.domain_knowledge["signatures"][domain][function_name]

        # Adapt the signature to our parameter names
        adapted_signature = self._adapt_signature(known_signature, parameters, types)
        if adapted_signature:
            return adapted_signature

    # If no known signature or adaptation failed, create one from parameters and return type
    signature = {
        "name": function_name,
        "parameters": parameters,
        "return_type": types.get("result", "Any")
    }

    return signature

def _adapt_signature(self, known_signature: Dict[str, Any], parameters: List[Dict[str, Any]],
                     types: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Adapt a known signature to match our parameter names."""
    if not known_signature or "parameters" not in known_signature:
        return None

    known_params = known_signature["parameters"]

    # If parameter counts don't match, can't adapt
    if len(known_params) != len(parameters):
        return None

    # Create adapted parameters
    adapted_params = []
    for i, known_param in enumerate(known_params):
        our_param = parameters[i]

        adapted_param = {
            "name": our_param.get("name", known_param.get("name", "")),
            "type": types.get(our_param.get("name", ""), known_param.get("type", "Any"))
        }

        adapted_params.append(adapted_param)

    # Create adapted signature
    return {
        "name": known_signature.get("name", "function"),
        "parameters": adapted_params,
        "return_type": types.get("result", known_signature.get("return_type", "Any"))
    }

def _create_enhanced_specification(self, original_spec: str, inferred_context: Dict[str, Any]) -> str:
    """Create an enhanced specification with inferred information."""
    # Extract information from inferred context
    function_name = inferred_context.get("function_name", "generated_function")
    parameters = inferred_context.get("parameters", [])
    types = inferred_context.get("types", {})
    constraints = inferred_context.get("constraints", [])
    examples = inferred_context.get("examples", [])

    # Create enhanced specification
    enhanced_spec = original_spec + "\n\n# Inferred information:\n"

    # Add function signature
    param_str = ", ".join([f"{p.get('name', '')}: {types.get(p.get('name', ''), 'Any')}" for p in parameters])
    return_type = types.get("result", "Any")
    enhanced_spec += f"\nFunction: def {function_name}({param_str}) -> {return_type}\n"

    # Add types
    if types:
        enhanced_spec += "\nTypes:\n"
        for name, type_name in types.items():
            if name != "result":
                enhanced_spec += f"- {name}: {type_name}\n"
        enhanced_spec += f"- return: {return_type}\n"

    # Add constraints
    if constraints:
        enhanced_spec += "\nConstraints:\n"
        for constraint in constraints:
            enhanced_spec += f"- {constraint}\n"

    # Add examples
    if examples:
        enhanced_spec += "\nExamples:\n"
        for i, example in enumerate(examples):
            input_str = ", ".join([f"{k}={v}" for k, v in example.get("input", {}).items()])
            output_val = example.get("output", "")
            enhanced_spec += f"- Example {i+1}: For input {input_str}, output should be {output_val}\n"

    return enhanced_specparam = parameters[i].get("name", "")
    param_mapping[known_param] = our_param
else:
# Try to map parameters by type compatibility
for known_param, known_value in known_example["input"].items():
    for our_param in parameters:
        our_param_name = our_param.get("name", "")
        our_param_type = types.get(our_param_name, "Any")

        if isinstance(known_value, int) and our_param_type == "int":
            param_mapping[known_param] = our_param_name
            break
        elif isinstance(known_value, float) and our_param_type == "float":
            param_mapping[known_param] = our_param_name
            break
        elif isinstance(known_value, bool) and our_param_type == "bool":
            param_mapping[known_param] = our_param_name
            break
        elif isinstance(known_value, str) and our_param_type == "str":
            param_mapping[known_param] = our_param_name
            break
        elif isinstance(known_value, list) and our_param_type.startswith("List"):
            param_mapping[known_param] = our_param_name
            break
        elif isinstance(known_value, dict) and our_param_type.startswith("Dict"):
            param_mapping[known_param] = our_param_name
            break

# If we couldn't map all parameters, return None
if len(param_mapping) != len(known_inputs):
    return None

# Create adapted example
adapted_inputs = {}
for known_param, known_value in known_example["input"].items():
    if known_param in param_mapping:
        our_#!/usr/bin/env python3
"""
Specification inference component for the Program Synthesis System.

This component analyzes natural language descriptions and partial specifications
to infer more complete formal specifications with types, constraints, and examples.
"""

import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from program_synthesis_system.src.shared import BaseComponent
from program_synthesis_system.src.shared import FormalSpecification


class SpecInference(BaseComponent):
    """Infers formal specifications from natural language and partial specifications."""

    def __init__(self, **params):
        """Initialize the specification inference component."""
        super().__init__(**params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration parameters
        self.max_examples = self.get_param("max_examples", 3)
        self.type_inference_enabled = self.get_param("type_inference_enabled", True)
        self.constraint_inference_enabled = self.get_param("constraint_inference_enabled", True)
        self.example_inference_enabled = self.get_param("example_inference_enabled", True)
        self.function_signature_inference = self.get_param("function_signature_inference", True)

        # Advanced inference options
        self.use_symbolic_execution = self.get_param("use_symbolic_execution", False)
        self.use_knowledge_base = self.get_param("use_knowledge_base", True)

        # Load domain-specific knowledge bases if enabled
        self.domain_knowledge = {}
        if self.use_knowledge_base:
            self._load_domain_knowledge()

        self.logger.info("Specification inference component initialized")

    def _load_domain_knowledge(self):
        """Load domain-specific knowledge bases."""
        # Domain-specific type mappings
        self.domain_knowledge["types"] = {
            "math": {
                "number": "int",
                "integer": "int",
                "float": "float",
                "decimal": "float",
                "boolean": "bool",
                "matrix": "List[List[float]]",
                "vector": "List[float]"
            },
            "string": {
                "string": "str",
                "text": "str",
                "char": "str",
                "character": "str",
                "substring": "str",
                "token": "str"
            },
            "collections": {
                "list": "List[Any]",
                "array": "List[Any]",
                "set": "Set[Any]",
                "map": "Dict[str, Any]",
                "dictionary": "Dict[str, Any]",
                "tree": "Dict[str, Any]",
                "graph": "Dict[str, List[Any]]"
            },
            "io": {
                "file": "str",
                "path": "str",
                "stream": "BinaryIO",
                "buffer": "bytes",
                "input": "str",
                "output": "str"
            }
        }

        # Domain-specific constraint patterns
        self.domain_knowledge["constraints"] = {
            "math": [
                r"(\w+)\s+(?:must be|should be|is)\s+positive",
                r"(\w+)\s+(?:must be|should be|is)\s+negative",
                r"(\w+)\s+(?:must be|should be|is)\s+non-negative",
                r"(\w+)\s+(?:must be|should be|is)\s+non-positive",
                r"(\w+)\s+(?:must be|should be|is)\s+between\s+(\d+)\s+and\s+(\d+)",
                r"(\w+)\s+(?:must be|should be|is)\s+(?:divisible by|a multiple of)\s+(\d+)",
                r"(\w+)\s+(?:must be|should be|is)\s+(?:even|odd)"
            ],
            "string": [
                r"(\w+)\s+(?:must|should|has to)\s+contain(?:s)?\s+['\"](.*?)['\"]",
                r"(\w+)\s+(?:must be|should be|is)\s+(?:uppercase|lowercase)",
                r"(\w+)\s+(?:must|should|has to)\s+(?:have|match|follow)\s+(?:pattern|format|regex)\s+['\"](.*?)['\"]",
                r"(\w+)\s+(?:must have|should have|has)\s+(?:length|size)\s+(?:of|equal to)?\s+(\d+)",
                r"(\w+)\s+(?:must have|should have|has)\s+(?:maximum|minimum)\s+(?:length|size)\s+(?:of)?\s+(\d+)"
            ],
            "collections": [
                r"(\w+)\s+(?:must be|should be|is)\s+(?:sorted|ordered)",
                r"(\w+)\s+(?:must have|should have|has)\s+(?:length|size)\s+(?:of|equal to)?\s+(\d+)",
                r"(\w+)\s+(?:must not be|should not be|is not)\s+empty",
                r"(\w+)\s+(?:must contain|should contain|contains)\s+only\s+unique\s+elements",
                r"(\w+)\s+(?:must contain|should contain|contains)\s+(\w+)"
            ]
        }

        # Common function signatures by domain
        self.domain_knowledge["signatures"] = {
            "math": {
                "add": {
                    "parameters": [
                        {"name": "a", "type": "int"},
                        {"name": "b", "type": "int"}
                    ],
                    "return_type": "int"
                },
                "subtract": {
                    "parameters": [
                        {"name": "a", "type": "int"},
                        {"name": "b", "type": "int"}
                    ],
                    "return_type": "int"
                },
                "multiply": {
                    "parameters": [
                        {"name": "a", "type": "int"},
                        {"name": "b", "type": "int"}
                    ],
                    "return_type": "int"
                },
                "divide": {
                    "parameters": [
                        {"name": "a", "type": "int"},
                        {"name": "b", "type": "int"}
                    ],
                    "return_type": "float"
                },
                "factorial": {
                    "parameters": [
                        {"name": "n", "type": "int"}
                    ],
                    "return_type": "int"
                },
                "fibonacci": {
                    "parameters": [
                        {"name": "n", "type": "int"}
                    ],
                    "return_type": "int"
                },
                "is_prime": {
                    "parameters": [
                        {"name": "n", "type": "int"}
                    ],
                    "return_type": "bool"
                }
            },
            "string": {
                "reverse": {
                    "parameters": [
                        {"name": "s", "type": "str"}
                    ],
                    "return_type": "str"
                },
                "contains": {
                    "parameters": [
                        {"name": "s", "type": "str"},
                        {"name": "substring", "type": "str"}
                    ],
                    "return_type": "bool"
                },
                "count_occurrences": {
                    "parameters": [
                        {"name": "s", "type": "str"},
                        {"name": "substring", "type": "str"}
                    ],
                    "return_type": "int"
                },
                "is_palindrome": {
                    "parameters": [
                        {"name": "s", "type": "str"}
                    ],
                    "return_type": "bool"
                }
            },
            "collections": {
                "find_max": {
                    "parameters": [
                        {"name": "arr", "type": "List[int]"}
                    ],
                    "return_type": "int"
                },
                "find_min": {
                    "parameters": [
                        {"name": "arr", "type": "List[int]"}
                    ],
                    "return_type": "int"
                },
                "calculate_average": {
                    "parameters": [
                        {"name": "arr", "type": "List[float]"}
                    ],
                    "return_type": "float"
                },
                "contains": {
                    "parameters": [
                        {"name": "arr", "type": "List[Any]"},
                        {"name": "element", "type": "Any"}
                    ],
                    "return_type": "bool"
                },
                "filter": {
                    "parameters": [
                        {"name": "arr", "type": "List[Any]"},
                        {"name": "condition", "type": "Callable"}
                    ],
                    "return_type": "List[Any]"
                }
            }
        }

        # Common examples by function name
        self.domain_knowledge["examples"] = {
            "add": [
                {"input": {"a": 2, "b": 3}, "output": 5},
                {"input": {"a": -1, "b": 1}, "output": 0}
            ],
            "subtract": [
                {"input": {"a": 5, "b": 3}, "output": 2},
                {"input": {"a": 0, "b": 5}, "output": -5}
            ],
            "reverse": [
                {"input": {"s": "hello"}, "output": "olleh"},
                {"input": {"s": "ab"}, "output": "ba"}
            ],
            "is_palindrome": [
                {"input": {"s": "radar"}, "output": True},
                {"input": {"s": "hello"}, "output": False}
            ],
            "find_max": [
                {"input": {"arr": [1, 3, 2, 5, 4]}, "output": 5},
                {"input": {"arr": [-1, -3, -2]}, "output": -1}
            ]
        }

    def enhance_specification(self, specification: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance a specification with inferred types, constraints, and examples.

        Args:
            specification: The original specification text
            context: Additional context for inference

        Returns:
            Dictionary with enhanced specification and inferred context
        """
        self.logger.info("Enhancing specification")
        start_time = time.time()

        # Initialize context if not provided
        if context is None:
            context = {}

        # Extract domain from specification or context
        domain = self._infer_domain(specification, context)

        # Extract or infer function name
        function_name = self._extract_function_name(specification) or context.get("function_name", "generated_function")

        # Extract or infer parameters
        parameters = self._extract_parameters(specification) or context.get("parameters", [])

        # Infer types for parameters if enabled
        inferred_types = {}
        if self.type_inference_enabled:
            inferred_types = self._infer_types(specification, parameters, domain)

        # Infer constraints if enabled
        inferred_constraints = []
        if self.constraint_inference_enabled:
            inferred_constraints = self._infer_constraints(specification, parameters, inferred_types, domain)

        # Infer examples if enabled
        inferred_examples = []
        if self.example_inference_enabled:
            inferred_examples = self._infer_examples(specification, function_name, parameters, inferred_types, domain)

        # Infer function signature if enabled and needed
        function_signature = {}
        if self.function_signature_inference and not context.get("function_signature"):
            function_signature = self._infer_function_signature(
                specification, function_name, parameters, inferred_types, domain
            )

        # Create enhanced context
        inferred_context = {
            "domain": domain,
            "function_name": function_name,
            "parameters": parameters,
            "types": inferred_types,
            "constraints": inferred_constraints,
            "examples": inferred_examples
        }

        if function_signature:
            inferred_context["function_signature"] = function_signature

        # Create enhanced specification
        enhanced_spec = self._create_enhanced_specification(specification, inferred_context)

        end_time = time.time()
        self.logger.info(f"Enhanced specification in {end_time - start_time:.2f} seconds")

        return {
            "enhanced_spec": enhanced_spec,
            "inferred_context": inferred_context
        }

    def _infer_domain(self, specification: str, context: Dict[str, Any]) -> str:
        """Infer the domain of the specification."""
        # Check if domain is in context
        if "domain" in context:
            return context["domain"]

        # Simple keyword-based domain inference
        domain_keywords = {
            "math": ["math", "arithmetic", "calculation", "formula", "number", "algebra"],
            "string": ["string", "text", "character", "substring", "concat"],
            "collections": ["list", "array", "set", "map", "dictionary", "tree", "graph"],
            "io": ["file", "input", "output", "read", "write", "stream", "buffer"]
        }

        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in specification.lower())
            domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0:
                return best_domain[0]

        # Default domain
        return "general"

    def _extract_function_name(self, specification: str) -> Optional[str]:
        """Extract function name from the specification."""
        # Look for explicit function name mention
        patterns = [
            r"function(?:\s+name)?\s+(?:is|should be|will be)?\s+['\"]*(\w+)['\"]*",
            r"implement(?:ing|ed)?\s+(?:a|the)\s+(?:function|method)\s+['\"]*(\w+)['\"]*",
            r"create(?:ing|ed)?\s+(?:a|the)\s+(?:function|method)\s+['\"]*(\w+)['\"]*",
            r"write(?:ing|e)?\s+(?:a|the)\s+(?:function|method)\s+['\"]*(\w+)['\"]*",
            r"def\s+(\w+)\s*\("
        ]

        for pattern in patterns:
            match = re.search(pattern, specification, re.IGNORECASE)
            if match:
                return match.group(1)

        # Extract from sentence context
        function_triggers = ["function", "method", "implementation", "code", "algorithm"]
        for trigger in function_triggers:
            pattern = rf"{trigger}\s+(?:for|that|to|which)\s+(\w+)s"
            match = re.search(pattern, specification, re.IGNORECASE)
            if match:
                return match.group(1)

        # Extract from context clues
        sentences = re.split(r'[.!?]', specification)
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            if 'function' in words or 'method' in words:
                # Look for camelCase or snake_case words that might be function names
                for word in words:
                    if (any(c.isupper() for c in word) and word[0].islower()) or '_' in word:
                        if word not in ['function', 'method', 'implement', 'create', 'write']:
                            return word

        return None

    def _extract_parameters(self, specification: str) -> List[Dict[str, str]]:
        """Extract parameters from the specification."""
        parameters = []

        # Look for explicit parameter declarations
        param_patterns = [
            r"parameter(?:s)?[\s:]+(?:includes?|are|is)?\s+([\w\s,]+)",
            r"takes(?:\s+in)?[\s:]+(?:parameters?|arguments?|inputs?)?\s+([\w\s,]+)",
            r"inputs?[\s:]+(?:includes?|are|is)?\s+([\w\s,]+)",
            r"arguments?[\s:]+(?:includes?|are|is)?\s+([\w\s,]+)",
            r"def\s+\w+\s*\(([\w\s,]+)\)"
        ]

        for pattern in param_patterns:
            match = re.search(pattern, specification, re.IGNORECASE)
            if match:
                param_str = match.group(1)
                param_list = [p.strip() for p in re.split(r'[,\s]+and\s+|\s*,\s*', param_str) if p.strip()]

                for param in param_list:
                    # Check if parameter has type annotation
                    type_match = re.search(r'(\w+)\s*(?::|as|of type)\s*(\w+)', param)
                    if type_match:
                        param_name = type_match.group(1)
                        param_type = type_match.group(2)
                        parameters.append({"name": param_name, "type": param_type})
                    else:
                        parameters.append({"name": param})

                break

        # If no parameters found, try to extract from context
        if not parameters:
            # Look for words that might be parameters
            param_candidates = []
            sentences = re.split(r'[.!?]', specification)
            for sentence in sentences:
                # Look for phrases like "x is an integer"
                type_matches = re.finditer(r'(\w+)\s+(?:is|as)\s+(?:an?|the)\s+(\w+)', sentence, re.IGNORECASE)
                for match in type_matches:
                    param_name = match.group(1)
                    param_type = match.group(2)
                    if param_name.islower() and not param_name in ['function', 'method', 'implementation']:
                        param_candidates.append({"name": param_name, "type": param_type})

            # Add unique parameters
            seen_params = set()
            for param in param_candidates:
                if param["name"] not in seen_params:
                    parameters.append(param)
                    seen_params.add(param["name"])

        return parameters

    def _infer_types(self, specification: str, parameters: List[Dict[str, Any]], domain: str) -> Dict[str, str]:
        """Infer types for parameters and return value."""
        inferred_types = {}

        # Process explicit type mentions in parameters
        for param in parameters:
            param_name = param.get("name", "")

            # If parameter already has a type, use it
            if "type" in param:
                param_type = param["type"].lower()
                inferred_types[param_name] = self._normalize_type(param_type)

        # Look for type hints in the specification
        for param in parameters:
            param_name = param.get("name", "")

            # Skip if type already inferred
            if param_name in inferred_types:
                continue

            # Look for explicit type declarations
            type_patterns = [
                rf"{param_name}\s+(?:is|as|should be)\s+(?:an?|the)\s+(\w+)",
                rf"{param_name}\s+(?:is|as|should be)\s+(?:of type|of)\s+(\w+)",
                rf"{param_name}\s*:\s*(\w+)"
            ]

            for pattern in type_patterns:
                match = re.search(pattern, specification, re.IGNORECASE)
                if match:
                    type_name = match.group(1).lower()
                    inferred_types[param_name] = self._normalize_type(type_name)
                    break

        # Use domain knowledge for remaining parameters
        if domain in self.domain_knowledge["types"]:
            for param in parameters:
                param_name = param.get("name", "")

                # Skip if type already inferred
                if param_name in inferred_types:
                    continue

                # Look for domain-specific type hints
                for hint, type_name in self.domain_knowledge["types"][domain].items():
                    hint_pattern = rf"{param_name}\s+(?:is|as|should be)?\s+(?:an?|the)?\s+{hint}"
                    if re.search(hint_pattern, specification, re.IGNORECASE):
                        inferred_types[param_name] = type_name
                        break

        # Infer return type
        return_patterns = [
            r"returns?\s+(?:an?|the)?\s+(\w+)",
            r"returns?\s+(?:value of type|value type|type)\s+(\w+)",
            r"output\s+(?:is|should be|type)?\s+(?:an?|the)?\s+(\w+)",
            r"result\s+(?:is|should be|type)?\s+(?:an?|the)?\s+(\w+)",
            r"(?:->|:)\s*(\w+)"
        ]

        for pattern in return_patterns:
            match = re.search(pattern, specification, re.