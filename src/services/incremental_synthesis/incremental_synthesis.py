#!/usr/bin/env python3
"""
Incremental synthesis component for the Program Synthesis System.

This component implements divide-and-conquer strategies for complex synthesis tasks,
breaking specifications into smaller sub-problems and combining solutions.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import copy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import z3
from program_synthesis_system.src.shared import BaseComponent
from src.services.shared.models import FormalSpecification, SynthesisResult


class IncrementalSynthesis(BaseComponent):
    """Implements incremental synthesis strategies for complex specifications."""

    def __init__(self, **params):
        """Initialize the incremental synthesis component."""
        super().__init__(**params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration parameters
        self.max_components = self.get_param("max_components", 5)
        self.min_component_size = self.get_param("min_component_size", 1)
        self.use_dependency_analysis = self.get_param("use_dependency_analysis", True)
        self.use_semantic_clustering = self.get_param("use_semantic_clustering", True)
        self.enable_caching = self.get_param("enable_caching", True)
        self.combine_strategy = self.get_param("combine_strategy", "sequential")

        # Initialize caching if enabled
        self.component_cache = {} if self.enable_caching else None

        self.logger.info(f"Incremental synthesis initialized with max {self.max_components} services")

    def decompose(self, formal_spec: FormalSpecification) -> List[FormalSpecification]:
        """
        Decompose a formal specification into smaller services.

        Args:
            formal_spec: The formal specification to decompose

        Returns:
            List of smaller component specifications
        """
        self.logger.info(f"Decomposing specification with {len(formal_spec.constraints)} constraints")
        start_time = time.time()

        # Check if the specification is already cached
        if self.enable_caching:
            cache_key = self._compute_cache_key(formal_spec)
            if cache_key in self.component_cache:
                self.logger.info(f"Using cached decomposition for specification")
                return self.component_cache[cache_key]

        # If the specification is small enough, don't decompose
        if len(formal_spec.constraints) <= self.min_component_size:
            self.logger.info(f"Specification too small to decompose, returning as is")
            return [formal_spec]

        # Choose decomposition strategy based on configuration
        if self.use_dependency_analysis:
            components = self._decompose_by_dependencies(formal_spec)
        elif self.use_semantic_clustering:
            components = self._decompose_by_semantic_clustering(formal_spec)
        else:
            # Default to simple partitioning
            components = self._decompose_by_partitioning(formal_spec)

        # Limit the number of services
        if len(components) > self.max_components:
            self.logger.info(f"Limiting from {len(components)} to {self.max_components} services")
            components = self._merge_components(components, self.max_components)

        # Cache the result if caching is enabled
        if self.enable_caching:
            self.component_cache[cache_key] = components

        end_time = time.time()
        self.logger.info(f"Decomposed into {len(components)} services in {end_time - start_time:.2f} seconds")

        return components

    def combine(self, component_results: List[SynthesisResult]) -> SynthesisResult:
        """
        Combine synthesis results from services into a final solution.

        Args:
            component_results: Results from synthesizing individual services

        Returns:
            Combined synthesis result
        """
        self.logger.info(f"Combining {len(component_results)} component results")
        start_time = time.time()

        # Choose combination strategy based on configuration
        if self.combine_strategy == "sequential":
            result = self._combine_sequential(component_results)
        elif self.combine_strategy == "parallel":
            result = self._combine_parallel(component_results)
        elif self.combine_strategy == "conditional":
            result = self._combine_conditional(component_results)
        else:
            # Default to sequential combination
            result = self._combine_sequential(component_results)

        end_time = time.time()
        self.logger.info(f"Combined results in {end_time - start_time:.2f} seconds")

        return result

    def _decompose_by_dependencies(self, formal_spec: FormalSpecification) -> List[FormalSpecification]:
        """Decompose a specification by analyzing variable dependencies."""
        self.logger.info("Decomposing by variable dependencies")

        # Extract variable dependencies from constraints
        dependencies = self._extract_variable_dependencies(formal_spec)

        # Find connected services in the dependency graph
        connected_components = self._find_connected_components(dependencies)

        # Create component specifications based on connected services
        components = []
        for component_vars in connected_components:
            component_spec = self._create_component_spec(formal_spec, component_vars)
            if component_spec is not None:
                components.append(component_spec)

        # If no valid services were created, return the original specification
        if not components:
            self.logger.warning("Failed to create valid services, returning original specification")
            return [formal_spec]

        return components

    def _extract_variable_dependencies(self, formal_spec: FormalSpecification) -> Dict[str, Set[str]]:
        """Extract dependencies between variables in constraints."""
        dependencies = {}

        # Initialize dependency graph
        for var_name in formal_spec.types:
            if var_name != "result":  # Skip result type
                dependencies[var_name] = set()

        # For each constraint, add edges between variables used together
        for constraint in formal_spec.constraints:
            vars_in_constraint = self._extract_variables_from_constraint(constraint)

            # Add connections between all pairs of variables in this constraint
            for var1 in vars_in_constraint:
                if var1 not in dependencies:
                    dependencies[var1] = set()

                for var2 in vars_in_constraint:
                    if var1 != var2:
                        dependencies[var1].add(var2)

        return dependencies

    def _extract_variables_from_constraint(self, constraint: Any) -> Set[str]:
        """Extract variables used in a constraint expression."""
        variables = set()

        # If this is a Z3 expression, extract variables
        if hasattr(constraint, "children"):
            # Process Z3 expression
            queue = [constraint]
            while queue:
                node = queue.pop(0)

                # Add node if it's a variable
                if z3.is_const(node) and not z3.is_bool(node) and not z3.is_int_value(node) and not z3.is_real_value(node):
                    variables.add(str(node))

                # Add children to queue
                if hasattr(node, "children"):
                    queue.extend(node.children())

        return variables

    def _find_connected_components(self, dependencies: Dict[str, Set[str]]) -> List[Set[str]]:
        """Find connected services in the dependency graph."""
        visited = set()
        components = []

        def dfs(node, component):
            """Depth-first search to find connected services."""
            visited.add(node)
            component.add(node)

            for neighbor in dependencies.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, component)

        # Find connected services using DFS
        for node in dependencies:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)

        return components

    def _create_component_spec(self, formal_spec: FormalSpecification,
                               component_vars: Set[str]) -> Optional[FormalSpecification]:
        """Create a component specification for the given variables."""
        # Create a subset of types for this component
        component_types = {}
        for var_name in component_vars:
            if var_name in formal_spec.types:
                component_types[var_name] = formal_spec.types[var_name]

        # Add result type if it's in the original specification
        if "result" in formal_spec.types:
            component_types["result"] = formal_spec.types["result"]

        # Filter constraints that only use variables in this component
        component_constraints = []
        for constraint in formal_spec.constraints:
            vars_in_constraint = self._extract_variables_from_constraint(constraint)
            if vars_in_constraint.issubset(component_vars):
                component_constraints.append(constraint)

        # Only create the component if it has at least one constraint
        if not component_constraints:
            return None

        # Filter examples that only use variables in this component
        component_examples = []
        for example in formal_spec.examples:
            example_inputs = set(example.get("input", {}).keys())
            if example_inputs.issubset(component_vars):
                component_examples.append(example)

        # Create a new AST for the component
        component_ast = copy.deepcopy(formal_spec.ast)

        # Filter parameters to only include those in this component
        if "parameters" in component_ast:
            component_ast["parameters"] = [
                param for param in component_ast["parameters"]
                if isinstance(param, dict) and param.get("name") in component_vars
            ]

        # Create the component specification
        component_spec = FormalSpecification(
            ast=component_ast,
            constraints=component_constraints,
            types=component_types,
            examples=component_examples
        )

        return component_spec

    def _decompose_by_semantic_clustering(self, formal_spec: FormalSpecification) -> List[FormalSpecification]:
        """Decompose a specification by clustering semantically related constraints."""
        self.logger.info("Decomposing by semantic clustering")

        # In a real implementation, this would use semantic similarity metrics
        # to cluster related constraints

        # For demonstration, we'll use a simpler approach based on constraint types
        constraint_groups = self._group_constraints_by_type(formal_spec)

        # Create component specifications based on constraint groups
        components = []
        for constraint_group in constraint_groups:
            # Create a new specification with this constraint group
            component_spec = copy.deepcopy(formal_spec)
            component_spec.constraints = constraint_group

            # Extract variables used in this group
            component_vars = set()
            for constraint in constraint_group:
                component_vars.update(self._extract_variables_from_constraint(constraint))

            # Filter types to only include those used in this component
            component_spec.types = {var_name: var_type
                                    for var_name, var_type in formal_spec.types.items()
                                    if var_name in component_vars or var_name == "result"}

            # Only add the component if it has constraints
            if constraint_group:
                components.append(component_spec)

        # If no valid services were created, return the original specification
        if not components:
            self.logger.warning("Failed to create valid services, returning original specification")
            return [formal_spec]

        return components

    def _group_constraints_by_type(self, formal_spec: FormalSpecification) -> List[List[Any]]:
        """Group constraints by their semantic type."""
        # Categorize constraints into types
        range_constraints = []
        equality_constraints = []
        inequality_constraints = []
        logical_constraints = []
        other_constraints = []

        for constraint in formal_spec.constraints:
            if hasattr(constraint, "decl"):
                if constraint.decl().name() in ('>=', '<=', '>', '<'):
                    range_constraints.append(constraint)
                elif constraint.decl().name() in ('==', '='):
                    equality_constraints.append(constraint)
                elif constraint.decl().name() in ('!=', '<>'):
                    inequality_constraints.append(constraint)
                elif constraint.decl().name() in ('and', 'or', 'not', 'implies', '=>'):
                    logical_constraints.append(constraint)
                else:
                    other_constraints.append(constraint)
            else:
                other_constraints.append(constraint)

        # Combine similar types or create balanced groups
        groups = []

        # Add non-empty groups
        if range_constraints:
            groups.append(range_constraints)
        if equality_constraints:
            groups.append(equality_constraints)
        if inequality_constraints:
            groups.append(inequality_constraints)
        if logical_constraints:
            groups.append(logical_constraints)
        if other_constraints:
            groups.append(other_constraints)

        return groups

    def _decompose_by_partitioning(self, formal_spec: FormalSpecification) -> List[FormalSpecification]:
        """Decompose a specification by simple partitioning."""
        self.logger.info("Decomposing by simple partitioning")

        # Simple strategy: create roughly equal-sized partitions
        constraints = formal_spec.constraints
        num_components = min(self.max_components, max(2, len(constraints) // self.min_component_size))

        # Distribute constraints evenly
        constraints_per_component = len(constraints) // num_components
        remainder = len(constraints) % num_components

        # Create services
        components = []
        start_idx = 0

        for i in range(num_components):
            # Determine number of constraints for this component
            num_constraints = constraints_per_component
            if i < remainder:
                num_constraints += 1

            # Extract constraints for this component
            end_idx = start_idx + num_constraints
            component_constraints = constraints[start_idx:end_idx]
            start_idx = end_idx

            # Create a new specification with these constraints
            component_spec = copy.deepcopy(formal_spec)
            component_spec.constraints = component_constraints

            # Only add the component if it has constraints
            if component_constraints:
                components.append(component_spec)

        return components

    def _merge_components(self, components: List[FormalSpecification],
                          target_count: int) -> List[FormalSpecification]:
        """Merge services to reduce their number to the target."""
        self.logger.info(f"Merging {len(components)} services to {target_count}")

        if len(components) <= target_count:
            return components

        # Find pairs of services to merge based on shared variables
        component_similarities = []

        for i in range(len(components)):
            for j in range(i+1, len(components)):
                # Calculate similarity between services
                similarity = self._calculate_component_similarity(components[i], components[j])
                component_similarities.append((i, j, similarity))

        # Sort by similarity (descending)
        component_similarities.sort(key=lambda x: x[2], reverse=True)

        # Merge services until we reach the target count
        merged_components = components.copy()
        merged_indices = set()

        for i, j, similarity in component_similarities:
            # Skip if either component has already been merged
            if i in merged_indices or j in merged_indices:
                continue

            # Merge the services
            merged_component = self._merge_two_components(merged_components[i], merged_components[j])

            # Replace component i with the merged component
            merged_components[i] = merged_component

            # Mark component j as merged
            merged_indices.add(j)

            # Check if we've reached the target count
            if len(merged_components) - len(merged_indices) <= target_count:
                break

        # Return the merged services (excluding merged indices)
        result = [merged_components[i] for i in range(len(merged_components)) if i not in merged_indices]

        return result

    def _calculate_component_similarity(self, comp1: FormalSpecification,
                                        comp2: FormalSpecification) -> float:
        """Calculate similarity between two services."""
        # Extract variables used in each component
        vars1 = set(comp1.types.keys())
        vars2 = set(comp2.types.keys())

        # Remove result type if present
        if "result" in vars1:
            vars1.remove("result")
        if "result" in vars2:
            vars2.remove("result")

        # Calculate Jaccard similarity between variable sets
        intersection = len(vars1.intersection(vars2))
        union = len(vars1.union(vars2))

        if union == 0:
            return 0.0

        return intersection / union

    def _merge_two_components(self, comp1: FormalSpecification,
                              comp2: FormalSpecification) -> FormalSpecification:
        """Merge two component specifications."""
        # Create a new specification with merged attributes
        merged_spec = copy.deepcopy(comp1)

        # Merge constraints
        merged_spec.constraints = comp1.constraints + comp2.constraints

        # Merge types (comp1 types take precedence)
        for var_name, var_type in comp2.types.items():
            if var_name not in merged_spec.types:
                merged_spec.types[var_name] = var_type

        # Merge examples
        merged_examples = {}

        # First add comp1 examples
        for example in comp1.examples:
            example_key = frozenset(example.get("input", {}).items())
            merged_examples[example_key] = example

        # Then add comp2 examples (if no conflict)
        for example in comp2.examples:
            example_key = frozenset(example.get("input", {}).items())
            if example_key not in merged_examples:
                merged_examples[example_key] = example

        merged_spec.examples = list(merged_examples.values())

        # Merge AST parameters
        if "parameters" in comp1.ast and "parameters" in comp2.ast:
            merged_params = {}

            # Add comp1 parameters
            for param in comp1.ast["parameters"]:
                if isinstance(param, dict) and "name" in param:
                    merged_params[param["name"]] = param

            # Add comp2 parameters if not already present
            for param in comp2.ast["parameters"]:
                if isinstance(param, dict) and "name" in param:
                    if param["name"] not in merged_params:
                        merged_params[param["name"]] = param

            merged_spec.ast["parameters"] = list(merged_params.values())

        return merged_spec

    def _combine_sequential(self, component_results: List[SynthesisResult]) -> SynthesisResult:
        """Combine services sequentially, with outputs flowing into inputs."""
        self.logger.info("Combining services sequentially")

        if not component_results:
            raise ValueError("No component results to combine")

        if len(component_results) == 1:
            return component_results[0]

        # Start with the AST from the first component
        combined_ast = copy.deepcopy(component_results[0].program_ast)

        # For sequential combination, we'll create a sequence of function calls
        combined_body = []

        # Build sequential calls to each component function
        for i, result in enumerate(component_results):
            # Add this component's body to the combined body
            combined_body.extend(self._get_component_body(result.program_ast))

            # Add function call if not the last component
            if i < len(component_results) - 1:
                function_name = result.program_ast.get("name", f"component_{i}")
                combined_body.append({
                    "type": "variable_declaration",
                    "name": f"result_{i}",
                    "value": {
                        "type": "function_call",
                        "function": function_name,
                        "arguments": self._get_function_arguments(result.program_ast)
                    }
                })

        # Add a return statement for the final result
        combined_body.append({
            "type": "return",
            "value": {
                "type": "variable",
                "name": f"result_{len(component_results) - 2}" if len(component_results) > 1 else "result"
            }
        })

        # Update the combined AST
        combined_ast["body"] = combined_body

        # Calculate the combined confidence score (weighted average)
        total_weight = sum(result.time_taken for result in component_results)
        if total_weight == 0:
            combined_confidence = sum(result.confidence_score for result in component_results) / len(component_results)
        else:
            combined_confidence = sum(result.confidence_score * (result.time_taken / total_weight)
                                      for result in component_results)

        # Calculate the total time taken
        total_time = sum(result.time_taken for result in component_results)

        # Create the combined result
        combined_result = SynthesisResult(
            program_ast=combined_ast,
            confidence_score=combined_confidence,
            time_taken=total_time,
            strategy="incremental_sequential"
        )

        return combined_result

    def _combine_parallel(self, component_results: List[SynthesisResult]) -> SynthesisResult:
        """Combine services to execute in parallel and merge results."""
        self.logger.info("Combining services in parallel")

        if not component_results:
            raise ValueError("No component results to combine")

        if len(component_results) == 1:
            return component_results[0]

        # Start with the AST from the first component
        combined_ast = copy.deepcopy(component_results[0].program_ast)

        # For parallel combination, we'll declare all functions and then combine their results
        combined_body = []

        # First add all component functions
        for i, result in enumerate(component_results):
            component_body = self._get_component_body(result.program_ast)
            if component_body:
                # Rename the function to avoid conflicts
                result.program_ast["name"] = f"component_{i}"

                # Add function definition
                combined_body.append({
                    "type": "function_definition",
                    "name": result.program_ast["name"],
                    "parameters": result.program_ast.get("parameters", []),
                    "body": component_body
                })

        # Now add the main function body
        main_body = []

        # Call each component function in parallel
        for i, result in enumerate(component_results):
            main_body.append({
                "type": "variable_declaration",
                "name": f"result_{i}",
                "value": {
                    "type": "function_call",
                    "function": f"component_{i}",
                    "arguments": self._get_function_arguments(result.program_ast)
                }
            })

        # Combine the results
        main_body.append({
            "type": "variable_declaration",
            "name": "combined_result",
            "value": {
                "type": "function_call",
                "function": "combine_results",
                "arguments": [{"type": "variable", "name": f"result_{i}"}
                              for i in range(len(component_results))]
            }
        })

        # Add a return statement for the combined result
        main_body.append({
            "type": "return",
            "value": {
                "type": "variable",
                "name": "combined_result"
            }
        })

        # Add combination function and main function to the combined body
        combined_body.append({
            "type": "function_definition",
            "name": "combine_results",
            "parameters": [{"name": f"result_{i}"} for i in range(len(component_results))],
            "body": [{
                "type": "return",
                "value": {
                    "type": "function_call",
                    "function": "merge_results",
                    "arguments": [{"type": "variable", "name": f"result_{i}"}
                                  for i in range(len(component_results))]
                }
            }]
        })

        combined_body.extend(main_body)

        # Update the combined AST
        combined_ast["body"] = combined_body

        # Calculate the combined confidence score (weighted average)
        total_weight = sum(result.time_taken for result in component_results)
        if total_weight == 0:
            combined_confidence = sum(result.confidence_score for result in component_results) / len(component_results)
        else:
            combined_confidence = sum(result.confidence_score * (result.time_taken / total_weight)
                                      for result in component_results)

        # Calculate the total time taken
        total_time = sum(result.time_taken for result in component_results)

        # Create the combined result
        combined_result = SynthesisResult(
            program_ast=combined_ast,
            confidence_score=combined_confidence,
            time_taken=total_time,
            strategy="incremental_parallel"
        )

        return combined_result

    def _combine_conditional(self, component_results: List[SynthesisResult]) -> SynthesisResult:
        """Combine services with conditional branching based on input conditions."""
        self.logger.info("Combining services with conditional branching")

        if not component_results:
            raise ValueError("No component results to combine")

        if len(component_results) == 1:
            return component_results[0]

        # Start with the AST from the first component
        combined_ast = copy.deepcopy(component_results[0].program_ast)

        # For conditional combination, we'll add conditions to select the right component
        combined_body = []

        # First add all component functions
        for i, result in enumerate(component_results):
            component_body = self._get_component_body(result.program_ast)
            if component_body:
                # Rename the function to avoid conflicts
                result.program_ast["name"] = f"component_{i}"

                # Add function definition
                combined_body.append({
                    "type": "function_definition",
                    "name": result.program_ast["name"],
                    "parameters": result.program_ast.get("parameters", []),
                    "body": component_body
                })

        # Add conditionals to select the right component
        main_body = []

        # Add a simple decision function based on input ranges
        # In a real implementation, this would be more sophisticated
        main_body.append({
            "type": "variable_declaration",
            "name": "selected_component",
            "value": {
                "type": "function_call",
                "function": "select_component",
                "arguments": self._get_function_arguments(component_results[0].program_ast)
            }
        })

        # Add conditional calls to services
        main_body.append({
            "type": "if_statement",
            "conditions": [
                {"condition": {"type": "binary_operation", "operator": "==",
                               "left": {"type": "variable", "name": "selected_component"},
                               "right": {"type": "literal", "value": i}},
                 "body": [{
                     "type": "variable_declaration",
                     "name": "result",
                     "value": {
                         "type": "function_call",
                         "function": f"component_{i}",
                         "arguments": self._get_function_arguments(result.program_ast)
                     }
                 }]}
                for i, result in enumerate(component_results)
            ],
            "else_body": [{
                "type": "variable_declaration",
                "name": "result",
                "value": {
                    "type": "function_call",
                    "function": "component_0",  # Default to first component
                    "arguments": self._get_function_arguments(component_results[0].program_ast)
                }
            }]
        })

        # Add a return statement for the result
        main_body.append({
            "type": "return",
            "value": {
                "type": "variable",
                "name": "result"
            }
        })

        # Add selection function and main function to the combined body
        combined_body.append({
            "type": "function_definition",
            "name": "select_component",
            "parameters": combined_ast.get("parameters", []),
            "body": [{
                "type": "return",
                "value": {
                    "type": "literal",
                    "value": 0  # Default to first component
                }
            }]
        })

        combined_body.extend(main_body)

        # Update the combined AST
        combined_ast["body"] = combined_body

        # Calculate the combined confidence score (weighted average)
        total_weight = sum(result.time_taken for result in component_results)
        if total_weight == 0:
            combined_confidence = sum(result.confidence_score for result in component_results) / len(component_results)
        else:
            combined_confidence = sum(result.confidence_score * (result.time_taken / total_weight)
                                      for result in component_results)

        # Calculate the total time taken
        total_time = sum(result.time_taken for result in component_results)

        # Create the combined result
        combined_result = SynthesisResult(
            program_ast=combined_ast,
            confidence_score=combined_confidence,
            time_taken=total_time,
            strategy="incremental_conditional"
        )

        return combined_result

    def _get_component_body(self, ast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract the body from a component AST."""
        if "body" in ast and isinstance(ast["body"], list):
            return ast["body"]
        return []

    def _get_function_arguments(self, ast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create argument expressions for a function call."""
        args = []

        if "parameters" in ast and isinstance(ast["parameters"], list):
            for param in ast["parameters"]:
                if isinstance(param, dict) and "name" in param:
                    args.append({
                        "type": "variable",
                        "name": param["name"]
                    })
                elif isinstance(param, str):
                    args.append({
                        "type": "variable",
                        "name": param
                    })

        return args

    def _compute_cache_key(self, formal_spec: FormalSpecification) -> str:
        """Compute a cache key for the specification."""
        # Simple hash of the constraint count and variable count
        constraints_hash = len(formal_spec.constraints)
        vars_hash = len(formal_spec.types)

        return f"{constraints_hash}_{vars_hash}"