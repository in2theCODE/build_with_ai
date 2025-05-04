#!/usr/bin/env python3
"""
Advanced Model-Based Constraint Relaxation Engine.

This component uses model-guided techniques and optimizations to intelligently
relax constraints when synthesis or verification fails.
"""

import logging
import time
from typing import Dict, Any, List, Optional

import z3

from src.services.shared.models.base import BaseMessage
from src.services.shared.models.messages import VerificationResult
from src.services.shared.models.types import FormalSpecification

logger = logging.getLogger(__name__)


class ModelBasedConstraintRelaxer:
    """
    Advanced constraint relaxer that uses model-guided techniques to intelligently
    relax constraints when synthesis or verification fails.
    """

    def __init__(self, **params):
        """Initialize the model-based constraint relaxer with advanced parameters."""
        self.logger = logger

        # Configuration parameters with sensible defaults
        self.max_relaxation_iterations = params.get("max_relaxation_iterations", 5)
        self.timeout_seconds = params.get("timeout_seconds", 30)
        self.use_optimization = params.get("use_optimization", True)
        self.use_unsat_core = params.get("use_unsat_core", True)
        self.use_maxsat = params.get("use_maxsat", True)
        self.min_constraints_to_keep = params.get("min_constraints_to_keep", 1)

        # Strategy weights for different relaxation techniques
        self.strategy_weights = params.get("strategy_weights", {
            "unsat_core": 0.5,
            "model_guided": 0.3,
            "maxsat": 0.2
        })

        # Normalize weights
        total = sum(self.strategy_weights.values())
        if total > 0:
            for key in self.strategy_weights:
                self.strategy_weights[key] /= total

        self.logger.info(
            f"Model-based constraint relaxer initialized with {self.max_relaxation_iterations} max iterations")

    verification_result = VerificationResult

    async def relax_constraints(self, formal_spec: FormalSpecification,
                                verification_result: Optional[Dict[str, Any]] = None) -> Optional[FormalSpecification]:
        """
        Relax constraints in the specification using advanced model-guided techniques.

        Args:
            formal_spec: The original formal specification
            verification_result: Optional verification result with counterexamples

        Returns:
            A new specification with relaxed constraints, or None if relaxation failed
        """
        self.logger.info("Starting model-based constraint relaxation process")
        start_time = time.time()

        # Record metrics for the original specification
        constraint_count = len(formal_spec.constraints)

        # Create a working copy of the specification
        relaxed_spec = formal_spec.model_copy()

        # Select the best relaxation strategy based on the specification
        strategy = self._select_relaxation_strategy(formal_spec, verification_result)
        self.logger.info(f"Selected relaxation strategy: {strategy}")

        # Apply the selected strategy
        if strategy == "unsat_core" and self.use_unsat_core:
            result = await self._relax_using_unsat_core(relaxed_spec, verification_result)
        elif strategy == "maxsat" and self.use_maxsat:
            result = await self._relax_using_maxsat(relaxed_spec, verification_result)
        else:  # Default to model-guided relaxation
            result = await self._relax_using_model_guided(relaxed_spec, verification_result)

        if not result:
            self.logger.warning("Primary relaxation strategy failed, trying fallback approach")
            # Try a different strategy as fallback
            if strategy != "model_guided":
                result = await self._relax_using_model_guided(relaxed_spec, verification_result)
            else:
                result = await self._relax_using_unsat_core(relaxed_spec, verification_result)

        end_time = time.time()
        time_taken = end_time - start_time

        if result:
            final_constraint_count = len(result.constraints)

            self.logger.info(f"Constraint relaxation completed in {time_taken:.2f} seconds")
            self.logger.info(f"Original constraint count: {constraint_count}, "
                             f"Relaxed constraint count: {final_constraint_count}")
            return result
        else:
            self.logger.warning(f"Constraint relaxation failed after {time_taken:.2f} seconds")
            return None

    def _select_relaxation_strategy(self, formal_spec: FormalSpecification,
                                    verification_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Select the most appropriate relaxation strategy based on the specification.

        Args:
            formal_spec: The formal specification
            verification_result: Optional verification result

        Returns:
            Name of the selected strategy
        """
        # If we have verification results with counterexamples, prefer model-guided
        if verification_result and verification_result.get('counterexamples'):
            return "model_guided"

        # If we have many constraints, UNSAT core approach is often more efficient
        if len(formal_spec.constraints) > 10:
            return "unsat_core"

        # For complex constraints with many variables, MaxSAT often works better
        variable_count = self._count_variables(formal_spec)
        if variable_count > 15:
            return "maxsat"

        # Default to model-guided approach
        return "model_guided"

    def _count_variables(self, formal_spec: FormalSpecification) -> int:
        """
        Count the number of unique variables in a specification.

        Args:
            formal_spec: The formal specification

        Returns:
            Count of unique variables
        """
        # Use the types dictionary to count variables
        return len(formal_spec.types)

    async def _relax_using_unsat_core(self, formal_spec: FormalSpecification,
                                      verification_result: Optional[Dict[str, Any]] = None) -> Optional[
        FormalSpecification]:
        """
        Relax constraints using the UNSAT core technique.

        Args:
            formal_spec: The formal specification
            verification_result: Optional verification result

        Returns:
            A relaxed specification or None if relaxation failed
        """
        self.logger.info("Applying UNSAT core relaxation strategy")

        # Check if we already have UNSAT core information from verification
        unsat_core_ids = []
        if verification_result and 'unsat_core' in verification_result:
            unsat_core_ids = verification_result['unsat_core']

        if not unsat_core_ids:
            # We need to compute the UNSAT core using Z3
            unsat_core_ids = await self._compute_unsat_core(formal_spec)

        if not unsat_core_ids:
            self.logger.warning("Failed to identify UNSAT core constraints")
            return None

        # Create a new specification with relaxed constraints
        relaxed_spec = formal_spec.model_copy()

        # Keep track of which constraints to keep
        constraints_to_keep = []

        # Process constraints based on unsat core
        for i, constraint in enumerate(relaxed_spec.constraints):
            # Create constraint identifier if not available
            constraint_id = getattr(constraint, 'id', f"constraint_{i}")

            # Check if this constraint is in the unsat core
            if constraint_id in unsat_core_ids:
                # This constraint is part of the problem
                # Check if it's a hard constraint that must be kept
                is_hard = getattr(constraint, 'is_hard', False)

                if is_hard:
                    constraints_to_keep.append(constraint)
            else:
                # This constraint is not causing the UNSAT, keep it
                constraints_to_keep.append(constraint)

        # Check if we've kept enough constraints
        if len(constraints_to_keep) < self.min_constraints_to_keep:
            self.logger.warning(f"Too few constraints remaining ({len(constraints_to_keep)}), relaxation failed")
            return None

        # Update the specification with the kept constraints
        # Since FormalSpecification is immutable, create a new one
        relaxed_spec = FormalSpecification(
            ast=relaxed_spec.ast,
            constraints=constraints_to_keep,
            types=relaxed_spec.types,
            examples=relaxed_spec.examples
        )

        # Verify that the relaxed specification is satisfiable
        if await self._is_satisfiable(relaxed_spec):
            self.logger.info(
                f"Successfully relaxed specification, removed {len(formal_spec.constraints) - len(constraints_to_keep)} constraints")
            return relaxed_spec
        else:
            self.logger.warning("Relaxed specification is still unsatisfiable")
            return None

    async def _compute_unsat_core(self, formal_spec: FormalSpecification) -> List[str]:
        """
        Compute the UNSAT core of a specification using Z3.

        Args:
            formal_spec: The formal specification

        Returns:
            List of constraint IDs in the UNSAT core
        """
        self.logger.info("Computing UNSAT core using Z3")

        try:
            # Create Z3 solver with timeout
            solver = z3()
            solver.set("timeout", self.timeout_seconds * 1000)  # Timeout in milliseconds

            # Convert constraints to Z3 format and add them as tracked assertions
            for i, constraint in enumerate(formal_spec.constraints):
                # Get or generate constraint ID
                constraint_id = getattr(constraint, 'id', f"constraint_{i}")

                # Convert to Z3 formula
                z3_constraint = self._constraint_to_z3(constraint, formal_spec.types)

                if z3_constraint is not None:
                    # Add as tracked assertion
                    solver.assert_and_track(z3_constraint, constraint_id)

            # Check satisfiability
            result = solver.check()

            if result == z3.unsat:
                # Extract UNSAT core
                unsat_core = solver.unsat_core()
                return [str(c) for c in unsat_core]  # Convert Z3 symbols to strings
            else:
                self.logger.info(f"Specification is {result}, no UNSAT core to extract")
                return []

        except Exception as e:
            self.logger.error(f"Error computing UNSAT core: {str(e)}")
            return []

    def _constraint_to_z3(self, constraint: Any, type_info: Dict[str, str]) -> Optional[z3.BoolRef]:
        """
        Convert a constraint to Z3 format.

        Args:
            constraint: The constraint object
            type_info: Type information for variables

        Returns:
            Z3 formula representing the constraint or None if conversion fails
        """
        try:
            # Handle different constraint representations
            if hasattr(constraint, 'to_z3'):
                # If the constraint has a to_z3 method, use it
                return constraint.to_z3()

            # Handle constraints represented as strings
            if isinstance(constraint, str):
                return self._parse_constraint_string(constraint, type_info)

            # Handle constraints represented as dictionaries
            if isinstance(constraint, dict) and 'expression' in constraint:
                expr = constraint['expression']
                if isinstance(expr, str):
                    return self._parse_constraint_string(expr, type_info)
                return expr  # Assume it's already a Z3 expression

            # If the constraint is already a Z3 expression, return it directly
            if isinstance(constraint, z3.BoolRef):
                return constraint

            # Use __str__ method if available and nothing else works
            if hasattr(constraint, '__str__'):
                return self._parse_constraint_string(str(constraint), type_info)

            self.logger.warning(f"Unsupported constraint type: {type(constraint)}")
            return None

        except Exception as e:
            self.logger.error(f"Error converting constraint to Z3: {str(e)}")
            return None

    def _parse_constraint_string(self, constraint_str: str, type_info: Dict[str, str]) -> Optional[z3.BoolRef]:
        """
        Parse a constraint string into a Z3 formula.

        Args:
            constraint_str: String representation of the constraint
            type_info: Type information for variables

        Returns:
            Z3 formula or None if parsing fails
        """
        try:
            # Create Z3 variables based on type information
            z3_vars = {}
            for var_name, var_type in type_info.items():
                if var_type.lower() == 'int':
                    z3_vars[var_name] = z3.Int(var_name)
                elif var_type.lower() in ('real', 'float'):
                    z3_vars[var_name] = z3.Real(var_name)
                elif var_type.lower() == 'bool':
                    z3_vars[var_name] = z3.Bool(var_name)
                else:
                    # Default to Int for unknown types
                    z3_vars[var_name] = z3.Int(var_name)

            # Create context for evaluating the constraint
            context = dict(z3_vars)

            # Add Z3 functions to context
            context.update({
                'And': z3.And,
                'Or': z3.Or,
                'Not': z3.Not,
                'Implies': z3.Implies,
                'If': z3.If,
                '==': lambda x, y: x == y,
                '!=': lambda x, y: x != y,
                '<': lambda x, y: x < y,
                '<=': lambda x, y: x <= y,
                '>': lambda x, y: x > y,
                '>=': lambda x, y: x >= y,
            })

            # WARNING: This uses eval and is not safe for untrusted input
            # In a production system, use a proper parser
            z3_expr = eval(constraint_str, {"__builtins__": {}}, context)

            # Ensure the result is a boolean expression
            if not isinstance(z3_expr, z3.BoolRef):
                z3_expr = z3_expr != 0  # Convert to boolean

            return z3_expr

        except Exception as e:
            self.logger.error(f"Error parsing constraint string: {str(e)}")
            return None

    async def _is_satisfiable(self, formal_spec: FormalSpecification) -> bool:
        """
        Check if a formal specification is satisfiable.

        Args:
            formal_spec: The formal specification

        Returns:
            True if satisfiable, False otherwise
        """
        try:
            # Create Z3 solver with timeout
            solver = z3.Solver()
            solver.set("timeout", self.timeout_seconds * 1000)  # Timeout in milliseconds

            # Convert constraints to Z3 format and add them
            for constraint in formal_spec.constraints:
                z3_constraint = self._constraint_to_z3(constraint, formal_spec.types)
                if z3_constraint is not None:
                    solver.add(z3_constraint)

            # Check satisfiability
            result = solver.check()
            return result == z3.sat

        except Exception as e:
            self.logger.error(f"Error checking satisfiability: {str(e)}")
            return False

    async def _relax_using_maxsat(self, formal_spec: FormalSpecification,
                                  verification_result: Optional[Dict[str, Any]] = None) -> Optional[
        FormalSpecification]:
        """
        Relax constraints using MaxSAT optimization.

        Args:
            formal_spec: The formal specification
            verification_result: Optional verification result

        Returns:
            A relaxed specification or None if relaxation failed
        """
        self.logger.info("Applying MaxSAT relaxation strategy")

        try:
            # Create Z3 optimizer
            optimizer = z3.Optimize()
            optimizer.set("timeout", self.timeout_seconds * 1000)  # Timeout in milliseconds

            # Track constraints and their weights
            soft_constraints = []
            hard_constraints = []

            # Process each constraint
            for i, constraint in enumerate(formal_spec.constraints):
                # Get constraint ID
                constraint_id = getattr(constraint, 'id', f"constraint_{i}")

                # Convert to Z3 formula
                z3_constraint = self._constraint_to_z3(constraint, formal_spec.types)
                if z3_constraint is None:
                    continue

                # Check if this is a hard constraint
                is_hard = getattr(constraint, 'is_hard', False)

                if is_hard:
                    hard_constraints.append((z3_constraint, constraint))
                else:
                    # Get weight (importance) of this constraint
                    weight = getattr(constraint, 'weight', 1.0)
                    soft_constraints.append((z3_constraint, weight, constraint))

            # Add hard constraints directly (must be satisfied)
            for z3_constraint, _ in hard_constraints:
                optimizer.add(z3_constraint)

            # Add soft constraints with weights
            weight_vars = []
            for z3_constraint, weight, _ in soft_constraints:
                # Create a boolean variable for this constraint
                b = z3.Bool(f"weight_var_{len(weight_vars)}")
                weight_vars.append(b)

                # Add implication: if b is true, the constraint must be satisfied
                optimizer.add(z3.Implies(b, z3_constraint))

                # Add to the objective function with weight
                optimizer.add_soft(b, weight)

            # Solve
            result = optimizer.check()

            if result == z3.sat:
                # Get model
                model = optimizer.model()

                # Determine which constraints to keep
                constraints_to_keep = []

                # Always keep hard constraints
                for _, constraint in hard_constraints:
                    constraints_to_keep.append(constraint)

                # Keep satisfied soft constraints
                for i, (_, _, constraint) in enumerate(soft_constraints):
                    if i < len(weight_vars):
                        if model.evaluate(weight_vars[i]):
                            constraints_to_keep.append(constraint)

                # Check if we've kept enough constraints
                if len(constraints_to_keep) < self.min_constraints_to_keep:
                    self.logger.warning(
                        f"Too few constraints remaining ({len(constraints_to_keep)}), relaxation failed")
                    return None

                # Create new specification with kept constraints
                relaxed_spec = FormalSpecification(
                    ast=formal_spec.ast,
                    constraints=constraints_to_keep,
                    types=formal_spec.types,
                    examples=formal_spec.examples
                )

                self.logger.info(
                    f"Successfully relaxed specification using MaxSAT, kept {len(constraints_to_keep)} constraints")
                return relaxed_spec
            else:
                self.logger.warning(f"MaxSAT solver returned {result}, relaxation failed")
                return None

        except Exception as e:
            self.logger.error(f"Error in MaxSAT relaxation: {str(e)}")
            return None

    async def _relax_using_model_guided(self, formal_spec: FormalSpecification,
                                        verification_result: Optional[Dict[str, Any]] = None) -> Optional[
        FormalSpecification]:
        """
        Relax constraints using model-guided approach with counterexamples.

        Args:
            formal_spec: The formal specification
            verification_result: Optional verification result with counterexamples

        Returns:
            A relaxed specification or None if relaxation failed
        """
        self.logger.info("Applying model-guided relaxation strategy")

        # Extract counterexamples from verification result
        counterexamples = []
        if verification_result and 'counterexamples' in verification_result:
            counterexamples = verification_result['counterexamples']

        # If no counterexamples available, try to generate some
        if not counterexamples:
            self.logger.info("No counterexamples available, generating samples")
            counterexamples = await self._generate_samples(formal_spec)

        if not counterexamples:
            self.logger.warning("Failed to generate counterexamples for model-guided relaxation")
            return None

        # Track constraint violations
        violation_counts = {}
        for i, constraint in enumerate(formal_spec.constraints):
            constraint_id = getattr(constraint, 'id', f"constraint_{i}")
            violation_counts[constraint_id] = {
                'count': 0,
                'constraint': constraint,
                'is_hard': getattr(constraint, 'is_hard', False)
            }

        # Evaluate each constraint against each counterexample
        for example in counterexamples:
            for i, constraint in enumerate(formal_spec.constraints):
                constraint_id = getattr(constraint, 'id', f"constraint_{i}")

                # Skip if this is a hard constraint
                if violation_counts[constraint_id]['is_hard']:
                    continue

                # Check if this constraint is satisfied by the example
                if not await self._evaluate_constraint(constraint, example, formal_spec.types):
                    violation_counts[constraint_id]['count'] += 1

        # Sort constraints by violation count (most violated first)
        sorted_violations = sorted(
            violation_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        # Start with all constraints
        constraints_to_keep = [c for c in formal_spec.constraints]

        # Try removing constraints one by one, starting with most violated
        for constraint_id, info in sorted_violations:
            # Skip hard constraints
            if info['is_hard']:
                continue

            # Try removing this constraint
            candidate_constraints = [c for c in constraints_to_keep if
                                     getattr(c, 'id',
                                             f"constraint_{formal_spec.constraints.index(c)}") != constraint_id]

            # Check if we would have enough constraints left
            if len(candidate_constraints) < self.min_constraints_to_keep:
                continue

            # Create a test specification
            test_spec = FormalSpecification(
                ast=formal_spec.ast,
                constraints=candidate_constraints,
                types=formal_spec.types,
                examples=formal_spec.examples
            )

            # Check if this specification is satisfiable
            if await self._is_satisfiable(test_spec):
                # Update the constraints to keep
                constraints_to_keep = candidate_constraints
                self.logger.info(f"Removed constraint {constraint_id} (violated {info['count']} times)")

                # If we've fixed the unsatisfiability, we can stop
                # Otherwise, continue removing constraints
                if len(counterexamples) > 0 and await self._check_examples(test_spec, counterexamples):
                    self.logger.info("Specification now satisfies all counterexamples")
                    break

        # Create final relaxed specification
        if len(constraints_to_keep) < len(formal_spec.constraints):
            relaxed_spec = FormalSpecification(
                ast=formal_spec.ast,
                constraints=constraints_to_keep,
                types=formal_spec.types,
                examples=formal_spec.examples
            )

            self.logger.info(
                f"Successfully relaxed specification using model-guided approach, removed {len(formal_spec.constraints) - len(constraints_to_keep)} constraints")
            return relaxed_spec
        else:
            self.logger.warning("Could not relax specification using model-guided approach")
            return None

    async def _evaluate_constraint(self, constraint: Any, example: Dict[str, Any],
                                   type_info: Dict[str, str]) -> bool:
        """
        Evaluate a constraint against a specific example.

        Args:
            constraint: The constraint to evaluate
            example: The example data
            type_info: Type information for variables

        Returns:
            True if constraint is satisfied, False otherwise
        """
        try:
            # If constraint has an evaluate method, use it
            if hasattr(constraint, 'evaluate'):
                return constraint.evaluate(example)

            # Convert to Z3 formula
            z3_constraint = self._constraint_to_z3(constraint, type_info)
            if z3_constraint is None:
                return False

            # Create solver
            solver = z3.Solver()

            # Add the constraint
            solver.add(z3_constraint)

            # Add the example values as constraints
            for var_name, value in example.items():
                if var_name in type_info:
                    var_type = type_info[var_name].lower()

                    if var_type == 'int':
                        z3_var = z3.Int(var_name)
                        solver.add(z3_var == value)
                    elif var_type in ('real', 'float'):
                        z3_var = z3.Real(var_name)
                        solver.add(z3_var == value)
                    elif var_type == 'bool':
                        z3_var = z3.Bool(var_name)
                        solver.add(z3_var == value)

            # Check satisfiability
            result = solver.check()
            return result == z3.sat

        except Exception as e:
            self.logger.error(f"Error evaluating constraint: {str(e)}")
            return False

    async def _check_examples(self, spec: FormalSpecification, examples: List[Dict[str, Any]]) -> bool:
        """
        Check if a specification satisfies all examples.

        Args:
            spec: The specification to check
            examples: List of examples to check against

        Returns:
            True if all examples are satisfied, False otherwise
        """
        for example in examples:
            # Convert to Z3 formulas
            solver = z3.Solver()

            # Add all constraints
            for constraint in spec.constraints:
                z3_constraint = self._constraint_to_z3(constraint, spec.types)
                if z3_constraint is not None:
                    solver.add(z3_constraint)


            # Check satisfiability
            result = solver.check()
            if result != z3.sat:
                return False

        return True

    async def _generate_samples(self, formal_spec: FormalSpecification) -> List[Dict[str, Any]]:
        """
        Generate sample points/counterexamples for model-guided relaxation.

        Args:
            formal_spec: The formal specification

        Returns:
            List of sample points as dictionaries
        """
        samples = []

        try:
            # Create Z3 solver
            solver = z3.Solver()

            # Create Z3 variables for all variables in the specification
            z3_vars = {}
            for var_name, var_type in formal_spec.types.items():
                var_type = var_type.lower()

                if var_type == 'int':
                    z3_vars[var_name] = z3.Int(var_name)
                elif var_type in ('real', 'float'):
                    z3_vars[var_name] = z3.Real(var_name)
                elif var_type == 'bool':
                    z3_vars[var_name] = z3.Bool(var_name)
                else:
                    # Default to Int for unknown types
                    z3_vars[var_name] = z3.Int(var_name)

            # Try to generate diverse samples
            for _ in range(5):  # Generate up to 5 samples
                solver.push()

                # Add constraints to ensure diversity from previous samples
                for sample in samples:
                    # Create a constraint that at least one variable must be different
                    different_conditions = []
                    for var_name, value in sample.items():
                        if var_name in z3_vars:
                            z3_var = z3_vars[var_name]
                            if isinstance(value, bool):
                                different_conditions.append(z3_var != value)
                            elif isinstance(value, int):
                                different_conditions.append(z3_var != value)
                            elif isinstance(value, float):
                                different_conditions.append(z3_var != value)

                    if different_conditions:
                        solver.add(z3.Or(*different_conditions))

                # Check satisfiability
                if solver.check() == z3.sat:
                    # Get model
                    model = solver.model()

                    # Extract values
                    sample = {}
                    for var_name, z3_var in z3_vars.items():
                        value = model.evaluate(z3_var)

                        # Convert Z3 values to Python types
                        if z3.is_int_value(value):
                            sample[var_name] = value.as_long()
                        elif z3.is_true(value):
                            sample[var_name] = True
                        elif z3.is_false(value):
                            sample[var_name] = False
                        elif z3.is_real_value(value):
                            # Handle rational numbers
                            if value.denominator_as_long() == 1:
                                sample[var_name] = value.numerator_as_long()
                            else:
                                sample[var_name] = float(value.numerator_as_long()) / float(value.denominator_as_long())
                        else:
                            sample[var_name] = str(value)

                    samples.append(sample)

                solver.pop()

            return samples

        except Exception as e:
            self.logger.error(f"Error generating samples: {str(e)}")
            return []