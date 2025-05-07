#!/usr/bin/env python3
"""
Utilities for working with Z3 solver.
"""

import logging
from typing import Any, List, Optional, Set, Tuple

import z3  # type: ignore


# Import all Z3 functions with type ignore comments
try:
    # First batch of imports
    # Second batch of imports
    from z3 import And  # type: ignore
    from z3 import Bool  # type: ignore
    from z3 import ExprRef  # type: ignore
    from z3 import is_and  # type: ignore
    from z3 import is_app  # type: ignore
    from z3 import is_arith  # type: ignore
    from z3 import is_bool  # type: ignore
    from z3 import is_const  # type: ignore
    from z3 import is_eq  # type: ignore
    from z3 import is_int_value  # type: ignore
    from z3 import is_real  # type: ignore
    from z3 import is_true  # type: ignore
    from z3 import Optimize  # type: ignore
    from z3 import sat  # type: ignore
    from z3 import Solver  # type: ignore
    from z3 import unsat  # type: ignore
except ImportError:
    logging.error("Z3 library not properly installed. Please install with 'pip install z3-solver'")
    raise

logger = logging.getLogger(__name__)


# Wrapper functions to avoid direct references to z3 module functions
def _solver():
    return z3.Solver()


def _check_sat(solver):
    return solver.check() == z3.sat


def _check_unsat(solver):
    return solver.check() == z3.unsat


def _is_expr_ref(expr):
    return isinstance(expr, z3.ExprRef)


def _is_const(node):
    return z3.is_const(node)


def _is_bool(node):
    return z3.is_bool(node)


def _is_int_value(node):
    return z3.is_int_value(node)


def _is_real(node):
    return z3.is_real(node)


def _create_bool(name):
    return z3.Bool(name)


def _is_true(expr, model):
    return z3.is_true(model.eval(expr, model_completion=True))


def _optimize():
    return z3.Optimize()


def _is_eq(expr):
    return z3.is_eq(expr)


def _is_arith(expr):
    return z3.is_arith(expr)


def _is_and(expr):
    return z3.is_and(expr)


def _and(*args):
    return z3.And(*args)


def _is_app(expr):
    return z3.is_app(expr)


# Main functions using the wrappers
def get_model(constraints: List[Any]) -> Optional[Any]:  # type: ignore
    """
    Get a model for a set of constraints if satisfiable.

    Args:
        constraints: List of Z3 constraints

    Returns:
        Z3 model or None if unsatisfiable
    """
    solver = _solver()

    for constraint in constraints:
        solver.add(constraint)

    if _check_sat(solver):
        return solver.model()
    else:
        return None


def extract_variables(expr: Any) -> Set[str]:  # type: ignore
    """
    Extract all variables from a Z3 expression.

    Args:
        expr: Z3 expression

    Returns:
        Set of variable names
    """
    variables = set()

    if not _is_expr_ref(expr):
        return variables

    # Use BFS to traverse the expression
    queue = [expr]
    while queue:
        node = queue.pop(0)

        # Check if this is a variable
        if (
            _is_const(node)
            and not _is_bool(node)
            and not _is_int_value(node)
            and not _is_real(node)
        ):
            variables.add(str(node))

        # Add all children to the queue
        if hasattr(node, "children") and callable(node.children):
            queue.extend(node.children())

    return variables


def is_satisfiable(constraints: List[Any]) -> bool:  # type: ignore
    """
    Check if a set of constraints is satisfiable.

    Args:
        constraints: List of Z3 constraints

    Returns:
        True if satisfiable, False otherwise
    """
    solver = _solver()

    for constraint in constraints:
        solver.add(constraint)

    return _check_sat(solver)


def get_unsat_core(constraints: List[Any]) -> List[int]:  # type: ignore
    """
    Get the indices of constraints in the UNSAT core.

    Args:
        constraints: List of Z3 constraints

    Returns:
        List of indices of constraints in the UNSAT core
    """
    solver = _solver()

    # Create tracking variables
    tracking_vars = []
    for i, constraint in enumerate(constraints):
        track_var = _create_bool(f"track_{i}")
        tracking_vars.append(track_var)
        solver.assert_and_track(constraint, track_var)

    # Check satisfiability
    if _check_unsat(solver):
        unsat_core = solver.unsat_core()
        unsat_indices = []

        # Map the core back to constraint indices
        for i, track_var in enumerate(tracking_vars):
            if track_var in unsat_core:
                unsat_indices.append(i)

        return unsat_indices
    else:
        return []


def optimize_constraints(
    constraints: List[Any], weights: Optional[List[int]] = None
) -> Tuple[List[int], Optional[Any]]:  # type: ignore
    """
    Use Z3 optimizer to find the maximum number of satisfiable constraints.

    Args:
        constraints: List of Z3 constraints
        weights: Optional list of weights for each constraint

    Returns:
        Tuple of (satisfied constraint indices, model)
    """
    opt = _optimize()

    # Use default weights if none provided
    if weights is None:
        weights = [1] * len(constraints)

    # Add soft constraints
    handles = []
    for i, (constraint, weight) in enumerate(zip(constraints, weights)):
        handles.append(opt.add_soft(constraint, weight))

    # Find the optimal solution
    if _check_sat(opt):
        model = opt.model()

        # Find which constraints are satisfied
        satisfied = []
        for i, constraint in enumerate(constraints):
            if _is_true(constraint, model):
                satisfied.append(i)

        return satisfied, model
    else:
        return [], None


def relax_constraint(constraint: Any) -> Optional[Any]:  # type: ignore
    """
    Try to relax a constraint to make it easier to satisfy.

    Args:
        constraint: Z3 constraint

    Returns:
        Relaxed constraint or None if no relaxation is possible
    """
    if not _is_expr_ref(constraint):
        return None

    # Try different relaxation strategies

    # 1. Convert equality to inequality
    if _is_eq(constraint):
        left, right = constraint.children()

        # Only for numeric types
        if _is_arith(left) and _is_arith(right):
            # x == y becomes x <= y (or x >= y)
            return left <= right

    # 2. Expand numeric bounds
    if _is_app(constraint):
        decl_name = constraint.decl().name()

        if decl_name in (">=", ">", "<", "<="):
            left, right = constraint.children()

            # Only if right is a constant
            if _is_int_value(right) or _is_real(right):
                value = right.as_long() if _is_int_value(right) else right.as_decimal(prec=10)

                # Convert to float for calculations
                value = float(value)

                # Apply relaxation
                if decl_name == ">=":
                    # Decrease lower bound by 20%
                    new_value = value * 0.8
                    return left >= new_value
                elif decl_name == ">":
                    # Decrease lower bound or convert to >=
                    new_value = value * 0.8
                    return left >= new_value
                elif decl_name == "<=":
                    # Increase upper bound by 20%
                    new_value = value * 1.2
                    return left <= new_value
                elif decl_name == "<":
                    # Increase upper bound or convert to <=
                    new_value = value * 1.2
                    return left <= new_value

    # 3. For complex expressions, try simplifying
    if _is_and(constraint):
        children = constraint.children()

        if len(children) > 1:
            # Remove one child (the most complex one)
            # Simple heuristic: count variables
            child_vars = [(i, len(extract_variables(child))) for i, child in enumerate(children)]
            child_vars.sort(key=lambda x: x[1], reverse=True)

            # Remove the most complex child
            complex_idx = child_vars[0][0]
            new_children = [child for i, child in enumerate(children) if i != complex_idx]

            return _and(*new_children)

    # No relaxation technique worked
    return None
