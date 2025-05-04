# Define or import validation functions
def is_int_value(v):
    """Validate if a value is an integer."""
    return isinstance(v, int)


def is_int(v):
    """Validate if a value is an integer."""
    return isinstance(v, int)


def is_real(v):
    """Validate if a value is a real number (float or int)."""
    return isinstance(v, (float, int))


# Define logical operators for constraints
class And:
    """Logical AND operator for constraints."""

    def __init__(self, *args):
        self.args = args

    def evaluate(self, context):
        return all(arg.evaluate(context) for arg in self.args)


class Or:
    """Logical OR operator for constraints."""

    def __init__(self, *args):
        self.args = args

    def evaluate(self, context):
        return any(arg.evaluate(context) for arg in self.args)


class Not:
    """Logical NOT operator for constraints."""

    def __init__(self, arg):
        self.arg = arg

    def evaluate(self, context):
        return not self.arg.evaluate(context)


# Define type classes
class Int:
    """Integer type constraint."""

    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value):
        if not isinstance(value, int):
            return False
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


class Real:
    """Real number type constraint."""

    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value):
        if not isinstance(value, (float, int)):
            return False
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


class Bool:
    """Boolean type constraint."""

    def validate(self, value):
        return isinstance(value, bool)


# Define solver class
class Solver:
    """Constraint solver for validation."""

    def __init__(self, constraints=None):
        self.constraints = constraints or []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def solve(self, context):
        """Solve the constraints given the context."""
        for constraint in self.constraints:
            if not constraint.evaluate(context):
                return False
        return True