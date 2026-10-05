# Import from external Z3 library
from z3 import Int, Solver, Distinct, And, Or, sat, unsat, unknown
from typing import Dict, List, Optional, Any, Tuple

def extract_solution_info(solution: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Extract attributes and attribute values from a solution.
    
    Args:
        solution: Solution to extract information from
        
    Returns:
        Tuple of (attributes list, attribute values dictionary)
    """
    header = solution.get("header", [])
    rows = solution.get("rows", [])

    attributes = header[1:] if header else []
    attr_values: Dict[str, Any] = {attr: set() for attr in attributes}

    for row in rows:
        if len(row) >= len(header):
            for i, attr in enumerate(attributes):
                attr_values[attr].add(row[i + 1])

    for attr in attr_values:
        attr_values[attr] = list(attr_values[attr])

    return attributes, attr_values

class Z3ConstraintChecker:
    """
    Z3 constraint checker for puzzle solutions.
    
    This class checks if a puzzle solution satisfies all Z3 constraints.
    """
    def __init__(self):
        self.solver = Solver()
        self.constraints = []
        self.var_map: Dict[str, Dict[str, Any]] = {} 
        self.attr_values: Dict[str, List[str]] = {}   
        self.house_count: int = 0
        self.attributes: List[str] = []

    def set_house_count(self, count: int):
        """Set the number of houses/entities."""
        self.house_count = count

    def set_attributes(self, attributes: List[str]):
        """Set the attributes to check."""
        self.attributes = attributes

    def set_attribute_values(self, attr_values: Dict[str, List[str]]):
        """Set the possible values for each attribute."""
        self.attr_values = attr_values
        self.var_map = {}
        for attr, values in attr_values.items():
            self.var_map[attr] = {}
            for val in values:
                self.var_map[attr][val] = Int(f"{attr}_{val}")

    def add_base_constraints(self):
        """Add base constraints for all attributes."""
        for attr, var_dict in self.var_map.items():
            for val, var in var_dict.items():
                self.solver.add(And(var >= 1, var <= self.house_count))

        for attr, var_dict in self.var_map.items():
            if var_dict:
                self.solver.add(Distinct(list(var_dict.values())))

    def add_constraint(self, constraint_type: str, *args, **kwargs):
        """Add a specific constraint type.
        
        Supported constraint types:
        - same_house: attr1, val1, attr2, val2
        - left_of: attr1, val1, attr2, val2
        - next_to: attr1, val1, attr2, val2
        - not_same_house: attr1, val1, attr2, val2
        - house_is: attr, val, house
        - not_house: attr, val, house
        """
        if constraint_type == "same_house":
            attr1, val1, attr2, val2 = args
            self.solver.add(self.var_map[attr1][val1] == self.var_map[attr2][val2])
        elif constraint_type == "left_of":
            attr1, val1, attr2, val2 = args
            self.solver.add(self.var_map[attr1][val1] < self.var_map[attr2][val2])
        elif constraint_type == "next_to":
            attr1, val1, attr2, val2 = args
            self.solver.add(
                Or(
                    self.var_map[attr1][val1] == self.var_map[attr2][val2] + 1,
                    self.var_map[attr1][val1] == self.var_map[attr2][val2] - 1
                )
            )
        elif constraint_type == "not_same_house":
            attr1, val1, attr2, val2 = args
            self.solver.add(self.var_map[attr1][val1] != self.var_map[attr2][val2])
        elif constraint_type == "house_is":
            attr, val, house = args
            self.solver.add(self.var_map[attr][val] == house)
        elif constraint_type == "not_house":
            attr, val, house = args
            self.solver.add(self.var_map[attr][val] != house)

    def check_solution(self, solution: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if a solution satisfies all Z3 constraints.
        
        Args:
            solution: Solution to check
            
        Returns:
            Tuple of (is_valid, feedback_message)
        """
        temp_solver = Solver()
        temp_solver.add(self.solver.assertions())

        header = solution.get("header", [])
        rows = solution.get("rows", [])

        if not header or not rows:
            return False, "Solution missing header or rows"

        house_mapping: Dict[int, Dict[str, str]] = {}
        for row in rows:
            if len(row) < len(header):
                continue
            house_num = row[0]
            try:
                house_idx = int(house_num)
            except Exception:
                continue
            house_mapping[house_idx] = {}
            for i, attr in enumerate(header[1:], 1):
                house_mapping[house_idx][attr] = row[i]

        for house_idx, attr_values in house_mapping.items():
            for attr, val in attr_values.items():
                if attr in self.var_map and val in self.var_map[attr]:
                    temp_solver.add(self.var_map[attr][val] == house_idx)

        result = temp_solver.check()
        if result == sat:
            return True, "All Z3 constraints are satisfied"
        else:
            return False, self._generate_feedback(temp_solver, solution)

    def _generate_feedback(self, solver, solution: Dict[str, Any]) -> str:
        """Generate feedback for an invalid solution.
        
        Args:
            solver: Z3 solver instance
            solution: Invalid solution
            
        Returns:
            Feedback message
        """
        feedback = "Constraint violations found by Z3.\n"

        if solver.check() == unsat:
            core = solver.unsat_core()
            if core:
                feedback += "Unsatisfiable core constraints:\n"
                for i, constraint in enumerate(core):
                    feedback += f"{i+1}. {constraint}\n"

        return feedback

    def analyze_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a solution for constraint violations and issues.
        
        Args:
            solution: Solution to analyze
            
        Returns:
            Dictionary with analysis results
        """
        header = solution.get("header", [])
        rows = solution.get("rows", [])

        issues: List[str] = []

        if not header or not rows:
            feedback = "Solution missing header or rows"
            return {
                "valid": False,
                "feedback": feedback,
                "issues": ["Missing header or rows"]
            }

        expected_len = len(header)

        for i, row in enumerate(rows):
            if len(row) != expected_len:
                feedback = f"Row {i+1} has incorrect length. Expected {expected_len}, got {len(row)}"
                return {
                    "valid": False,
                    "feedback": feedback,
                    "issues": [feedback]
                }

        try:
            house_nums = [int(row[0]) for row in rows]
        except Exception:
            feedback = "House numbers (first column) contain non-integer values"
            return {
                "valid": False,
                "feedback": feedback,
                "issues": [feedback]
            }

        if sorted(house_nums) != list(range(1, len(rows) + 1)):
            feedback = "House numbers are not consecutive starting from 1"
            return {
                "valid": False,
                "feedback": feedback,
                "issues": [feedback]
            }

        z3_valid, z3_feedback = self.check_solution(solution)

        attributes_in_header = header[1:]  

        for attr_idx, attr in enumerate(attributes_in_header, start=1):
            seen_vals: Dict[str, List[int]] = {}
            for row in rows:
                try:
                    house_idx = int(row[0])
                except Exception:
                    continue
                if len(row) <= attr_idx:
                    continue
                val = row[attr_idx]
                seen_vals.setdefault(val, []).append(house_idx)

            for val, houses in seen_vals.items():
                if len(houses) > 1:
                    issues.append(
                        f"Value '{val}' of attribute '{attr}' appears in multiple houses: {houses}"
                    )

            expected_vals = self.attr_values.get(attr)
            if expected_vals:
                missing = [v for v in expected_vals if v not in seen_vals]
                if missing:
                    issues.append(
                        f"Attribute '{attr}' is missing values: {missing}"
                    )
                illegal = [v for v in seen_vals if v not in expected_vals]
                if illegal:
                    issues.append(
                        f"Attribute '{attr}' has illegal values: {illegal} (expected: {expected_vals})"
                    )

        valid = z3_valid and not issues

        if issues:
            feedback = "; ".join(issues)
            if z3_feedback and z3_feedback != "All Z3 constraints are satisfied":
                feedback = z3_feedback + " " + feedback
        else:
            feedback = z3_feedback or "All constraints are satisfied"

        return {
            "valid": valid,
            "feedback": feedback,
            "issues": issues
        }
