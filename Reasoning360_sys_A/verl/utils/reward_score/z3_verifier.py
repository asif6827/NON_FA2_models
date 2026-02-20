# Import from external Z3 library
from z3 import Int, Solver, Distinct, And, Or, sat, unsat, unknown
from typing import Dict, List, Optional, Any, Tuple, Union

class Z3Verifier:
    """
    Z3 verifier for puzzle solutions, integrated from final_code-1.
    
    This class provides Z3-based verification for puzzle solutions, including:
    - Constraint checking
    - Solution validation
    - Score calculation
    - Detailed feedback generation
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
        """Generate feedback for an invalid solution."""
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
            Dictionary with analysis results including validity, feedback, and issues
        """
        header = solution.get("header", [])
        rows = solution.get("rows", [])
        
        issues: List[str] = []
        
        if not header or not rows:
            feedback = "Solution missing header or rows"
            return {
                "valid": False,
                "feedback": feedback,
                "issues": ["Missing header or rows"],
                "score": 0.0
            }
        
        expected_len = len(header)
        
        for i, row in enumerate(rows):
            if len(row) != expected_len:
                feedback = f"Row {i+1} has incorrect length. Expected {expected_len}, got {len(row)}"
                return {
                    "valid": False,
                    "feedback": feedback,
                    "issues": [feedback],
                    "score": 0.0
                }
        
        try:
            house_nums = [int(row[0]) for row in rows]
        except Exception:
            feedback = "House numbers (first column) contain non-integer values"
            return {
                "valid": False,
                "feedback": feedback,
                "issues": [feedback],
                "score": 0.0
            }
        
        if sorted(house_nums) != list(range(1, len(rows) + 1)):
            feedback = "House numbers are not consecutive starting from 1"
            return {
                "valid": False,
                "feedback": feedback,
                "issues": [feedback],
                "score": 0.0
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
                if isinstance(val, list):
                    # Handle the case where val is a list (perhaps convert it to a tuple or do something else)
                    val = tuple(val)  # Convert list to tuple if needed
                try:
                    seen_vals.setdefault(val, []).append(house_idx)
                except Exception:
                    print(f"EXCEPTION CAUSED in Z3-VERIFIER val type: {type(val)}, val: {val}")
            
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
        
        # Calculate Z3 verification score
        # 1.0 if valid, otherwise 0.0 (can be extended for partial scoring)
        z3_score = 1.0 if valid else 0.0
        
        return {
            "valid": valid,
            "feedback": feedback,
            "issues": issues,
            "score": z3_score
        }
    
    def verify_solution(self, solution: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify a puzzle solution using Z3.
        
        Args:
            solution: Solution to verify
            meta: Optional metadata containing entity_count, attributes, and attribute_values
            
        Returns:
            Dictionary with verification results and score
        """
        if meta:
            # Initialize from metadata if provided
            self.set_house_count(meta.get("entity_count", 5))
            self.set_attributes(meta.get("attributes", []))
            self.set_attribute_values(meta.get("attribute_values", {}))
            self.add_base_constraints()
        
        # Analyze the solution
        analysis_result = self.analyze_solution(solution)
        
        # Additional checks can be added here
        
        return {
            "z3_valid": analysis_result["valid"],
            "z3_feedback": analysis_result["feedback"],
            "z3_issues": analysis_result["issues"],
            "z3_score": analysis_result["score"]
        }
    
    def compute_z3_score(self, solution: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute Z3 verification score for a solution.
        
        Args:
            solution: Solution to score
            meta: Optional metadata
            
        Returns:
            Z3 verification score (0.0-1.0)
        """
        verification_result = self.verify_solution(solution, meta)
        return verification_result["z3_score"]


def extract_solution_info(solution: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Extract attributes and attribute values from a solution.
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


def verify_solution_with_z3(solution: Dict[str, Any], ground_truth: Dict[str, Any] = None, 
                           meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Verify a solution using Z3, with optional ground truth comparison.
    
    Args:
        solution: Solution to verify
        ground_truth: Optional ground truth solution for comparison
        meta: Optional metadata
        
    Returns:
        Dictionary with verification results including Z3 score and optional ACC score
    """
    try:
        # Create Z3 verifier instance
        verifier = Z3Verifier()
        
        # Perform Z3 verification
        z3_result = verifier.verify_solution(solution, meta)
        
        # Calculate ACC score if ground truth is provided
        acc_score = 0.0
        if ground_truth is not None:
            # 确保 ground_truth 是字典类型
            if not isinstance(ground_truth, dict):
                ground_truth = None
            
        if ground_truth:
            # Simple ACC calculation: check if solution matches ground truth
            def are_equal(obj1, obj2):
                import numpy as np
                # Check if both are numpy arrays
                if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
                    return bool(np.array_equal(obj1, obj2))
                # Check if one is numpy array and the other is not
                elif isinstance(obj1, np.ndarray) or isinstance(obj2, np.ndarray):
                    # Convert numpy array to list for comparison
                    if isinstance(obj1, np.ndarray):
                        obj1 = obj1.tolist()
                    if isinstance(obj2, np.ndarray):
                        obj2 = obj2.tolist()
                    return are_equal(obj1, obj2)
                # Check if both are lists
                elif isinstance(obj1, list) and isinstance(obj2, list):
                    if len(obj1) != len(obj2):
                        return False
                    result = True
                    for o1, o2 in zip(obj1, obj2):
                        if not are_equal(o1, o2):
                            result = False
                            break
                    return result
                # Check if both are dictionaries
                elif isinstance(obj1, dict) and isinstance(obj2, dict):
                    if obj1.keys() != obj2.keys():
                        return False
                    result = True
                    for k in obj1.keys():
                        if not are_equal(obj1[k], obj2[k]):
                            result = False
                            break
                    return result
                # Default comparison
                else:
                    try:
                        # Handle numpy scalar values
                        if hasattr(obj1, 'item'):
                            obj1 = obj1.item()
                        if hasattr(obj2, 'item'):
                            obj2 = obj2.item()
                        return bool(obj1 == obj2)
                    except (ValueError, TypeError):
                        return False
            
            if are_equal(solution, ground_truth):
                acc_score = 1.0
            else:
                # More sophisticated ACC calculation could be implemented here
                # For example, cell-wise accuracy
                header = solution.get("header", [])
                rows = solution.get("rows", [])
                gt_header = ground_truth.get("header", [])
                gt_rows = ground_truth.get("rows", [])
                
                # Handle numpy arrays by converting to lists
                import numpy as np
                if isinstance(header, np.ndarray):
                    header = header.tolist()
                if isinstance(gt_header, np.ndarray):
                    gt_header = gt_header.tolist()
                if isinstance(rows, np.ndarray):
                    rows = rows.tolist()
                if isinstance(gt_rows, np.ndarray):
                    gt_rows = gt_rows.tolist()
                
                if header == gt_header and len(rows) == len(gt_rows):
                    correct_cells = 0
                    total_cells = 0
                    
                    for row, gt_row in zip(rows, gt_rows):
                        if len(row) == len(gt_row):
                            total_cells += len(row)
                            correct_cells += sum(1 for r, gt_r in zip(row, gt_row) if r == gt_r)
                    
                    if total_cells > 0:
                        acc_score = correct_cells / total_cells
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in verify_solution_with_z3: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认结果
        return {
            "z3_valid": False,
            "z3_feedback": f"Error in Z3 verification: {str(e)}",
            "z3_issues": [f"Error: {str(e)}"],
            "z3_score": 0.0,
            "acc_score": 0.0
        }
    
    return {
        **z3_result,
        "acc_score": acc_score
    }


def main():
    """Example usage of the Z3Verifier."""
    # Example solution
    example_solution = {
        "header": ["House", "Name", "Pet", "Color"],
        "rows": [
            ["1", "Alice", "cat", "red"],
            ["2", "Bob", "dog", "blue"],
            ["3", "Charlie", "bird", "green"]
        ]
    }
    
    # Example metadata
    example_meta = {
        "entity_count": 3,
        "attributes": ["Name", "Pet", "Color"],
        "attribute_values": {
            "Name": ["Alice", "Bob", "Charlie"],
            "Pet": ["cat", "dog", "bird"],
            "Color": ["red", "blue", "green"]
        }
    }
    
    # Verify the solution
    result = verify_solution_with_z3(example_solution, example_meta)
    
    print("Z3 Verification Result:")
    print(f"Valid: {result['z3_valid']}")
    print(f"Feedback: {result['z3_feedback']}")
    print(f"Issues: {result['z3_issues']}")
    print(f"Z3 Score: {result['z3_score']}")
    print(f"ACC Score: {result['acc_score']}")


if __name__ == "__main__":
    main()
