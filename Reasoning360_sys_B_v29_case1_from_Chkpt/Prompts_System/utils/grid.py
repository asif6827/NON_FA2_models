from typing import Dict, List, Any, Optional

def _normalize_atom(x: Any) -> str:
    """
    Normalize a single value to a string.
    """
    s = str(x).strip().lower()
    return s

def try_extract_grid(data: Any) -> Any:
    """
    Extract grid from data, handling different formats.
    
    Args:
        data: Data to extract grid from
        
    Returns:
        Extracted grid or original data if no grid found
    """
    if not isinstance(data, dict):
        return data

    if "header" in data and "rows" in data:
        return data

    sol = data.get("solution")
    if isinstance(sol, dict) and "header" in sol and "rows" in sol:
        return sol

    return data

def normalize_grid(data: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize grid data for comparison.
    
    Args:
        data: Grid data to normalize
        
    Returns:
        Normalized grid or None if normalization failed
    """
    if not isinstance(data, dict):
        return None
    if "header" not in data or "rows" not in data:
        return None

    try:
        header = [_normalize_atom(h) for h in data["header"]]

        ignore_cols = {"house"}
        keep_indices = [i for i, h in enumerate(header) if h not in ignore_cols]
        if not keep_indices:
            return None

        rows_norm = []
        for row in data["rows"]:
            row_norm_full = [_normalize_atom(x) for x in row]
            row_norm = [row_norm_full[i] for i in keep_indices]
            rows_norm.append(row_norm)

        header_kept = [header[i] for i in keep_indices]
        rows_norm_sorted = sorted(rows_norm)
        return {"header": header_kept, "rows": rows_norm_sorted}
    except Exception:
        return None

def normalize_value(value: Any) -> Any:
    """
    Recursively normalize values in a dictionary or list.
    """
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    else:
        return _normalize_atom(value)

def score_with_ground_truth(current_solution: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a solution against ground truth.
    
    Args:
        current_solution: Current solution to score
        ground_truth: Ground truth solution
        
    Returns:
        Dictionary with score, correct_cells, and total_cells
    """
    try:
        gt_for_grid = try_extract_grid(ground_truth)
        cur_for_grid = try_extract_grid(current_solution)
        norm_gt = normalize_grid(gt_for_grid)
        norm_cur = normalize_grid(cur_for_grid)
        if norm_gt is not None and norm_cur is not None:
            rows_gt = norm_gt["rows"]
            rows_cur = norm_cur["rows"]
            cols = len(norm_gt["header"]) if isinstance(norm_gt.get("header"), list) else 0
            total_rows = len(rows_gt)
            total_cells = total_rows * cols
            correct = 0
            if total_cells > 0:
                for i in range(total_rows):
                    gt_row = rows_gt[i] if i < len(rows_gt) else []
                    cur_row = rows_cur[i] if i < len(rows_cur) else []
                    for j in range(cols):
                        gt_val = gt_row[j] if j < len(gt_row) else None
                        cur_val = cur_row[j] if j < len(cur_row) else None
                        if gt_val == cur_val:
                            correct += 1
                score = correct / total_cells
            else:
                score = 0.0
            return {"score": score, "correct_cells": correct, "total_cells": total_cells}
        normalized_gt = normalize_value(ground_truth)
        normalized_sol = normalize_value(current_solution)
        ok = 1.0 if normalized_gt == normalized_sol else 0.0
        return {"score": ok, "correct_cells": int(ok), "total_cells": 1}
    except Exception:
        return {"score": 0.0, "correct_cells": 0, "total_cells": 0}

def check_with_ground_truth(current_solution: Dict[str, Any], ground_truth: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    """
    Check if a solution matches ground truth.
    
    Args:
        current_solution: Current solution to check
        ground_truth: Ground truth solution
        
    Returns:
        True if solution matches ground truth, False otherwise, or None if checking failed
    """
    try:
        gt_for_grid = try_extract_grid(ground_truth)
        cur_for_grid = try_extract_grid(current_solution)

        norm_gt = normalize_grid(gt_for_grid)
        norm_cur = normalize_grid(cur_for_grid)

        if norm_gt is not None and norm_cur is not None:
            return norm_gt == norm_cur

        normalized_gt = normalize_value(ground_truth)
        normalized_sol = normalize_value(current_solution)
        return normalized_gt == normalized_sol

    except Exception:
        return None
