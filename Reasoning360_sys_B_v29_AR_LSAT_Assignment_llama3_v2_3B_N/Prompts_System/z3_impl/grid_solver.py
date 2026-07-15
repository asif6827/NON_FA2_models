from dataclasses import dataclass
from z3 import Solver, Int, And, Or, Distinct, sat, unsat
from typing import Dict, List, Optional, Any, Tuple

@dataclass
class GridModel:
    """
    Dataclass for representing a grid model for Z3 solving.
    
    Attributes:
        meta: Metadata about the model
        solver: Z3 solver instance
        vars: Dictionary of variables, organized by attribute
        enc: Encoding mapping from attribute values to integers
        enc_rev: Reverse encoding mapping from integers to attribute values
    """
    meta: Dict[str, Any]
    solver: Solver
    vars: Dict[str, List[Any]]          # vars["book"][i] = house(i+1)_book
    enc: Dict[str, Dict[str, int]]      # enc[attr][value] = code
    enc_rev: Dict[str, Dict[int, str]]  # enc_rev[attr][code] = value

def build_grid_skeleton(meta: Dict[str, Any]) -> GridModel:
    """
    Build a grid model skeleton from metadata.
    
    Args:
        meta: Metadata containing entity_count, attributes, and attribute_values
        
    Returns:
        GridModel instance
    """
    H = meta["entity_count"]
    # Use default value "house" if entity_type is not provided
    entity_type = meta.get("entity_type", "house").lower()
    attributes: List[str] = meta["attributes"]
    attr_vals: Dict[str, List[str]] = meta["attribute_values"]

    enc: Dict[str, Dict[str, int]] = {}
    enc_rev: Dict[str, Dict[int, str]] = {}
    for attr in attributes:
        values = attr_vals[attr]
        enc[attr] = {v: i + 1 for i, v in enumerate(values)}
        enc_rev[attr] = {i + 1: v for i, v in enumerate(values)}

    vars: Dict[str, List[Any]] = {}
    for attr in attributes:
        vars[attr] = [
            Int(f"{entity_type}{idx+1}_{attr}")
            for idx in range(H)
        ]

    s = Solver()
    # Set timeout for Z3 solver (5 seconds)
    s.set(timeout=5000)
    # Enable unsat core generation for debugging
    s.set(unsat_core=True)

    domain = list(range(1, H + 1))
    for attr in attributes:
        col = vars[attr]
        for v in col:
            s.add(Or([v == d for d in domain]))
        s.add(Distinct(col))

    return GridModel(meta=meta, solver=s, vars=vars, enc=enc, enc_rev=enc_rev)


def _encode(model: GridModel, ref: Dict[str, str]) -> Tuple[List[Any], int]:
    """
    Encode an attribute-value reference to Z3 variables.
    
    Args:
        model: GridModel instance
        ref: Attribute-value reference
        
    Returns:
        Tuple of (variable list, encoded value)
    """
    attr = ref["attr"].lower().strip()
    value = ref["value"].strip()

    matched_attr = None
    attr_mapping_strategies = [
        attr,  
        attr.rstrip('s'),  
        attr + 's',  
        attr.replace(' ', '_'),  
        attr.replace('-', '_') 
    ]

    for attr_candidate in attr_mapping_strategies:
        if attr_candidate in model.vars:
            matched_attr = attr_candidate
            break

    if not matched_attr:
        for model_attr in model.vars.keys():
            if attr in model_attr or model_attr in attr:
                matched_attr = model_attr
                break

    if not matched_attr:
        raise KeyError(f"Unknown attribute in DSL: {attr}")

    matched_value = None
    value_mapping_strategies = [
        value, 
        value.lower(),  
        value.capitalize(),  
        value.upper(),  
        value.strip(),  
        value.lower().strip(),  
        value.capitalize().strip()  
    ]

    for value_candidate in value_mapping_strategies:
        if value_candidate in model.enc[matched_attr]:
            matched_value = value_candidate
            break

    if not matched_value:
        for model_value in model.enc[matched_attr].keys():
            if value.lower() in model_value.lower() or model_value.lower() in value.lower():
                matched_value = model_value
                break

    if not matched_value:
        raise KeyError(f"Unknown value for attr {matched_attr}: {value}")

    col = model.vars[matched_attr]
    code = model.enc[matched_attr][matched_value]
    return col, code


def _same_person(model: GridModel, a: Dict[str, str], b: Dict[str, str]):
    """
    Create a Z3 constraint for same person.
    
    Args:
        model: GridModel instance
        a: First attribute-value reference
        b: Second attribute-value reference
        
    Returns:
        Z3 constraint
    """
    H = model.meta["entity_count"]
    col_a, code_a = _encode(model, a)
    col_b, code_b = _encode(model, b)
    return Or(*[
        And(col_a[i] == code_a, col_b[i] == code_b)
        for i in range(H)
    ])


def _house_is(model: GridModel, a: Dict[str, str], house_idx: int):
    """
    Create a Z3 constraint for house is.
    
    Args:
        model: GridModel instance
        a: Attribute-value reference
        house_idx: House index
        
    Returns:
        Z3 constraint
    """
    col, code = _encode(model, a)
    return col[house_idx - 1] == code


def _not_house(model: GridModel, a: Dict[str, str], house_idx: int):
    """
    Create a Z3 constraint for not house.
    
    Args:
        model: GridModel instance
        a: Attribute-value reference
        house_idx: House index
        
    Returns:
        Z3 constraint
    """
    col, code = _encode(model, a)
    return col[house_idx - 1] != code


def _left_of(model: GridModel, a: Dict[str, str], b: Dict[str, str]):
    """
    Create a Z3 constraint for left of.
    
    Args:
        model: GridModel instance
        a: First attribute-value reference
        b: Second attribute-value reference
        
    Returns:
        Z3 constraint
    """
    H = model.meta["entity_count"]
    col_a, code_a = _encode(model, a)
    col_b, code_b = _encode(model, b)
    return Or(*[
        And(col_a[i] == code_a, col_b[j] == code_b)
        for i in range(H) for j in range(H)
        if i < j
    ])


def _right_of(model: GridModel, a: Dict[str, str], b: Dict[str, str]):
    """
    Create a Z3 constraint for right of.
    
    Args:
        model: GridModel instance
        a: First attribute-value reference
        b: Second attribute-value reference
        
    Returns:
        Z3 constraint
    """
    H = model.meta["entity_count"]
    col_a, code_a = _encode(model, a)
    col_b, code_b = _encode(model, b)
    return Or(*[
        And(col_a[i] == code_a, col_b[j] == code_b)
        for i in range(H) for j in range(H)
        if i > j
    ])


def _next_to(model: GridModel, a: Dict[str, str], b: Dict[str, str]):
    """
    Create a Z3 constraint for next to.
    
    Args:
        model: GridModel instance
        a: First attribute-value reference
        b: Second attribute-value reference
        
    Returns:
        Z3 constraint
    """
    H = model.meta["entity_count"]
    col_a, code_a = _encode(model, a)
    col_b, code_b = _encode(model, b)
    return Or(*[
        And(col_a[i] == code_a, col_b[i + 1] == code_b)
        for i in range(H - 1)
    ] + [
        And(col_a[i] == code_a, col_b[i - 1] == code_b)
        for i in range(1, H)
    ])


def _distance(model: GridModel, a: Dict[str, str], b: Dict[str, str], dist: int):
    """
    Create a Z3 constraint for distance.
    
    Args:
        model: GridModel instance
        a: First attribute-value reference
        b: Second attribute-value reference
        dist: Distance between entities
        
    Returns:
        Z3 constraint
    """
    H = model.meta["entity_count"]
    col_a, code_a = _encode(model, a)
    col_b, code_b = _encode(model, b)
    return Or(*[
        And(col_a[i] == code_a, col_b[j] == code_b)
        for i in range(H) for j in range(H)
        if abs(i - j) == dist + 1
    ])


def compile_constraint(model: GridModel, c: Dict[str, Any]):
    """
    Compile a constraint dictionary to a Z3 constraint.
    
    Args:
        model: GridModel instance
        c: Constraint dictionary
        
    Returns:
        Z3 constraint
    """
    op = c["op"]

    if op == "same_person":
        return _same_person(model, c["a"], c["b"])

    elif op == "house_is":
        return _house_is(model, c["a"], int(c["house"]))

    elif op == "not_house":
        return _not_house(model, c["a"], int(c["house"]))

    elif op == "left_of":
        return _left_of(model, c["a"], c["b"])

    elif op == "right_of":
        return _right_of(model, c["a"], c["b"])

    elif op == "next_to":
        return _next_to(model, c["a"], c["b"])

    elif op == "distance":
        return _distance(model, c["a"], c["b"], int(c["distance"]))

    else:
        raise ValueError(f"Unknown op in DSL: {op}")


def compile_and_add_constraints(model: GridModel, constraints: List[Dict[str, Any]]):
    """
    Compile and add constraints to the Z3 solver.
    
    Args:
        model: GridModel instance
        constraints: List of constraint dictionaries
    """
    for i, c in enumerate(constraints):
        try:
            expr = compile_constraint(model, c)
            model.solver.add(expr)
        except Exception as e:
            print(f"[Warning] Skipped invalid constraint {i}: {c}. Error: {e}")


def decode_solution(model: GridModel) -> Dict[str, Any]:
    """
    Decode a Z3 solution to a grid format.
    
    Args:
        model: GridModel instance with a solved Z3 model
        
    Returns:
        Solution dictionary with header and rows
    """
    H = model.meta["entity_count"]
    attributes = model.meta["attributes"]
    m = model.solver.model()

    header = ["House"] + attributes
    rows = []
    for i in range(H):
        row = [str(i + 1)]
        for attr in attributes:
            var = model.vars[attr][i]
            code = m[var].as_long()
            value = model.enc_rev[attr][code]
            row.append(value)
        rows.append(row)

    return {"header": header, "rows": rows}
