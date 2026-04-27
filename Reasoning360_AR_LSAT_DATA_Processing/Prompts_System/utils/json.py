import re
import json
from typing import Optional, Dict, Any


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction handling markdown blocks and raw text.
    
    Args:
        text: Text to extract JSON from
        
    Returns:
        Extracted JSON or None if extraction failed
    """
    if not text:
        return None
    
    # First try: extract from markdown code blocks
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    # Second try: extract from raw text
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            return json.loads(json_str)
    except Exception:
        pass
    
    # Third try: handle cases with "json" prefix
    text = text.strip()
    if text.lower().startswith("json"):
        text = text[4:].lstrip(" :\n\t")
        try:
            return json.loads(text)
        except Exception:
            pass
    
    # Fourth try: extract array format if no object found
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            return json.loads(candidate)
    except Exception:
        pass
    
    return None
