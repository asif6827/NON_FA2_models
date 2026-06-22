import json

SOLUTION_PROMPT_SYSTEM = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

Your task is to:
1. Analyze the clues step by step.
2. Derive a correct final solution.
3. Return the result STRICTLY as a single valid JSON object.

CRITICAL FORMAT REQUIREMENTS:
- Output ONLY a JSON object, NO natural language, NO markdown, NO code fences.
- The top-level JSON MUST have exactly two keys: "reasoning" and "solution".
- "reasoning" MUST be a SHORT English explanation (1–5 sentences, not more).
- "solution" MUST be an object with:
  - "header": a list of column names (e.g. ["House", "Name", "Pet", "..."])
  - "rows": a list of rows, where each row is a list of strings, one per column.

Example of the REQUIRED SHAPE (this is ONLY an example, not the answer):

{
  "reasoning": "Your step-by-step logic here, but concise.",
  "solution": {
    "header": ["House", "Name", "Pet", "..."],
    "rows": [
      ["1", "Eric", "cat", "..."],
      ["2", "Arnold", "dog", "..."]
    ]
  }
}

Do NOT include any text before or after the JSON.
"""

SOLUTION_PROMPT_USER = """PUZZLE:
{puzzle}

Please provide your reasoning and solution:"""


def main(in_path, out_path):
    data_processed = []

    
    with open(in_path, 'r') as f:
        content = json.load(f)
        if isinstance(content, list):
            data = content

    for line in data:
        entry = {}

        entry['instruction'] = SOLUTION_PROMPT_SYSTEM + SOLUTION_PROMPT_USER.format(puzzle=line['puzzle'])
        entry['apply_chat_template'] = False
        entry['puzzle_id'] = line['id']
        entry['ground_truth'] = line['solution']
        entry['reward_model'] = line['solution']
        entry['config'] = line['size']
        entry['extra_info'] = line['created_at']

        data_processed.append(entry)


    

    with open(out_path, "w", encoding="utf-8") as f:
        for item in data_processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    in_path = '/home/asif/data3/HF_cache/ZebraLogic/Zebra_Puzzle_small_320.json'
    out_path = '/home/asif/data3/HF_cache/ZebraPuzzle_guru/zebra_puzzle.json'
    main(in_path, out_path)