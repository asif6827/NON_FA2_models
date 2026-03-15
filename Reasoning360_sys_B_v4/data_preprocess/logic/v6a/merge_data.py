import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_json_list(paths):
    merged_data = None

    for path in paths:
        data = load_json(path)

        if merged_data is None:
            merged_data = data
        else:
            if isinstance(merged_data, list) and isinstance(data, list):
                merged_data.extend(data)
            elif isinstance(merged_data, dict) and isinstance(data, dict):
                merged_data.update(data)
            else:
                raise ValueError("All JSON files must have the same structure.")

    return merged_data


def main():
    json_paths = [
        "/home/asif/data3/HF_cache/ZebraLogic/Zebra_Puzzle_medium_280.json",
        "/home/asif/data3/HF_cache/ZebraLogic/Zebra_Puzzle_large_200.json"
    ]

    output_path = "/home/asif/data3/HF_cache/ZebraLogic/Zebra_Puzzle_ML_480.json"

    merged_data = merge_json_list(json_paths)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged JSON written to {output_path}")


if __name__ == "__main__":
    main()
