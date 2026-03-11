"""Convert raw data to instruction tuning format."""
import json
import argparse
from pathlib import Path


def convert_to_alpaca(input_file: str, output_file: str) -> None:
    """Convert a raw QA dataset to Alpaca JSONL format."""
    with open(input_file) as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append({
            "instruction": item.get("question", item.get("prompt", "")),
            "input": "",
            "output": item.get("answer", item.get("response", "")),
        })

    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Converted {len(samples)} samples → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    convert_to_alpaca(args.input, args.output)
