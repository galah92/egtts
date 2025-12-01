"""
Format predictions to match evaluation format (SQL\tDB_ID).
"""

import json
from pathlib import Path


def format_predictions():
    """Add db_id to predictions file."""
    spider_data_dir = Path("data/spider/spider_data")
    dev_file = spider_data_dir / "dev.json"
    pred_file = Path("spider_results.txt")
    output_file = Path("pred_dev.txt")

    print(f"Loading {dev_file}...")
    with open(dev_file) as f:
        examples = json.load(f)

    print(f"Loading {pred_file}...")
    with open(pred_file) as f:
        predictions = [line.strip() for line in f]

    print(f"✓ Loaded {len(examples)} examples and {len(predictions)} predictions")

    if len(examples) != len(predictions):
        print("⚠ Warning: Mismatch in counts!")
        return

    # Write predictions in format: SQL\tDB_ID
    with open(output_file, "w") as f:
        for example, pred_sql in zip(examples, predictions):
            db_id = example["db_id"]
            # Format: SQL\tDB_ID (tab-separated)
            f.write(f"{pred_sql}\t{db_id}\n")

    print(f"✓ Predictions formatted to: {output_file}")
    print(f"  {len(predictions)} queries written")


if __name__ == "__main__":
    format_predictions()
