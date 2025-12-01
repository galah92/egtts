"""
Create gold standard file from Spider dev.json.

Extracts the query and db_id fields and formats them for evaluation.
"""

import json
from pathlib import Path


def create_gold_standard():
    """Extract gold queries from dev.json in evaluation format."""
    spider_data_dir = Path("data/spider/spider_data")
    dev_file = spider_data_dir / "dev.json"
    output_file = Path("gold_dev.txt")

    print(f"Loading {dev_file}...")
    with open(dev_file) as f:
        examples = json.load(f)

    print(f"✓ Loaded {len(examples)} examples")

    # Write gold standard in format: SQL\tDB_ID
    with open(output_file, "w") as f:
        for example in examples:
            query = example["query"]
            db_id = example["db_id"]
            # Format: SQL\tDB_ID (tab-separated)
            f.write(f"{query}\t{db_id}\n")

    print(f"✓ Gold standard saved to: {output_file}")
    print(f"  {len(examples)} queries written")


if __name__ == "__main__":
    create_gold_standard()
