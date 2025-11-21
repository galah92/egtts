#!/usr/bin/env python3
"""
Setup BIRD official evaluation environment by creating required files.
"""

import json
from pathlib import Path


def create_ground_truth_sql(data_path: str, output_path: str):
    """Create ground truth SQL file from mini_dev JSON."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    with open(output_path, 'w') as f:
        for example in data:
            sql = example['SQL']
            db_id = example['db_id']
            # Write SQL on single line with tab separator and db_id
            sql_single_line = ' '.join(sql.replace('\n', ' ').split())
            f.write(f"{sql_single_line}\t{db_id}\n")

    print(f"✓ Created ground truth SQL: {output_path} ({len(data)} queries)")


def create_difficulty_jsonl(data_path: str, output_path: str):
    """Create difficulty JSONL file from mini_dev JSON."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    with open(output_path, 'w') as f:
        for example in data:
            entry = {
                "db_id": example['db_id'],
                "difficulty": example.get('difficulty', 'moderate'),  # Default if missing
                "question": example['question'],
                "SQL": example['SQL']
            }
            f.write(json.dumps(entry) + '\n')

    print(f"✓ Created difficulty JSONL: {output_path} ({len(data)} queries)")


def main():
    # Paths
    data_path = Path("data/bird/mini_dev_sqlite.json")
    sqlite_dir = Path("data/mini_dev_official/sqlite")

    # Create sqlite directory if it doesn't exist
    sqlite_dir.mkdir(parents=True, exist_ok=True)

    # Create ground truth SQL
    gold_sql_path = sqlite_dir / "mini_dev_sqlite_gold.sql"
    create_ground_truth_sql(str(data_path), str(gold_sql_path))

    # Create difficulty JSONL
    diff_jsonl_path = sqlite_dir / "mini_dev_sqlite.jsonl"
    create_difficulty_jsonl(str(data_path), str(diff_jsonl_path))

    print("\n✓ Setup complete! Official evaluation environment is ready.")
    print(f"  Database: {sqlite_dir}/dev_databases")
    print(f"  Ground truth: {gold_sql_path}")
    print(f"  Difficulty: {diff_jsonl_path}")


if __name__ == '__main__':
    main()
