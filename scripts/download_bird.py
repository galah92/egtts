"""
Download and explore BIRD Mini-Dev dataset.

BIRD focuses on VES (Valid Efficiency Score) which evaluates:
1. Correctness (execution accuracy)
2. Efficiency (query plan cost)
"""

from datasets import load_dataset
import json
from pathlib import Path


def download_bird():
    """Download BIRD Mini-Dev dataset from HuggingFace."""
    print("=" * 80)
    print("DOWNLOADING BIRD MINI-DEV DATASET")
    print("=" * 80)

    print("\nLoading dataset from HuggingFace (birdsql/bird_mini_dev)...")
    dataset = load_dataset("birdsql/bird_mini_dev")

    # Access SQLite version
    sqlite_data = dataset["mini_dev_sqlite"]
    print(f"✓ Loaded {len(sqlite_data)} SQLite instances")

    # Save to local JSON for inspection
    output_dir = Path("data/bird_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to list of dicts
    data_list = list(sqlite_data)

    output_file = output_dir / "mini_dev_sqlite.json"
    with open(output_file, "w") as f:
        json.dump(data_list, f, indent=2)

    print(f"✓ Saved to: {output_file}")

    # Print first example
    print("\n" + "=" * 80)
    print("EXAMPLE INSTANCE")
    print("=" * 80)

    example = data_list[0]
    print(f"\nQuestion: {example.get('question', 'N/A')}")
    print(f"Database ID: {example.get('db_id', 'N/A')}")
    print(f"SQL: {example.get('SQL', 'N/A')}")
    print(f"Evidence: {example.get('evidence', 'N/A')}")

    # Print all keys to understand structure
    print("\n" + "=" * 80)
    print("DATASET STRUCTURE")
    print("=" * 80)
    print(f"\nKeys in each instance: {list(example.keys())}")

    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    # Count difficulty levels if available
    if 'difficulty' in example:
        difficulties = {}
        for item in data_list:
            diff = item.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1

        print("\nDifficulty Distribution:")
        for diff, count in difficulties.items():
            print(f"  {diff}: {count} ({count/len(data_list)*100:.1f}%)")

    # Count databases
    db_ids = {}
    for item in data_list:
        db_id = item.get('db_id', 'unknown')
        db_ids[db_id] = db_ids.get(db_id, 0) + 1

    print(f"\nUnique Databases: {len(db_ids)}")
    print(f"Total Instances: {len(data_list)}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Explore BIRD evaluation metrics (R-VES)")
    print("2. Create run_bird_full.py script")
    print("3. Run Baseline and M4 on BIRD")
    print("4. Compare efficiency scores")

    return data_list


if __name__ == "__main__":
    download_bird()
