#!/usr/bin/env python3
"""
Convert our benchmark results to BIRD official evaluation format.

Official format:
{
    "0": "SQL query\\t----- bird -----\\tdb_id",
    "1": "SQL query\\t----- bird -----\\tdb_id",
    ...
}

Our format:
{
    "strategy": "baseline",
    "examples": [
        {
            "index": 0,
            "db_id": "...",
            "predicted_sql": "...",
            ...
        },
        ...
    ]
}
"""

import argparse
import json


def convert_to_official_format(results_path: str, output_path: str) -> None:
    """
    Convert our results JSON to official BIRD evaluation format.

    Args:
        results_path: Path to our results JSON file
        output_path: Path to save converted JSON file
    """
    with open(results_path, "r") as f:
        data = json.load(f)

    # Convert to official format
    official_format = {}

    for example in data["examples"]:
        idx = example["index"]
        db_id = example["db_id"]

        # Handle cases where prediction failed
        if "predicted_sql" in example:
            sql = example["predicted_sql"]
        elif "error" in example:
            # Use a dummy query for failed predictions
            sql = "SELECT 1"  # Will fail execution but won't crash evaluation
        else:
            sql = "SELECT 1"

        # Format: "SQL\t----- bird -----\tdb_id"
        official_format[str(idx)] = f"{sql}\t----- bird -----\t{db_id}"

    # Write output
    with open(output_path, "w") as f:
        json.dump(official_format, f, indent=4)

    print(f"✓ Converted {len(official_format)} predictions")
    print(f"  Input:  {results_path}")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert EGTTS results to official BIRD evaluation format"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to our results JSON file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save converted JSON file"
    )

    args = parser.parse_args()

    convert_to_official_format(args.input, args.output)


if __name__ == "__main__":
    main()
