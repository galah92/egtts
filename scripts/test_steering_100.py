"""
Test Execution-Guided Steering (EG-SQL) on first 100 Spider examples.

This tests the new clause-aware beam search approach that validates
SQL at clause boundaries and prunes invalid beams early.
"""

import json
import sqlite3
import time
from pathlib import Path

from tqdm import tqdm

from egtts import load_model
from egtts.steering import generate_with_steering


def get_database_schema(db_path: Path) -> str:
    """Extract CREATE TABLE statements from database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        schema_parts = []
        for (table_name,) in tables:
            cursor.execute(
                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            )
            result = cursor.fetchone()
            if result:
                schema_parts.append(result[0])

        conn.close()
        return "\n\n".join(schema_parts)
    except Exception as e:
        print(f"Warning: Failed to extract schema from {db_path}: {e}")
        return ""


def main():
    print("=" * 80)
    print("EG-SQL: EXECUTION-GUIDED STEERING - Test on 100 Examples")
    print("=" * 80)

    # Load Spider data
    spider_data_dir = Path("data/spider/spider_data")
    dev_file = spider_data_dir / "dev.json"

    print(f"\nLoading {dev_file}...")
    with open(dev_file) as f:
        examples = json.load(f)

    # Take first 100 examples
    examples = examples[:100]
    print(f"✓ Testing on first {len(examples)} examples")

    # Load model
    print("\nLoading Qwen2.5-Coder-7B-Instruct...")
    model, tokenizer = load_model()
    print("✓ Model loaded")

    # Results storage
    results = []
    metadata = []

    errors = 0
    total_time = 0.0
    total_checkpoints = 0
    total_pruned = 0

    print(f"\n{'=' * 80}")
    print(f"Processing {len(examples)} examples...")
    print(f"{'=' * 80}\n")

    # Process examples
    for idx, example in enumerate(tqdm(examples, desc="EG-SQL Generation")):
        db_id = example["db_id"]
        question = example["question"]
        gold_sql = example["query"]

        # Get database path
        db_path = spider_data_dir / "database" / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            results.append("")
            metadata.append({
                "index": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": "",
                "error": "Database not found",
                "generation_time_ms": 0,
            })
            errors += 1
            continue

        # Get schema
        schema = get_database_schema(db_path)

        if not schema:
            results.append("")
            metadata.append({
                "index": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": "",
                "error": "Failed to extract schema",
                "generation_time_ms": 0,
            })
            errors += 1
            continue

        # Generate SQL with execution-guided steering
        start_time = time.perf_counter()

        try:
            predicted_sql, gen_metadata = generate_with_steering(
                model, tokenizer, question, schema, str(db_path), num_beams=5
            )
            generation_time = (time.perf_counter() - start_time) * 1000

            # Track statistics
            total_checkpoints += gen_metadata.get("checkpoints", 0)
            total_pruned += gen_metadata.get("pruned_beams", 0)

            metadata.append({
                "index": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": predicted_sql,
                "generation_time_ms": generation_time,
                **gen_metadata
            })

        except Exception as e:
            predicted_sql = ""
            generation_time = (time.perf_counter() - start_time) * 1000
            errors += 1
            metadata.append({
                "index": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": "",
                "error": str(e),
                "generation_time_ms": generation_time,
            })
            results.append("")
            continue

        results.append(predicted_sql)
        total_time += generation_time

    # Save results
    output_file = Path("spider_results_steering_100.txt")
    with open(output_file, "w") as f:
        for sql in results:
            f.write(sql + "\n")

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")

    successful = len(examples) - errors
    print(f"\nTotal examples: {len(examples)}")
    print(f"Successful generations: {successful} ({successful/len(examples)*100:.1f}%)")
    print(f"Failed generations: {errors} ({errors/len(examples)*100:.1f}%)")

    if successful > 0:
        avg_time = total_time / successful
        avg_checkpoints = total_checkpoints / successful
        avg_pruned = total_pruned / successful

        print(f"\nGeneration Statistics:")
        print(f"  Average time: {avg_time:.1f}ms")
        print(f"  Average checkpoints: {avg_checkpoints:.1f}")
        print(f"  Average beams pruned: {avg_pruned:.1f}")
        print(f"\nTotal time: {total_time/1000:.1f}s")

    print(f"\n✓ Results saved to: {output_file}")

    # Save metadata
    metadata_file = Path("results/steering_100_metadata.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, "w") as f:
        json.dump({
            "strategy": "EG-SQL (Execution-Guided Steering)",
            "total_examples": len(examples),
            "successful": successful,
            "failed": errors,
            "avg_generation_time_ms": total_time / successful if successful > 0 else 0,
            "avg_checkpoints": total_checkpoints / successful if successful > 0 else 0,
            "avg_pruned_beams": total_pruned / successful if successful > 0 else 0,
            "total_time_ms": total_time,
            "examples": metadata,
        }, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_file}")

    print(f"\n{'=' * 80}")
    print("Next steps:")
    print("1. Run Spider evaluation:")
    print(f"   cd spider_eval && python evaluation.py \\")
    print(f"     --gold ../dev_gold_100.sql \\")
    print(f"     --pred ../spider_results_steering_100.txt \\")
    print(f"     --db ../data/spider/spider_data/database \\")
    print(f"     --table ../data/spider/spider_data/tables.json \\")
    print(f"     --etype all")
    print("2. Compare with M3 baseline (expected: better accuracy)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
