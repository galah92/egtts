"""
Test M5 (Confidence-Aware Re-ranking) on first 100 Spider dev examples.

This is a quick validation to see if M5 improves over M4.
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from egtts import ExplainGuidedGenerator, load_model


def get_database_schema(db_path: Path) -> str:
    """Extract CREATE TABLE statements from database."""
    import sqlite3

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


def run_m5_test():
    """Run M5 on first 100 Spider dev examples."""
    print("=" * 80)
    print("M5 (CONFIDENCE-AWARE RE-RANKING) - Test on 100 Examples")
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

    # Initialize generator
    generator = ExplainGuidedGenerator(model, tokenizer)
    print("✓ Generator initialized")

    # Results storage
    results = []
    metadata = []

    errors = 0
    total_time = 0.0

    print(f"\n{'=' * 80}")
    print(f"Processing {len(examples)} examples...")
    print(f"{'=' * 80}\n")

    # Process examples
    for idx, example in enumerate(tqdm(examples, desc="Generating SQL (M5)")):
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

        # Generate SQL with M5 (Confidence-Aware Re-ranking)
        start_time = time.perf_counter()

        try:
            result = generator.generate_with_confidence_aware_reranking(
                question, schema, str(db_path), num_beams=5
            )
            predicted_sql = result.sql
            generation_time = (time.perf_counter() - start_time) * 1000

            metadata.append({
                "index": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": predicted_sql,
                "valid": result.valid,
                "beam_selected": result.iterations,
                "error_history": result.error_history,
                "generation_time_ms": generation_time,
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
    output_file = Path("spider_results_m5_100.txt")
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
        print(f"\nAverage generation time: {avg_time:.1f}ms")
        print(f"Total time: {total_time/1000:.1f}s")

    print(f"\n✓ Results saved to: {output_file}")

    # Save metadata
    metadata_file = Path("results/m5_test_100_metadata.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, "w") as f:
        json.dump({
            "strategy": "M5 (Confidence-Aware Re-ranking)",
            "total_examples": len(examples),
            "successful": successful,
            "failed": errors,
            "avg_generation_time_ms": total_time / successful if successful > 0 else 0,
            "total_time_ms": total_time,
            "examples": metadata,
        }, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_file}")

    print(f"\n{'=' * 80}")
    print("Next steps:")
    print("1. Run Spider evaluation on spider_results_m5_100.txt")
    print("2. Compare M5 accuracy vs Baseline/M3/M4 on these 100 examples")
    print("3. Analyze decision reasons in metadata to see when M5 picks efficient vs probable")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_m5_test()
