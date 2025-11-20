"""
Full Spider Benchmark: Run M4 (Cost-Aware Efficiency Guidance) on all 1,034 dev examples.

Saves results in Spider evaluation format (one SQL per line).

Supports --baseline flag for greedy decoding without validation.
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

from tqdm import tqdm

from egtts import ExplainGuidedGenerator, load_model


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


def run_spider_full_benchmark(baseline: bool = False, strategy: str = "M4"):
    """
    Run full Spider benchmark on all 1,034 dev examples.

    Args:
        baseline: If True, use greedy decoding (beam 0 only) without validation.
                  If False, use the specified strategy.
        strategy: Strategy to use - "M3" (validation only) or "M4" (cost-aware)
    """
    if baseline:
        method = "BASELINE (Greedy Decoding)"
    elif strategy == "M3":
        method = "M3 (Validation Only - First Valid Beam)"
    else:
        method = "M4 (Cost-Aware Efficiency Guidance)"
    print("=" * 80)
    print(f"SPIDER FULL BENCHMARK: {method}")
    print("=" * 80)

    # Load Spider data
    spider_data_dir = Path("data/spider/spider_data")
    dev_file = spider_data_dir / "dev.json"

    print(f"\nLoading Spider dev.json...")
    with open(dev_file) as f:
        examples = json.load(f)

    total = len(examples)
    print(f"✓ Loaded {total} examples")

    # Load model
    print("\nLoading Qwen2.5-Coder-7B-Instruct...")
    model, tokenizer = load_model()
    print("✓ Model loaded")

    # Initialize generator
    generator = ExplainGuidedGenerator(model, tokenizer)
    print("✓ Generator initialized")

    # Results storage
    results = []  # List of generated SQL queries (one per line)
    metadata = []  # Detailed metadata for analysis

    errors = 0
    total_time = 0.0

    print(f"\n{'=' * 80}")
    print(f"Processing {total} examples...")
    print(f"{'=' * 80}\n")

    # Process all examples with progress bar
    for idx, example in enumerate(tqdm(examples, desc="Generating SQL")):
        db_id = example["db_id"]
        question = example["question"]
        gold_sql = example["query"]

        # Get database path
        db_path = spider_data_dir / "database" / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            # Database not found - save empty query and continue
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
            # Failed to extract schema - save empty query
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

        # Generate SQL
        start_time = time.perf_counter()

        try:
            if baseline:
                # Baseline: Greedy decoding (beam 0 only, no validation)
                predicted_sql = generator.generate(question, schema)
                generation_time = (time.perf_counter() - start_time) * 1000

                metadata.append({
                    "index": idx,
                    "db_id": db_id,
                    "question": question,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "generation_time_ms": generation_time,
                })
            elif strategy == "M3":
                # M3: Validation only - pick first valid beam (no cost sorting)
                result = generator.generate_with_schema_guidance(
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
                    "generation_time_ms": generation_time,
                })
            else:
                # M4: Cost-aware guidance with beam search
                result = generator.generate_with_cost_guidance(
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
                    "generation_time_ms": generation_time,
                })

        except Exception as e:
            # Generation failed - save empty query
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

        # Save result
        results.append(predicted_sql)
        total_time += generation_time

    # Save results in Spider evaluation format (one SQL per line)
    if baseline:
        suffix = "baseline"
    elif strategy == "M3":
        suffix = "m3"
    else:
        suffix = "m4"
    output_file = Path(f"spider_results_{suffix}.txt")
    with open(output_file, "w") as f:
        for sql in results:
            # Spider evaluation expects one query per line
            # Empty lines for failed generations
            f.write(sql + "\n")

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")

    successful = total - errors
    print(f"\nTotal examples: {total}")
    print(f"Successful generations: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed generations: {errors} ({errors/total*100:.1f}%)")

    if successful > 0:
        avg_time = total_time / successful
        print(f"\nAverage generation time: {avg_time:.1f}ms")
        print(f"Total time: {total_time/1000:.1f}s ({total_time/60000:.1f} minutes)")

    print(f"\n✓ Results saved to: {output_file}")

    # Save detailed metadata
    metadata_file = Path(f"results/spider_full_metadata_{suffix}.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, "w") as f:
        json.dump({
            "total_examples": total,
            "successful": successful,
            "failed": errors,
            "avg_generation_time_ms": total_time / successful if successful > 0 else 0,
            "total_time_ms": total_time,
            "examples": metadata,
        }, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_file}")

    print(f"\n{'=' * 80}")
    print("Next steps:")
    print("1. Use official Spider evaluation script to compute accuracy")
    print("2. Compare with gold queries in dev.json")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spider full benchmark")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use baseline (greedy decoding, no validation)"
    )
    parser.add_argument(
        "--strategy",
        choices=["M3", "M4"],
        default="M4",
        help="Strategy to use: M3 (validation only) or M4 (cost-aware, default)"
    )
    args = parser.parse_args()

    run_spider_full_benchmark(baseline=args.baseline, strategy=args.strategy)
