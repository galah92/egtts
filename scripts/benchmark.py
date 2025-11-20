"""
Unified benchmark script for Text-to-SQL evaluation.

Supports:
- Multiple datasets: Spider, BIRD
- Multiple strategies: Baseline, M3, M4, M5
- Configurable test sizes and parameters
- Automatic evaluation and metadata tracking

Examples:
    # Run M3 on first 100 Spider examples
    python scripts/benchmark.py --dataset spider --strategy M3 --limit 100

    # Run baseline on full Spider dev set
    python scripts/benchmark.py --dataset spider --strategy baseline

    # Run M5 with custom threshold
    python scripts/benchmark.py --dataset spider --strategy M5 --threshold 0.05 --limit 100

    # Run on BIRD dataset
    python scripts/benchmark.py --dataset bird --strategy M3 --limit 50
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

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


def load_dataset(dataset: str, limit: Optional[int] = None) -> tuple[list[dict], Path]:
    """
    Load dataset examples.

    Args:
        dataset: Dataset name ('spider' or 'bird')
        limit: Maximum number of examples to load (None for all)

    Returns:
        Tuple of (examples list, dataset directory path)
    """
    if dataset == "spider":
        data_dir = Path("data/spider/spider_data")
        dev_file = data_dir / "dev.json"
    elif dataset == "bird":
        data_dir = Path("data/bird/bird_data")
        dev_file = data_dir / "dev/dev.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"\nLoading {dataset.upper()} dataset from {dev_file}...")
    with open(dev_file) as f:
        examples = json.load(f)

    if limit:
        examples = examples[:limit]
        print(f"✓ Loaded {len(examples)} examples (limited from {limit})")
    else:
        print(f"✓ Loaded {len(examples)} examples")

    return examples, data_dir


def get_database_path(dataset: str, data_dir: Path, db_id: str) -> Path:
    """Get database path for given dataset and db_id."""
    if dataset == "spider":
        return data_dir / "database" / db_id / f"{db_id}.sqlite"
    elif dataset == "bird":
        return data_dir / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def run_benchmark(
    dataset: str = "spider",
    strategy: str = "M3",
    limit: Optional[int] = None,
    num_beams: int = 5,
    threshold: float = -0.22,
):
    """
    Run benchmark on specified dataset with given strategy.

    Args:
        dataset: Dataset name ('spider' or 'bird')
        strategy: Strategy to use ('baseline', 'M3', 'M4', 'M5')
        limit: Maximum number of examples (None for all)
        num_beams: Number of beams for beam search (default: 5)
        threshold: Confidence threshold for M5 strategy (default: -0.22)
    """
    # Setup
    strategy_names = {
        "baseline": "Baseline (Greedy Decoding)",
        "M3": "M3 (Validation Only - First Valid Beam)",
        "M4": "M4 (Cost-Aware Efficiency Guidance)",
        "M5": f"M5 (Confidence-Aware Re-ranking, threshold={threshold})",
    }

    print("=" * 80)
    print(f"{dataset.upper()} BENCHMARK: {strategy_names[strategy]}")
    print("=" * 80)

    # Load dataset
    examples, data_dir = load_dataset(dataset, limit)
    total = len(examples)

    # Load model
    print("\nLoading Qwen2.5-Coder-7B-Instruct...")
    model, tokenizer = load_model()
    print("✓ Model loaded")

    # Initialize generator
    generator = ExplainGuidedGenerator(model, tokenizer)
    print("✓ Generator initialized")

    # Results storage
    results = []  # Generated SQL queries
    metadata = []  # Detailed metadata

    errors = 0
    total_time = 0.0

    print(f"\n{'=' * 80}")
    print(f"Processing {total} examples with strategy={strategy}...")
    print(f"{'=' * 80}\n")

    # Process examples
    for idx, example in enumerate(tqdm(examples, desc=f"{strategy} Generation")):
        db_id = example["db_id"]
        question = example["question"]
        gold_sql = example.get("query", example.get("SQL", ""))  # BIRD uses 'SQL'

        # Get database path
        db_path = get_database_path(dataset, data_dir, db_id)

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

        # Generate SQL
        start_time = time.perf_counter()

        try:
            if strategy == "baseline":
                # Baseline: Greedy decoding
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
                # M3: Validation only
                result = generator.generate_with_schema_guidance(
                    question, schema, str(db_path), num_beams=num_beams
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

            elif strategy == "M4":
                # M4: Cost-aware
                result = generator.generate_with_cost_guidance(
                    question, schema, str(db_path), num_beams=num_beams
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

            elif strategy == "M5":
                # M5: Confidence-aware re-ranking
                result = generator.generate_with_confidence_aware_reranking(
                    question, schema, str(db_path),
                    num_beams=num_beams,
                    confidence_threshold=threshold
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

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

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
    suffix = f"{dataset}_{strategy.lower()}"
    if limit:
        suffix += f"_{limit}"
    if strategy == "M5":
        threshold_str = f"t{int(threshold * 100):03d}" if threshold >= 0 else f"tn{int(abs(threshold) * 100):03d}"
        suffix += f"_{threshold_str}"

    output_file = Path(f"results/{suffix}_predictions.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for sql in results:
            f.write(sql + "\n")

    # Print summary
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

    # Save metadata
    metadata_file = Path(f"results/{suffix}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump({
            "dataset": dataset,
            "strategy": strategy,
            "num_beams": num_beams,
            "threshold": threshold if strategy == "M5" else None,
            "total_examples": total,
            "successful": successful,
            "failed": errors,
            "avg_generation_time_ms": total_time / successful if successful > 0 else 0,
            "total_time_ms": total_time,
            "examples": metadata,
        }, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_file}")

    # Next steps
    print(f"\n{'=' * 80}")
    print("Next steps:")
    if dataset == "spider":
        print("Run evaluation:")
        print(f"  uv run python spider_eval/evaluation.py \\")
        print(f"    --gold dev_gold.sql \\")
        print(f"    --pred {output_file} \\")
        print(f"    --db data/spider/spider_data/database \\")
        print(f"    --table data/spider/spider_data/tables.json \\")
        print(f"    --etype all")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark for Text-to-SQL evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run M3 on first 100 Spider examples
  python scripts/benchmark.py --dataset spider --strategy M3 --limit 100

  # Run baseline on full Spider dev set
  python scripts/benchmark.py --dataset spider --strategy baseline

  # Run M5 with custom threshold
  python scripts/benchmark.py --dataset spider --strategy M5 --threshold 0.05 --limit 100

  # Run on BIRD dataset
  python scripts/benchmark.py --dataset bird --strategy M3 --limit 50
        """
    )

    parser.add_argument(
        "--dataset",
        choices=["spider", "bird"],
        default="spider",
        help="Dataset to use (default: spider)"
    )

    parser.add_argument(
        "--strategy",
        choices=["baseline", "M3", "M4", "M5"],
        default="M3",
        help="Strategy to use (default: M3)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (default: all)"
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams for beam search (default: 5)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.22,
        help="Confidence threshold for M5 strategy (default: -0.22)"
    )

    args = parser.parse_args()

    run_benchmark(
        dataset=args.dataset,
        strategy=args.strategy,
        limit=args.limit,
        num_beams=args.num_beams,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
