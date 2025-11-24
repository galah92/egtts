#!/usr/bin/env python3
"""
BIRD Benchmark for M15: Incremental Consensus Generation (MAKER).

This script evaluates the M15 strategy which decomposes SQL generation into
3 atomic phases with majority voting at each stage:

Phase 1: Table Selection (FROM/JOIN) - Vote on required tables, validate with EXPLAIN
Phase 2: Filter Generation (WHERE) - Vote on filter conditions
Phase 3: Projection (SELECT/GROUP BY) - Vote on final query structure

Key hypothesis: By locking in table choices via consensus first, we prevent
downstream column hallucinations that plague M10.

Configuration:
- N=12 samples per phase (36 total calls per query)
- Uses M10's augmented schema for grounding
- Fallback to M10 if Phase 1 fails to find valid join path

Comparison metric: Accuracy vs M10 (62% on 50 examples)
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from egtts import load_model
from egtts.maker import IncrementalGenerator, M15Result


def load_bird_mini_dev(data_dir: Path) -> list[dict]:
    """Load BIRD Mini-Dev dataset."""
    json_path = data_dir / "mini_dev_sqlite.json"
    with open(json_path) as f:
        data = json.load(f)
    return data


def get_bird_database_path(db_id: str, data_dir: Path) -> Path:
    """Get path to BIRD database."""
    db_path = data_dir / "dev_databases" / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    return db_path


def execute_sql_with_timeout(sql: str, db_path: Path, timeout: float = 30.0) -> tuple[Any, float, str]:
    """
    Execute SQL query and measure execution time.

    Returns:
        (results, execution_time_ms, error_message)
    """
    start_time = time.perf_counter()
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")

        def progress_handler():
            if time.perf_counter() - start_time > timeout:
                return 1
            return 0

        conn.set_progress_handler(progress_handler, 1000)

        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()

        execution_time = (time.perf_counter() - start_time) * 1000
        return results, execution_time, ""

    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            execution_time = (time.perf_counter() - start_time) * 1000
            return None, execution_time, f"Timeout after {timeout}s"
        execution_time = (time.perf_counter() - start_time) * 1000
        return None, execution_time, str(e)
    except Exception as e:
        execution_time = (time.perf_counter() - start_time) * 1000
        return None, execution_time, str(e)


def check_correctness(pred_results: Any, gold_results: Any) -> bool:
    """Check if predicted results match gold results (set equality)."""
    if pred_results is None or gold_results is None:
        return False
    return set(pred_results) == set(gold_results)


def run_m15_strategy(
    generator: IncrementalGenerator,
    question: str,
    db_path: Path,
    num_samples_per_phase: int = 12
) -> tuple[str, dict]:
    """
    Run M15 strategy with 3-phase incremental consensus.

    Args:
        generator: IncrementalGenerator instance
        question: Natural language question
        db_path: Database path
        num_samples_per_phase: Samples per phase (default: 12)

    Returns:
        Tuple of (sql, metadata_dict)
    """
    result = generator.generate(question, str(db_path), num_samples_per_phase)

    metadata = {
        "strategy": "M15_IncrementalConsensus",
        "total_latency_ms": result.total_latency_ms,
        "fallback_used": result.fallback_used,
        "phase1_tables": result.phase1.tables,
        "phase1_votes": result.phase1.votes,
        "phase1_valid": result.phase1.valid,
        "phase2_where": result.phase2.where_clause,
        "phase2_votes": result.phase2.votes,
        "phase3_votes": result.phase3.votes,
        "phase3_valid": result.phase3.valid,
        "total_samples": num_samples_per_phase * 3,
    }

    return result.sql, metadata


def run_m10_fallback(generator, question: str, db_path: Path) -> tuple[str, dict]:
    """
    Run M10 strategy as fallback comparison.

    This uses the standard augmented schema + plan voting approach.
    """
    from egtts.guided import ExplainGuidedGenerator

    # Wrap in ExplainGuidedGenerator for M10 method
    m10_gen = ExplainGuidedGenerator(generator.model, generator.tokenizer)

    result = m10_gen.generate_with_augmented_schema(
        question=question,
        db_path=str(db_path),
        num_samples=15
    )

    metadata = {
        "strategy": "M10_Fallback",
        "latency_ms": result.latency_ms,
        "valid": result.valid,
        "iterations": result.iterations,
    }

    return result.sql, metadata


def main():
    parser = argparse.ArgumentParser(description="BIRD Benchmark for M15 Strategy")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/bird"),
        help="Path to BIRD data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of examples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--samples-per-phase",
        type=int,
        default=12,
        help="Number of samples per phase (default: 12)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Model name (default: Qwen2.5-Coder-7B)"
    )
    parser.add_argument(
        "--compare-m10",
        action="store_true",
        help="Also run M10 for comparison"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)

    # Create generator
    generator = IncrementalGenerator(model, tokenizer, temperature=0.7)

    # Load BIRD dataset
    print(f"Loading BIRD Mini-Dev from {args.data_dir}")
    dataset = load_bird_mini_dev(args.data_dir)

    # Limit examples
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"Evaluating first {args.limit} examples")

    # Run evaluation
    results = []
    correct_m15 = 0
    correct_m10 = 0
    failed_m15 = 0

    print(f"\nRunning M15 evaluation with {args.samples_per_phase} samples per phase...")

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        question = example["question"]
        db_id = example["db_id"]
        gold_sql = example["SQL"]

        # Get database path
        db_path = get_bird_database_path(db_id, args.data_dir)

        # Execute gold query
        gold_results, gold_time_ms, gold_error = execute_sql_with_timeout(gold_sql, db_path)

        if gold_error:
            print(f"\nWarning: Gold query failed for example {idx}: {gold_error}")
            continue

        # Run M15
        try:
            pred_sql_m15, metadata_m15 = run_m15_strategy(
                generator, question, db_path, args.samples_per_phase
            )

            if not pred_sql_m15:
                failed_m15 += 1
                pred_results_m15 = None
                pred_error_m15 = "No SQL generated"
            else:
                pred_results_m15, pred_time_ms, pred_error_m15 = execute_sql_with_timeout(
                    pred_sql_m15, db_path
                )

            correct_m15_flag = check_correctness(pred_results_m15, gold_results)
            if correct_m15_flag:
                correct_m15 += 1

        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            failed_m15 += 1
            pred_sql_m15 = ""
            pred_error_m15 = str(e)
            correct_m15_flag = False
            metadata_m15 = {}

        # Optionally run M10 for comparison
        pred_sql_m10 = None
        correct_m10_flag = False
        metadata_m10 = {}

        if args.compare_m10:
            try:
                pred_sql_m10, metadata_m10 = run_m10_fallback(generator, question, db_path)
                pred_results_m10, _, _ = execute_sql_with_timeout(pred_sql_m10, db_path)
                correct_m10_flag = check_correctness(pred_results_m10, gold_results)
                if correct_m10_flag:
                    correct_m10 += 1
            except Exception as e:
                print(f"\nM10 error on example {idx}: {e}")

        # Record result
        result_entry = {
            "idx": idx,
            "question": question,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "pred_sql_m15": pred_sql_m15,
            "correct_m15": correct_m15_flag,
            "metadata_m15": metadata_m15,
        }

        if args.compare_m10:
            result_entry["pred_sql_m10"] = pred_sql_m10
            result_entry["correct_m10"] = correct_m10_flag
            result_entry["metadata_m10"] = metadata_m10

        results.append(result_entry)

    # Calculate metrics
    total = len(results)
    accuracy_m15 = correct_m15 / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("M15 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total examples: {total}")
    print(f"M15 Correct: {correct_m15}")
    print(f"M15 Failed: {failed_m15}")
    print(f"M15 Accuracy: {accuracy_m15:.1%}")

    if args.compare_m10:
        accuracy_m10 = correct_m10 / total if total > 0 else 0
        improvement = accuracy_m15 - accuracy_m10
        print(f"\nM10 Correct: {correct_m10}")
        print(f"M10 Accuracy: {accuracy_m10:.1%}")
        print(f"Improvement: {improvement:+.1%}")

    # Save results
    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_file = args.output_dir / f"bird_m15_{args.limit}.json"

    output_data = {
        "config": {
            "strategy": "M15_IncrementalConsensus",
            "model": args.model_name,
            "limit": args.limit,
            "samples_per_phase": args.samples_per_phase,
            "total_samples_per_query": args.samples_per_phase * 3,
            "compare_m10": args.compare_m10,
        },
        "metrics": {
            "total": total,
            "correct_m15": correct_m15,
            "failed_m15": failed_m15,
            "accuracy_m15": accuracy_m15,
        },
        "results": results,
    }

    if args.compare_m10:
        output_data["metrics"]["correct_m10"] = correct_m10
        output_data["metrics"]["accuracy_m10"] = accuracy_m10
        output_data["metrics"]["improvement"] = improvement

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
