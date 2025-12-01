#!/usr/bin/env python3
"""
BIRD VES (Valid Efficiency Score) Benchmark

Evaluates SQL generation strategies on the BIRD Mini-Dev benchmark.

Strategies:
- baseline: Greedy decoding
- M4: Cost-aware beam selection
- M7: Plan-based majority voting
- M8: Massive diversity (32 samples)
- M10: Schema augmentation + plan voting (best)
- M12: Execution-based self-correction

Usage:
    uv run python scripts/run_bird_ves.py --strategy M10 --limit 50
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, TypedDict

from tqdm import tqdm

from egtts import load_model
from egtts.guided import ExplainGuidedGenerator
from egtts.model import create_sql_prompt, generate_sql


class BenchmarkResults(TypedDict):
    strategy: str
    total: int
    correct: int
    incorrect: int
    failed: int
    total_ves: float
    examples: list[dict[str, Any]]
    accuracy: float
    avg_ves: float


def load_bird_mini_dev(data_dir: Path) -> list[dict]:
    """Load BIRD Mini-Dev dataset."""
    json_path = data_dir / "mini_dev_sqlite.json"
    with open(json_path) as f:
        return json.load(f)


def get_bird_database_path(db_id: str, data_dir: Path) -> Path:
    """Get path to BIRD database."""
    db_path = data_dir / "dev_databases" / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    return db_path


def extract_schema(db_path: Path) -> str:
    """Extract schema from database as CREATE TABLE statements."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        cursor.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';"
        )
        create_stmt = cursor.fetchone()[0]
        schema_parts.append(create_stmt)

    conn.close()
    return "\n\n".join(schema_parts)


def execute_sql_with_timeout(
    sql: str, db_path: Path, timeout: float = 30.0
) -> tuple[Any, float, str]:
    """Execute SQL query and measure execution time."""
    start_time = time.perf_counter()
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")

        def progress_handler():
            return 1 if time.perf_counter() - start_time > timeout else 0

        conn.set_progress_handler(progress_handler, 1000)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()

        return results, (time.perf_counter() - start_time) * 1000, ""

    except sqlite3.OperationalError as e:
        exec_time = (time.perf_counter() - start_time) * 1000
        if "interrupted" in str(e):
            return None, exec_time, f"Timeout after {timeout}s"
        return None, exec_time, str(e)
    except Exception as e:
        return None, (time.perf_counter() - start_time) * 1000, str(e)


def calculate_ves(
    pred_results: Any, gold_results: Any, pred_time_ms: float, gold_time_ms: float
) -> float:
    """Calculate Valid Efficiency Score (VES)."""
    if pred_results is None or gold_results is None:
        return 0.0
    if set(pred_results) != set(gold_results):
        return 0.0
    if pred_time_ms == 0:
        return 0.0
    return (gold_time_ms / pred_time_ms) ** 0.5


# Strategy runners


def run_baseline(
    generator: ExplainGuidedGenerator, question: str, schema: str, evidence: str
) -> tuple[str, dict]:
    """Baseline: Greedy decoding."""
    start_time = time.perf_counter()
    prompt = create_sql_prompt(
        question, schema, generator.tokenizer, evidence=evidence if evidence else None
    )
    sql = generate_sql(
        generator.model,
        generator.tokenizer,
        prompt,
        max_new_tokens=512,
        do_sample=False,
    )
    return sql, {
        "strategy": "baseline",
        "generation_time_ms": (time.perf_counter() - start_time) * 1000,
    }


def run_m4(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
) -> tuple[str, dict]:
    """M4: Cost-aware beam selection."""
    start_time = time.perf_counter()
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question
    result = generator.generate_with_cost_guidance(
        question_with_hint, schema, str(db_path), num_beams=5
    )
    return result.sql, {
        "strategy": "M4",
        "generation_time_ms": (time.perf_counter() - start_time) * 1000,
        "valid": result.valid,
    }


def run_m7(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
) -> tuple[str, dict]:
    """M7: Plan-based majority voting."""
    start_time = time.perf_counter()
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question
    result = generator.generate_with_plan_voting(
        question_with_hint, schema, str(db_path), num_samples=15
    )

    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast

            try:
                vote_stats = ast.literal_eval(str(entry).replace("Vote stats: ", ""))
            except (ValueError, SyntaxError):
                pass

    return result.sql, {
        "strategy": "M7",
        "generation_time_ms": (time.perf_counter() - start_time) * 1000,
        "valid": result.valid,
        "votes": vote_stats.get("winning_votes", 0),
        "valid_candidates": vote_stats.get("valid_candidates", 0),
    }


def run_m8(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
) -> tuple[str, dict]:
    """M8: Massive diversity plan-bagging."""
    start_time = time.perf_counter()
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question
    result = generator.generate_with_massive_diversity(
        question_with_hint, schema, str(db_path), num_samples=32
    )

    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast

            try:
                vote_stats = ast.literal_eval(str(entry).replace("Vote stats: ", ""))
            except (ValueError, SyntaxError):
                pass

    return result.sql, {
        "strategy": "M8",
        "generation_time_ms": (time.perf_counter() - start_time) * 1000,
        "valid": result.valid,
        "votes": vote_stats.get("winning_votes", 0),
        "valid_candidates": vote_stats.get("valid_candidates", 0),
    }


def run_m10(
    generator: ExplainGuidedGenerator, question: str, evidence: str, db_path: Path
) -> tuple[str, dict]:
    """M10: Schema augmentation + plan voting (best strategy)."""
    start_time = time.perf_counter()
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question
    result = generator.generate_with_augmented_schema(
        question_with_hint, str(db_path), num_samples=15
    )

    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast

            try:
                vote_stats = ast.literal_eval(str(entry).replace("Vote stats: ", ""))
            except (ValueError, SyntaxError):
                pass

    return result.sql, {
        "strategy": "M10",
        "generation_time_ms": (time.perf_counter() - start_time) * 1000,
        "valid": result.valid,
        "votes": vote_stats.get("winning_votes", 0),
        "valid_candidates": vote_stats.get("valid_candidates", 0),
    }


def run_m12(
    generator: ExplainGuidedGenerator, question: str, evidence: str, db_path: Path
) -> tuple[str, dict]:
    """M12: Execution-based self-correction."""
    start_time = time.perf_counter()
    result = generator.generate_with_execution_correction(
        question, str(db_path), hint=evidence, num_samples=15, max_corrections=2
    )

    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast

            try:
                vote_stats = ast.literal_eval(str(entry).replace("Vote stats: ", ""))
            except (ValueError, SyntaxError):
                pass

    return result.sql, {
        "strategy": "M12",
        "generation_time_ms": (time.perf_counter() - start_time) * 1000,
        "valid": result.valid,
        "corrections_needed": vote_stats.get("corrections_needed", 0),
    }


def run_benchmark(
    data_dir: Path,
    strategy: str,
    limit: int | None,
    output_dir: Path,
    model_name: str,
    quantization: str | None,
) -> BenchmarkResults:
    """Run the VES benchmark."""
    # Load model
    model, tokenizer = load_model(model_name, quantization=quantization)
    generator = ExplainGuidedGenerator(model, tokenizer)

    # Load dataset
    data = load_bird_mini_dev(data_dir)
    if limit:
        data = data[:limit]

    print(f"Running {strategy} on {len(data)} examples")

    total = len(data)
    correct = 0
    incorrect = 0
    failed = 0
    total_ves = 0.0
    examples: list[dict[str, Any]] = []

    for idx, example in enumerate(tqdm(data, desc=f"Running {strategy}")):
        db_id = example["db_id"]
        question = example["question"]
        evidence = example.get("evidence", "")
        gold_sql = example["SQL"]

        try:
            db_path = get_bird_database_path(db_id, data_dir)
            schema = extract_schema(db_path)
        except FileNotFoundError as e:
            failed += 1
            examples.append({"index": idx, "error": str(e)})
            continue

        # Execute gold query
        gold_results, gold_time_ms, gold_error = execute_sql_with_timeout(
            gold_sql, db_path
        )
        if gold_error:
            failed += 1
            examples.append({"index": idx, "error": f"Gold query failed: {gold_error}"})
            continue

        # Generate prediction based on strategy
        if strategy == "baseline":
            pred_sql, metadata = run_baseline(generator, question, schema, evidence)
        elif strategy == "M4":
            pred_sql, metadata = run_m4(generator, question, schema, evidence, db_path)
        elif strategy == "M7":
            pred_sql, metadata = run_m7(generator, question, schema, evidence, db_path)
        elif strategy == "M8":
            pred_sql, metadata = run_m8(generator, question, schema, evidence, db_path)
        elif strategy == "M10":
            pred_sql, metadata = run_m10(generator, question, evidence, db_path)
        elif strategy == "M12":
            pred_sql, metadata = run_m12(generator, question, evidence, db_path)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Execute prediction
        pred_results, pred_time_ms, pred_error = execute_sql_with_timeout(
            pred_sql, db_path
        )

        # Calculate VES
        ves = calculate_ves(pred_results, gold_results, pred_time_ms, gold_time_ms)
        total_ves += ves

        if pred_error or ves == 0:
            incorrect += 1
            correctness = "error" if pred_error else "incorrect"
        else:
            correct += 1
            correctness = "correct"

        examples.append(
            {
                "index": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": pred_sql,
                "ves": ves,
                "correctness": correctness,
                "generation_time_ms": metadata.get("generation_time_ms", 0),
            }
        )

    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0.0
    avg_ves = total_ves / total if total > 0 else 0.0

    results: BenchmarkResults = {
        "strategy": strategy,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "failed": failed,
        "total_ves": total_ves,
        "examples": examples,
        "accuracy": accuracy,
        "avg_ves": avg_ves,
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"bird_ves_{strategy}_{total}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults for {strategy}:")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Avg VES: {avg_ves:.3f}")
    print(f"  Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run BIRD VES benchmark")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/bird"),
        help="Path to BIRD data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="M10",
        choices=["baseline", "M4", "M7", "M8", "M10", "M12"],
        help="Strategy to evaluate",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of examples"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Model to use",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="Quantization mode",
    )

    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        strategy=args.strategy,
        limit=args.limit,
        output_dir=args.output_dir,
        model_name=args.model,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()
