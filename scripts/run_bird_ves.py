#!/usr/bin/env python3
"""
BIRD VES (Valid Efficiency Score) Benchmark

This script evaluates SQL generation strategies on the BIRD Mini-Dev benchmark,
focusing on the Valid Efficiency Score (VES) metric which measures both
correctness and execution efficiency.

VES Calculation:
- If incorrect: VES = 0
- If correct: VES = sqrt(T_gold / T_pred)
  where T_gold is gold query execution time and T_pred is predicted query time

Strategies tested:
- Baseline: Greedy decoding (beam 0)
- M4: Cost-aware re-ranking (prefer index seeks over scans)
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
from egtts.database import explain_query
from egtts.guided import ExplainGuidedGenerator
from egtts.model import create_sql_prompt
from egtts.schema import build_schema_index


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


def extract_schema(db_path: Path) -> str:
    """Extract schema from database as CREATE TABLE statements."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    # Get CREATE statements for each table
    schema_parts = []
    for table in tables:
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
        create_stmt = cursor.fetchone()[0]
        schema_parts.append(create_stmt)

    conn.close()
    return "\n\n".join(schema_parts)


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
        
        # Use progress handler to check for timeout
        # SQLite's PRAGMA busy_timeout only handles lock waits, not execution time.
        # We need to interrupt long-running queries (e.g. infinite loops) to prevent hangs
        # during VES benchmarking where we must execute the query to measure performance.
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


def calculate_ves(
    pred_results: Any,
    gold_results: Any,
    pred_time_ms: float,
    gold_time_ms: float,
) -> float:
    """
    Calculate Valid Efficiency Score (VES).

    VES = 0 if incorrect
    VES = sqrt(T_gold / T_pred) if correct (allows >1.0 if faster than gold)
    """
    # Check correctness (set equality, ignore order)
    if pred_results is None or gold_results is None:
        return 0.0

    if set(pred_results) != set(gold_results):
        return 0.0

    # Calculate efficiency score
    if pred_time_ms == 0:
        return 0.0

    ratio = gold_time_ms / pred_time_ms
    return ratio ** 0.5  # sqrt for VES


def run_baseline(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    few_shot_examples: list = None,
) -> tuple[str, dict]:
    """Run baseline strategy (greedy decoding)."""
    start_time = time.perf_counter()

    # Create prompt with proper structure
    prompt = create_sql_prompt(
        question=question,
        schema=schema,
        tokenizer=generator.tokenizer,
        evidence=evidence if evidence else None,
        few_shot_examples=few_shot_examples
    )

    from egtts.model import generate_sql
    sql = generate_sql(
        generator.model,
        generator.tokenizer,
        prompt,
        max_new_tokens=512,
        do_sample=False
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    metadata = {
        "strategy": "baseline" if not few_shot_examples else f"few_shot_{len(few_shot_examples)}",
        "generation_time_ms": generation_time,
        "num_beams": 1,
        "num_few_shot": len(few_shot_examples) if few_shot_examples else 0,
    }

    return sql, metadata


def run_m4_cost_aware(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
    num_beams: int = 5,
) -> tuple[str, dict]:
    """Run M4 strategy (cost-aware re-ranking)."""
    start_time = time.perf_counter()

    # Create prompt with evidence hint
    prompt_text = f"Question: {question}\nHint: {evidence}\n\nSchema:\n{schema}"

    # Generate using M4 cost-aware strategy
    result = generator.generate_with_cost_guidance(
        prompt_text,
        "",
        str(db_path),
        num_beams=num_beams
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "M4",
        "valid": result.valid,
        "iterations": result.iterations,
        "latency_ms": result.latency_ms,
    }

    return result.sql, metadata


def run_explain_feedback(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
) -> tuple[str, dict]:
    """Run explain_feedback strategy (EXPLAIN output pushed to prompt for optimization)."""
    start_time = time.perf_counter()

    # Create prompt with evidence hint in question
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question

    # Generate using explain_feedback strategy
    result = generator.generate_with_explain_feedback(
        question_with_hint,
        schema,
        str(db_path),
        max_iterations=2
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "explain_feedback",
        "valid": result.valid,
        "iterations": result.iterations,
        "latency_ms": result.latency_ms,
        "error_history": result.error_history,
    }

    return result.sql, metadata


def run_plan_voting(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
    num_samples: int = 15,  # Increased for better consensus
) -> tuple[str, dict]:
    """Run M7 strategy (Plan-Based Majority Voting for accuracy)."""
    start_time = time.perf_counter()

    # Create prompt with evidence hint in question
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question

    # Generate using plan voting strategy
    result = generator.generate_with_plan_voting(
        question_with_hint,
        schema,
        str(db_path),
        num_samples=num_samples
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    # Parse vote stats from error_history for consensus analysis
    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            # Extract the dict part
            import ast
            try:
                stats_str = str(entry).replace("Vote stats: ", "")
                vote_stats = ast.literal_eval(stats_str)
            except (ValueError, SyntaxError):
                pass

    # Calculate consensus confidence
    valid_candidates = vote_stats.get("valid_candidates", num_samples)
    winning_votes = vote_stats.get("winning_votes", 0)
    consensus_confidence = winning_votes / valid_candidates if valid_candidates > 0 else 0

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "M7",
        "valid": result.valid,
        "votes": winning_votes,
        "num_samples": num_samples,
        "valid_candidates": valid_candidates,
        "consensus_confidence": consensus_confidence,
        "unique_signatures": vote_stats.get("unique_signatures", 0),
        "latency_ms": result.latency_ms,
    }

    return result.sql, metadata


def run_massive_diversity(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
    num_samples: int = 32,
) -> tuple[str, dict]:
    """Run M8 strategy (Massive Diversity Plan-Bagging)."""
    start_time = time.perf_counter()

    # Create prompt with evidence hint in question
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question

    # Generate using massive diversity strategy
    result = generator.generate_with_massive_diversity(
        question_with_hint,
        schema,
        str(db_path),
        num_samples=num_samples
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    # Parse vote stats from error_history
    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast
            try:
                stats_str = str(entry).replace("Vote stats: ", "")
                vote_stats = ast.literal_eval(stats_str)
            except (ValueError, SyntaxError):
                pass

    # Calculate consensus confidence
    valid_candidates = vote_stats.get("valid_candidates", num_samples)
    winning_votes = vote_stats.get("winning_votes", 0)
    consensus_confidence = winning_votes / valid_candidates if valid_candidates > 0 else 0

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "M8",
        "valid": result.valid,
        "votes": winning_votes,
        "num_samples": num_samples,
        "valid_candidates": valid_candidates,
        "consensus_confidence": consensus_confidence,
        "unique_signatures": vote_stats.get("unique_signatures", 0),
        "syntax_errors": vote_stats.get("syntax_errors", 0),
        "schema_errors": vote_stats.get("schema_errors", 0),
        "latency_ms": result.latency_ms,
    }

    return result.sql, metadata


def run_few_shot_simulation(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
    num_samples: int = 32,
) -> tuple[str, dict]:
    """Run M9 strategy (Few-Shot + Simulation Filter)."""
    start_time = time.perf_counter()

    # Generate using few-shot + simulation strategy
    result = generator.generate_with_few_shot_and_simulation(
        question,
        schema,
        str(db_path),
        hint=evidence,
        num_samples=num_samples
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    # Parse vote stats from error_history
    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast
            try:
                stats_str = str(entry).replace("Vote stats: ", "")
                vote_stats = ast.literal_eval(stats_str)
            except (ValueError, SyntaxError):
                pass

    # Calculate consensus confidence
    valid_candidates = vote_stats.get("valid_candidates", num_samples)
    winning_votes = vote_stats.get("winning_votes", 0)
    consensus_confidence = winning_votes / valid_candidates if valid_candidates > 0 else 0

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "M9",
        "valid": result.valid,
        "votes": winning_votes,
        "num_samples": num_samples,
        "valid_candidates": valid_candidates,
        "consensus_confidence": consensus_confidence,
        "unique_signatures": vote_stats.get("unique_signatures", 0),
        "filtered_by_simulation": vote_stats.get("filtered_by_simulation", 0),
        "expects_singular": vote_stats.get("expects_singular", False),
        "latency_ms": result.latency_ms,
    }

    return result.sql, metadata


def run_augmented_schema_with_probing(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
    num_samples: int = 15,
) -> tuple[str, dict]:
    """Run M13 strategy (Schema Augmentation + Data Probing)."""
    start_time = time.perf_counter()

    # Create prompt with evidence hint in question
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question

    # Generate using augmented schema + probing strategy
    result = generator.generate_with_augmented_schema_and_probing(
        question_with_hint,
        str(db_path),
        num_samples=num_samples
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    # Parse vote stats from error_history
    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast
            try:
                stats_str = str(entry).replace("Vote stats: ", "")
                vote_stats = ast.literal_eval(stats_str)
            except (ValueError, SyntaxError):
                pass

    # Calculate consensus confidence
    valid_candidates = vote_stats.get("valid_candidates", num_samples)
    winning_votes = vote_stats.get("winning_votes", 0)
    consensus_confidence = winning_votes / valid_candidates if valid_candidates > 0 else 0

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "M13",
        "valid": result.valid,
        "votes": winning_votes,
        "num_samples": num_samples,
        "valid_candidates": valid_candidates,
        "consensus_confidence": consensus_confidence,
        "unique_signatures": vote_stats.get("unique_signatures", 0),
        "probing_stats": vote_stats.get("probing_stats", {}),
        "latency_ms": result.latency_ms,
    }

    return result.sql, metadata


def run_cot(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
    num_samples: int = 16,
) -> tuple[str, dict]:
    """Run M11 strategy (Chain-of-Thought with Augmented Schema)."""
    start_time = time.perf_counter()

    # Create prompt with evidence hint in question
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question

    # Generate using CoT strategy
    result = generator.generate_with_cot(
        question_with_hint,
        str(db_path),
        hint=evidence,
        num_samples=num_samples
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    # Parse vote stats from error_history
    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast
            try:
                stats_str = str(entry).replace("Vote stats: ", "")
                vote_stats = ast.literal_eval(stats_str)
            except (ValueError, SyntaxError):
                pass

    # Calculate consensus confidence
    valid_candidates = vote_stats.get("valid_candidates", num_samples)
    winning_votes = vote_stats.get("winning_votes", 0)
    consensus_confidence = winning_votes / valid_candidates if valid_candidates > 0 else 0

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "M11",
        "valid": result.valid,
        "votes": winning_votes,
        "num_samples": num_samples,
        "valid_candidates": valid_candidates,
        "consensus_confidence": consensus_confidence,
        "unique_signatures": vote_stats.get("unique_signatures", 0),
        "schema_augmentation_time_ms": vote_stats.get("schema_augmentation_time_ms", 0),
        "latency_ms": result.latency_ms,
    }

    return result.sql, metadata


def run_dataflow(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
    num_samples: int = 16,
) -> tuple[str, dict]:
    """Run M14 strategy (Data Flow Chain-of-Thought)."""
    start_time = time.perf_counter()

    # Generate using dataflow strategy
    result = generator.generate_with_dataflow(
        question,
        str(db_path),
        hint=evidence,
        num_samples=num_samples,
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    # Parse vote stats from error_history
    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast
            try:
                stats_str = str(entry).replace("Vote stats: ", "")
                vote_stats = ast.literal_eval(stats_str)
            except (ValueError, SyntaxError):
                pass

    # Calculate consensus confidence
    valid_candidates = vote_stats.get("valid_candidates", num_samples)
    winning_votes = vote_stats.get("winning_votes", 0)
    consensus_confidence = winning_votes / valid_candidates if valid_candidates > 0 else 0

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "M14",
        "valid": result.valid,
        "votes": winning_votes,
        "num_samples": num_samples,
        "valid_candidates": valid_candidates,
        "consensus_confidence": consensus_confidence,
        "schema_augmentation_time_ms": vote_stats.get("schema_augmentation_time_ms", 0),
        "latency_ms": result.latency_ms,
    }

    return result.sql, metadata


def run_execution_correction(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
    num_samples: int = 15,
) -> tuple[str, dict]:
    """Run M12 strategy (Execution-Based Self-Correction)."""
    start_time = time.perf_counter()

    # Generate using execution correction strategy
    result = generator.generate_with_execution_correction(
        question,
        str(db_path),
        hint=evidence,
        num_samples=num_samples,
        max_corrections=2,
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    # Parse vote stats from error_history
    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast
            try:
                stats_str = str(entry).replace("Vote stats: ", "")
                vote_stats = ast.literal_eval(stats_str)
            except (ValueError, SyntaxError):
                pass

    # Calculate consensus confidence
    valid_candidates = vote_stats.get("valid_candidates", num_samples)
    winning_votes = vote_stats.get("winning_votes", 0)
    consensus_confidence = winning_votes / valid_candidates if valid_candidates > 0 else 0

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "M12",
        "valid": result.valid,
        "votes": winning_votes,
        "num_samples": num_samples,
        "valid_candidates": valid_candidates,
        "consensus_confidence": consensus_confidence,
        "corrections_needed": vote_stats.get("corrections_needed", 0),
        "result_rows": vote_stats.get("result_rows", 0),
        "latency_ms": result.latency_ms,
    }

    return result.sql, metadata


def run_augmented_schema(
    generator: ExplainGuidedGenerator,
    question: str,
    schema: str,
    evidence: str,
    db_path: Path,
    num_samples: int = 15,
) -> tuple[str, dict]:
    """Run M10 strategy (Schema Augmentation with sample data)."""
    start_time = time.perf_counter()

    # Create prompt with evidence hint in question
    question_with_hint = f"{question}\nHint: {evidence}" if evidence else question

    # Generate using augmented schema strategy
    result = generator.generate_with_augmented_schema(
        question_with_hint,
        str(db_path),
        num_samples=num_samples
    )

    generation_time = (time.perf_counter() - start_time) * 1000

    # Parse vote stats from error_history
    vote_stats = {}
    for entry in result.error_history:
        if "Vote stats:" in str(entry):
            import ast
            try:
                stats_str = str(entry).replace("Vote stats: ", "")
                vote_stats = ast.literal_eval(stats_str)
            except (ValueError, SyntaxError):
                pass

    # Calculate consensus confidence
    valid_candidates = vote_stats.get("valid_candidates", num_samples)
    winning_votes = vote_stats.get("winning_votes", 0)
    consensus_confidence = winning_votes / valid_candidates if valid_candidates > 0 else 0

    metadata = {
        "generation_time_ms": generation_time,
        "strategy": "M10",
        "valid": result.valid,
        "votes": winning_votes,
        "num_samples": num_samples,
        "valid_candidates": valid_candidates,
        "consensus_confidence": consensus_confidence,
        "unique_signatures": vote_stats.get("unique_signatures", 0),
        "schema_augmentation_time_ms": vote_stats.get("schema_augmentation_time_ms", 0),
        "latency_ms": result.latency_ms,
    }

    return result.sql, metadata


def run_ves_benchmark(
    data_dir: Path,
    strategy: str = "baseline",
    limit: int | None = None,
    num_beams: int = 5,
    output_dir: Path = Path("results"),
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    quantization: str = None,
) -> dict:
    """
    Run VES benchmark on BIRD Mini-Dev.

    Args:
        data_dir: Path to BIRD data directory
        strategy: "baseline" or "M4"
        limit: Limit number of examples (None = all)
        num_beams: Number of beams for M4 strategy
        output_dir: Directory for output files
        model_name: HuggingFace model identifier
        quantization: Quantization mode ("8bit", "4bit", or None)

    Returns:
        Results dictionary with VES metrics
    """
    print(f"Loading BIRD Mini-Dev from {data_dir}")
    examples = load_bird_mini_dev(data_dir)

    if limit:
        examples = examples[:limit]

    print(f"Evaluating {len(examples)} examples with strategy: {strategy}")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(model_name=model_name, quantization=quantization)
    generator = ExplainGuidedGenerator(model, tokenizer)

    results = {
        "strategy": strategy,
        "total_examples": len(examples),
        "num_beams": num_beams,
        "successful": 0,
        "failed": 0,
        "correct": 0,
        "incorrect": 0,
        "total_ves": 0.0,
        "avg_ves": 0.0,
        "avg_generation_time_ms": 0.0,
        "avg_gold_exec_time_ms": 0.0,
        "avg_pred_exec_time_ms": 0.0,
        "examples": []
    }

    total_generation_time = 0.0
    total_gold_exec_time = 0.0
    total_pred_exec_time = 0.0

    # Setup incremental save
    output_dir.mkdir(exist_ok=True, parents=True)
    partial_file = output_dir / f"bird_ves_{strategy}_{len(examples)}_partial.json"
    save_interval = 50  # Save every 50 examples

    for idx, example in enumerate(tqdm(examples, desc=f"Running {strategy}")):
        db_id = example["db_id"]
        question = example["question"]
        evidence = example.get("evidence", "")
        gold_sql = example["SQL"]

        try:
            # Get database path and schema
            db_path = get_bird_database_path(db_id, data_dir)
            schema = extract_schema(db_path)

            # Execute gold query first to get baseline timing
            gold_results, gold_time_ms, gold_error = execute_sql_with_timeout(
                gold_sql, db_path
            )

            if gold_error:
                print(f"\nWarning: Gold query failed for example {idx}: {gold_error}")
                results["examples"].append({
                    "index": idx,
                    "db_id": db_id,
                    "question": question,
                    "gold_sql": gold_sql,
                    "error": f"Gold query failed: {gold_error}",
                    "ves": 0.0
                })
                results["failed"] += 1
                continue

            # Generate predicted SQL based on strategy
            if strategy == "baseline":
                pred_sql, gen_metadata = run_baseline(
                    generator, question, schema, evidence
                )
            elif strategy.startswith("few_shot"):
                # Parse number of examples from strategy name (e.g., "few_shot_3")
                num_shots = int(strategy.split("_")[-1]) if "_" in strategy else 3
                from egtts.few_shot import get_few_shot_examples
                few_shot_examples = get_few_shot_examples(num_shots)
                pred_sql, gen_metadata = run_baseline(
                    generator, question, schema, evidence, few_shot_examples=few_shot_examples
                )
            elif strategy == "M4":
                pred_sql, gen_metadata = run_m4_cost_aware(
                    generator, question, schema, evidence, db_path, num_beams
                )
            elif strategy == "explain_feedback":
                pred_sql, gen_metadata = run_explain_feedback(
                    generator, question, schema, evidence, db_path
                )
            elif strategy == "M7":
                pred_sql, gen_metadata = run_plan_voting(
                    generator, question, schema, evidence, db_path
                )
            elif strategy == "M8":
                pred_sql, gen_metadata = run_massive_diversity(
                    generator, question, schema, evidence, db_path
                )
            elif strategy == "M9":
                pred_sql, gen_metadata = run_few_shot_simulation(
                    generator, question, schema, evidence, db_path
                )
            elif strategy == "M10":
                pred_sql, gen_metadata = run_augmented_schema(
                    generator, question, schema, evidence, db_path
                )
            elif strategy == "M11":
                pred_sql, gen_metadata = run_cot(
                    generator, question, schema, evidence, db_path
                )
            elif strategy == "M12":
                pred_sql, gen_metadata = run_execution_correction(
                    generator, question, schema, evidence, db_path
                )
            elif strategy == "M13":
                pred_sql, gen_metadata = run_augmented_schema_with_probing(
                    generator, question, schema, evidence, db_path
                )
            elif strategy == "M14":
                pred_sql, gen_metadata = run_dataflow(
                    generator, question, schema, evidence, db_path
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Execute predicted query
            pred_results, pred_time_ms, pred_error = execute_sql_with_timeout(
                pred_sql, db_path
            )

            # Calculate VES
            ves = calculate_ves(pred_results, gold_results, pred_time_ms, gold_time_ms)

            # Track metrics
            total_generation_time += gen_metadata["generation_time_ms"]
            total_gold_exec_time += gold_time_ms
            total_pred_exec_time += pred_time_ms
            results["total_ves"] += ves

            if pred_error:
                results["incorrect"] += 1
                correctness = "error"
            elif ves > 0:
                results["correct"] += 1
                correctness = "correct"
            else:
                results["incorrect"] += 1
                correctness = "incorrect"

            results["successful"] += 1

            # Store example results
            example_result = {
                "index": idx,
                "db_id": db_id,
                "question": question,
                "evidence": evidence,
                "gold_sql": gold_sql,
                "predicted_sql": pred_sql,
                "ves": ves,
                "correctness": correctness,
                "gold_exec_time_ms": gold_time_ms,
                "pred_exec_time_ms": pred_time_ms,
                "generation_time_ms": gen_metadata["generation_time_ms"],
            }

            # Add M7-specific consensus metrics
            if strategy == "M7":
                example_result["consensus_confidence"] = gen_metadata.get("consensus_confidence", 0)
                example_result["votes"] = gen_metadata.get("votes", 0)
                example_result["valid_candidates"] = gen_metadata.get("valid_candidates", 0)
                example_result["unique_signatures"] = gen_metadata.get("unique_signatures", 0)

            if pred_error:
                example_result["execution_error"] = pred_error

            if strategy == "M4" and "cost_metadata" in gen_metadata:
                example_result["cost_metadata"] = gen_metadata["cost_metadata"]

            results["examples"].append(example_result)

            # Incremental save every N examples
            if (idx + 1) % save_interval == 0:
                # Calculate current averages
                if results["successful"] > 0:
                    results["avg_ves"] = results["total_ves"] / results["successful"]
                    results["avg_generation_time_ms"] = total_generation_time / results["successful"]
                    results["avg_gold_exec_time_ms"] = total_gold_exec_time / results["successful"]
                    results["avg_pred_exec_time_ms"] = total_pred_exec_time / results["successful"]

                with open(partial_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\n[Progress saved: {idx + 1}/{len(examples)} examples]")

        except Exception as e:
            print(f"\nError processing example {idx}: {e}")
            results["examples"].append({
                "index": idx,
                "db_id": db_id,
                "question": question,
                "error": str(e),
                "ves": 0.0
            })
            results["failed"] += 1

    # Calculate averages
    if results["successful"] > 0:
        results["avg_ves"] = results["total_ves"] / results["successful"]
        results["avg_generation_time_ms"] = total_generation_time / results["successful"]
        results["avg_gold_exec_time_ms"] = total_gold_exec_time / results["successful"]
        results["avg_pred_exec_time_ms"] = total_pred_exec_time / results["successful"]

    results["accuracy"] = (
        results["correct"] / results["total_examples"]
        if results["total_examples"] > 0
        else 0.0
    )

    # Save results
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f"bird_ves_{strategy}_{len(examples)}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results for {strategy.upper()}")
    print(f"{'='*80}")
    print(f"Total Examples: {results['total_examples']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Accuracy: {results['accuracy']:.1%}")
    print(f"Average VES: {results['avg_ves']:.3f}")
    print(f"Avg Generation Time: {results['avg_generation_time_ms']:.1f}ms")
    print(f"Avg Gold Exec Time: {results['avg_gold_exec_time_ms']:.1f}ms")
    print(f"Avg Pred Exec Time: {results['avg_pred_exec_time_ms']:.1f}ms")
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run BIRD VES benchmark"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/bird"),
        help="Path to BIRD data directory"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="both",
        help="Strategy: baseline, M4, few_shot_3, few_shot_5, or both"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples (for testing)"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams for M4 strategy"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model identifier (default: 7B)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["8bit", "4bit", None],
        default=None,
        help="Quantization mode: 8bit, 4bit, or None (default: None/FP16)"
    )

    args = parser.parse_args()

    if args.strategy == "both":
        strategies = ["baseline", "M4"]
    else:
        strategies = [args.strategy]

    all_results = {}
    for strategy in strategies:
        results = run_ves_benchmark(
            data_dir=args.data_dir,
            strategy=strategy,
            limit=args.limit,
            num_beams=args.num_beams,
            output_dir=args.output_dir,
            model_name=args.model,
            quantization=args.quantization,
        )
        all_results[strategy] = results

    # If both strategies were run, print comparison
    if len(strategies) == 2:
        print(f"\n{'='*80}")
        print("COMPARISON: Baseline vs M4")
        print(f"{'='*80}")
        print(f"{'Metric':<30} {'Baseline':>15} {'M4':>15} {'Δ':>15}")
        print(f"{'-'*80}")

        baseline = all_results["baseline"]
        m4 = all_results["M4"]

        print(f"{'Accuracy':<30} {baseline['accuracy']:>14.1%} {m4['accuracy']:>14.1%} {(m4['accuracy']-baseline['accuracy']):>+14.1%}")
        print(f"{'Average VES':<30} {baseline['avg_ves']:>15.3f} {m4['avg_ves']:>15.3f} {(m4['avg_ves']-baseline['avg_ves']):>+15.3f}")
        print(f"{'Avg Generation Time (ms)':<30} {baseline['avg_generation_time_ms']:>15.1f} {m4['avg_generation_time_ms']:>15.1f} {(m4['avg_generation_time_ms']-baseline['avg_generation_time_ms']):>+15.1f}")
        print(f"{'Avg Pred Exec Time (ms)':<30} {baseline['avg_pred_exec_time_ms']:>15.1f} {m4['avg_pred_exec_time_ms']:>15.1f} {(m4['avg_pred_exec_time_ms']-baseline['avg_pred_exec_time_ms']):>+15.1f}")


if __name__ == "__main__":
    main()
