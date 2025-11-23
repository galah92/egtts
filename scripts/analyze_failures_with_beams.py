#!/usr/bin/env python3
"""
Failure Analysis with Beam-Level Details
=========================================
Re-runs failed examples and captures ALL generated candidates to determine:

1. Selection Failure: Correct SQL was in beams but we picked wrong one
2. Generation Failure: None of the beams were correct

Usage:
    uv run python scripts/analyze_failures_with_beams.py --results results/bird_ves_M8_50.json --limit 5
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def execute_sql(db_path: Path, sql: str, timeout: float = 5.0) -> tuple[Optional[list], Optional[str]]:
    """Execute SQL and return results or error."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def results_match(r1: Optional[list], r2: Optional[list]) -> bool:
    """Check if two result sets match."""
    if r1 is None or r2 is None:
        return False
    # Normalize and compare
    try:
        # Convert to comparable format
        s1 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in r1)
        s2 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in r2)
        return s1 == s2
    except:
        return r1 == r2


def analyze_failure_with_beams(
    example: dict,
    db_path: Path,
    generator,
    num_samples: int = 32,
) -> dict:
    """Re-run a failed example and capture all beams."""

    question = example["question"]
    evidence = example.get("evidence", "")
    gold_sql = example["gold_sql"]
    db_id = example["db_id"]

    # Load schema
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema_parts = []
    for table in tables:
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
        create_stmt = cursor.fetchone()[0]
        schema_parts.append(create_stmt)
    conn.close()
    schema = "\n\n".join(schema_parts)

    # Build prompt (same as in benchmark)
    if evidence:
        full_question = f"{question}\nHint: {evidence}"
    else:
        full_question = question

    # Generate diverse samples and capture ALL of them
    from egtts.model import create_sql_prompt
    prompt = create_sql_prompt(full_question, schema, generator.tokenizer)
    inputs = generator.tokenizer(prompt, return_tensors="pt").to(generator.model.device)
    input_length = inputs["input_ids"].shape[1]

    import torch

    # Generate with temperature sampling
    gen_start = time.perf_counter()
    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=generator.tokenizer.eos_token_id,
            eos_token_id=generator.tokenizer.eos_token_id,
        )
    gen_time = time.perf_counter() - gen_start

    # Decode all candidates
    def clean_sql(generated_text):
        """Clean up SQL: remove markdown formatting, etc."""
        sql = generated_text.strip()
        if "```sql" in sql.lower():
            sql = sql.split("```sql", 1)[1].split("```")[0].strip()
        elif "```" in sql:
            parts = sql.split("```")
            if len(parts) >= 2:
                sql = parts[1].strip()
        lines = sql.split("\n")
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("--") and not line.lower().startswith(("this query", "the query", "note:")):
                sql_lines.append(line)
            elif sql_lines:
                break
        return " ".join(sql_lines)

    all_candidates = []
    for i, output in enumerate(outputs):
        generated_tokens = output[input_length:]
        decoded = generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        sql = clean_sql(decoded)
        all_candidates.append(sql)

    # Execute gold SQL
    gold_result, gold_err = execute_sql(db_path, gold_sql)

    # Analyze each candidate
    beam_analysis = []
    correct_beam_indices = []
    valid_beam_indices = []

    plan_clusters = {}  # plan_signature -> list of beam indices

    for i, sql in enumerate(all_candidates):
        # Try to execute
        result, err = execute_sql(db_path, sql)

        # Check correctness
        is_correct = results_match(result, gold_result) if gold_result is not None else False
        is_valid = err is None

        # Get plan signature if valid
        plan_sig = None
        if is_valid:
            valid_beam_indices.append(i)
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
                plan_rows = cursor.fetchall()
                conn.close()
                plan_sig = str(plan_rows)
            except:
                plan_sig = "EXPLAIN_FAILED"

            if plan_sig not in plan_clusters:
                plan_clusters[plan_sig] = []
            plan_clusters[plan_sig].append(i)

        if is_correct:
            correct_beam_indices.append(i)

        beam_analysis.append({
            "index": i,
            "sql": sql[:200] + "..." if len(sql) > 200 else sql,
            "is_valid": is_valid,
            "is_correct": is_correct,
            "error": err[:100] if err else None,
            "result_preview": str(result)[:100] if result else None,
            "plan_signature": plan_sig[:100] if plan_sig else None,
        })

    # Determine failure type
    if correct_beam_indices:
        failure_type = "SELECTION (Tragedy)"
        diagnosis = f"Correct SQL was in beam(s) {correct_beam_indices[:5]} but we picked wrong one!"
    else:
        failure_type = "GENERATION (Incompetence)"
        diagnosis = f"None of the {num_samples} beams produced correct SQL."

    # Find largest cluster
    largest_cluster = max(plan_clusters.items(), key=lambda x: len(x[1])) if plan_clusters else (None, [])

    return {
        "index": example["index"],
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "gold_sql": gold_sql,
        "gold_result_preview": str(gold_result)[:200] if gold_result else None,
        "predicted_sql": example["predicted_sql"],
        "failure_type": failure_type,
        "diagnosis": diagnosis,
        "generation_time_s": gen_time,
        "total_candidates": len(all_candidates),
        "valid_candidates": len(valid_beam_indices),
        "correct_candidates": len(correct_beam_indices),
        "correct_beam_indices": correct_beam_indices[:10],  # First 10
        "num_plan_clusters": len(plan_clusters),
        "largest_cluster_size": len(largest_cluster[1]),
        "beam_analysis": beam_analysis[:10],  # First 10 for readability
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze failures with beam details")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--limit", type=int, default=5, help="Number of failures to analyze")
    parser.add_argument("--samples", type=int, default=32, help="Number of samples to generate")
    parser.add_argument("--data-dir", type=str, default="data/bird", help="Path to BIRD data")
    parser.add_argument("--output", type=str, default="results/failure_beam_report.txt")
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    with open(results_path) as f:
        data = json.load(f)

    # Filter failures
    failures = [ex for ex in data["examples"] if ex["correctness"] != "correct"]
    print(f"Found {len(failures)} failures out of {len(data['examples'])} examples")
    failures = failures[:args.limit]

    # Load model
    print("Loading model for beam analysis...")
    from egtts import load_model
    from egtts.guided import ExplainGuidedGenerator
    model, tokenizer = load_model()
    generator = ExplainGuidedGenerator(model, tokenizer)

    # Find database paths
    data_dir = Path(args.data_dir)
    databases_dir = data_dir / "dev_databases"

    # Analyze each failure
    analyses = []
    for i, ex in enumerate(failures):
        db_id = ex["db_id"]
        db_path = databases_dir / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            print(f"Warning: Database not found: {db_path}")
            continue

        print(f"\n[{i+1}/{len(failures)}] Analyzing failure #{ex['index']} ({db_id})...")
        analysis = analyze_failure_with_beams(ex, db_path, generator, args.samples)
        analyses.append(analysis)

        # Print summary immediately
        print(f"  Type: {analysis['failure_type']}")
        print(f"  Valid: {analysis['valid_candidates']}/{analysis['total_candidates']}")
        print(f"  Correct in beams: {analysis['correct_candidates']}")
        print(f"  Clusters: {analysis['num_plan_clusters']}, Largest: {analysis['largest_cluster_size']}")

    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("FAILURE ANALYSIS WITH BEAM DETAILS")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    selection_failures = sum(1 for a in analyses if "SELECTION" in a["failure_type"])
    generation_failures = sum(1 for a in analyses if "GENERATION" in a["failure_type"])

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Analyzed: {len(analyses)}")
    lines.append(f"Selection Failures (correct SQL was in beams): {selection_failures}")
    lines.append(f"Generation Failures (no correct SQL in beams): {generation_failures}")
    lines.append("")

    if selection_failures > 0:
        lines.append(">>> SELECTION FAILURES indicate our voting/ranking needs improvement!")
    if generation_failures > 0:
        lines.append(">>> GENERATION FAILURES indicate the model needs better prompting/fine-tuning")
    lines.append("")

    # Detailed analysis
    for a in analyses:
        lines.append("=" * 80)
        lines.append(f"FAILURE #{a['index']} - {a['failure_type']}")
        lines.append("=" * 80)
        lines.append(f"DB: {a['db_id']}")
        lines.append(f"Question: {a['question']}")
        if a['evidence']:
            lines.append(f"Evidence: {a['evidence']}")
        lines.append("")
        lines.append(f"Gold SQL: {a['gold_sql']}")
        lines.append(f"Gold Result: {a['gold_result_preview']}")
        lines.append("")
        lines.append(f"Our Pick: {a['predicted_sql']}")
        lines.append("")
        lines.append(f"Diagnosis: {a['diagnosis']}")
        lines.append(f"Stats: {a['valid_candidates']}/{a['total_candidates']} valid, {a['num_plan_clusters']} clusters")
        lines.append("")

        # Show first few beams
        lines.append("Sample Beams:")
        for beam in a['beam_analysis'][:5]:
            status = "✓ CORRECT" if beam['is_correct'] else ("✗ invalid" if not beam['is_valid'] else "valid but wrong")
            lines.append(f"  [{beam['index']:2}] {status}")
            lines.append(f"      SQL: {beam['sql'][:100]}...")
            if beam['error']:
                lines.append(f"      Error: {beam['error']}")
            if beam['result_preview'] and not beam['is_correct']:
                lines.append(f"      Result: {beam['result_preview']}")
        lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")

    # Also print to console
    print("\n" + report)


if __name__ == "__main__":
    main()
