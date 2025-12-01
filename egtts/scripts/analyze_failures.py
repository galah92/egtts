#!/usr/bin/env python3
"""
Failure Analysis Script for EGTTS
=================================
Diagnoses WHY the system fails, categorizing into:

1. Selection Failure (Tragedy): Correct SQL was in beams but wrong one selected
2. Generation Failure (Incompetence): None of the beams were correct
3. Schema Failure (Hallucination): Model used non-existent columns/tables

Usage:
    uv run python scripts/analyze_failures.py --results results/bird_ves_M8_50.json --limit 10
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from egtts.guided import ExplainGuidedGenerator


@dataclass
class FailureAnalysis:
    """Analysis of a single failure case."""

    index: int
    question: str
    gold_sql: str
    predicted_sql: str
    failure_type: str  # "selection", "generation", "schema", "semantic"
    details: dict


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    import re

    sql = sql.strip().rstrip(";").lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def execute_sql(db_path: Path, sql: str) -> tuple[Optional[list], Optional[str]]:
    """Execute SQL and return results or error."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def check_schema_errors(db_path: Path, sql: str) -> list[str]:
    """Check if SQL references non-existent tables/columns."""
    errors = []

    # Get actual schema
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0].lower() for row in cursor.fetchall()}

        # Get all columns per table
        columns_by_table = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns_by_table[table] = {row[1].lower() for row in cursor.fetchall()}

        conn.close()

        # Parse SQL for table/column references (simple heuristic)
        sql_lower = sql.lower()

        # Check for common hallucinated patterns
        import re

        # Look for "FROM table" or "JOIN table" patterns
        table_refs = re.findall(r"(?:from|join)\s+(\w+)", sql_lower)
        for ref in table_refs:
            if ref not in tables and ref not in ["select", "where", "on", "and", "or"]:
                errors.append(f"Unknown table: {ref}")

        return errors

    except Exception as e:
        return [f"Schema check error: {e}"]


def analyze_single_failure(
    example: dict,
    db_path: Path,
    generator: Optional[ExplainGuidedGenerator] = None,
) -> FailureAnalysis:
    """Analyze a single failure case."""

    question = example["question"]
    gold_sql = example["gold_sql"]
    predicted_sql = example["predicted_sql"]

    details = {
        "db_id": example["db_id"],
        "evidence": example.get("evidence", ""),
    }

    # Check for execution errors first
    if example.get("correctness") == "error":
        error_msg = example.get("execution_error", "Unknown error")

        # Check if it's a schema error
        schema_errors = check_schema_errors(db_path, predicted_sql)
        if schema_errors or "no such" in error_msg.lower():
            details["schema_errors"] = schema_errors
            details["execution_error"] = error_msg
            return FailureAnalysis(
                index=example["index"],
                question=question,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                failure_type="schema",
                details=details,
            )
        else:
            details["execution_error"] = error_msg
            return FailureAnalysis(
                index=example["index"],
                question=question,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                failure_type="syntax",
                details=details,
            )

    # Both execute - compare results
    gold_result, gold_err = execute_sql(db_path, gold_sql)
    pred_result, pred_err = execute_sql(db_path, predicted_sql)

    if pred_err:
        schema_errors = check_schema_errors(db_path, predicted_sql)
        if schema_errors or "no such" in pred_err.lower():
            details["schema_errors"] = schema_errors
            details["execution_error"] = pred_err
            return FailureAnalysis(
                index=example["index"],
                question=question,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                failure_type="schema",
                details=details,
            )

    # Semantic failure - wrong logic
    details["gold_result_preview"] = str(gold_result)[:200] if gold_result else "None"
    details["pred_result_preview"] = str(pred_result)[:200] if pred_result else "None"

    # Try to identify common patterns
    gold_norm = normalize_sql(gold_sql)
    pred_norm = normalize_sql(predicted_sql)

    # Check for aggregation mismatches
    if "sum(" in gold_norm or "count(" in gold_norm or "avg(" in gold_norm:
        if "group by" in gold_norm and "group by" not in pred_norm:
            details["pattern"] = "Missing GROUP BY aggregation"
        elif "group by" not in gold_norm and "group by" in pred_norm:
            details["pattern"] = "Spurious GROUP BY"

    # Check for SUBSTR/date handling issues
    if "substr" in gold_norm and "substr" not in pred_norm:
        details["pattern"] = "Missing SUBSTR for date extraction"
    elif "substr" in gold_norm and "like" in pred_norm:
        details["pattern"] = "Used LIKE instead of SUBSTR"

    # Check for column selection issues
    if gold_result and pred_result:
        if len(gold_result) > 0 and len(pred_result) > 0:
            gold_cols = len(gold_result[0]) if isinstance(gold_result[0], tuple) else 1
            pred_cols = len(pred_result[0]) if isinstance(pred_result[0], tuple) else 1
            if gold_cols != pred_cols:
                details["pattern"] = (
                    f"Column count mismatch: gold={gold_cols}, pred={pred_cols}"
                )

    return FailureAnalysis(
        index=example["index"],
        question=question,
        gold_sql=gold_sql,
        predicted_sql=predicted_sql,
        failure_type="semantic",
        details=details,
    )


def generate_report(analyses: list[FailureAnalysis], output_path: Path) -> str:
    """Generate a human-readable failure report."""

    lines = []
    lines.append("=" * 80)
    lines.append("FAILURE ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Summary statistics
    failure_counts = {}
    for a in analyses:
        failure_counts[a.failure_type] = failure_counts.get(a.failure_type, 0) + 1

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Failures Analyzed: {len(analyses)}")
    for ftype, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        pct = count / len(analyses) * 100
        lines.append(f"  {ftype.upper():15} : {count:3} ({pct:.1f}%)")
    lines.append("")

    # Diagnosis interpretation
    lines.append("DIAGNOSIS MATRIX")
    lines.append("-" * 40)
    if "schema" in failure_counts:
        lines.append(f"  Schema Failures: {failure_counts.get('schema', 0)}")
        lines.append("    → Fix: M9 (Schema Scout) - ground column names")
    if "semantic" in failure_counts:
        lines.append(f"  Semantic Failures: {failure_counts.get('semantic', 0)}")
        lines.append("    → Fix: Few-shot prompting or fine-tuning")
    if "syntax" in failure_counts:
        lines.append(f"  Syntax Failures: {failure_counts.get('syntax', 0)}")
        lines.append("    → Fix: M6 (LLM Repair) or syntax validation")
    lines.append("")

    # Detailed analysis
    lines.append("=" * 80)
    lines.append("DETAILED FAILURE ANALYSIS")
    lines.append("=" * 80)

    for i, a in enumerate(analyses, 1):
        lines.append("")
        lines.append(f"FAILURE #{i} (Index: {a.index})")
        lines.append("-" * 60)
        lines.append(f"Type: {a.failure_type.upper()}")
        lines.append(f"DB: {a.details.get('db_id', 'unknown')}")
        lines.append("")
        lines.append(f"Question: {a.question}")
        lines.append("")
        if a.details.get("evidence"):
            lines.append(f"Evidence: {a.details['evidence']}")
            lines.append("")
        lines.append("Gold SQL:")
        lines.append(f"  {a.gold_sql}")
        lines.append("")
        lines.append("Predicted SQL:")
        lines.append(f"  {a.predicted_sql}")
        lines.append("")

        if a.details.get("pattern"):
            lines.append(f"Pattern Detected: {a.details['pattern']}")
        if a.details.get("schema_errors"):
            lines.append(f"Schema Errors: {a.details['schema_errors']}")
        if a.details.get("execution_error"):
            lines.append(f"Execution Error: {a.details['execution_error']}")
        if a.details.get("gold_result_preview"):
            lines.append(f"Gold Result (preview): {a.details['gold_result_preview']}")
        if a.details.get("pred_result_preview"):
            lines.append(f"Pred Result (preview): {a.details['pred_result_preview']}")

        lines.append("")

    report = "\n".join(lines)

    # Write to file
    output_path.write_text(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze EGTTS failures")
    parser.add_argument(
        "--results", type=str, required=True, help="Path to results JSON"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of failures to analyze"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/bird", help="Path to BIRD data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/failure_report.txt",
        help="Output report path",
    )
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    # Filter failures
    failures = [ex for ex in data["examples"] if ex["correctness"] != "correct"]
    print(f"Found {len(failures)} failures out of {len(data['examples'])} examples")

    if not failures:
        print("No failures to analyze!")
        sys.exit(0)

    # Limit
    failures = failures[: args.limit]
    print(f"Analyzing first {len(failures)} failures...")

    # Find database paths
    data_dir = Path(args.data_dir)
    databases_dir = data_dir / "dev_databases"

    # Analyze each failure
    analyses = []
    for ex in failures:
        db_id = ex["db_id"]
        db_path = databases_dir / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            print(f"Warning: Database not found: {db_path}")
            continue

        analysis = analyze_single_failure(ex, db_path)
        analyses.append(analysis)
        print(f"  Analyzed #{ex['index']}: {analysis.failure_type}")

    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_report(analyses, output_path)

    print(f"\nReport saved to: {output_path}")
    print("\n" + "=" * 60)
    print("FIRST 3 FAILURES PREVIEW")
    print("=" * 60)

    # Print first 3 to console
    for a in analyses[:3]:
        print(f"\n--- Failure #{a.index} ({a.failure_type.upper()}) ---")
        print(f"Q: {a.question[:100]}...")
        print(f"Gold: {a.gold_sql[:100]}...")
        print(f"Pred: {a.predicted_sql[:100]}...")
        if a.details.get("pattern"):
            print(f"Pattern: {a.details['pattern']}")


if __name__ == "__main__":
    main()
