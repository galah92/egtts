"""
Milestone 4: Cost-Aware Query Selection for Efficiency

Compares three beam selection strategies:
1. Baseline: Greedy (beam 0 only)
2. Validity Only (M3): First schema+EXPLAIN valid beam
3. Efficiency Guided (M4): Lowest cost among valid beams

Targets the BIRD benchmark's VES (Valid Efficiency Score) metric.
"""

import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from egtts import ExplainGuidedGenerator, load_model
from egtts.database import explain_query


def get_database_schema(db_path: Path) -> str:
    """Extract CREATE TABLE statements from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    schema_parts = []
    for (table_name,) in tables:
        cursor.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        create_stmt = cursor.fetchone()[0]
        schema_parts.append(create_stmt)

    conn.close()
    return "\n\n".join(schema_parts)


def execute_sql(sql: str, db_path: Path) -> tuple[bool, list | str]:
    """Execute SQL and return results."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)


def results_match(results1: list, results2: list) -> bool:
    """Compare query results for equality."""
    if len(results1) != len(results2):
        return False

    # For single-value results, compare directly
    if len(results1) == 1 and len(results1[0]) == 1:
        return results1[0][0] == results2[0][0]

    # For multi-row results, compare as sets (order-independent)
    try:
        set1 = set(tuple(row) for row in results1)
        set2 = set(tuple(row) for row in results2)
        return set1 == set2
    except TypeError:
        # If unhashable, compare sorted lists
        return sorted(results1) == sorted(results2)


def has_table_scan(explain_plan: list[dict]) -> bool:
    """Check if query plan contains table scan."""
    plan_str = " ".join(str(row) for row in explain_plan).upper()
    # SQLite EXPLAIN uses "SCAN <table>" not "SCAN TABLE <table>"
    return "'SCAN " in plan_str and "USING INDEX" not in plan_str


def evaluate_milestone4(num_examples: int = 50):
    """
    Run Milestone 4 efficiency-guided evaluation.

    Args:
        num_examples: Number of Spider examples to evaluate
    """
    print("=" * 80)
    print("MILESTONE 4: EFFICIENCY-GUIDED QUERY SELECTION")
    print("=" * 80)

    # Load Spider data
    spider_data_dir = Path("data/spider/spider_data")
    dev_file = spider_data_dir / "dev.json"

    print(f"\nLoading {num_examples} examples from Spider...")
    with open(dev_file) as f:
        examples = json.load(f)[:num_examples]
    print(f"✓ Loaded {len(examples)} examples")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model()
    print("✓ Model loaded")

    # Initialize guided generator
    generator = ExplainGuidedGenerator(model, tokenizer)

    # Track results for three approaches
    baseline_results = []  # Beam 0 only
    validity_results = []  # M3: First valid
    efficiency_results = []  # M4: Lowest cost

    print(f"\n{'=' * 80}")
    print(f"Evaluating {num_examples} examples with 3 strategies")
    print(f"{'=' * 80}\n")

    for idx, example in enumerate(examples):
        db_id = example["db_id"]
        question = example["question"]
        gold_sql = example["query"]

        print(f"[{idx + 1}/{num_examples}] {db_id}: {question[:60]}...")

        db_path = spider_data_dir / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            print("  ⚠ Database not found, skipping")
            continue

        # Get schema
        schema = get_database_schema(db_path)

        # Execute gold SQL for comparison
        gold_success, gold_results = execute_sql(gold_sql, db_path)

        # Strategy 1: Baseline (beam 0 only)
        baseline_result = generator.generate(question, schema)
        baseline_success, baseline_exec = execute_sql(baseline_result, db_path)
        baseline_correct = False
        if baseline_success and gold_success:
            baseline_correct = results_match(baseline_exec, gold_results)

        # Get baseline plan for scan analysis
        baseline_plan = explain_query(baseline_result, str(db_path))
        baseline_has_scan = has_table_scan(baseline_plan.plan) if hasattr(baseline_plan, 'plan') else False

        baseline_results.append({
            "sql": baseline_result,
            "correct": baseline_correct,
            "has_scan": baseline_has_scan,
        })

        # Strategy 2: Validity Only (M3)
        validity_result = generator.generate_with_schema_guidance(
            question, schema, str(db_path), num_beams=5
        )
        validity_success, validity_exec = execute_sql(validity_result.sql, db_path)
        validity_correct = False
        if validity_success and gold_success:
            validity_correct = results_match(validity_exec, gold_results)

        # Get validity plan
        validity_plan = explain_query(validity_result.sql, str(db_path))
        validity_has_scan = has_table_scan(validity_plan.plan) if hasattr(validity_plan, 'plan') else False

        validity_results.append({
            **asdict(validity_result),
            "correct": validity_correct,
            "has_scan": validity_has_scan,
        })

        # Strategy 3: Efficiency Guided (M4)
        efficiency_result = generator.generate_with_cost_guidance(
            question, schema, str(db_path), num_beams=5
        )
        efficiency_success, efficiency_exec = execute_sql(efficiency_result.sql, db_path)
        efficiency_correct = False
        if efficiency_success and gold_success:
            efficiency_correct = results_match(efficiency_exec, gold_results)

        # Get efficiency plan and cost
        efficiency_plan = explain_query(efficiency_result.sql, str(db_path))
        efficiency_has_scan = has_table_scan(efficiency_plan.plan) if hasattr(efficiency_plan, 'plan') else False
        efficiency_cost = generator.calculate_plan_cost(efficiency_plan.plan) if hasattr(efficiency_plan, 'plan') else 999

        efficiency_results.append({
            **asdict(efficiency_result),
            "correct": efficiency_correct,
            "has_scan": efficiency_has_scan,
            "cost": efficiency_cost,
        })

        # Print summary
        print(f"  Baseline:    {'✓' if baseline_correct else '✗'} | Scan: {baseline_has_scan}")
        print(f"  Validity:    {'✓' if validity_correct else '✗'} | Scan: {validity_has_scan}")
        print(f"  Efficiency:  {'✓' if efficiency_correct else '✗'} | Scan: {efficiency_has_scan} | Cost: {efficiency_cost}")

    # Analysis
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")

    total = len(baseline_results)

    baseline_acc = sum(1 for r in baseline_results if r["correct"]) / total * 100
    validity_acc = sum(1 for r in validity_results if r["correct"]) / total * 100
    efficiency_acc = sum(1 for r in efficiency_results if r["correct"]) / total * 100

    baseline_scans = sum(1 for r in baseline_results if r["has_scan"]) / total * 100
    validity_scans = sum(1 for r in validity_results if r["has_scan"]) / total * 100
    efficiency_scans = sum(1 for r in efficiency_results if r["has_scan"]) / total * 100

    print("\n1. Execution Accuracy:")
    print(f"  Baseline (beam 0):       {baseline_acc:.1f}%")
    print(f"  Validity Only (M3):      {validity_acc:.1f}%")
    print(f"  Efficiency Guided (M4):  {efficiency_acc:.1f}%")

    print("\n2. Table Scan Rate (lower is better):")
    print(f"  Baseline (beam 0):       {baseline_scans:.1f}%")
    print(f"  Validity Only (M3):      {validity_scans:.1f}%")
    print(f"  Efficiency Guided (M4):  {efficiency_scans:.1f}%")
    print(f"  Reduction vs Baseline:   {baseline_scans - efficiency_scans:.1f}%")

    # Find cases where M4 picked different query than M3
    different_selections = []
    for i in range(total):
        if validity_results[i]["sql"] != efficiency_results[i]["sql"]:
            different_selections.append({
                "index": i,
                "validity_sql": validity_results[i]["sql"],
                "validity_scan": validity_results[i]["has_scan"],
                "efficiency_sql": efficiency_results[i]["sql"],
                "efficiency_scan": efficiency_results[i]["has_scan"],
                "efficiency_cost": efficiency_results[i]["cost"],
            })

    print(f"\n3. Different Selections (M3 vs M4): {len(different_selections)}")
    if different_selections:
        print(f"\nExample: Query #{different_selections[0]['index'] + 1}")
        print(f"  M3 Selected:  {different_selections[0]['validity_sql'][:80]}...")
        print(f"    Has Scan:   {different_selections[0]['validity_scan']}")
        print(f"  M4 Selected:  {different_selections[0]['efficiency_sql'][:80]}...")
        print(f"    Has Scan:   {different_selections[0]['efficiency_scan']}")
        print(f"    Cost Score: {different_selections[0]['efficiency_cost']}")

    # Success criteria
    print(f"\n{'=' * 80}")
    print("SUCCESS CRITERIA")
    print(f"{'=' * 80}")

    print(f"\n1. Accuracy maintained: {efficiency_acc >= validity_acc}")
    print(f"   M4: {efficiency_acc:.1f}% vs M3: {validity_acc:.1f}%")

    print(f"\n2. Scan rate reduced: {efficiency_scans < baseline_scans}")
    print(f"   M4: {efficiency_scans:.1f}% vs Baseline: {baseline_scans:.1f}%")

    print(f"\n3. Cost-aware selection working: {len(different_selections) > 0}")
    print(f"   {len(different_selections)} queries where M4 picked different beam than M3")

    # Save results
    output_file = Path("results/milestone4_evaluation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_examples": total,
        "accuracy": {
            "baseline": baseline_acc,
            "validity_only": validity_acc,
            "efficiency_guided": efficiency_acc,
        },
        "scan_rate": {
            "baseline": baseline_scans,
            "validity_only": validity_scans,
            "efficiency_guided": efficiency_scans,
        },
        "different_selections": len(different_selections),
    }

    with open(output_file, "w") as f:
        json.dump({
            "summary": summary,
            "baseline_results": baseline_results,
            "validity_results": validity_results,
            "efficiency_results": efficiency_results,
            "different_selections": different_selections,
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    import sys

    num_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    evaluate_milestone4(num_examples=num_examples)
