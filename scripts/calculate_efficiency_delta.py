"""
Calculate Efficiency Delta between Baseline and M4 strategies.

This script computes a "Proxy VES" (Valid Efficiency Score) by comparing
the execution plan complexity of Baseline vs M4 for queries where BOTH
strategies produce valid results.

Efficiency Delta = Total_Baseline_Cost / Total_M4_Cost

If Delta > 1.0, M4 is more efficient on average.

Inspired by BIRD benchmark's VES metric:
VES = Accuracy × sqrt(ExecutionTime_Gold / ExecutionTime_Generated)
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import sqlite3

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from egtts.database import ExplainSuccess, explain_query


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


def count_plan_complexity(explain_result: ExplainSuccess) -> dict:
    """
    Calculate complexity metrics from EXPLAIN plan.

    Returns:
        Dictionary with:
        - lines: Total execution plan steps
        - scans: Number of table scans
        - searches: Number of index searches
        - temp_btrees: Number of temporary B-trees
    """
    lines = len(explain_result.plan)
    scans = sum(1 for step in explain_result.plan if "SCAN" in step["detail"])
    searches = sum(1 for step in explain_result.plan if "SEARCH" in step["detail"])
    temp_btrees = sum(1 for step in explain_result.plan if "TEMP B-TREE" in step["detail"])

    return {
        "lines": lines,
        "scans": scans,
        "searches": searches,
        "temp_btrees": temp_btrees
    }


def calculate_efficiency_delta():
    """
    Compare efficiency of Baseline vs M4 strategies.

    Analyzes queries where both strategies succeed and compares their
    execution plan complexity.
    """
    print("="*80)
    print("EFFICIENCY DELTA ANALYSIS")
    print("="*80)
    print()

    # Load metadata files
    print("Loading metadata files...")
    baseline_path = Path("results/spider_full_metadata_baseline.json")
    m4_path = Path("results/spider_full_metadata_m4.json")

    if not baseline_path.exists():
        print(f"❌ Error: {baseline_path} not found")
        return

    if not m4_path.exists():
        print(f"❌ Error: {m4_path} not found")
        return

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    with open(m4_path) as f:
        m4_data = json.load(f)

    baseline_examples = {ex["index"]: ex for ex in baseline_data["examples"]}
    m4_examples = {ex["index"]: ex for ex in m4_data["examples"]}

    print(f"✓ Loaded {len(baseline_examples)} baseline examples")
    print(f"✓ Loaded {len(m4_examples)} M4 examples")
    print()

    # Find examples where both strategies have valid queries
    print("Analyzing query complexity for examples where BOTH strategies succeed...")
    print()

    baseline_total = defaultdict(int)
    m4_total = defaultdict(int)

    both_valid_count = 0
    examples_analyzed = []

    for idx in sorted(baseline_examples.keys()):
        if idx not in m4_examples:
            continue

        baseline_ex = baseline_examples[idx]
        m4_ex = m4_examples[idx]

        # Get database path
        db_id = baseline_ex["db_id"]
        db_path = Path("data/spider/spider_data/database") / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            continue

        # Get SQL queries
        baseline_sql = baseline_ex["predicted_sql"]
        m4_sql = m4_ex["predicted_sql"]

        # Analyze both queries
        baseline_result = explain_query(baseline_sql, str(db_path))
        m4_result = explain_query(m4_sql, str(db_path))

        # Only compare when BOTH are valid
        if isinstance(baseline_result, ExplainSuccess) and isinstance(m4_result, ExplainSuccess):
            baseline_complexity = count_plan_complexity(baseline_result)
            m4_complexity = count_plan_complexity(m4_result)

            # Accumulate totals
            for key in baseline_complexity:
                baseline_total[key] += baseline_complexity[key]
                m4_total[key] += m4_complexity[key]

            both_valid_count += 1

            # Track interesting cases (where M4 is simpler)
            if m4_complexity["lines"] < baseline_complexity["lines"]:
                examples_analyzed.append({
                    "index": idx,
                    "question": baseline_ex["question"],
                    "baseline_lines": baseline_complexity["lines"],
                    "m4_lines": m4_complexity["lines"],
                    "reduction": baseline_complexity["lines"] - m4_complexity["lines"],
                    "baseline_scans": baseline_complexity["scans"],
                    "m4_scans": m4_complexity["scans"]
                })

    print(f"✓ Found {both_valid_count} examples where BOTH strategies produce valid queries")
    print()

    # Calculate efficiency delta
    print("="*80)
    print("AGGREGATE EFFICIENCY METRICS")
    print("="*80)
    print()

    print(f"{'Metric':<20} {'Baseline':<15} {'M4':<15} {'Delta':<15}")
    print("-"*65)

    for metric in ["lines", "scans", "searches", "temp_btrees"]:
        baseline_val = baseline_total[metric]
        m4_val = m4_total[metric]

        if m4_val > 0:
            delta = baseline_val / m4_val
            improvement = ((baseline_val - m4_val) / baseline_val * 100) if baseline_val > 0 else 0

            print(f"{metric.capitalize():<20} {baseline_val:<15} {m4_val:<15} {delta:>6.3f}x ({improvement:+.1f}%)")
        else:
            print(f"{metric.capitalize():<20} {baseline_val:<15} {m4_val:<15} {'N/A':<15}")

    print()

    # Primary metric: Total Execution Steps
    baseline_steps = baseline_total["lines"]
    m4_steps = m4_total["lines"]

    if m4_steps > 0 and baseline_steps > 0:
        efficiency_delta = baseline_steps / m4_steps
        efficiency_improvement = ((baseline_steps - m4_steps) / baseline_steps) * 100

        print("="*80)
        print("PRIMARY METRIC: Total Execution Plan Steps")
        print("="*80)
        print()
        print(f"  Baseline Total Steps:  {baseline_steps:,}")
        print(f"  M4 Total Steps:        {m4_steps:,}")
        print(f"  Steps Saved:           {baseline_steps - m4_steps:,}")
        print()
        print(f"  🎯 Efficiency Delta:   {efficiency_delta:.4f}x")
        print(f"  📊 Improvement:        {efficiency_improvement:+.2f}%")
        print()

        if efficiency_delta > 1.0:
            print(f"  ✅ M4 is MORE EFFICIENT than Baseline by {efficiency_improvement:.2f}%")
        elif efficiency_delta < 1.0:
            loss = (1 - efficiency_delta) * 100
            print(f"  ⚠️  M4 is LESS EFFICIENT than Baseline by {loss:.2f}%")
        else:
            print(f"  ➖ M4 and Baseline have EQUAL efficiency")
        print()

    # Show examples where M4 was simpler
    if examples_analyzed:
        print("="*80)
        print(f"EXAMPLES WHERE M4 WAS SIMPLER ({len(examples_analyzed)} found)")
        print("="*80)
        print()

        # Sort by reduction amount
        examples_analyzed.sort(key=lambda x: x["reduction"], reverse=True)

        print(f"Top {min(10, len(examples_analyzed))} examples with largest complexity reduction:")
        print()

        for i, ex in enumerate(examples_analyzed[:10], 1):
            print(f"{i}. Example {ex['index']}: {ex['question'][:60]}...")
            print(f"   Baseline: {ex['baseline_lines']} steps ({ex['baseline_scans']} scans)")
            print(f"   M4:       {ex['m4_lines']} steps ({ex['m4_scans']} scans)")
            print(f"   Saved:    {ex['reduction']} steps")
            print()

        # Calculate how often M4 was simpler
        pct_simpler = (len(examples_analyzed) / both_valid_count) * 100
        print(f"📈 M4 was simpler in {len(examples_analyzed)}/{both_valid_count} cases ({pct_simpler:.1f}%)")
        print()

    # Secondary Analysis: Scan Reduction
    baseline_scans = baseline_total["scans"]
    m4_scans = m4_total["scans"]

    if baseline_scans > 0 and m4_scans > 0:
        scan_delta = baseline_scans / m4_scans
        scan_improvement = ((baseline_scans - m4_scans) / baseline_scans) * 100

        print("="*80)
        print("SECONDARY METRIC: Table Scans")
        print("="*80)
        print()
        print(f"  Baseline Total Scans:  {baseline_scans:,}")
        print(f"  M4 Total Scans:        {m4_scans:,}")
        print(f"  Scans Saved:           {baseline_scans - m4_scans:,}")
        print()
        print(f"  🎯 Scan Reduction:     {scan_delta:.4f}x")
        print(f"  📊 Improvement:        {scan_improvement:+.2f}%")
        print()

    # Save detailed results
    output_file = Path("results/efficiency_delta.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "both_valid_count": both_valid_count,
                "baseline_total": dict(baseline_total),
                "m4_total": dict(m4_total),
                "efficiency_delta": efficiency_delta if baseline_steps > 0 and m4_steps > 0 else None,
                "efficiency_improvement_pct": efficiency_improvement if baseline_steps > 0 and m4_steps > 0 else None,
                "m4_simpler_count": len(examples_analyzed),
                "m4_simpler_pct": pct_simpler if both_valid_count > 0 else 0
            },
            "simpler_examples": examples_analyzed
        }, f, indent=2)

    print(f"💾 Detailed results saved to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate efficiency delta between Baseline and M4 strategies"
    )

    args = parser.parse_args()
    calculate_efficiency_delta()


if __name__ == "__main__":
    main()
