"""
Analyze regression: Find cases where Baseline was correct but M4 was wrong.

This script helps debug why M4 performs worse than baseline by:
1. Loading baseline and M4 predictions
2. Loading gold standard queries
3. Finding examples where baseline matches gold but M4 doesn't
4. Showing the cost scores to check if M4 picked wrong queries due to cost
"""

import json
from pathlib import Path

import sqlite3
from dataclasses import dataclass


@dataclass
class RegressionExample:
    """Example where baseline was correct but M4 was wrong."""
    index: int
    db_id: str
    question: str
    gold_sql: str
    baseline_sql: str
    m4_sql: str
    baseline_cost: int
    m4_cost: int


def get_explain_plan(sql: str, db_path: str) -> list:
    """Get EXPLAIN QUERY PLAN output."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        plan = cursor.fetchall()
        conn.close()
        return plan
    except Exception as e:
        return []


def calculate_plan_cost(explain_output: list) -> int:
    """Calculate heuristic cost score from EXPLAIN QUERY PLAN output."""
    cost = 0
    plan_str = " ".join(str(row) for row in explain_output).upper()

    if "'SCAN " in plan_str and "USING INDEX" not in plan_str:
        cost += 100  # Table scan is expensive
    if "USE TEMP B-TREE" in plan_str:
        cost += 50  # Temporary structures add overhead

    return cost


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (basic normalization)."""
    # Remove extra whitespace
    sql = " ".join(sql.split())
    # Convert to uppercase for case-insensitive comparison
    sql = sql.upper()
    # Remove trailing semicolons
    sql = sql.rstrip(";")
    return sql


def main():
    """Find and analyze regression examples."""
    print("=" * 80)
    print("REGRESSION ANALYSIS: Baseline Correct, M4 Incorrect")
    print("=" * 80)

    # Load Spider dev.json for gold standard
    spider_data_dir = Path("data/spider/spider_data")
    dev_file = spider_data_dir / "dev.json"

    print(f"\nLoading {dev_file}...")
    with open(dev_file) as f:
        examples = json.load(f)

    # Load predictions
    print("Loading baseline predictions...")
    with open("spider_results_baseline.txt") as f:
        baseline_predictions = [line.strip() for line in f]

    print("Loading M4 predictions...")
    with open("spider_results_m4.txt") as f:
        m4_predictions = [line.strip() for line in f]

    print(f"\n✓ Loaded {len(examples)} examples")

    # Find regressions (baseline correct, M4 wrong)
    regressions = []

    print("\nSearching for regressions...")
    for idx, example in enumerate(examples):
        gold_sql = normalize_sql(example["query"])
        baseline_sql = normalize_sql(baseline_predictions[idx])
        m4_sql = normalize_sql(m4_predictions[idx])

        # Check if baseline matches gold but M4 doesn't
        if baseline_sql == gold_sql and m4_sql != gold_sql:
            db_id = example["db_id"]
            db_path = spider_data_dir / "database" / db_id / f"{db_id}.sqlite"

            # Get costs for both queries
            baseline_plan = get_explain_plan(baseline_predictions[idx], str(db_path))
            m4_plan = get_explain_plan(m4_predictions[idx], str(db_path))

            baseline_cost = calculate_plan_cost(baseline_plan)
            m4_cost = calculate_plan_cost(m4_plan)

            regressions.append(RegressionExample(
                index=idx,
                db_id=db_id,
                question=example["question"],
                gold_sql=example["query"],
                baseline_sql=baseline_predictions[idx],
                m4_sql=m4_predictions[idx],
                baseline_cost=baseline_cost,
                m4_cost=m4_cost,
            ))

    print(f"\n✓ Found {len(regressions)} regressions")

    # Show first 5 examples
    print("\n" + "=" * 80)
    print("TOP 5 REGRESSION EXAMPLES")
    print("=" * 80)

    for i, reg in enumerate(regressions[:5], 1):
        print(f"\n{'=' * 80}")
        print(f"EXAMPLE {i} (Index: {reg.index}, DB: {reg.db_id})")
        print(f"{'=' * 80}")

        print(f"\n📝 Question:")
        print(f"   {reg.question}")

        print(f"\n✅ Baseline Query (CORRECT - Cost: {reg.baseline_cost}):")
        print(f"   {reg.baseline_sql}")

        print(f"\n❌ M4 Query (INCORRECT - Cost: {reg.m4_cost}):")
        print(f"   {reg.m4_sql}")

        print(f"\n🎯 Gold Standard:")
        print(f"   {reg.gold_sql}")

        # Analysis
        if reg.m4_cost < reg.baseline_cost:
            print(f"\n💡 HYPOTHESIS CONFIRMED: M4 picked lower-cost query ({reg.m4_cost} < {reg.baseline_cost})")
            print(f"   M4 chose efficiency over correctness!")
        elif reg.m4_cost > reg.baseline_cost:
            print(f"\n⚠️  HYPOTHESIS REJECTED: M4 picked higher-cost query ({reg.m4_cost} > {reg.baseline_cost})")
            print(f"   Cost was not the deciding factor.")
        else:
            print(f"\n➡️  EQUAL COST: Both queries have same cost ({reg.m4_cost})")
            print(f"   Cost-based ranking didn't differentiate.")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")

    m4_lower_cost = sum(1 for r in regressions if r.m4_cost < r.baseline_cost)
    m4_higher_cost = sum(1 for r in regressions if r.m4_cost > r.baseline_cost)
    equal_cost = sum(1 for r in regressions if r.m4_cost == r.baseline_cost)

    print(f"\nTotal Regressions: {len(regressions)}")
    print(f"\nM4 chose lower-cost query:  {m4_lower_cost} ({m4_lower_cost/len(regressions)*100:.1f}%)")
    print(f"M4 chose higher-cost query: {m4_higher_cost} ({m4_higher_cost/len(regressions)*100:.1f}%)")
    print(f"Equal cost:                 {equal_cost} ({equal_cost/len(regressions)*100:.1f}%)")

    if m4_lower_cost > len(regressions) * 0.5:
        print(f"\n🔍 CONCLUSION: M4 is sacrificing correctness for efficiency!")
        print(f"   In {m4_lower_cost/len(regressions)*100:.1f}% of regressions, M4 picked the lower-cost (but wrong) query.")
    else:
        print(f"\n🔍 CONCLUSION: Cost is NOT the main issue.")
        print(f"   Only {m4_lower_cost/len(regressions)*100:.1f}% of regressions show M4 preferring lower cost.")
        print(f"   The problem may be in beam search diversity or validation.")

    print(f"\n{'=' * 80}")

    # Save full analysis to JSON
    analysis_file = Path("results/regression_analysis.json")
    analysis_file.parent.mkdir(parents=True, exist_ok=True)

    with open(analysis_file, "w") as f:
        json.dump({
            "total_regressions": len(regressions),
            "m4_lower_cost_count": m4_lower_cost,
            "m4_higher_cost_count": m4_higher_cost,
            "equal_cost_count": equal_cost,
            "examples": [
                {
                    "index": r.index,
                    "db_id": r.db_id,
                    "question": r.question,
                    "gold_sql": r.gold_sql,
                    "baseline_sql": r.baseline_sql,
                    "m4_sql": r.m4_sql,
                    "baseline_cost": r.baseline_cost,
                    "m4_cost": r.m4_cost,
                }
                for r in regressions
            ]
        }, f, indent=2)

    print(f"✓ Full analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
