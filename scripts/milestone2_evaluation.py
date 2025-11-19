"""
Milestone 2: EXPLAIN-Guided Iterative Refinement Evaluation

Evaluates iterative refinement approach on Spider dataset.
Compares baseline (one-shot) vs guided (with feedback loop).
"""

import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from egtts import ExplainGuidedGenerator, load_model


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
    """
    Execute SQL and return results.

    Args:
        sql: SQL query to execute
        db_path: Path to database

    Returns:
        Tuple of (success, results_or_error)
        - If success: (True, results as list of tuples)
        - If error: (False, error message)
    """
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
    """
    Compare query results for equality.

    Handles different orderings and formats.
    """
    # Convert to sets of tuples for order-independent comparison
    # Sort each result set to handle ordering differences
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


def evaluate_milestone2(num_examples: int = 100, max_iterations: int = 3):
    """
    Run Milestone 2 evaluation.

    Args:
        num_examples: Number of Spider examples to evaluate
        max_iterations: Maximum refinement iterations
    """
    print("=" * 80)
    print("MILESTONE 2: EXPLAIN-GUIDED ITERATIVE REFINEMENT")
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
    generator = ExplainGuidedGenerator(model, tokenizer, max_iterations=max_iterations)

    # Evaluation metrics
    results = []
    iteration_success = defaultdict(int)  # Success count at each iteration
    error_types = defaultdict(int)
    fixed_errors = defaultdict(int)
    persistent_errors = []

    print(f"\n{'=' * 80}")
    print(f"Evaluating {num_examples} examples with max {max_iterations} iterations")
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

        # Generate with feedback
        result = generator.generate_with_feedback(question, schema, str(db_path))

        # Execute both generated and gold SQL to check correctness
        gen_success, gen_results = execute_sql(result.sql, db_path)
        gold_success, gold_results = execute_sql(gold_sql, db_path)

        # Determine execution accuracy
        execution_correct = False
        if gen_success and gold_success:
            execution_correct = results_match(gen_results, gold_results)
        elif not gen_success and not gold_success:
            # Both failed - consider this a match (edge case)
            execution_correct = True

        # Track metrics
        if execution_correct:
            iteration_success[result.iterations] += 1
            if result.iterations == 0:
                print(f"  ✓ Correct on first try ({result.latency_ms:.0f}ms)")
            else:
                print(f"  ✓ Correct after {result.iterations} iterations ({result.latency_ms:.0f}ms)")
                # Track what error types were fixed
                if result.error_history:
                    first_error = result.error_history[0]
                    if "no such column" in first_error.lower():
                        fixed_errors["schema_column"] += 1
                    elif "no such table" in first_error.lower():
                        fixed_errors["schema_table"] += 1
                    elif "syntax error" in first_error.lower():
                        fixed_errors["syntax"] += 1
                    else:
                        fixed_errors["other"] += 1
        else:
            if result.iterations == 0:
                print(f"  ✗ Incorrect on first try ({result.latency_ms:.0f}ms)")
            else:
                print(f"  ✗ Still incorrect after {result.iterations} iterations ({result.latency_ms:.0f}ms)")

            persistent_errors.append(
                {
                    "db_id": db_id,
                    "question": question,
                    "generated_sql": result.sql,
                    "gold_sql": gold_sql,
                    "gen_valid": result.valid,
                    "gen_success": gen_success,
                    "gold_success": gold_success,
                    "errors": result.error_history,
                }
            )

            # Track error types
            if not result.valid and result.error_history:
                last_error = result.error_history[-1]
                if "no such column" in last_error.lower():
                    error_types["schema_column"] += 1
                elif "no such table" in last_error.lower():
                    error_types["schema_table"] += 1
                elif "syntax error" in last_error.lower():
                    error_types["syntax"] += 1
                else:
                    error_types["other"] += 1
            elif not gen_success:
                error_types["execution_error"] += 1
            else:
                error_types["wrong_answer"] += 1

        # Store detailed result
        results.append(
            {
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "execution_correct": execution_correct,
                "gen_executed": gen_success,
                "gold_executed": gold_success,
                **asdict(result),
            }
        )

    # Analysis
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")

    total = len(results)
    baseline_accuracy = iteration_success[0] / total * 100
    final_accuracy = sum(iteration_success.values()) / total * 100
    improvement = final_accuracy - baseline_accuracy

    print("\nExecution Accuracy (Benchmark Metric):")
    print(f"  Baseline (iteration 0): {iteration_success[0]}/{total} ({baseline_accuracy:.1f}%)")
    print(f"  Final (after refinement): {sum(iteration_success.values())}/{total} ({final_accuracy:.1f}%)")
    print(f"  Improvement: +{improvement:.1f}%")

    print("\nSuccess by Iteration:")
    for i in range(max_iterations + 1):
        count = iteration_success[i]
        pct = count / total * 100 if total > 0 else 0
        print(f"  Iteration {i}: {count}/{total} ({pct:.1f}%)")

    # Average iterations for correct queries
    correct_results = [r for r in results if r["execution_correct"]]
    if correct_results:
        avg_iterations = sum(r["iterations"] for r in correct_results) / len(correct_results)
        print(f"\nAverage iterations (correct): {avg_iterations:.2f}")

    # EXPLAIN validity vs execution correctness
    explain_valid = sum(1 for r in results if r["valid"])
    print("\nEXPLAIN Validity vs Execution Correctness:")
    print(f"  EXPLAIN valid: {explain_valid}/{total} ({explain_valid/total*100:.1f}%)")
    print(f"  Execution correct: {sum(iteration_success.values())}/{total} ({final_accuracy:.1f}%)")
    print(f"  Gap (valid but wrong): {explain_valid - sum(iteration_success.values())}")

    # Latency analysis
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    avg_gen_time = sum(sum(r["generation_times_ms"]) for r in results) / len(results)
    avg_explain_time = sum(sum(r["explain_times_ms"]) for r in results) / len(results)

    print("\nLatency:")
    print(f"  Average total: {avg_latency:.0f}ms")
    print(f"  Average generation: {avg_gen_time:.0f}ms")
    print(f"  Average EXPLAIN: {avg_explain_time:.0f}ms")

    # Error recovery analysis
    print("\nError Recovery:")
    total_fixed = sum(fixed_errors.values())
    print(f"  Total errors fixed: {total_fixed}")
    for error_type, count in fixed_errors.items():
        print(f"    {error_type}: {count}")

    print(f"\n  Persistent errors: {len(persistent_errors)}")
    for error_type, count in error_types.items():
        print(f"    {error_type}: {count}")

    # Success criteria check
    print(f"\n{'=' * 80}")
    print("SUCCESS CRITERIA")
    print(f"{'=' * 80}")

    # Criterion 1: ≥90% of invalid queries fixed
    initially_invalid = total - iteration_success[0]
    fixed_count = sum(iteration_success[i] for i in range(1, max_iterations + 1))
    fix_rate = fixed_count / initially_invalid * 100 if initially_invalid > 0 else 0

    print(f"\n1. Fix rate: {fixed_count}/{initially_invalid} ({fix_rate:.1f}%) (target: ≥90%)")
    if fix_rate >= 90:
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")

    # Criterion 2: Each iteration <5s
    max_iteration_time = max(r["latency_ms"] for r in results) / 1000
    print(f"\n2. Max latency: {max_iteration_time:.2f}s (target: <5s per iteration)")
    if max_iteration_time < 5:
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")

    # Criterion 3: Net improvement ≥20%
    print(f"\n3. Accuracy improvement: +{improvement:.1f}% (target: ≥20%)")
    if improvement >= 20:
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")

    # Save results
    output_file = Path("results/milestone2_evaluation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_examples": total,
        "baseline_accuracy": baseline_accuracy,
        "final_accuracy": final_accuracy,
        "improvement": improvement,
        "avg_iterations": avg_iterations if correct_results else 0,
        "iteration_breakdown": {f"iter_{i}": iteration_success[i] for i in range(max_iterations + 1)},
        "latency": {
            "avg_total_ms": avg_latency,
            "avg_generation_ms": avg_gen_time,
            "avg_explain_ms": avg_explain_time,
            "max_total_s": max_iteration_time,
        },
        "error_recovery": {
            "total_fixed": total_fixed,
            "fixed_by_type": dict(fixed_errors),
            "persistent_by_type": dict(error_types),
        },
        "success_criteria": {
            "fix_rate_pass": fix_rate >= 90,
            "latency_pass": max_iteration_time < 5,
            "improvement_pass": improvement >= 20,
        },
    }

    with open(output_file, "w") as f:
        json.dump({"summary": summary, "detailed_results": results}, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    import sys

    num_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    evaluate_milestone2(num_examples=num_examples)
