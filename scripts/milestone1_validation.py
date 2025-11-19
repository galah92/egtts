"""
Milestone 1: Feasibility & Signal Validation

Validates that EXPLAIN catches schema hallucinations in generated SQL.

This script implements the complete Milestone 1 pipeline:
1. Load Spider dataset
2. Generate baseline SQL (using gold queries)
3. Generate adversarial SQL (with injected hallucinations)
4. Run EXPLAIN on both sets
5. Produce quantitative analysis
"""

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from egtts import ExplainError, ExplainSuccess, explain_query


@dataclass
class ValidationResult:
    """Result from validating a single SQL query."""
    db_id: str
    question: str
    sql: str
    query_type: Literal["baseline", "adversarial"]
    explain_status: Literal["success", "error"]
    error_type: str | None
    error_message: str | None
    execution_time_ms: float
    hallucination_type: str | None  # e.g., "nonexistent_column", "nonexistent_table"


def load_spider_examples(split_file: Path, limit: int | None = None):
    """
    Load Spider examples from JSON file.

    Args:
        split_file: Path to dev.json or train_spider.json
        limit: Optional limit on number of examples

    Returns:
        List of examples
    """
    with open(split_file) as f:
        examples = json.load(f)

    if limit:
        examples = examples[:limit]

    return examples


def get_database_schema(db_path: Path) -> str:
    """
    Extract database schema from SQLite database.

    Returns CREATE TABLE statements as a formatted string.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    schema_parts = []
    for (table_name,) in tables:
        # Get CREATE statement for each table
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        create_stmt = cursor.fetchone()[0]
        schema_parts.append(create_stmt)

    conn.close()
    return "\n\n".join(schema_parts)


def create_adversarial_query(sql: str, db_path: Path, hallucination_type: str = "column") -> tuple[str, str]:
    """
    Create an adversarial query by injecting non-existent entities.

    Args:
        sql: Original valid SQL query
        db_path: Path to database
        hallucination_type: Type of hallucination ("column" or "table")

    Returns:
        Tuple of (adversarial_sql, hallucination_description)
    """
    if hallucination_type == "column":
        # Common column names that likely don't exist
        fake_columns = ["confidence_score", "priority_level", "status_code", "metadata"]

        # Inject a fake column into the SELECT clause
        if "SELECT" in sql.upper():
            # Simple injection: add fake column to SELECT
            fake_col = fake_columns[hash(sql) % len(fake_columns)]
            adversarial = sql.replace("SELECT", f"SELECT {fake_col}, ", 1)
            return adversarial, f"nonexistent_column:{fake_col}"

    elif hallucination_type == "table":
        # Inject a fake table into FROM clause
        fake_tables = ["user_preferences", "audit_log", "configuration", "metadata_cache"]
        fake_table = fake_tables[hash(sql) % len(fake_tables)]

        if "FROM" in sql.upper():
            # This is a simplification - real implementation would need SQL parsing
            adversarial = sql.replace("FROM", f"FROM {fake_table}, ", 1)
            return adversarial, f"nonexistent_table:{fake_table}"

    return sql, "none"


def validate_queries(
    spider_data_dir: Path,
    limit: int = 50,
    use_model: bool = False
):
    """
    Run Milestone 1 validation experiment.

    Args:
        spider_data_dir: Path to spider_data directory
        limit: Number of examples to test
        use_model: Whether to use model for generation (expensive) or use gold queries

    Returns:
        List of ValidationResult objects
    """
    print("=" * 80)
    print("MILESTONE 1: FEASIBILITY & SIGNAL VALIDATION")
    print("=" * 80)

    # Load Spider dev split
    dev_file = spider_data_dir / "dev.json"
    print(f"\nLoading Spider examples from {dev_file}...")
    examples = load_spider_examples(dev_file, limit=limit)
    print(f"Loaded {len(examples)} examples")

    results = []

    # Process each example
    for idx, example in enumerate(examples):
        print(f"\n[{idx + 1}/{len(examples)}] Processing {example['db_id']}...")

        db_path = spider_data_dir / "database" / example["db_id"] / f"{example['db_id']}.sqlite"

        if not db_path.exists():
            print(f"  ⚠ Database not found: {db_path}")
            continue

        question = example["question"]
        gold_sql = example["query"]

        # 1. Test baseline (gold query)
        print("  Testing baseline query...")
        result = explain_query(gold_sql, str(db_path))

        baseline_result = ValidationResult(
            db_id=example["db_id"],
            question=question,
            sql=gold_sql,
            query_type="baseline",
            explain_status="success" if isinstance(result, ExplainSuccess) else "error",
            error_type=result.error_type if isinstance(result, ExplainError) else None,
            error_message=result.error_message if isinstance(result, ExplainError) else None,
            execution_time_ms=result.execution_time_ms,
            hallucination_type=None
        )
        results.append(baseline_result)

        if isinstance(result, ExplainSuccess):
            print(f"    ✓ Valid ({result.execution_time_ms:.2f}ms)")
        else:
            print(f"    ✗ Error: {result.error_message}")

        # 2. Test adversarial query (with hallucination)
        adversarial_sql, hallucination_desc = create_adversarial_query(gold_sql, db_path, "column")

        print(f"  Testing adversarial query (hallucination: {hallucination_desc})...")
        result = explain_query(adversarial_sql, str(db_path))

        adversarial_result = ValidationResult(
            db_id=example["db_id"],
            question=question,
            sql=adversarial_sql,
            query_type="adversarial",
            explain_status="success" if isinstance(result, ExplainSuccess) else "error",
            error_type=result.error_type if isinstance(result, ExplainError) else None,
            error_message=result.error_message if isinstance(result, ExplainError) else None,
            execution_time_ms=result.execution_time_ms,
            hallucination_type=hallucination_desc
        )
        results.append(adversarial_result)

        if isinstance(result, ExplainError):
            print(f"    ✓ Caught error ({result.execution_time_ms:.2f}ms): {result.error_message}")
        else:
            print("    ✗ Should have caught error but succeeded")

    return results


def analyze_results(results: list[ValidationResult]):
    """
    Produce quantitative analysis from validation results.

    Answers the questions from PLAN.md:
    1. What percentage of hallucinated queries result in OperationalError?
    2. Does EXPLAIN catch these errors instantly (<100ms)?
    3. Does the error message specifically name the missing entity?
    """
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    baseline_results = [r for r in results if r.query_type == "baseline"]
    adversarial_results = [r for r in results if r.query_type == "adversarial"]

    # Baseline statistics
    baseline_errors = [r for r in baseline_results if r.explain_status == "error"]
    print("\nBaseline Queries (Gold SQL):")
    print(f"  Total: {len(baseline_results)}")
    print(f"  Errors: {len(baseline_errors)} ({len(baseline_errors)/len(baseline_results)*100:.1f}%)")
    if baseline_errors:
        print("  ⚠ Note: Gold queries should not error - possible data issue")

    # Adversarial statistics
    adversarial_errors = [r for r in adversarial_results if r.explain_status == "error"]
    adversarial_success = [r for r in adversarial_results if r.explain_status == "success"]

    print("\nAdversarial Queries (With Hallucinations):")
    print(f"  Total: {len(adversarial_results)}")

    caught_pct = len(adversarial_errors) / len(adversarial_results) * 100
    print(f"  Caught by EXPLAIN: {len(adversarial_errors)} ({caught_pct:.1f}%)")

    escaped_pct = len(adversarial_success) / len(adversarial_results) * 100
    print(f"  Escaped detection: {len(adversarial_success)} ({escaped_pct:.1f}%)")

    # Q1: What percentage result in OperationalError?
    operational_errors = [r for r in adversarial_errors if r.error_type == "OperationalError"]
    op_err_pct = len(operational_errors) / len(adversarial_errors) * 100
    print(f"\n  OperationalErrors: {len(operational_errors)} ({op_err_pct:.1f}% of errors)")

    # Q2: Response time analysis
    adversarial_times = [r.execution_time_ms for r in adversarial_results]
    avg_time = sum(adversarial_times) / len(adversarial_times)
    max_time = max(adversarial_times)
    under_100ms = len([t for t in adversarial_times if t < 100])

    print("\n  Execution Time:")
    print(f"    Average: {avg_time:.2f}ms")
    print(f"    Maximum: {max_time:.2f}ms")
    print(f"    Under 100ms: {under_100ms}/{len(adversarial_times)} ({under_100ms/len(adversarial_times)*100:.1f}%)")

    # Q3: Error message specificity
    print("\n  Error Message Analysis:")
    for r in adversarial_errors[:5]:  # Show first 5
        print(f"    - {r.error_message}")

    # Success criteria check
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA (from PLAN.md)")
    print("=" * 80)

    detection_rate = len(adversarial_errors) / len(adversarial_results) * 100
    speed_compliance = under_100ms / len(adversarial_times) * 100

    print(f"\n1. Detection rate: {detection_rate:.1f}% (target: >95%)")
    if detection_rate > 95:
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")

    print(f"\n2. Speed: {speed_compliance:.1f}% under 100ms")
    if avg_time < 100:
        print("   ✓ PASS (average < 100ms)")
    else:
        print("   ✗ FAIL")

    print("\n3. Error specificity: Check messages above")
    specific_errors = [
        r for r in adversarial_errors
        if "no such column" in r.error_message.lower() or "no such table" in r.error_message.lower()
    ]
    print(f"   {len(specific_errors)}/{len(adversarial_errors)} errors name specific entity")
    if len(specific_errors) / len(adversarial_errors) > 0.9:
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")


def save_results(results: list[ValidationResult], output_file: Path):
    """Save validation results to JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    SPIDER_DATA_DIR = Path("data/spider/spider_data")
    LIMIT = 50  # Number of examples to test
    OUTPUT_FILE = Path("results/milestone1_validation.json")

    # Run validation
    results = validate_queries(
        spider_data_dir=SPIDER_DATA_DIR,
        limit=LIMIT,
        use_model=False  # Use gold queries for speed
    )

    # Analyze and report
    analyze_results(results)

    # Save results
    save_results(results, OUTPUT_FILE)
