"""
End-to-end pipeline test: Load model, generate SQL, verify with EXPLAIN.

This script validates that the full pipeline works:
1. Load Qwen2.5-Coder-7B-Instruct
2. Load Spider examples
3. Generate SQL from natural language questions
4. Verify generated SQL with EXPLAIN
5. Compare against gold SQL
"""

import json
import sqlite3
from pathlib import Path

from egtts import (
    ExplainError,
    ExplainSuccess,
    create_sql_prompt,
    explain_query,
    generate_sql,
    load_model,
)


def get_database_schema(db_path: Path) -> str:
    """Extract CREATE TABLE statements from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    schema_parts = []
    for (table_name,) in tables:
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        create_stmt = cursor.fetchone()[0]
        schema_parts.append(create_stmt)

    conn.close()
    return "\n\n".join(schema_parts)


def test_pipeline(num_examples: int = 3):
    """
    Test the full pipeline on a few Spider examples.

    Args:
        num_examples: Number of examples to test
    """
    print("=" * 80)
    print("END-TO-END PIPELINE TEST")
    print("=" * 80)

    # Load Spider examples
    spider_data_dir = Path("data/spider/spider_data")
    dev_file = spider_data_dir / "dev.json"

    print(f"\n[1/4] Loading Spider examples from {dev_file}...")
    with open(dev_file) as f:
        examples = json.load(f)[:num_examples]
    print(f"  ✓ Loaded {len(examples)} examples")

    # Load model
    print("\n[2/4] Loading Qwen2.5-Coder-7B-Instruct...")
    print("  (This may take 1-2 minutes on first load)")
    model, tokenizer = load_model()
    print(f"  ✓ Model loaded on {model.device}")

    # Process each example
    print(f"\n[3/4] Generating SQL for {num_examples} questions...")
    print("=" * 80)

    results = []

    for idx, example in enumerate(examples):
        print(f"\n[Example {idx + 1}/{num_examples}]")
        print(f"Database: {example['db_id']}")
        print(f"Question: {example['question']}")

        db_path = spider_data_dir / "database" / example["db_id"] / f"{example['db_id']}.sqlite"

        if not db_path.exists():
            print(f"  ✗ Database not found: {db_path}")
            continue

        # Get schema
        schema = get_database_schema(db_path)

        # Create prompt with chat template
        prompt = create_sql_prompt(example["question"], schema, tokenizer)

        # Generate SQL
        print("\n  Generating SQL...")
        generated_sql = generate_sql(
            model,
            tokenizer,
            prompt,
            max_new_tokens=256,
            do_sample=False
        )

        print("\n  Generated SQL:")
        print(f"    {generated_sql}")

        # Verify with EXPLAIN
        print("\n  Verifying with EXPLAIN...")
        result = explain_query(generated_sql, str(db_path))

        if isinstance(result, ExplainSuccess):
            print(f"    ✓ Valid ({result.execution_time_ms:.2f}ms)")
            print(f"    Plan: {result.plan[0]['detail'] if result.plan else 'N/A'}")
        else:
            print(f"    ✗ Error ({result.execution_time_ms:.2f}ms)")
            print(f"    Type: {result.error_type}")
            print(f"    Message: {result.error_message}")

        # Compare with gold SQL
        gold_sql = example["query"]
        print("\n  Gold SQL:")
        print(f"    {gold_sql}")

        # Verify gold SQL too
        gold_result = explain_query(gold_sql, str(db_path))
        if isinstance(gold_result, ExplainSuccess):
            print(f"  Gold verification: ✓ Valid ({gold_result.execution_time_ms:.2f}ms)")
        else:
            print(f"  Gold verification: ✗ Error - {gold_result.error_message}")

        results.append({
            "db_id": example["db_id"],
            "question": example["question"],
            "generated_sql": generated_sql,
            "gold_sql": gold_sql,
            "generated_valid": isinstance(result, ExplainSuccess),
            "gold_valid": isinstance(gold_result, ExplainSuccess),
            "error": result.error_message if isinstance(result, ExplainError) else None
        })

        print("\n" + "-" * 80)

    # Summary
    print("\n[4/4] Summary")
    print("=" * 80)

    total = len(results)
    generated_valid = sum(1 for r in results if r["generated_valid"])
    gold_valid = sum(1 for r in results if r["gold_valid"])

    print(f"\nTotal examples: {total}")
    print(f"Generated SQL valid: {generated_valid}/{total} ({generated_valid/total*100:.1f}%)")
    print(f"Gold SQL valid: {gold_valid}/{total} ({gold_valid/total*100:.1f}%)")

    if generated_valid < total:
        print("\nErrors found in generated SQL:")
        for r in results:
            if not r["generated_valid"]:
                print(f"  - {r['db_id']}: {r['error']}")

    print("\n" + "=" * 80)
    print("Pipeline test complete!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Allow specifying number of examples as command line arg
    num_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    test_pipeline(num_examples=num_examples)
