"""
Find concrete examples where lower-probability beams have simpler execution plans.

This script hunts for cases where:
- Beam 0 (highest probability) has a complex execution plan
- Beam X (lower probability) has a simpler plan with fewer operations

Plan Complexity Metric: Number of lines in EXPLAIN QUERY PLAN output
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from egtts.database import ExplainSuccess, explain_query
from egtts.model import load_model, create_sql_prompt, generate_sql


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
        Dictionary with complexity metrics:
        - lines: Total number of plan steps
        - scans: Number of table scans
        - searches: Number of index searches
        - joins: Number of join operations
    """
    lines = len(explain_result.plan)
    scans = sum(1 for step in explain_result.plan if "SCAN" in step["detail"])
    searches = sum(1 for step in explain_result.plan if "SEARCH" in step["detail"])
    joins = sum(1 for step in explain_result.plan if "JOIN" in step["detail"].upper())

    return {
        "lines": lines,
        "scans": scans,
        "searches": searches,
        "joins": joins
    }


def clean_sql(sql: str) -> str:
    """Clean generated SQL by removing markdown and extra whitespace."""
    sql = sql.strip()

    # Remove markdown code blocks
    if "```sql" in sql.lower():
        sql = sql.split("```sql")[-1].split("```")[0].strip()
    elif "```" in sql:
        parts = sql.split("```")
        sql = parts[1].strip() if len(parts) > 1 else sql

    # Join multi-line SQL into single line
    sql = " ".join(line.strip() for line in sql.split("\n") if line.strip())

    return sql


def generate_beams(model, tokenizer, question: str, schema: str, num_beams: int = 5) -> list[str]:
    """Generate multiple SQL candidates using beam search."""
    prompt = create_sql_prompt(question, schema, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    beams = []
    for output in outputs:
        # Decode and clean
        full_text = tokenizer.decode(output, skip_special_tokens=True)

        # Extract SQL after the prompt
        if "<|im_start|>assistant" in full_text:
            sql = full_text.split("<|im_start|>assistant")[-1].strip()
        else:
            sql = full_text

        sql = clean_sql(sql)
        beams.append(sql)

    return beams


def find_optimization_examples(
    limit: int = 200,
    num_beams: int = 5,
    min_complexity_reduction: int = 2,
    debug: bool = False
):
    """
    Search for examples where lower beams have simpler plans.

    Args:
        limit: Number of Spider examples to check
        num_beams: Number of beams to generate
        min_complexity_reduction: Minimum line reduction to report
    """
    print("Loading model...")
    model, tokenizer = load_model()

    print("Loading Spider dataset...")
    dev_file = Path("data/spider/spider_data/dev.json")
    with open(dev_file) as f:
        examples = json.load(f)[:limit]

    print(f"\nSearching {len(examples)} examples with {num_beams} beams each...")
    print(f"Looking for complexity reduction >= {min_complexity_reduction} lines\n")

    optimization_cases = []

    for idx, example in enumerate(tqdm(examples, desc="Processing")):
        question = example["question"]
        db_id = example["db_id"]
        db_path = Path("data/spider/spider_data/database") / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            print(f"\nWarning: Database not found for {db_id} at {db_path}")
            continue

        # Get schema
        schema = get_database_schema(db_path)

        # Generate beams
        try:
            beams = generate_beams(model, tokenizer, question, schema, num_beams)
        except Exception as e:
            print(f"\nError generating beams for example {idx}: {e}")
            continue

        # Analyze each beam
        beam_results = []
        for beam_idx, sql in enumerate(beams):
            result = explain_query(sql, str(db_path))

            if isinstance(result, ExplainSuccess):
                complexity = count_plan_complexity(result)
                beam_results.append({
                    "beam_idx": beam_idx,
                    "sql": sql,
                    "valid": True,
                    "complexity": complexity,
                    "plan": result.plan
                })
            else:
                beam_results.append({
                    "beam_idx": beam_idx,
                    "sql": sql,
                    "valid": False,
                    "error": result.error_message
                })

        # Find valid beams only
        valid_beams = [b for b in beam_results if b["valid"]]

        if debug and valid_beams:
            print(f"\n[DEBUG] Example {idx}: {question[:60]}...")
            for vb in valid_beams:
                print(f"  Beam {vb['beam_idx']}: {vb['complexity']['lines']} lines, "
                      f"{vb['complexity']['scans']} scans, {vb['complexity']['searches']} searches")

        if len(valid_beams) < 2:
            continue

        # Check if beam 0 is more complex than any other beam
        beam0 = valid_beams[0]
        beam0_lines = beam0["complexity"]["lines"]

        for other_beam in valid_beams[1:]:
            other_lines = other_beam["complexity"]["lines"]
            reduction = beam0_lines - other_lines

            if reduction >= min_complexity_reduction:
                # Found an optimization case!
                case = {
                    "example_idx": idx,
                    "question": question,
                    "db_id": db_id,
                    "beam0": {
                        "sql": beam0["sql"],
                        "complexity": beam0["complexity"],
                        "plan": beam0["plan"]
                    },
                    "simpler_beam": {
                        "beam_idx": other_beam["beam_idx"],
                        "sql": other_beam["sql"],
                        "complexity": other_beam["complexity"],
                        "plan": other_beam["plan"]
                    },
                    "complexity_reduction": reduction
                }
                optimization_cases.append(case)

                print(f"\n{'='*80}")
                print(f"🎯 OPTIMIZATION FOUND (Example {idx})")
                print(f"{'='*80}")
                print(f"Question: {question}")
                print(f"\n📊 Beam 0 (High Probability) - {beam0_lines} plan steps:")
                print(f"SQL: {beam0['sql']}")
                print(f"Complexity: {beam0['complexity']}")
                print("\nEXPLAIN Plan:")
                for step in beam0["plan"]:
                    print(f"  {step['detail']}")

                print(f"\n✨ Beam {other_beam['beam_idx']} (Lower Probability) - {other_lines} plan steps:")
                print(f"SQL: {other_beam['sql']}")
                print(f"Complexity: {other_beam['complexity']}")
                print("\nEXPLAIN Plan:")
                for step in other_beam["plan"]:
                    print(f"  {step['detail']}")

                print(f"\n💡 Complexity Reduction: {reduction} fewer plan steps")
                print(f"{'='*80}\n")

                # Only report the first simpler beam for this example
                break

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Examples processed: {len(examples)}")
    print(f"Optimization cases found: {len(optimization_cases)}")

    if optimization_cases:
        print("\nComplexity reductions:")
        for case in optimization_cases:
            print(f"  Example {case['example_idx']}: -{case['complexity_reduction']} steps "
                  f"(Beam 0: {case['beam0']['complexity']['lines']} → "
                  f"Beam {case['simpler_beam']['beam_idx']}: {case['simpler_beam']['complexity']['lines']})")
    else:
        print(f"\n❌ No optimization cases found with reduction >= {min_complexity_reduction}")
        print("Try running with a lower --min-reduction value")

    # Save results
    output_file = Path("results/optimization_examples.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "parameters": {
                "limit": limit,
                "num_beams": num_beams,
                "min_complexity_reduction": min_complexity_reduction
            },
            "total_examples": len(examples),
            "optimization_cases_found": len(optimization_cases),
            "cases": optimization_cases
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    return optimization_cases


def main():
    parser = argparse.ArgumentParser(
        description="Find examples where lower-probability beams have simpler execution plans"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of Spider examples to check (default: 200)"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams to generate (default: 5)"
    )
    parser.add_argument(
        "--min-reduction",
        type=int,
        default=2,
        help="Minimum plan complexity reduction to report (default: 2)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information for each example"
    )

    args = parser.parse_args()

    find_optimization_examples(
        limit=args.limit,
        num_beams=args.num_beams,
        min_complexity_reduction=args.min_reduction,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
