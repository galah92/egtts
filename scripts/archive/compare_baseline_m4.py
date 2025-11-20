"""
Compare Baseline vs M4 results on Spider benchmark.

This script:
1. Formats baseline predictions with db_id
2. Runs official Spider evaluation on baseline
3. Extracts metrics from both evaluations
4. Generates comparison table
"""

import json
import re
import subprocess
from pathlib import Path


def format_predictions(results_file: str, output_file: str):
    """Format predictions to include db_id for evaluation."""
    spider_data_dir = Path("data/spider/spider_data")
    dev_file = spider_data_dir / "dev.json"

    print(f"Loading {dev_file}...")
    with open(dev_file) as f:
        examples = json.load(f)

    print(f"Loading {results_file}...")
    with open(results_file) as f:
        predictions = [line.strip() for line in f]

    print(f"✓ Loaded {len(examples)} examples and {len(predictions)} predictions")

    # Write predictions in format: SQL\tDB_ID
    with open(output_file, "w") as f:
        for example, pred_sql in zip(examples, predictions):
            db_id = example["db_id"]
            f.write(f"{pred_sql}\t{db_id}\n")

    print(f"✓ Formatted predictions saved to: {output_file}")


def run_evaluation(pred_file: str, output_file: str):
    """Run official Spider evaluation."""
    print(f"\nRunning evaluation on {pred_file}...")

    cmd = [
        "uv", "run", "python", "spider_eval/evaluation.py",
        "--gold", "gold_dev.txt",
        "--pred", pred_file,
        "--db", "data/spider/spider_data/database",
        "--table", "data/spider/spider_data/tables.json",
        "--etype", "all"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Save output
    with open(output_file, "w") as f:
        f.write(result.stdout)

    print(f"✓ Evaluation saved to: {output_file}")
    return result.stdout


def extract_metrics(eval_output: str) -> dict:
    """Extract key metrics from evaluation output."""
    metrics = {}

    # Find the final results table
    lines = eval_output.split('\n')

    # Look for execution accuracy line
    for i, line in enumerate(lines):
        if 'execution' in line.lower() and not line.strip().startswith('='):
            parts = line.split()
            # Format: execution   0.677   0.478   0.580   0.175   0.494
            if len(parts) >= 6:
                metrics['easy_ex'] = float(parts[1])
                metrics['medium_ex'] = float(parts[2])
                metrics['hard_ex'] = float(parts[3])
                metrics['extra_ex'] = float(parts[4])
                metrics['overall_ex'] = float(parts[5])

    # Look for exact match line
    for i, line in enumerate(lines):
        if 'exact match' in line.lower() and not line.strip().startswith('='):
            parts = line.split()
            if len(parts) >= 6:
                metrics['easy_em'] = float(parts[2])
                metrics['medium_em'] = float(parts[3])
                metrics['hard_em'] = float(parts[4])
                metrics['extra_em'] = float(parts[5])
                metrics['overall_em'] = float(parts[6])

    return metrics


def generate_comparison(baseline_metrics: dict, m4_metrics: dict):
    """Generate comparison table."""

    report = """# Baseline vs M4 Comparison

## Executive Summary

**Overall Execution Accuracy:**
- **Baseline (Greedy Decoding):** {baseline_overall:.1%}
- **M4 (Cost-Aware Guidance):** {m4_overall:.1%}
- **Improvement:** {improvement:+.1%} ({improvement_relative:+.1%} relative)

---

## Detailed Comparison

### Execution Accuracy (EX)

| Difficulty | Baseline | M4 | Absolute Δ | Relative Δ |
|-----------|----------|-----|------------|-----------|
| **Easy** | {baseline_easy:.1%} | {m4_easy:.1%} | {easy_abs:+.1%} | {easy_rel:+.1%} |
| **Medium** | {baseline_medium:.1%} | {m4_medium:.1%} | {medium_abs:+.1%} | {medium_rel:+.1%} |
| **Hard** | {baseline_hard:.1%} | {m4_hard:.1%} | {hard_abs:+.1%} | {hard_rel:+.1%} |
| **Extra** | {baseline_extra:.1%} | {m4_extra:.1%} | {extra_abs:+.1%} | {extra_rel:+.1%} |
| **Overall** | **{baseline_overall:.1%}** | **{m4_overall:.1%}** | **{improvement:+.1%}** | **{improvement_relative:+.1%}** |

### Exact Match (EM)

| Difficulty | Baseline | M4 | Absolute Δ | Relative Δ |
|-----------|----------|-----|------------|-----------|
| **Easy** | {baseline_easy_em:.1%} | {m4_easy_em:.1%} | {easy_abs_em:+.1%} | {easy_rel_em:+.1%} |
| **Medium** | {baseline_medium_em:.1%} | {m4_medium_em:.1%} | {medium_abs_em:+.1%} | {medium_rel_em:+.1%} |
| **Hard** | {baseline_hard_em:.1%} | {m4_hard_em:.1%} | {hard_abs_em:+.1%} | {hard_rel_em:+.1%} |
| **Extra** | {baseline_extra_em:.1%} | {m4_extra_em:.1%} | {extra_abs_em:+.1%} | {extra_rel_em:+.1%} |
| **Overall** | **{baseline_overall_em:.1%}** | **{m4_overall_em:.1%}** | **{improvement_em:+.1%}** | **{improvement_relative_em:+.1%}** |

---

## Analysis

### Key Findings

1. **Overall Improvement:** M4 achieves {improvement:.1%} absolute improvement over baseline
   - Relative improvement: {improvement_relative:.1%} ({improvement_relative_pct:.1f}% better)

2. **Per-Difficulty Analysis:**
   - **Easy:** {easy_message}
   - **Medium:** {medium_message}
   - **Hard:** {hard_message}
   - **Extra:** {extra_message}

3. **Method Comparison:**
   - **Baseline:** Greedy decoding (beam 0 only), no validation
   - **M4:** 5-beam search with schema validation, EXPLAIN validation, and cost-aware selection

### Conclusion

The M4 method demonstrates **{conclusion}** improvement over greedy baseline decoding, validating the effectiveness of:
- Multi-beam search for diversity
- Schema validation to eliminate hallucinations
- EXPLAIN-based validation for correctness
- Cost-aware beam selection for efficiency

The improvement is most significant in {best_category} queries ({best_improvement:.1%}), showing that execution-guided decoding particularly helps with {best_category_analysis}.

---

## Files

- `spider_results_baseline.txt` - Baseline predictions (greedy decoding)
- `spider_results_m4.txt` - M4 predictions (cost-aware guidance)
- `eval_output_baseline.txt` - Baseline evaluation results
- `eval_output_m4.txt` - M4 evaluation results (renamed from eval_output.txt)
"""

    # Calculate improvements
    improvement = m4_metrics['overall_ex'] - baseline_metrics['overall_ex']
    improvement_relative = improvement / baseline_metrics['overall_ex'] if baseline_metrics['overall_ex'] > 0 else 0

    improvement_em = m4_metrics['overall_em'] - baseline_metrics['overall_em']
    improvement_relative_em = improvement_em / baseline_metrics['overall_em'] if baseline_metrics['overall_em'] > 0 else 0

    # Per-difficulty improvements
    easy_abs = m4_metrics['easy_ex'] - baseline_metrics['easy_ex']
    easy_rel = easy_abs / baseline_metrics['easy_ex'] if baseline_metrics['easy_ex'] > 0 else 0

    medium_abs = m4_metrics['medium_ex'] - baseline_metrics['medium_ex']
    medium_rel = medium_abs / baseline_metrics['medium_ex'] if baseline_metrics['medium_ex'] > 0 else 0

    hard_abs = m4_metrics['hard_ex'] - baseline_metrics['hard_ex']
    hard_rel = hard_abs / baseline_metrics['hard_ex'] if baseline_metrics['hard_ex'] > 0 else 0

    extra_abs = m4_metrics['extra_ex'] - baseline_metrics['extra_ex']
    extra_rel = extra_abs / baseline_metrics['extra_ex'] if baseline_metrics['extra_ex'] > 0 else 0

    # EM improvements
    easy_abs_em = m4_metrics['easy_em'] - baseline_metrics['easy_em']
    easy_rel_em = easy_abs_em / baseline_metrics['easy_em'] if baseline_metrics['easy_em'] > 0 else 0

    medium_abs_em = m4_metrics['medium_em'] - baseline_metrics['medium_em']
    medium_rel_em = medium_abs_em / baseline_metrics['medium_em'] if baseline_metrics['medium_em'] > 0 else 0

    hard_abs_em = m4_metrics['hard_em'] - baseline_metrics['hard_em']
    hard_rel_em = hard_abs_em / baseline_metrics['hard_em'] if baseline_metrics['hard_em'] > 0 else 0

    extra_abs_em = m4_metrics['extra_em'] - baseline_metrics['extra_em']
    extra_rel_em = extra_abs_em / baseline_metrics['extra_em'] if baseline_metrics['extra_em'] > 0 else 0

    # Determine best improvement category
    improvements = {
        'Easy': easy_abs,
        'Medium': medium_abs,
        'Hard': hard_abs,
        'Extra': extra_abs
    }
    best_category = max(improvements, key=improvements.get)
    best_improvement = improvements[best_category]

    # Category analysis
    category_analysis = {
        'Easy': 'straightforward queries',
        'Medium': 'moderately complex queries requiring JOINs and aggregations',
        'Hard': 'complex multi-table queries with nested conditions',
        'Extra': 'very challenging queries with set operations and advanced SQL'
    }
    best_category_analysis = category_analysis[best_category]

    # Per-difficulty messages
    easy_message = f"{easy_abs:+.1%} improvement" if easy_abs > 0 else f"{easy_abs:.1%} change"
    medium_message = f"{medium_abs:+.1%} improvement" if medium_abs > 0 else f"{medium_abs:.1%} change"
    hard_message = f"{hard_abs:+.1%} improvement" if hard_abs > 0 else f"{hard_abs:.1%} change"
    extra_message = f"{extra_abs:+.1%} improvement" if extra_abs > 0 else f"{extra_abs:.1%} change"

    # Overall conclusion
    if improvement > 0.05:
        conclusion = "significant"
    elif improvement > 0.02:
        conclusion = "meaningful"
    elif improvement > 0:
        conclusion = "modest"
    else:
        conclusion = "no"

    # Calculate percentage for display
    improvement_relative_pct = improvement_relative * 100

    return report.format(
        baseline_overall=baseline_metrics['overall_ex'],
        m4_overall=m4_metrics['overall_ex'],
        improvement=improvement,
        improvement_relative=improvement_relative,
        improvement_relative_pct=improvement_relative_pct,

        baseline_easy=baseline_metrics['easy_ex'],
        m4_easy=m4_metrics['easy_ex'],
        easy_abs=easy_abs,
        easy_rel=easy_rel,

        baseline_medium=baseline_metrics['medium_ex'],
        m4_medium=m4_metrics['medium_ex'],
        medium_abs=medium_abs,
        medium_rel=medium_rel,

        baseline_hard=baseline_metrics['hard_ex'],
        m4_hard=m4_metrics['hard_ex'],
        hard_abs=hard_abs,
        hard_rel=hard_rel,

        baseline_extra=baseline_metrics['extra_ex'],
        m4_extra=m4_metrics['extra_ex'],
        extra_abs=extra_abs,
        extra_rel=extra_rel,

        baseline_easy_em=baseline_metrics['easy_em'],
        m4_easy_em=m4_metrics['easy_em'],
        easy_abs_em=easy_abs_em,
        easy_rel_em=easy_rel_em,

        baseline_medium_em=baseline_metrics['medium_em'],
        m4_medium_em=m4_metrics['medium_em'],
        medium_abs_em=medium_abs_em,
        medium_rel_em=medium_rel_em,

        baseline_hard_em=baseline_metrics['hard_em'],
        m4_hard_em=m4_metrics['hard_em'],
        hard_abs_em=hard_abs_em,
        hard_rel_em=hard_rel_em,

        baseline_extra_em=baseline_metrics['extra_em'],
        m4_extra_em=m4_metrics['extra_em'],
        extra_abs_em=extra_abs_em,
        extra_rel_em=extra_rel_em,

        baseline_overall_em=baseline_metrics['overall_em'],
        m4_overall_em=m4_metrics['overall_em'],
        improvement_em=improvement_em,
        improvement_relative_em=improvement_relative_em,

        easy_message=easy_message,
        medium_message=medium_message,
        hard_message=hard_message,
        extra_message=extra_message,

        conclusion=conclusion,
        best_category=best_category,
        best_improvement=best_improvement,
        best_category_analysis=best_category_analysis,
    )


def main():
    """Run full comparison."""
    print("=" * 80)
    print("BASELINE vs M4 COMPARISON")
    print("=" * 80)

    # Step 1: Format baseline predictions
    print("\n[Step 1] Formatting baseline predictions...")
    format_predictions("spider_results_baseline.txt", "pred_dev_baseline.txt")

    # Step 2: Run baseline evaluation
    print("\n[Step 2] Running baseline evaluation...")
    baseline_output = run_evaluation("pred_dev_baseline.txt", "eval_output_baseline.txt")

    # Step 3: Load M4 eval output
    print("\n[Step 3] Loading M4 results...")
    # Check if eval_output.txt exists (needs renaming) or use eval_output_m4.txt directly
    if Path("eval_output.txt").exists():
        Path("eval_output.txt").rename("eval_output_m4.txt")

    with open("eval_output_m4.txt") as f:
        m4_output = f.read()
    print("✓ M4 evaluation output: eval_output_m4.txt")

    # Step 4: Extract metrics
    print("\n[Step 4] Extracting metrics...")
    baseline_metrics = extract_metrics(baseline_output)
    m4_metrics = extract_metrics(m4_output)

    print(f"✓ Baseline EX: {baseline_metrics['overall_ex']:.1%}")
    print(f"✓ M4 EX: {m4_metrics['overall_ex']:.1%}")

    # Step 5: Generate comparison report
    print("\n[Step 5] Generating comparison report...")
    comparison = generate_comparison(baseline_metrics, m4_metrics)

    with open("baseline_vs_m4_comparison.md", "w") as f:
        f.write(comparison)

    print("✓ Comparison saved to: baseline_vs_m4_comparison.md")

    # Print summary
    improvement = m4_metrics['overall_ex'] - baseline_metrics['overall_ex']
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nBaseline EX:  {baseline_metrics['overall_ex']:.1%}")
    print(f"M4 EX:        {m4_metrics['overall_ex']:.1%}")
    print(f"Improvement:  {improvement:+.1%} ({improvement/baseline_metrics['overall_ex']:+.1%} relative)")
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
