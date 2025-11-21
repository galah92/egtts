#!/usr/bin/env python3
"""
Run BIRD official evaluation using our predictions.

This script runs the official BIRD evaluation metrics:
- Execution Accuracy (EX)
- Reward-based Valid Efficiency Score (R-VES)
- Soft F1-Score
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_evaluation(strategy: str, metric: str = "all"):
    """
    Run official BIRD evaluation.

    Args:
        strategy: "baseline" or "M4"
        metric: "ex", "ves", "f1", or "all"
    """
    # Paths
    project_root = Path(__file__).parent.parent
    eval_dir = project_root / "data" / "mini_dev_official" / "evaluation"
    db_root = project_root / "data" / "mini_dev_official" / "sqlite" / "dev_databases"
    ground_truth = project_root / "data" / "mini_dev_official" / "sqlite" / "mini_dev_sqlite_gold.sql"
    diff_json = project_root / "data" / "mini_dev_official" / "sqlite" / "mini_dev_sqlite.jsonl"
    pred_path = project_root / "results" / f"official_{strategy}_500.json"
    output_dir = project_root / "results" / "official_eval"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check files exist
    if not pred_path.exists():
        print(f"Error: Prediction file not found: {pred_path}")
        print(f"Run: python3 scripts/convert_to_official_format.py \\")
        print(f"  --input results/bird_ves_{strategy}_500.json \\")
        print(f"  --output {pred_path}")
        sys.exit(1)

    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        print(f"Run: git clone https://github.com/bird-bench/mini_dev.git data/mini_dev_official")
        sys.exit(1)

    print("=" * 80)
    print(f"BIRD Official Evaluation - {strategy.upper()}")
    print("=" * 80)
    print(f"Predictions: {pred_path}")
    print(f"Database: {db_root}")
    print("")

    metrics_to_run = []
    if metric == "all":
        metrics_to_run = [("ex", "Execution Accuracy"), ("ves", "R-VES"), ("f1", "Soft F1")]
    else:
        metric_names = {"ex": "Execution Accuracy", "ves": "R-VES", "f1": "Soft F1"}
        metrics_to_run = [(metric, metric_names[metric])]

    for metric_name, metric_full in metrics_to_run:
        print(f"\n{'='*80}")
        print(f"Running {metric_full} evaluation...")
        print(f"{'='*80}\n")

        script = eval_dir / f"evaluation_{metric_name}.py"
        output_log = output_dir / f"{strategy}_{metric_name}.txt"

        cmd = [
            sys.executable,  # Use current Python interpreter
            str(script),
            "--db_root_path", str(db_root),
            "--predicted_sql_path", str(pred_path),
            "--ground_truth_path", str(ground_truth),
            "--diff_json_path", str(diff_json),
            "--num_cpus", "16",
            "--meta_time_out", "30.0",
            "--sql_dialect", "SQLite",
            "--output_log_path", str(output_log),
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"\n✓ {metric_full} results saved to: {output_log}")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {metric_full} evaluation failed")
            print(f"Error: {e}")
            continue

    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Results directory: {output_dir}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Run BIRD official evaluation on predictions"
    )
    parser.add_argument(
        "--strategy",
        choices=["baseline", "M4"],
        required=True,
        help="Strategy to evaluate"
    )
    parser.add_argument(
        "--metric",
        choices=["ex", "ves", "f1", "all"],
        default="all",
        help="Metric to run (default: all)"
    )

    args = parser.parse_args()
    run_evaluation(args.strategy, args.metric)


if __name__ == '__main__':
    main()
