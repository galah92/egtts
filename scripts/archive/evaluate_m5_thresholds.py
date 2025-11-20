"""
Evaluate M5 with different thresholds and analyze decision patterns.
"""

import json
import subprocess
from pathlib import Path
from collections import Counter


def run_evaluation(pred_file: str, output_name: str):
    """Run Spider evaluation on predictions."""
    cmd = [
        "python", "spider_eval/evaluation.py",
        "--gold", "dev_gold_100.sql",
        "--pred", pred_file,
        "--db", "data/spider/spider_data/database",
        "--table", "data/spider/spider_data/tables.json",
        "--etype", "all"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse execution accuracy from output
    output = result.stdout + result.stderr

    # Look for "Execution Accuracy: XX.X%"
    for line in output.split('\n'):
        if 'execution accuracy' in line.lower():
            print(f"{output_name}: {line.strip()}")
            return line

    return None


def analyze_decisions(metadata_file: str, threshold: float):
    """Analyze decision patterns from metadata."""
    with open(metadata_file) as f:
        data = json.load(f)

    examples = data["examples"]

    # Count decision types
    probable_count = 0
    efficient_count = 0
    gaps = []

    for ex in examples:
        if "error_history" in ex and ex["error_history"]:
            # Look for M5 decision in error_history
            for msg in ex["error_history"]:
                if "M5 Decision:" in msg:
                    if "Probable" in msg:
                        probable_count += 1
                        # Extract gap value
                        if "gap=" in msg:
                            gap_str = msg.split("gap=")[1].split()[0]
                            try:
                                gap = float(gap_str)
                                gaps.append(gap)
                            except:
                                pass
                    elif "Efficient" in msg:
                        efficient_count += 1
                        # Extract gap value
                        if "gap=" in msg:
                            gap_str = msg.split("gap=")[1].split()[0]
                            try:
                                gap = float(gap_str)
                                gaps.append(gap)
                            except:
                                pass
                    break

    total = probable_count + efficient_count

    print(f"\n{'=' * 80}")
    print(f"M5 Decision Analysis - Threshold={threshold}")
    print(f"{'=' * 80}")
    print(f"\nTotal decisions: {total}")
    print(f"Chose Probable: {probable_count} ({probable_count/total*100:.1f}%)")
    print(f"Chose Efficient: {efficient_count} ({efficient_count/total*100:.1f}%)")

    if gaps:
        print(f"\nGap statistics:")
        print(f"  Mean gap: {sum(gaps)/len(gaps):.4f}")
        print(f"  Min gap: {min(gaps):.4f}")
        print(f"  Max gap: {max(gaps):.4f}")

        # Gap distribution
        bins = [0.0, 0.05, 0.10, 0.15, 0.20, float('inf')]
        bin_labels = ["0.00-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20+"]
        bin_counts = [0] * len(bin_labels)

        for gap in gaps:
            for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
                if lower <= gap < upper:
                    bin_counts[i] += 1
                    break

        print(f"\nGap distribution:")
        for label, count in zip(bin_labels, bin_counts):
            pct = count / len(gaps) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

    return {
        "threshold": threshold,
        "probable_count": probable_count,
        "efficient_count": efficient_count,
        "gaps": gaps
    }


def main():
    print("=" * 80)
    print("M5 THRESHOLD COMPARISON")
    print("=" * 80)

    # Evaluate all threshold variants
    thresholds = [
        (-0.22, "spider_results_m5_100.txt", "results/m5_test_100_metadata.json"),
        (0.05, "spider_results_m5_t005_100.txt", "results/m5_t005_test_100_metadata.json"),
        (0.10, "spider_results_m5_t010_100.txt", "results/m5_t010_test_100_metadata.json"),
    ]

    results = []

    for threshold, pred_file, metadata_file in thresholds:
        print(f"\n{'=' * 80}")
        print(f"Threshold: {threshold}")
        print(f"{'=' * 80}")

        # Run evaluation
        print(f"\nRunning Spider evaluation on {pred_file}...")
        run_evaluation(pred_file, f"M5 (t={threshold})")

        # Analyze decisions
        analysis = analyze_decisions(metadata_file, threshold)
        results.append(analysis)

    # Comparative summary
    print(f"\n{'=' * 80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n{'Threshold':<12} {'Probable':<12} {'Efficient':<12} {'Efficiency %':<15}")
    print("-" * 60)
    for r in results:
        total = r["probable_count"] + r["efficient_count"]
        eff_pct = r["efficient_count"] / total * 100 if total > 0 else 0
        print(f"{r['threshold']:<12.2f} {r['probable_count']:<12} {r['efficient_count']:<12} {eff_pct:<15.1f}%")

    print(f"\n{'=' * 80}")
    print("Key Insights:")
    print(f"{'=' * 80}")
    print("1. As threshold increases, more cases switch to efficient beam")
    print("2. Compare execution accuracy across thresholds to find optimal balance")
    print("3. Check if efficiency gains come at the cost of accuracy")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
