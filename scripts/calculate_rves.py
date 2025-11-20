#!/usr/bin/env python3
"""
Calculate R-VES (Reward-based Valid Efficiency Score) from existing benchmark results.

Based on BIRD's official evaluation_ves.py:
https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ves.py
"""

import json
import math
from pathlib import Path


def calculate_reward(time_ratio: float, is_correct: bool) -> float:
    """
    Calculate reward based on time_ratio using BIRD's R-VES thresholds.

    Args:
        time_ratio: gold_time / pred_time
        is_correct: whether the prediction was correct

    Returns:
        reward value between 0 and 1.25
    """
    if not is_correct:
        return 0.0

    if time_ratio >= 2.0:
        return 1.25
    elif time_ratio >= 1.0:
        return 1.0
    elif time_ratio >= 0.5:
        return 0.75
    elif time_ratio >= 0.25:
        return 0.5
    else:
        return 0.25


def calculate_rves(results_path: str) -> dict:
    """
    Calculate R-VES from benchmark results JSON file.

    Args:
        results_path: Path to results JSON file

    Returns:
        Dictionary with R-VES metrics
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    total_reward_score = 0.0
    num_queries = len(data['examples'])
    correct_count = 0
    reward_distribution = {
        '0.00': 0,     # Incorrect
        '0.25': 0,     # Very slow (< 0.25x)
        '0.50': 0,     # Slow (0.25-0.5x)
        '0.75': 0,     # Slightly slower (0.5-1x)
        '1.00': 0,     # Similar speed (1-2x)
        '1.25': 0,     # Much faster (>= 2x)
    }

    for result in data['examples']:
        # Failed queries have 'error' field instead of 'correctness'
        if 'error' in result:
            is_correct = False
        else:
            is_correct = result.get('correctness') == 'correct'

        if is_correct:
            gold_time = result['gold_exec_time_ms']
            pred_time = result['pred_exec_time_ms']

            # Calculate time ratio (gold / pred)
            # If pred is faster, ratio > 1
            # If pred is slower, ratio < 1
            time_ratio = gold_time / pred_time if pred_time > 0 else 0

            reward = calculate_reward(time_ratio, is_correct)
            correct_count += 1
        else:
            reward = 0.0
            time_ratio = 0.0

        # Track reward distribution
        reward_key = f"{reward:.2f}"
        reward_distribution[reward_key] += 1

        # R-VES = sqrt(reward) * 100, averaged across all queries
        total_reward_score += math.sqrt(reward) * 100

    rves = total_reward_score / num_queries

    return {
        'strategy': data['strategy'],
        'total_examples': num_queries,
        'correct': correct_count,
        'incorrect': num_queries - correct_count,
        'rves': rves,
        'reward_distribution': reward_distribution,
        'avg_generation_time_ms': data.get('avg_generation_time_ms', 0),
    }


def main():
    """Calculate R-VES for baseline and M4 strategies."""

    baseline_path = Path('results/bird_ves_baseline_500.json')
    m4_path = Path('results/bird_ves_M4_500.json')

    print("=" * 80)
    print("R-VES CALCULATION (Reward-based Valid Efficiency Score)")
    print("=" * 80)
    print()

    if baseline_path.exists():
        print("Calculating R-VES for BASELINE strategy...")
        baseline_rves = calculate_rves(str(baseline_path))

        print(f"  Total examples:    {baseline_rves['total_examples']}")
        print(f"  Correct:           {baseline_rves['correct']}")
        print(f"  R-VES Score:       {baseline_rves['rves']:.2f}")
        print(f"  Avg Gen Time:      {baseline_rves['avg_generation_time_ms']:.0f}ms")
        print()
        print("  Reward Distribution:")
        for reward, count in sorted(baseline_rves['reward_distribution'].items()):
            pct = count / baseline_rves['total_examples'] * 100
            print(f"    {reward}: {count:3d} ({pct:5.1f}%)")
        print()
    else:
        print(f"ERROR: {baseline_path} not found!")
        baseline_rves = None

    if m4_path.exists():
        print("Calculating R-VES for M4 strategy...")
        m4_rves = calculate_rves(str(m4_path))

        print(f"  Total examples:    {m4_rves['total_examples']}")
        print(f"  Correct:           {m4_rves['correct']}")
        print(f"  R-VES Score:       {m4_rves['rves']:.2f}")
        print(f"  Avg Gen Time:      {m4_rves['avg_generation_time_ms']:.0f}ms")
        print()
        print("  Reward Distribution:")
        for reward, count in sorted(m4_rves['reward_distribution'].items()):
            pct = count / m4_rves['total_examples'] * 100
            print(f"    {reward}: {count:3d} ({pct:5.1f}%)")
        print()
    else:
        print(f"ERROR: {m4_path} not found!")
        m4_rves = None

    if baseline_rves and m4_rves:
        print("=" * 80)
        print("COMPARISON: M4 vs Baseline")
        print("=" * 80)

        rves_improvement = ((m4_rves['rves'] - baseline_rves['rves']) /
                           baseline_rves['rves'] * 100)
        rves_pp_improvement = m4_rves['rves'] - baseline_rves['rves']

        print(f"R-VES Improvement:  {rves_improvement:+.1f}% relative "
              f"({rves_pp_improvement:+.2f} points absolute)")
        print()

        # Save comparison to file
        comparison = {
            'baseline': baseline_rves,
            'm4': m4_rves,
            'improvements': {
                'rves_relative_pct': rves_improvement,
                'rves_absolute_points': rves_pp_improvement,
            }
        }

        output_path = Path('/tmp/rves_comparison.json')
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"Detailed results saved to: {output_path}")
        print()


if __name__ == '__main__':
    main()
