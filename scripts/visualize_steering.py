"""
Visualize EG-SQL beam search decisions.

Creates text-based tree visualizations showing how beams are pruned
based on EXPLAIN validation at clause boundaries.

Usage:
    python scripts/visualize_steering.py --metadata results/steering_100_metadata.json --example 5
    python scripts/visualize_steering.py --metadata results/steering_100_metadata.json --db concert_singer
    python scripts/visualize_steering.py --metadata results/steering_100_metadata.json --all-pruned
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def format_sql_compact(sql: str, max_len: int = 60) -> str:
    """Format SQL for compact display."""
    # Remove code fences
    sql = sql.replace("```sql", "").replace("```", "").strip()
    # Join multi-line
    sql = " ".join(line.strip() for line in sql.split("\n") if line.strip())
    # Truncate if too long
    if len(sql) > max_len:
        sql = sql[:max_len - 3] + "..."
    return sql


def visualize_example(example: Dict[str, Any], verbose: bool = False) -> str:
    """
    Visualize beam search tree for a single example.

    Args:
        example: Example metadata with checkpoint_history
        verbose: Show all checkpoints, not just pruning events

    Returns:
        Formatted tree visualization
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append(f"Example {example['index']}: {example['db_id']}")
    lines.append(f"Question: {example['question']}")
    lines.append("=" * 80)
    lines.append("")

    # Final result
    predicted = format_sql_compact(example.get('predicted_sql', ''), max_len=100)
    lines.append(f"Generated SQL: {predicted}")
    lines.append(f"Generation time: {example.get('generation_time_ms', 0):.1f}ms")
    lines.append(f"Tokens: {example.get('tokens_generated', 0)}")
    lines.append(f"Checkpoints: {example.get('checkpoints', 0)}")
    lines.append(f"Beams pruned: {example.get('pruned_beams', 0)}")
    lines.append("")

    # Checkpoint history
    checkpoints = example.get('checkpoint_history', [])

    if not checkpoints:
        lines.append("No checkpoint history available.")
        return "\n".join(lines)

    lines.append("Beam Search Timeline:")
    lines.append("-" * 80)

    # Group checkpoints by unique SQL to show beam diversity
    sql_groups = {}
    for i, checkpoint in enumerate(checkpoints):
        sql = checkpoint.get('sql', '')
        trigger = checkpoint.get('trigger', 'UNKNOWN')
        valid = checkpoint.get('valid', True)

        if sql not in sql_groups:
            sql_groups[sql] = []
        sql_groups[sql].append({
            'checkpoint_idx': i,
            'trigger': trigger,
            'valid': valid
        })

    # Display checkpoints
    checkpoint_num = 0
    for sql, occurrences in sql_groups.items():
        checkpoint_num += 1

        # Get first occurrence info
        first = occurrences[0]
        trigger = first['trigger']
        valid = first['valid']
        count = len(occurrences)

        # Only show pruning events unless verbose
        if not verbose and valid:
            continue

        # Format SQL
        sql_display = format_sql_compact(sql, max_len=70)

        # Status indicator
        if valid:
            status = "✅ Valid"
            prefix = "├──"
        else:
            status = "❌ PRUNED"
            prefix = "├──"

        # Show checkpoint
        if count > 1:
            lines.append(f"{prefix} Checkpoint {checkpoint_num} (×{count} beams): {status}")
        else:
            lines.append(f"{prefix} Checkpoint {checkpoint_num}: {status}")

        lines.append(f"│   Trigger: {trigger}")
        lines.append(f"│   SQL: {sql_display}")

        if not valid:
            lines.append(f"│   └── Beam terminated (invalid)")

        lines.append("│")

    if not verbose:
        valid_count = sum(1 for sql, occs in sql_groups.items() if occs[0]['valid'])
        if valid_count > 0:
            lines.append(f"├── ({valid_count} valid checkpoints not shown - use --verbose)")

    lines.append("└── End")
    lines.append("")

    return "\n".join(lines)


def find_pruning_examples(metadata: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find examples with interesting pruning behavior.

    Args:
        metadata: Full metadata from test run
        limit: Maximum number of examples to return

    Returns:
        List of examples with pruning events
    """
    examples = metadata.get('examples', [])

    # Find examples with pruning
    pruned_examples = [
        ex for ex in examples
        if ex.get('pruned_beams', 0) > 0
    ]

    # Sort by number of pruned beams (most interesting first)
    pruned_examples.sort(key=lambda x: x.get('pruned_beams', 0), reverse=True)

    return pruned_examples[:limit]


def main():
    parser = argparse.ArgumentParser(description="Visualize EG-SQL beam search decisions")
    parser.add_argument(
        '--metadata',
        type=str,
        default='results/steering_100_metadata.json',
        help='Path to metadata JSON file'
    )
    parser.add_argument(
        '--example',
        type=int,
        help='Specific example index to visualize'
    )
    parser.add_argument(
        '--db',
        type=str,
        help='Show examples from specific database (e.g., concert_singer)'
    )
    parser.add_argument(
        '--all-pruned',
        action='store_true',
        help='Show all examples with pruning events'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show all checkpoints, not just pruning events'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Maximum number of examples to show (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save visualization to file instead of printing'
    )

    args = parser.parse_args()

    # Load metadata
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return 1

    with open(metadata_path) as f:
        metadata = json.load(f)

    examples = metadata.get('examples', [])

    if not examples:
        print("No examples found in metadata.")
        return 1

    # Select examples to visualize
    selected_examples = []

    if args.example is not None:
        # Specific example
        if 0 <= args.example < len(examples):
            selected_examples = [examples[args.example]]
        else:
            print(f"Error: Example index {args.example} out of range (0-{len(examples)-1})")
            return 1

    elif args.db:
        # Filter by database
        selected_examples = [
            ex for ex in examples
            if ex.get('db_id') == args.db
        ][:args.limit]

        if not selected_examples:
            print(f"No examples found for database: {args.db}")
            return 1

    elif args.all_pruned:
        # All examples with pruning
        selected_examples = find_pruning_examples(metadata, limit=args.limit)

        if not selected_examples:
            print("No examples with pruning events found.")
            print("This is expected if EG-SQL didn't prune any beams.")
            return 1

    else:
        # Default: show most interesting pruning examples
        selected_examples = find_pruning_examples(metadata, limit=args.limit)

        if not selected_examples:
            print("No examples with pruning events found.")
            print("Showing first 5 examples instead...")
            selected_examples = examples[:args.limit]

    # Generate visualizations
    output_lines = []

    # Summary
    output_lines.append("")
    output_lines.append("╔" + "═" * 78 + "╗")
    output_lines.append("║" + " " * 20 + "EG-SQL BEAM SEARCH VISUALIZATION" + " " * 26 + "║")
    output_lines.append("╚" + "═" * 78 + "╝")
    output_lines.append("")

    total = metadata.get('total_examples', 0)
    successful = metadata.get('successful', 0)
    avg_time = metadata.get('avg_generation_time_ms', 0)
    avg_checkpoints = metadata.get('avg_checkpoints', 0)
    avg_pruned = metadata.get('avg_pruned_beams', 0)

    output_lines.append(f"Total examples: {total}")
    output_lines.append(f"Successful: {successful} ({successful/total*100:.1f}%)")
    output_lines.append(f"Average generation time: {avg_time:.1f}ms")
    output_lines.append(f"Average checkpoints: {avg_checkpoints:.1f}")
    output_lines.append(f"Average beams pruned: {avg_pruned:.1f}")
    output_lines.append(f"Showing {len(selected_examples)} example(s)")
    output_lines.append("")

    # Visualize each example
    for example in selected_examples:
        viz = visualize_example(example, verbose=args.verbose)
        output_lines.append(viz)
        output_lines.append("")

    # Output
    output_text = "\n".join(output_lines)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output_text)
        print(f"✓ Visualization saved to: {output_path}")
    else:
        print(output_text)

    return 0


if __name__ == '__main__':
    exit(main())
