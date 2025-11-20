# EG-SQL Beam Search Visualization

## Overview

The `scripts/visualize_steering.py` tool creates text-based visualizations of EG-SQL's execution-guided beam search, showing how beams are validated and pruned at SQL clause boundaries.

## Usage

### Show Examples with Pruning

```bash
# Show top 5 examples with most beam pruning
uv run python scripts/visualize_steering.py

# Show all examples with pruning
uv run python scripts/visualize_steering.py --all-pruned

# Limit to top 3
uv run python scripts/visualize_steering.py --all-pruned --limit 3
```

### Show Specific Example

```bash
# By index
uv run python scripts/visualize_steering.py --example 43

# By database
uv run python scripts/visualize_steering.py --db concert_singer

# With verbose output (show all checkpoints, not just pruning)
uv run python scripts/visualize_steering.py --example 5 --verbose
```

### Save to File

```bash
# For thesis figures
uv run python scripts/visualize_steering.py --all-pruned --limit 3 --output docs/pruning_examples.txt
```

## Output Format

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EG-SQL BEAM SEARCH VISUALIZATION                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Total examples: 100
Successful: 100 (100.0%)
Average generation time: 15234.5ms
Average checkpoints: 8.3
Average beams pruned: 2.1
Showing 2 example(s)

================================================================================
Example 23: car_1
Question: Find all car models lighter than average weight.
================================================================================

Generated SQL: SELECT T1.model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ...
Generation time: 18456.2ms
Tokens: 35
Checkpoints: 12
Beams pruned: 3

Beam Search Timeline:
--------------------------------------------------------------------------------
├── Checkpoint 1 (×5 beams): ✅ Valid
│   Trigger: KEYWORD
│   SQL: SELECT * FROM
│
├── Checkpoint 2: ❌ PRUNED
│   Trigger: KEYWORD
│   SQL: SELECT * FROM cars WHERE Weight < (SELECT AVG(Weight) FROM
│   └── Beam terminated (invalid)
│
├── Checkpoint 3 (×4 beams): ✅ Valid
│   Trigger: KEYWORD
│   SQL: SELECT T1.model FROM CAR_NAMES AS T1 JOIN
│
├── (7 valid checkpoints not shown - use --verbose)
└── End
```

## What It Shows

### Summary Section
- Total examples processed
- Success rate
- Average metrics (time, checkpoints, pruning)
- Number of examples in visualization

### Per-Example Details
- **Header**: Example index, database ID, question
- **Generated SQL**: Final output (truncated if long)
- **Metadata**: Generation time, tokens, checkpoints, pruned beams

### Beam Search Timeline
- **Checkpoints**: Validation points at SQL clause boundaries
- **Triggers**: What caused validation (KEYWORD or EOS)
- **Status**: ✅ Valid or ❌ PRUNED
- **SQL**: The partial query at that checkpoint
- **Beam count**: How many beams hit this checkpoint (×5 beams)

## Interpreting Results

### Good Pruning Example
```
├── Checkpoint 5: ❌ PRUNED
│   Trigger: KEYWORD
│   SQL: SELECT * FROM invalid_table WHERE
│   └── Beam terminated (invalid)
```
This shows EG-SQL correctly pruned a beam that hallucinated a non-existent table.

### No Pruning (All Valid)
```
Beam Search Timeline:
--------------------------------------------------------------------------------
├── (10 valid checkpoints not shown - use --verbose)
└── End
```
All beams passed validation. Use `--verbose` to see details.

### High Pruning Rate
```
Beams pruned: 15
Checkpoints: 8
```
Average ~2 beams pruned per checkpoint - aggressive pruning, possibly too strict.

### Low Pruning Rate
```
Beams pruned: 1
Checkpoints: 12
```
Minimal pruning - validation is permissive, or examples are simple.

## Use Cases

### 1. Thesis Figures
Create publication-quality examples:
```bash
uv run python scripts/visualize_steering.py \
  --db concert_singer \
  --verbose \
  --output figures/fig3_pruning_example.txt
```

### 2. Debugging
Find examples where pruning helped or hurt:
```bash
# Find aggressive pruning
uv run python scripts/visualize_steering.py --all-pruned --limit 10

# Check specific failure
uv run python scripts/visualize_steering.py --example 87 --verbose
```

### 3. Analysis
Compare pruning across databases:
```bash
for db in concert_singer pets_1 car_1; do
  echo "=== $db ==="
  uv run python scripts/visualize_steering.py --db $db --limit 3
done
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--metadata PATH` | Metadata JSON file | `results/steering_100_metadata.json` |
| `--example N` | Show specific example index | - |
| `--db NAME` | Filter by database ID | - |
| `--all-pruned` | Show all examples with pruning | false |
| `--verbose` | Show all checkpoints, not just pruning | false |
| `--limit N` | Max examples to show | 5 |
| `--output PATH` | Save to file instead of printing | - |

## Metadata Requirements

The visualization tool requires metadata files generated by `scripts/test_steering_100.py` with the following structure:

```json
{
  "strategy": "EG-SQL",
  "total_examples": 100,
  "successful": 98,
  "avg_generation_time_ms": 15000,
  "avg_checkpoints": 8.2,
  "avg_pruned_beams": 2.1,
  "examples": [
    {
      "index": 0,
      "db_id": "concert_singer",
      "question": "How many singers?",
      "predicted_sql": "SELECT COUNT(*) FROM singer",
      "generation_time_ms": 12345.6,
      "tokens_generated": 15,
      "checkpoints": 5,
      "pruned_beams": 0,
      "checkpoint_history": [
        {
          "sql": "SELECT COUNT(*) FROM",
          "trigger": "KEYWORD",
          "valid": true
        }
      ]
    }
  ]
}
```

The `checkpoint_history` field is critical - it captures the actual SQL at each validation point.

## Future Enhancements

Potential improvements:
1. **Graphviz output**: Generate DOT files for publication-quality diagrams
2. **Error messages**: Include actual EXPLAIN error text for pruned beams
3. **Beam divergence**: Show when beams split into different SQL paths
4. **Cost visualization**: Display query cost estimates at checkpoints
5. **Interactive mode**: TUI for exploring beam search interactively

## See Also

- `scripts/test_steering_100.py` - Generates metadata
- `src/egtts/steering.py` - EG-SQL implementation
- `docs/EGSQL_FAILURE_ANALYSIS.md` - Why EG-SQL failed (or succeeded!)
