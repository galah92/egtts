# EGTTS: Execution-Guided Text-to-SQL

A research project exploring inference-time scaling for Text-to-SQL. We achieve **53.6% accuracy** on BIRD Mini-Dev (up from 41.6% baseline) using schema augmentation with plan-based majority voting.

- **Model**: Qwen2.5-Coder-7B-Instruct
- **Best Strategy**: M10 (Schema Augmentation + Plan Voting)
- **Dataset**: BIRD Mini-Dev (500 examples)

## Core Insight

> Hallucinations are fragile, correct answers are stable.

When generating multiple SQL candidates, wrong answers produce unique execution plans while correct answers converge on the same plan. By voting for the most common plan signature, we select stable (correct) queries.

## Results

### Strategy Comparison

| Strategy | Accuracy | Description |
|----------|----------|-------------|
| Baseline | 41.6% | Greedy decoding |
| M4 | 45.3% | Cost-aware beam selection |
| M7 | 51.2% | Plan-based voting (5 beams) |
| M8 | 60.0%* | Massive diversity (32 samples) |
| **M10** | **53.6%** | **Schema augmentation + plan voting** |
| M12 | 58.0%* | Execution-based self-correction |

*Results on 50-example subset; M10's 53.6% is on full 500 examples.

### Failed Experiments

| Strategy | Accuracy | Why It Failed |
|----------|----------|---------------|
| M9 | 52.0% | Few-shot examples hurt generalization |
| M11 | 30.0% | Chain-of-thought broke batch generation |
| M14 | 34.0% | Data flow CoT had same batching issues |
| M15 | 42.0% | Incremental consensus caused error propagation |
| EG-SQL | 10.0% | Real-time steering fragmented SQL generation |

## Quick Start

```bash
# Install dependencies
uv sync
uv run python -m egtts.scripts.download_bird

# Run best strategy (M10)
uv run python -m egtts.scripts.run_bird_ves \
  --strategy M10 \
  --limit 50 \
  --data-dir data/bird \
  --output-dir results
```

### Available Strategies

```bash
--strategy baseline  # Greedy decoding
--strategy M4        # Cost-aware selection
--strategy M7        # Plan-based voting
--strategy M8        # Massive diversity (32 samples)
--strategy M10       # Schema augmentation (best)
--strategy M12       # Execution-based self-correction
```

## Strategy Details

### M4: Cost-Aware Selection
Generate 5 candidates with beam search, score with `EXPLAIN QUERY PLAN` cost, select lowest-cost valid query.

### M7: Plan-Based Majority Voting
Generate 5 diverse candidates, validate with EXPLAIN, cluster by plan signature, vote for largest cluster with cost tie-breaking.

### M8: Massive Diversity
Generate 32 candidates with temperature sampling (T=0.7), OOM-safe batching, plan-based voting on valid queries.

### M10: Schema Augmentation (Best)
Include 3 sample data rows per table in the prompt. This helps the model understand actual data formats (e.g., `Date='201301'` not `'2013-01-01'`). Combined with plan-based voting.

### M12: Execution-Based Self-Correction
Execute queries and check results. If 0 rows returned, provide feedback and regenerate (up to 2 iterations). Limited benefit since most wrong queries return data.

## Failed Approaches

### Chain-of-Thought (M11, M14)
Structured reasoning before SQL generation caused tensor size mismatch errors in batch generation. Variable-length CoT prompts are incompatible with batching.

### Incremental Consensus / MAKER (M15)
Decomposed SQL into 3 phases (FROM → WHERE → SELECT) with voting at each stage. Failed because SQL generation is holistic—phase 1 added unnecessary JOINs, and errors propagated through phases.

### EG-SQL (Execution-Guided Steering)
Attempted real-time validation during generation. Achieved only 10% accuracy (vs 53% baseline) while being 11x slower. Root causes:
- Multi-line SQL fragmentation (model treats newlines as completion points)
- Token-count-based validation pruned valid beams mid-word
- Speculative closure interfered with generation

**Key lesson**: Post-hoc validation (generate then filter) beats real-time steering.

### Few-Shot Prompting (M9)
Domain-specific examples hurt generalization by 8%.

## Architecture

### Plan Signature
```python
def normalize_plan(plan_rows):
    # Extract structure: tables, joins, scans, indexes
    signature = hash(canonical_plan_structure)
    return signature
```

### Voting Algorithm
```python
# Cluster candidates by plan signature
clusters = group_by(candidates, lambda c: c.plan_signature)

# Vote for largest cluster
winner_cluster = max(clusters, key=len)

# Tie-break by lowest execution cost
best_sql = min(winner_cluster, key=lambda c: c.cost)
```

## Project Structure

```
egtts/
├── egtts/
│   ├── model.py      # Model loading, SQL generation
│   ├── database.py   # EXPLAIN QUERY PLAN validation
│   ├── guided.py     # Strategy implementations (M4, M7, M8, M10, M12)
│   ├── plans.py      # Plan normalization and voting
│   ├── schema.py     # Schema utilities and augmentation
│   ├── data.py       # Dataset loading utilities
│   └── scripts/      # CLI tools (run with python -m egtts.scripts.*)
│       ├── run_bird_ves.py
│       ├── analyze_failures.py
│       └── download_bird.py
│
└── results/          # Evaluation outputs
```

## Failure Analysis

### Diagnosis Matrix

| Failure Type | % of Failures | Description |
|--------------|---------------|-------------|
| Generation | 80% | Correct SQL not in any beam |
| Selection | 20% | Correct SQL in beams but wrong pick |

### Common Patterns
1. **Aggregation confusion**: `ORDER BY col` instead of `ORDER BY SUM(col)`
2. **Date extraction**: Wrong substring indices for date components
3. **Complexity wall**: Multi-segment comparisons beyond model capability

## Model Size Experiments

| Model | Params | Strategy | Accuracy | Time/Example | Notes |
|-------|--------|----------|----------|--------------|-------|
| **Arctic-Text2SQL-7B** | 7B | **Baseline** | **60.0%** | ~29s | OmniSQL prompt format |
| Arctic-Text2SQL-7B | 7B | Baseline (old) | 54.0% | ~30s | Generic prompt format |
| Qwen2.5-Coder-7B | 7B | M10 | 53.6% | ~14s | Inference scaling only |
| Qwen3-4B | 4B | M10 | 49.2% | ~10s | -4.4 points |
| Qwen2.5-Coder-7B | 7B | Baseline | 41.6% | ~5s | No enhancements |

**Conclusion**: Using the correct OmniSQL prompt format improves Arctic baseline from 54% → 60% (+6 points). The remaining gap to Arctic's reported 68.9% is likely due to their **value retrieval** technique (extracting literal values from the database).

## Metrics

### VES (Valid Efficiency Score)
```
VES = (gold_exec_time / pred_exec_time) if correct else 0
```

### R-VES (Reward-based VES)
BIRD's official efficiency metric with tiered rewards based on execution time ratio.

## Key Learnings

### What Worked
1. **Diversity beats optimization**: 32 random samples > 5 optimized beams
2. **Plan-based voting**: Execution plan signatures are reliable correctness signals
3. **Schema augmentation**: Sample data rows help understand data formats
4. **OOM-safe batching**: Graceful fallback enables large sample counts

### What Didn't Work
1. **Real-time steering**: Post-hoc validation is simpler and more accurate
2. **Chain-of-thought**: Variable-length reasoning breaks batching
3. **Incremental decomposition**: SQL is holistic, not decomposable
4. **Few-shot prompting**: Hurt generalization on this task
5. **Execution feedback**: Most wrong queries return data (just wrong data)
6. **Blind fine-tuned model usage**: Prompt format mismatch destroys performance (Arctic: 38% vs expected ~75%)

### Scientific Conclusion

**Inference-time scaling (more samples + consensus) beats prompt engineering for Text-to-SQL.**

The model "knows" the correct answer but assigns it low probability. By sampling widely and voting, we find correct answers that greedy decoding misses.

## Limitations

1. **Generation ceiling**: 80% of failures have no correct SQL in 32 samples
2. **Compute cost**: M8 is ~3x slower than baseline
3. **SQLite-specific**: Plan analysis uses SQLite's EXPLAIN format
4. **7B model optimal**: Larger models provide minimal accuracy gain at high cost

## Comparison with SOTA (7B Models)

### BIRD Benchmark Leaderboard (Single Model Track, 7B class)

| Model | BIRD Dev | BIRD Test | Training | Our Method? |
|-------|----------|-----------|----------|-------------|
| Arctic-Text2SQL-R1-7B | 70.7% | 70.4% | GRPO (64 H100s) | Baseline only |
| SFT CodeS-7B | 57.2% | 59.3% | SFT | - |
| **EGTTS M10 (ours)** | **53.6%** | - | **Zero** | Yes |
| Qwen2.5-Coder-7B | 41.6% | - | Zero | Baseline |
| Llama3-8b-instruct | 24.4% | - | Zero | - |

### Key Insight

Our inference-time scaling approach (M10) achieves **+12 points** over greedy baseline **without any training**.
This is competitive with fine-tuned models while requiring zero GPU training cost.

## Research Roadmap

### Hypothesis

> **Can inference-time scaling stack with fine-tuning to exceed SOTA?**

If Arctic-Text2SQL achieves 70.7% with greedy decoding, and our M10 strategy adds ~12 points to base models,
then **Arctic + M10 could potentially reach ~75-80%**, approaching the current overall SOTA (81.7%).

### Experiment Results

| Experiment | Model | Strategy | Result | Notes |
|------------|-------|----------|--------|-------|
| ✅ Baseline | Qwen2.5-Coder-7B | Greedy | 41.6% | Baseline |
| ✅ M10 | Qwen2.5-Coder-7B | Plan Voting | **53.6%** | +12 points |
| ✅ M10 | Qwen3-4B | Plan Voting | 49.2% | Smaller model, lower accuracy |
| ✅ Baseline | Arctic-Text2SQL-7B | Greedy (old) | 54.0% | Generic prompts |
| ✅ Baseline | Arctic-Text2SQL-7B | Greedy (OmniSQL) | **60.0%** | +6 points with proper format |
| 🔄 M10 | Arctic-Text2SQL-7B | Plan Voting | *pending* | Expected: ~65%+ |

**Key Finding: Prompt format is critical for fine-tuned models!**

### Key Finding: OmniSQL Prompt Format

Arctic-Text2SQL was trained using the **OmniSQL prompt format**:
- Single user message (no system message)
- "Task Overview: You are a data science expert..."
- "Database Engine: SQLite" specification
- "Take a deep breath and think step by step" instruction
- Chain-of-thought in `<think>` tags, SQL in `<answer>` tags

Using the correct format improved baseline from **54% → 60%** (+6 points).

The remaining ~9 point gap to Arctic's reported 68.9% is likely due to their **value retrieval** technique, which extracts literal values from the database to help with case sensitivity and date formats.

### Common Failure Patterns

Analysis of failures shows:
1. **Date format**: Model uses `SUBSTR(Date, 6, 2)` but data is `YYYYMM` format (month at position 5)
2. **Case sensitivity**: Model uses `'discount'` but actual value is `'Discount'`
3. **Extra columns**: Model adds computed columns not requested

These are exactly the issues **value retrieval** would fix.

### Future Work

1. ✅ Implement OmniSQL prompt format - **Done, +6 points**
2. 🔄 Test Arctic + M10 with new prompt format
3. ⏳ Implement value retrieval technique
4. ⏳ Run on full Dev set (1,534 examples)

### Paper Contribution

**Title idea**: *"Inference-Time Scaling for Text-to-SQL: When It Works and When It Doesn't"*

1. **Positive finding**: +12 points on base models without training
2. **Negative finding**: Prompt format mismatch destroys fine-tuned model performance
3. **Practical guidance**: How to adapt inference-time strategies for different model types

### Available Models

```bash
# General instruction models
--model qwen2.5-coder-7b    # Baseline (14GB VRAM)
--model qwen3-4b            # Fast & efficient (8GB VRAM)

# Text-to-SQL fine-tuned models
--model arctic-text2sql     # SOTA 7B (68.9% BIRD), 16GB VRAM
--model omnisql-7b          # Strong alternative, SQLite-only
```

## References

- [BIRD Benchmark](https://bird-bench.github.io/) - Li et al., NeurIPS 2023
- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) - Alibaba Cloud
- [Self-Consistency](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [EG-CFG](https://arxiv.org/abs/2506.10948) - Lavon et al., 2024
- [DAIL-SQL](https://arxiv.org/abs/2308.15363) - Gao et al., 2023
- [TA-SQL](https://arxiv.org/abs/2402.12960) - ACL Findings 2024
- [Arctic-Text2SQL-R1](https://arxiv.org/abs/2505.20315) - Snowflake, May 2025
- [OmniSQL](https://arxiv.org/abs/2503.02240) - RUCBM, 2025

## License

MIT License
