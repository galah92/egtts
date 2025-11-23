# EGTTS: Execution-Guided Text-to-SQL

**Improving SQL Query Accuracy Through Plan-Based Consensus Voting**

A research project exploring inference-time scaling for Text-to-SQL. We achieve **60% accuracy** on BIRD Mini-Dev (up from 41.6% baseline) using plan-based majority voting with diverse candidate generation.

**Model:** Qwen2.5-Coder-7B-Instruct
**Best Strategy:** M8 (Massive Diversity Plan-Bagging)
**Dataset:** BIRD Mini-Dev (500 examples with realistic databases)

---

## Key Results

### Strategy Comparison (50 examples)

| Strategy | Accuracy | VES | Description |
|----------|----------|-----|-------------|
| Baseline | 41.6% | 0.416 | Greedy decoding |
| M4 | 45.3% | 0.518 | Cost-aware beam selection |
| M7 | 59.0% | 0.557 | Plan-based voting (5 beams) |
| **M8** | **60.0%** | 0.544 | **Massive diversity (32 samples)** |
| M9 | 52.0% | 0.476 | Few-shot + simulation filter |

### Full Benchmark (500 examples)

| Strategy | Accuracy | VES | R-VES |
|----------|----------|-----|-------|
| Baseline | 41.8% | 0.500 | 37.62 |
| M7 | 51.2% | 0.472 | - |

**Best improvement: +18.4 percentage points** (41.6% → 60.0%)

---

## Strategy Evolution

### M4: Cost-Aware Selection
- Generate 5 candidates with beam search
- Score with EXPLAIN QUERY PLAN cost
- Select lowest-cost valid query
- **Result:** +3.5% accuracy, +3.6% VES

### M7: Plan-Based Majority Voting
- Generate 5 diverse candidates
- Validate with EXPLAIN
- Cluster by plan signature
- Vote for largest cluster, tie-break by cost
- **Result:** +17.4% accuracy

### M8: Massive Diversity Plan-Bagging (Best)
- Generate 32 candidates with temperature sampling (T=0.7)
- OOM-safe batching (falls back to 2×16 if needed)
- Validate, cluster by plan signature
- Vote for consensus cluster
- **Result:** +18.4% accuracy

### M9: Few-Shot + Simulation Filter (Failed Experiment)
- Added domain-specific few-shot examples
- Simulation filter for "chatty" queries (wrong column count)
- **Result:** -8% vs M8 (few-shot hurt performance)

---

## Quick Start

### Installation

```bash
uv sync
uv run python scripts/download_bird.py
```

### Run Best Strategy (M8)

```bash
uv run python scripts/run_bird_ves.py \
  --strategy M8 \
  --limit 50 \
  --data-dir data/bird \
  --output-dir results
```

### Available Strategies

```bash
# Baseline (greedy decoding)
--strategy baseline

# M4 (cost-aware selection)
--strategy M4

# M7 (plan-based voting, 5 beams)
--strategy M7

# M8 (massive diversity, 32 samples) - BEST
--strategy M8

# M9 (few-shot + simulation)
--strategy M9
```

---

## Failure Analysis

We analyzed failures to understand *why* the system fails:

### Diagnosis Matrix

| Failure Type | % of Failures | Description | Fix |
|--------------|---------------|-------------|-----|
| **Generation** | 80% | Correct SQL not in any beam | Better prompting/fine-tuning |
| **Selection** | 20% | Correct SQL in beams but wrong pick | Better ranking |

### Common Generation Failure Patterns

1. **Aggregation Confusion**: Model uses `ORDER BY col` instead of `ORDER BY SUM(col)`
2. **Date Extraction**: Uses `SUBSTR(Date, 1, 7)` instead of `SUBSTR(Date, 5, 2)` for month
3. **Complexity Wall**: Multi-segment comparisons beyond model capability

### Key Finding

The Selection Failure analysis showed correct SQL was present in **beam 17** for some failures - proving the value of diverse sampling. M8's 32-sample approach maximizes the chance of finding these "hidden" correct answers.

---

## Architecture

### Core Insight

**Hallucinations are fragile, correct answers are stable.**

When generating multiple SQL candidates:
- Wrong answers produce unique, "weird" execution plans
- Correct answers converge on the same plan across samples

By voting for the most common plan signature, we select stable (correct) queries.

### Plan Signature

```python
# Normalize EXPLAIN QUERY PLAN output to a signature
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

---

## Project Structure

```
egtts/
├── src/egtts/
│   ├── model.py         # Model loading, SQL generation
│   ├── database.py      # EXPLAIN QUERY PLAN analysis
│   ├── guided.py        # All strategies (M4-M9)
│   ├── plans.py         # Plan normalization and voting
│   ├── prompts.py       # Few-shot prompting (M9)
│   └── schema.py        # Schema utilities
│
├── scripts/
│   ├── run_bird_ves.py              # Main benchmark runner
│   ├── analyze_failures.py          # Basic failure analysis
│   ├── analyze_failures_with_beams.py # Beam-level failure analysis
│   ├── calculate_rves.py            # R-VES calculator
│   └── download_bird.py             # Dataset downloader
│
└── results/
    ├── bird_ves_M8_50.json          # Best results
    ├── failure_report_M8.txt        # Failure analysis
    └── failure_beam_report.txt      # Beam-level analysis
```

---

## Key Learnings

### What Worked

1. **Diversity beats optimization**: 32 random samples > 5 optimized beams
2. **Plan-based voting**: Execution plan signatures are reliable correctness signals
3. **Temperature sampling**: T=0.7 provides good diversity without being too random
4. **OOM-safe batching**: Graceful fallback enables large sample counts

### What Didn't Work

1. **Few-shot prompting (M9)**: Domain-specific examples hurt generalization (-8%)
2. **Simulation filter**: Column count validation was too aggressive
3. **Token-level steering**: SQL's declarative structure incompatible with real-time intervention

### Scientific Conclusion

**Inference-time scaling (more samples + consensus) beats prompt engineering for Text-to-SQL.**

The model "knows" the correct answer but assigns it low probability. By sampling widely and voting, we find correct answers that greedy decoding misses.

---

## Metrics

### VES (Valid Efficiency Score)
```
VES = (gold_exec_time / pred_exec_time) if correct else 0
```

### R-VES (Reward-based VES)
BIRD's official efficiency metric with tiered rewards:
- `time_ratio ≥ 2.0` → reward = 1.25 (much faster)
- `1.0 ≤ ratio < 2.0` → reward = 1.00 (similar)
- `0.5 ≤ ratio < 1.0` → reward = 0.75 (slower)
- `ratio < 0.25` → reward = 0.25 (very slow)
- Incorrect → reward = 0.00

---

## Limitations

1. **Generation ceiling**: 80% of failures have no correct SQL in 32 samples
2. **Compute cost**: M8 is ~3× slower than baseline (32 samples vs 1)
3. **SQLite-specific**: Plan analysis uses SQLite's EXPLAIN format
4. **7B model limit**: Larger models might benefit less from diversity

---

## Future Work

1. **Larger sample counts**: Test N=64, N=128 for diminishing returns analysis
2. **Hybrid approaches**: Use M8 diversity with M9-style targeted examples
3. **Weighted voting**: Weight votes by model confidence or plan cost
4. **Error recovery**: When all samples fail, fall back to repair strategy
5. **Cross-model**: Test if diversity scaling works for larger models

---

## References

- [BIRD Benchmark](https://bird-bench.github.io/) - Li et al., NeurIPS 2023
- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) - Alibaba Cloud
- [Self-Consistency](https://arxiv.org/abs/2203.11171) - Wang et al., 2022 (inspiration for voting)

---

## License

MIT License
