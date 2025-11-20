# BIRD VES Benchmark Methodology

## Overview

This document describes the Valid Efficiency Score (VES) benchmark implementation for evaluating SQL generation strategies on the BIRD Mini-Dev dataset.

## Dataset: BIRD Mini-Dev

**Source:** BIRD (Big Bench for Large-Scale Database Grounded Text-to-SQL Evaluation)
**Version:** Mini-Dev (500 carefully selected examples from BIRD Dev)
**Download:** https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip

### Key Features
- 500 high-quality SELECT-only instances
- 11 end-user level databases (e.g., california_schools, debit_card_specializing, european_football_2)
- Includes `evidence` field with external knowledge/hints critical for accuracy
- Representative subset of full BIRD distribution
- Avoids the 33GB download of full BIRD dataset

### Data Structure
```json
{
  "question_id": 1471,
  "db_id": "debit_card_specializing",
  "question": "What is the ratio of customers who pay in EUR against customers who pay in CZK?",
  "evidence": "ratio of customers who pay in EUR against customers who pay in CZK = count(Currency = 'EUR') / count(Currency = 'CZK').",
  "SQL": "SELECT CAST(SUM(IIF(Currency = 'EUR', 1, 0)) AS FLOAT) / SUM(IIF(Currency = 'CZK', 1, 0)) AS ratio FROM customers",
  "difficulty": "simple"
}
```

**Critical:** The `evidence` field must be included in the prompt as a hint, or accuracy will significantly degrade.

## VES Metric Definition

### Valid Efficiency Score (VES)

VES measures both **correctness** and **execution efficiency** of generated SQL queries.

#### Formula
```
VES = 0                        if incorrect
VES = √(T_gold / T_pred)       if correct
```

Where:
- `T_gold` = Execution time of gold (reference) SQL query
- `T_pred` = Execution time of predicted SQL query

#### Interpretation
- **VES = 0**: Query is incorrect (wrong results or execution error)
- **VES < 1.0**: Query is correct but slower than gold
- **VES = 1.0**: Query matches gold performance
- **VES > 1.0**: Query is correct and faster than gold (efficiency improvement!)

#### BIRD Official VES Scoring
The official BIRD benchmark uses a bucketed reward system:
- `time_ratio >= 2.0`: reward = 1.25
- `1.0 <= time_ratio < 2.0`: reward = 1.0
- `0.5 <= time_ratio < 1.0`: reward = 0.75
- `0.25 <= time_ratio < 0.5`: reward = 0.5
- `time_ratio < 0.25`: reward = 0.25

Our implementation uses the raw `√(T_gold / T_pred)` for finer granularity.

## Strategies Evaluated

### Baseline: Greedy Decoding
- **Method:** Standard greedy decoding (num_beams=1)
- **Prompt:** `Question: {question}\nHint: {evidence}\n\nSchema: {schema}`
- **Rationale:** Fastest generation, model's top-1 choice

### M4: Cost-Aware Re-ranking
- **Method:** Generate K=5 beams, re-rank by EXPLAIN cost metrics
- **Cost Calculation:**
  - SCAN operations: +100 points (table scans are expensive)
  - Temp B-tree: +50 points
  - Index seeks (SEARCH): 0 points (optimal)
- **Selection:** Pick lowest-cost valid beam
- **Rationale:** Should prefer queries with index seeks over full table scans

## Benchmark Pipeline

### 1. Data Preparation
```python
# Load BIRD Mini-Dev dataset
examples = load_bird_mini_dev("data/bird/mini_dev_sqlite.json")

# Get database path
db_path = "data/bird/dev_databases/{db_id}/{db_id}.sqlite"

# Extract schema from database
schema = extract_schema(db_path)  # CREATE TABLE statements
```

### 2. Gold Query Execution (Baseline Timing)
```python
# Execute gold SQL to establish performance baseline
gold_results, gold_time_ms, gold_error = execute_sql_with_timeout(
    gold_sql, db_path, timeout=30.0
)
```

**Timeout:** 30 seconds per query
**Purpose:** Measure reference execution time for VES calculation

### 3. Prediction Generation
```python
# Baseline strategy
sql = generate_sql(model, tokenizer, prompt, do_sample=False)

# M4 strategy (cost-aware)
sql, metadata = generator.generate_with_cost_guidance(
    question, schema, db_path, num_beams=5
)
```

### 4. Predicted Query Execution
```python
# Execute predicted SQL
pred_results, pred_time_ms, pred_error = execute_sql_with_timeout(
    pred_sql, db_path, timeout=30.0
)
```

### 5. VES Calculation
```python
# Check correctness (set equality, order-independent)
if set(pred_results) != set(gold_results):
    ves = 0.0
else:
    # Calculate efficiency score
    ves = sqrt(gold_time_ms / pred_time_ms)
```

### 6. Metrics Aggregation
- **Accuracy:** Percentage of correct results
- **Average VES:** Mean VES across all examples
- **Avg Generation Time:** Time to generate SQL
- **Avg Execution Time:** Time to execute SQL

## Expected Outcomes

### Hypothesis
**M4 should match Baseline accuracy but achieve higher VES by preferring efficient execution plans.**

#### Rationale
- BIRD databases are larger than Spider (>1K rows per table vs Spider's small test DBs)
- Full table scans become noticeably slower on larger datasets
- Index seeks provide measurable speedup
- EXPLAIN cost metrics should correlate with actual execution time

### What We're Testing
1. **Does cost-aware re-ranking improve VES?**
   - M4 should select queries with lower execution times
   - Even if accuracy is same, VES should be higher

2. **Accuracy vs Efficiency Trade-off**
   - Does M4 sacrifice accuracy for efficiency?
   - Or can it maintain accuracy while improving VES?

3. **Correlation: EXPLAIN Cost ↔ Execution Time**
   - Do SCAN operations actually take longer?
   - Do index seeks provide measurable speedup?

## Output Files

### Results JSON
```json
{
  "strategy": "M4",
  "total_examples": 50,
  "accuracy": 0.48,
  "avg_ves": 0.856,
  "avg_generation_time_ms": 3124.5,
  "avg_gold_exec_time_ms": 173.2,
  "avg_pred_exec_time_ms": 198.7,
  "examples": [
    {
      "index": 0,
      "db_id": "debit_card_specializing",
      "question": "...",
      "evidence": "...",
      "gold_sql": "...",
      "predicted_sql": "...",
      "ves": 0.925,
      "correctness": "correct",
      "gold_exec_time_ms": 12.3,
      "pred_exec_time_ms": 14.5,
      "generation_time_ms": 3050.2,
      "cost_metadata": {
        "selected_beam_index": 2,
        "cost": 50
      }
    }
  ]
}
```

### Comparison Report
```
COMPARISON: Baseline vs M4
================================================================================
Metric                         Baseline              M4               Δ
--------------------------------------------------------------------------------
Accuracy                         48.0%           48.0%          +0.0%
Average VES                      0.845           0.912         +0.067
Avg Generation Time (ms)        4562.5          3124.5       -1438.0
Avg Pred Exec Time (ms)          234.6           198.7         -35.9
```

## Key Differences from Spider Benchmark

| Aspect | Spider | BIRD |
|--------|--------|------|
| **Dataset Size** | 1,034 dev examples | 500 mini-dev examples |
| **Database Scale** | Small test DBs (<100 rows) | Realistic DBs (1K-100K rows) |
| **Primary Metric** | Execution Accuracy | VES (Accuracy + Efficiency) |
| **Evidence Field** | Not present | Critical for accuracy |
| **Efficiency Signal** | Not measured | Core evaluation metric |
| **Use Case** | Correctness validation | Production readiness |

## Limitations

### Current Implementation
1. **Single iteration:** No retry on timeout (could add iterative execution like official BIRD)
2. **No outlier filtering:** Official BIRD runs queries 3-5 times and filters outliers
3. **SQLite only:** Official BIRD supports MySQL and PostgreSQL
4. **Set equality only:** Doesn't check column order or NULL handling nuances

### Known Challenges
1. **Timeout sensitivity:** 30s timeout may be too strict for complex queries
2. **Cache effects:** First query might be slower due to cold cache
3. **Non-deterministic timing:** Execution time can vary due to system load

## Future Improvements

### Short-term
1. Add iterative execution (3-5 runs) with outlier filtering
2. Implement official BIRD bucketed reward system
3. Add detailed error analysis (timeout vs wrong results vs execution errors)

### Long-term
1. Support MySQL and PostgreSQL (full BIRD compatibility)
2. Scale to full BIRD Dev (11,000+ examples)
3. Test M3 (validation-only) strategy for comparison
4. Explore hybrid strategies (M3 + M4 combination)

## References

1. **BIRD Paper:** "Can LLM Already Serve as A Database Interface?" (Li et al., 2023)
2. **BIRD Leaderboard:** https://bird-bench.github.io/
3. **Official Evaluation Code:** https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird
4. **Mini-Dev Dataset:** https://huggingface.co/datasets/birdsql/bird_mini_dev

---

**Document Version:** 1.0
**Last Updated:** 2025-11-20
**Status:** Benchmark in progress (50 examples test run)
