# EGTTS Onboarding Guide

Welcome! This guide will get you up to speed on the Execution-Guided Text-to-SQL (EGTTS) project.

## What We're Doing

### Goal
Build a text-to-SQL system that uses **SQLite's EXPLAIN QUERY PLAN** to validate and optimize SQL generation without executing queries on real data.

### Current Status (Nov 2024)

**✅ Completed:**
- **Baseline**: Greedy decoding (49-51% accuracy)
- **M3 (Best)**: Post-hoc beam validation (53% accuracy)
- **M4/M5**: Cost/confidence-aware ranking (48% - underperforms M3)
- **Refactoring**: Unified codebase, consolidated 17 scripts → 4

**❌ Failed Experiments:**
- **EG-SQL**: Execution-guided steering (10% accuracy - worse than baseline)
- **M4/M5**: Cost/confidence-aware ranking (48% - worse than M3)

### Key Findings

1. ✅ **Beam search + validation helps** (M3: 53% vs Baseline: 49%)
2. ❌ **Cost optimization hurts** (M4/M5: 48%)
3. ✅ **First valid beam is best** (model confidence > query cost)
4. ❌ **Real-time steering fails** (EG-SQL: 10% - post-hoc filtering is better)

---

## How We Work

### Development Workflow

```bash
# 1. Setup
uv sync                          # Install dependencies
uv run python scripts/download_spider.sh  # Get data

# 2. Quick test (5 examples)
uv run python scripts/benchmark.py --dataset spider --strategy M3 --limit 5

# 3. Full benchmark (100 examples)
uv run python scripts/benchmark.py --dataset spider --strategy M3 --limit 100

# 4. Evaluate results
uv run python spider_eval/evaluation.py \
  --gold data/spider/spider_data/dev_gold.sql \
  --pred results/spider_m3_100_predictions.txt \
  --db data/spider/spider_data/database \
  --table data/spider/spider_data/tables.json \
  --etype all
```

### Tools & Stack

- **Language**: Python 3.11+
- **Package Manager**: `uv` (fast, modern pip replacement)
- **Model**: Qwen2.5-Coder-7B-Instruct (7B parameters)
- **Framework**: PyTorch + transformers
- **Datasets**: Spider, BIRD
- **Validation**: SQLite EXPLAIN QUERY PLAN

### Entry Points

1. **Main benchmark runner**: `scripts/benchmark.py`
   - Handles all strategies (baseline, M3, M4, M5)
   - Configurable: dataset, limit, num_beams, threshold
   - Outputs predictions + metadata

2. **EG-SQL steering test**: `scripts/test_steering_100.py`
   - Tests execution-guided steering
   - Clause-aware beam search
   - Real-time validation

3. **Utilities**:
   - `scripts/create_gold.py`: Generate gold standard files
   - `scripts/format_predictions.py`: Format results for evaluation
   - `scripts/download_bird.py`: Download BIRD dataset

---

## Code Structure

```
src/egtts/
├── guided.py         # Post-hoc strategies (M3, M4, M5)
│   ├── ExplainGuidedGenerator
│   ├── generate()                    # Baseline
│   ├── generate_with_schema_guidance()  # M3
│   ├── generate_with_cost_guidance()    # M4
│   └── generate_with_confidence_aware_reranking()  # M5
│
├── steering.py       # EG-SQL execution-guided steering
│   ├── ClauseAwareBeamSearch         # Main class
│   ├── speculative_closure()         # Make partial SQL valid
│   └── generate_with_steering()      # Entry point
│
├── database.py       # EXPLAIN validation
│   ├── explain_query()               # Run EXPLAIN QUERY PLAN
│   ├── calculate_plan_cost()         # Extract cost metrics
│   └── ExplainSuccess / ExplainError
│
├── schema.py         # Schema utilities
│   ├── SchemaIndex                   # Fast lookup structure
│   └── build_schema_index()          # Parse CREATE TABLE
│
├── model.py          # Model loading
│   ├── load_model()                  # Load Qwen2.5-Coder
│   └── create_sql_prompt()           # Format input
│
└── data.py           # Dataset loading
    ├── load_spider()
    └── get_database_path()
```

### Key Classes

**`ExplainGuidedGenerator` (guided.py)**
- Main generator for post-hoc strategies
- Handles beam search + filtering
- Methods: `_validate_schema_references()`, `_select_best_valid_beam()`

**`ClauseAwareBeamSearch` (steering.py)**
- Real-time validation during generation
- Custom beam search loop
- Validates at SQL clause boundaries (FROM, WHERE, etc.)

**`SchemaIndex` (schema.py)**
- Fast lookup: tables, columns
- Used for schema validation
- Built from database CREATE statements

---

## Strategies Explained

### Baseline: Greedy Decoding
**How it works**: Standard `model.generate()` with greedy decoding

**Pros**: Fast, simple
**Cons**: 49-51% accuracy, no validation

### M3: Validation Only (Recommended)
**How it works**:
1. Generate K=5 beams
2. Validate each with EXPLAIN
3. Return first valid beam

**Pros**: Best accuracy (53%)
**Cons**: Post-hoc (wastes compute on invalid beams)

**Code**: `generate_with_schema_guidance()` in `guided.py`

### M4: Cost-Aware
**How it works**:
1. Generate K=5 beams
2. Validate with EXPLAIN
3. Pick **lowest-cost** valid beam

**Why it fails**: Low cost ≠ correct query
**Accuracy**: 48% (worse than M3)

### M5: Confidence-Aware
**How it works**:
1. Generate K=5 beams
2. Validate with EXPLAIN
3. Use **model confidence** to gate cost decisions

**Why it fails**: Confidence doesn't correlate with correctness
**Accuracy**: 48% (same as M4, worse than M3)

See [M5_THRESHOLD_ANALYSIS.md](M5_THRESHOLD_ANALYSIS.md) for experiments.

### EG-SQL: Execution-Guided Steering (FAILED - 10% Accuracy)
**How it worked**:
1. Generate SQL **token-by-token**
2. At checkpoints (every 20 tokens):
   - Apply **speculative closure** (make partial SQL valid)
   - Run EXPLAIN QUERY PLAN
   - **Prune invalid beams** immediately
3. Continue generation with only valid beams

**Why it failed**:
- **Multi-line fragmentation**: Model generates incomplete SQL fragments
- **Token-based checkpoints**: Validates mid-word, killing valid beams
- **11x slower**: 17.1s vs M3's 1.5s per query
- **10% accuracy**: vs M3's 53%

**Key learning**: Post-hoc validation >> real-time steering for LLMs

**See**: [EGSQL_FAILURE_ANALYSIS.md](EGSQL_FAILURE_ANALYSIS.md) for full details

---

## Running Experiments

### Compare Strategies

```bash
# Baseline
uv run python scripts/benchmark.py --dataset spider --strategy baseline --limit 100

# M3 (best)
uv run python scripts/benchmark.py --dataset spider --strategy M3 --limit 100

# M4
uv run python scripts/benchmark.py --dataset spider --strategy M4 --limit 100

# Compare
cat results/spider_baseline_100_metadata.json
cat results/spider_m3_100_metadata.json
cat results/spider_m4_100_metadata.json
```

### Test EG-SQL

```bash
# Run steering test (100 examples)
uv run python scripts/test_steering_100.py

# Check results
cat results/steering_100_metadata.json

# Evaluate
uv run python spider_eval/evaluation.py \
  --gold dev_gold_100.sql \
  --pred spider_results_steering_100.txt \
  --db data/spider/spider_data/database \
  --table data/spider/spider_data/tables.json \
  --etype all
```

### Debugging

```python
# In Python REPL
from egtts import load_model, generate_with_steering
from egtts.schema import build_schema_index

# Load model
model, tokenizer = load_model()

# Test on single example
question = "How many singers do we have?"
schema = "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT)"
db_path = "data/spider/spider_data/database/concert_singer/concert_singer.sqlite"

# Generate
sql, metadata = generate_with_steering(model, tokenizer, question, schema, db_path)
print(sql)
print(metadata)
```

---

## Output Files

### Benchmark Outputs

Every benchmark run creates two files:

1. **Predictions**: `results/{dataset}_{strategy}_{limit}_predictions.txt`
   - One SQL query per line
   - Compatible with Spider evaluation

2. **Metadata**: `results/{dataset}_{strategy}_{limit}_metadata.json`
   ```json
   {
     "strategy": "M3",
     "total_examples": 100,
     "successful": 95,
     "failed": 5,
     "avg_generation_time_ms": 1523.4,
     "examples": [
       {
         "index": 0,
         "question": "How many singers?",
         "predicted_sql": "SELECT COUNT(*) FROM singer",
         "gold_sql": "SELECT count(*) FROM singer",
         "generation_time_ms": 1205.3,
         "valid_beams": 3,
         "selected_beam_index": 0
       }
     ]
   }
   ```

---

## Common Tasks

### Add New Strategy

1. Add method to `ExplainGuidedGenerator` in `guided.py`:
```python
def generate_with_my_strategy(self, question, schema, db_path, num_beams=5):
    # Your logic here
    beams = self._generate_beams(question, schema, num_beams)
    # ... validate, rank, select ...
    return best_sql, metadata
```

2. Add to benchmark runner in `scripts/benchmark.py`:
```python
elif strategy == "MY_STRATEGY":
    sql, metadata = generator.generate_with_my_strategy(...)
```

### Add New Dataset

1. Add loader to `data.py`:
```python
def load_my_dataset(data_dir):
    # Return list of examples
    pass
```

2. Add to benchmark runner:
```python
elif dataset == "my_dataset":
    examples = load_my_dataset(...)
```

### Analyze Results

```python
import json

# Load metadata
with open("results/spider_m3_100_metadata.json") as f:
    data = json.load(f)

# Analyze
examples = data["examples"]
valid_count = sum(1 for ex in examples if "error" not in ex)
avg_time = sum(ex["generation_time_ms"] for ex in examples) / len(examples)

print(f"Valid: {valid_count}/{len(examples)}")
print(f"Avg time: {avg_time:.1f}ms")
```

---

## Resources

- **Spider**: https://yale-lily.github.io/spider
- **BIRD**: https://bird-bench.github.io/
- **Qwen2.5-Coder**: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- **SQLite EXPLAIN**: https://www.sqlite.org/eqp.html

## Next Steps

1. Read [M5_THRESHOLD_ANALYSIS.md](M5_THRESHOLD_ANALYSIS.md) for experimental details
2. Check steering test results when complete
3. Compare EG-SQL vs M3 accuracy
4. If steering works: scale to full Spider dev set

---

**Questions?** Check the code comments or ask the team!
