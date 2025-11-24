# Strategy M15: Incremental Consensus Generation (MAKER)

## Overview

M15 adapts the "Solving a Million-Step Task" framework to Text-to-SQL generation. Instead of generating the entire SQL query at once, we decompose generation into 3 atomic phases and apply **majority voting** at each phase before proceeding to the next.

## Core Hypothesis

**Column hallucinations occur because the model generates SELECT columns before committing to specific tables.**

By locking in the table scope via consensus first (Phase 1), we provide a stable foundation that prevents the model from hallucinating columns that don't exist in those tables.

## Three-Phase Architecture

### Phase 1: Scope Consensus (FROM/JOIN)
**Goal**: Identify required tables and join structure

**Process**:
1. Generate N=12 diverse proposals for table lists and join conditions
2. Vote by set similarity (table order doesn't matter)
3. Validate each candidate cluster using `EXPLAIN SELECT 1 FROM ...`
4. Pick the cluster with most votes that has a valid join path

**Output**: Locked table set + join conditions

**Why this helps**:
- Multiple samples converge on correct tables (stable)
- Hallucinated table references are noise (low probability)
- EXPLAIN validation ensures the join path is syntactically correct

### Phase 2: Filter Consensus (WHERE)
**Goal**: Generate WHERE clause conditions

**Process**:
1. Provide locked tables from Phase 1 in prompt
2. Generate N=12 diverse WHERE clauses
3. Normalize and vote (lowercase, trim spaces)
4. Pick most common filter logic

**Output**: Locked WHERE clause

**Why this helps**:
- Model only focuses on filter logic, not table selection
- Reduced cognitive load (one concern at a time)
- Filters are more stable than complex aggregations

### Phase 3: Projection Consensus (SELECT/GROUP BY)
**Goal**: Complete the SQL query

**Process**:
1. Provide locked tables and filters in prompt
2. Generate N=12 complete SQL queries
3. Validate with EXPLAIN and compute plan signatures
4. Use plan-based voting (from M7) to select winner
5. Tie-break by lowest execution cost

**Output**: Final SQL query

**Why this helps**:
- Model generates SELECT with full context (tables + filters locked)
- Plan voting filters out hallucinations and syntax variations
- Cost tie-breaking improves efficiency

## Configuration

- **Samples per phase**: N=12 (total 36 calls per query)
- **Schema**: Uses M10's augmented schema (sample data rows)
- **Temperature**: 0.7 for diversity
- **Fallback**: If Phase 1 fails to find valid join path, fall back to M10

## Implementation Details

### Phase 1 Validation
```python
# Test join path with dummy SELECT
test_query = f"SELECT 1 FROM {tables[0]} JOIN {tables[1]} ON {join_conditions} LIMIT 1"
result = explain_query(test_query, db_path)
valid = isinstance(result, ExplainSuccess)
```

### Phase 2 Normalization
```python
# Normalize WHERE clauses for voting
normalized = where_clause.lower()
normalized = re.sub(r'\s+', ' ', normalized)
```

### Phase 3 Voting
```python
# Use M7's plan-based voting
from plans import normalize_plan, vote_by_plan
candidates_with_plans = [(sql, normalize_plan(plan), cost)]
winning_sql, _, stats = vote_by_plan(candidates_with_plans)
```

## Expected Benefits vs M10

**M10 Failure Modes**:
1. Hallucinated columns (67% of failures)
2. Wrong table selection (15% of failures)
3. Incorrect JOIN logic (10% of failures)

**How M15 Addresses These**:
1. **Phase 1 consensus** prevents wrong table selection
2. **EXPLAIN validation** ensures valid join paths
3. **Locked table context in Phase 2/3** reduces column hallucinations
4. **Plan voting in Phase 3** filters out remaining errors

**Hypothesis**: M15 should improve accuracy by 5-10 percentage points over M10 (62% → 67-72%)

## Trade-offs

**Pros**:
- Decomposed problem is easier for 7B model
- Error containment (Phase 1 errors don't propagate)
- Focused prompts reduce cognitive load
- Interpretable (can debug each phase)

**Cons**:
- 3x more inference calls (36 vs 15 for M10)
- Longer latency (3 sequential phases)
- Phase 1 can fail to find valid join path (fallback needed)
- WHERE clause parsing is heuristic-based

## Evaluation Metrics

**Primary**: Accuracy on BIRD Mini-Dev (50 examples)
- M10 baseline: 62% (31/50 correct)
- M15 target: 67%+ (34+ correct)

**Secondary**:
- Phase 1 success rate (valid join paths)
- Phase 2 consensus strength (vote ratios)
- Phase 3 plan voting stats (unique signatures)
- Fallback usage frequency

## Files

- **Implementation**: `src/egtts/maker.py`
- **Benchmark**: `scripts/run_bird_m15.py`
- **Results**: `results/bird_m15_50.json`

## Usage

```bash
# Run M15 on 50 examples with M10 comparison
uv run python scripts/run_bird_m15.py \
  --limit 50 \
  --samples-per-phase 12 \
  --compare-m10

# Check results
cat results/bird_m15_50.json
```

## Potential Improvements

If M15 underperforms:
1. **Increase Phase 1 samples** (N=16) for better table consensus
2. **Add column validation** in Phase 2 (check if filter columns exist in locked tables)
3. **Use execution feedback** in Phase 3 (M12-style correction)
4. **Hybrid approach**: Use M15 for complex queries, M10 for simple ones

If M15 succeeds:
1. **Optimize latency** with parallel phase generation (speculative execution)
2. **Extend to 4 phases** (split Phase 3 into aggregation + projection)
3. **Fine-tune prompts** based on failure analysis
4. **Distill into single-pass model** using M15 as training signal

## References

- "Solving a Million-Step Task" (ICLR 2024) - MAKER framework
- M7: Plan-based majority voting
- M10: Schema augmentation with sample data
- BIRD Benchmark: Text-to-SQL with realistic databases
