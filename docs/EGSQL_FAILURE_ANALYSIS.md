# EG-SQL Failure Analysis

## Executive Summary

**EG-SQL (Execution-Guided Steering) achieved 10% accuracy vs M3's 53% baseline**, while being 11x slower (17.1s vs 1.5s per query). This document analyzes why the approach failed and what we learned.

## Results Comparison

| Strategy | Execution Accuracy | Easy | Medium | Hard | Extra | Avg Time |
|----------|-------------------|------|--------|------|-------|----------|
| **M3 (Post-hoc)** | **53%** | - | - | - | - | 1.5s |
| M4 (Cost-aware) | 48% | - | - | - | - | ~2s |
| M5 (Confidence) | 48% | - | - | - | - | ~2s |
| **EG-SQL** | **10%** | 33% | 12% | 0% | 0% | 17.1s |
| Baseline (Greedy) | 49-51% | - | - | - | - | <1s |

## Root Causes of Failure

### 1. Multi-Line Fragmentation (Primary Issue)

**Problem**: The model generates SQL with newlines, treating them as completion points.

**Evidence** (from `spider_results_steering_100.txt`):
```
Line 11: SELECT Country, COUNT(*) AS Number_of_Singers
Line 12: FROM singer
Line 13: GROUP BY Country;
```

The beam search stops after line 11, treating it as a complete query. The remaining clauses (FROM, GROUP BY) never get generated.

**Root Cause**: The custom beam search loop in `steering.py` marks beams as `alive=False` when encountering what it thinks is an EOS token (likely newlines or certain punctuation).

**Impact**: ~60% of queries are incomplete fragments rather than valid SQL.

### 2. Token-Count-Based Validation Checkpoints

**Problem**: Validation triggers every 20 tokens (`tokens_generated % 20 == 0`), regardless of SQL state.

**Example Failure Scenario**:
```python
# Token 20: "SELECT * FROM concert_singer WHERE ag"
# ↓ Validation triggered
# Speculative closure: "SELECT * FROM concert_singer WHERE ag LIMIT 1"
# ↓ SQLite error: "no such column: ag"
# ↓ Beam PRUNED (was mid-word "age")
```

**Impact**: Valid beams are killed while typing column names, operators, or values.

### 3. Speculative Closure Interference

**Problem**: The dummy suffixes added by `speculative_closure()` interfere with generation.

**Example**:
```sql
-- Partial SQL: "SELECT * FROM table WHERE"
-- After closure: "SELECT * FROM table WHERE 1=1 LIMIT 1"
```

When the model is trying to generate `WHERE age > 20`, the closure makes it appear complete, changing the model's continuation probabilities.

**Impact**: The model's token predictions are distorted by the closure artifacts.

### 4. Low Pruning Rate (0.1 beams/query)

**Finding**: Average pruned beams is only 0.1 per query (metadata shows 2.3 checkpoints, minimal pruning).

**Interpretation**:
- Either validation is too permissive (speculative closure makes everything valid)
- Or examples are too simple (unlikely, given 0% on hard queries)
- Or beams are being killed by EOS before reaching checkpoints

**Impact**: The core mechanism (pruning invalid beams early) isn't working as intended.

## Architecture Issues

### Fundamental Mismatch

**EG-SQL assumes**: SQL generation follows a clean clause-by-clause structure
```
SELECT ... → FROM ... → WHERE ... → GROUP BY ... → [DONE]
```

**Reality**: LLMs generate SQL in complex, non-linear ways:
- Multi-line formatting with early stops
- Nested subqueries (can't validate partial inner queries)
- Conditional logic (OR, CASE) that breaks clause boundaries
- Comments and formatting tokens

### Beam Search Complexity

The custom beam search loop has several architectural problems:

1. **State management**: Tracking `alive`, `score`, `text`, `tokens` separately is error-prone
2. **Batch processing**: Padding and attention masks for variable-length beams is complex
3. **Duplicate expansion**: When pruning occurs, duplicating beams to fill slots creates identical search paths
4. **No early stopping**: Unlike HuggingFace's beam search, no sophisticated early stopping heuristics

### Validation Overhead

Each EXPLAIN call takes ~10-50ms:
- 2.3 checkpoints per query
- 5 beams per checkpoint
- = ~11.5 EXPLAIN calls per query
- = ~115-575ms of pure validation overhead

Plus the custom beam search is slower than HuggingFace's optimized implementation.

## What We Learned

### ✅ Positive Findings

1. **EXPLAIN is reliable**: SQLite's EXPLAIN QUERY PLAN works well for validation (when given complete SQL)
2. **Speculative closure is creative**: The idea of making partial SQL valid is novel
3. **Infrastructure is solid**: The schema index, database utilities, and test framework work well

### ❌ Negative Findings

1. **Real-time validation doesn't help**: Guiding generation with EXPLAIN is harder than post-hoc filtering
2. **LLM generation is holistic**: Models don't generate SQL in strict clause order
3. **Beam search is complex**: Custom implementations are bug-prone and slow
4. **Speed matters less than accuracy**: 11x slower is acceptable IF accuracy improves (it didn't)

## Why M3 Works Better

**M3's Advantages**:
1. **Complete queries**: Waits for full generation before validation
2. **Simple logic**: Generate K beams, pick first valid one
3. **Preserves model confidence**: First valid beam is usually the model's top choice
4. **Fast**: Uses HuggingFace's optimized beam search
5. **Robust**: No custom beam management, no fragmentation issues

**M3's Philosophy**: "Let the model do what it does best, then filter"

**EG-SQL's Philosophy**: "Guide the model during generation" ← This doesn't work

## Could EG-SQL Be Fixed?

### Option 1: Fix Multi-Line Fragmentation

**Approach**: Prevent EOS on newlines, force single-line SQL

**Challenges**:
- Requires tokenizer modifications or custom stopping criteria
- Model was trained on multi-line SQL, forcing single-line may hurt quality
- Doesn't solve the token-count checkpoint issue

### Option 2: Clause-Boundary Detection

**Approach**: Validate only at SQL keyword boundaries (FROM, WHERE, etc.)

**Challenges**:
- Keywords can appear in strings, comments, column names ("user_where_clause")
- Requires SQL parsing (which is what we're trying to avoid)
- Nested subqueries break the clause assumption

### Option 3: Post-Generation Validation per Beam

**Approach**: Generate full beams, then validate each

**Result**: This is just M3 with extra steps and custom beam search overhead

### Option 4: Use Parser-Based Validation

**Approach**: Use a SQL parser (like `sqlparse`) to detect when clauses are complete

**Challenges**:
- Adds dependency and complexity
- Parser needs to handle partial/invalid SQL
- Still doesn't solve the model's multi-line generation pattern

## Recommendation

**Abandon EG-SQL and focus on M3 improvements.**

Potential M3 enhancements:
1. **Increase beam width** (K=10 instead of K=5)
2. **Schema-aware prompting** (include schema in prompt more effectively)
3. **Few-shot examples** (add 2-3 examples to prompt)
4. **Model fine-tuning** (train on Spider for domain adaptation)
5. **Ensemble methods** (combine multiple M3 runs)

## Conclusion

EG-SQL was a well-motivated idea: validate during generation to explore more valid paths. However, the execution revealed fundamental mismatches between how we think SQL should be generated (clause-by-clause) and how LLMs actually generate it (holistically, with formatting).

**Key Insight**: Post-hoc validation (M3) is simpler, faster, and more accurate than real-time steering (EG-SQL).

**Final Score**:
- **M3: 53% accuracy, 1.5s** ✅ Winner
- **EG-SQL: 10% accuracy, 17.1s** ❌ Failure

The experiment was valuable for understanding the limits of execution guidance. We should archive EG-SQL and double down on improving M3.

---

**Test Results**: 100/100 Spider examples (2024-11-20)
- Metadata: `results/steering_100_metadata.json`
- Predictions: `spider_results_steering_100.txt`
- Evaluation: 10/100 execution accuracy (10%)
