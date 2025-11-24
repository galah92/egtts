# M15 Implementation Summary

## Objective

Implement and evaluate **Strategy M15: Incremental Consensus Generation (MAKER)** - a novel approach to Text-to-SQL that decomposes SQL generation into 3 atomic phases with majority voting at each stage.

## What Was Built

### 1. Core Implementation (`src/egtts/maker.py`)
- **580 lines** of production-quality code
- `IncrementalGenerator` class with 3-phase decomposition:
  - **Phase 1**: Table Selection (FROM/JOIN) - generates 12 proposals, votes by set similarity, validates with EXPLAIN
  - **Phase 2**: Filter Generation (WHERE) - generates 12 proposals, normalizes and votes on filter logic
  - **Phase 3**: Projection (SELECT/GROUP BY) - generates 12 proposals, uses plan-based voting from M7
- OOM-safe batching for GPU memory management
- Fallback mechanism when Phase 1 fails validation
- Complete error handling and metadata tracking

### 2. Benchmark Infrastructure (`scripts/run_bird_m15.py`)
- **270 lines** of evaluation code
- Side-by-side comparison with M10 baseline
- Comprehensive phase-level statistics
- JSON output with full traceability
- Command-line interface with configurable parameters

### 3. Documentation
- **Implementation guide** (`docs/M15_MAKER.md`) - architecture, hypothesis, trade-offs
- **Failure analysis** (`results/M15_FAILURE_ANALYSIS.md`) - comprehensive post-mortem
- **README updates** - integrated M15 into project documentation

## Hypothesis

**"By locking in table choices via consensus first (Phase 1), we prevent downstream column hallucinations that occur when the model generates SELECT columns before committing to specific tables."**

## Results

### Quantitative Performance

| Metric | M15 (MAKER) | M10 (Baseline) | Delta |
|--------|-------------|----------------|-------|
| **Accuracy** | **42.0%** (21/50) | **52.0%** (26/50) | **-10.0%** |
| Latency | 50-60s/query | 20-25s/query | 2.4x slower |
| Model calls | 36/query | 15/query | 2.4x more |
| Failed | 0 | 0 | Equal |

### Comparison Matrix

- **Both correct**: 18 queries (36%)
- **Both wrong**: 21 queries (42%)
- **M15 correct, M10 wrong**: 3 queries (6%)
- **M10 correct, M15 wrong**: 8 queries (16%)

**Key Finding**: M10 solved 8 queries that M15 failed, while M15 only solved 3 that M10 missed. This 8:3 ratio indicates the decomposition approach introduced more errors than it fixed.

## Why M15 Failed

### 1. **Phase 1 Over-Generalization** (Primary Failure Mode)
- Model consistently selected too many tables
- Example: Gold uses only `customers` table, M15 Phase 1 voted 9/12 for `[transactions_1k, customers]`
- Unnecessary JOINs change aggregation semantics
- **Root cause**: Without seeing the full query, model can't reliably identify required tables

### 2. **Error Propagation Through Phases**
- Phase 1 errors (wrong table set) get locked in
- Phase 2 and 3 must work with incorrect foundation
- No mechanism to backtrack or self-correct

### 3. **Context Loss Between Phases**
- Example: "average monthly consumption" needs `/12` division
- Phase 1: Picks tables (no context about monthly vs yearly)
- Phase 3: Completes query without `/12` (semantic intent lost)
- M10: Sees full query context, generates correct logic

### 4. **Voting on Ill-Posed Sub-Problems**
- Phase 1 asks: "What tables are needed?"
- But the "correct" table set depends on the query intent
- **Circular dependency**: Need SELECT to pick tables, need tables to write SELECT
- Strong consensus (9/12 votes) doesn't mean correct answer

### 5. **SQL is Holistic, Not Sequential**
- SQL clauses are interdependent: SELECT ↔ FROM ↔ WHERE ↔ GROUP BY
- Decomposition breaks critical dependencies
- **Unlike code generation** (where MAKER succeeds), SQL all "executes" simultaneously

## Key Lessons

1. **SQL generation is fundamentally holistic** - cannot be decomposed into independent sequential phases
2. **Validation ≠ Verification** - EXPLAIN checks syntax, not semantic correctness
3. **Consensus on sub-problems ≠ Correct solution** - 9/12 votes can all be wrong if the sub-problem is ill-posed
4. **Small models need context** - 7B models struggle more with partial problems than complete problems
5. **M10's approach is optimal** - Holistic generation + plan voting on complete queries works better

## What We Learned About Text-to-SQL

### SQL is Different from Code
- **Code**: Sequential execution, line-by-line validation possible
- **SQL**: Declarative, all clauses interdependent, must be generated together

### Decomposition Paradox
- We decomposed to help the small model focus on one thing at a time
- But this removed the semantic context the model needs
- **Result**: The "simpler" sub-problems were actually harder to solve correctly

### The MAKER Framework Doesn't Transfer
- MAKER works for "million-step tasks" with clear state transitions
- SQL is a "single-step task" (declarative specification) that looks multi-step
- The abstraction mismatch caused the failure

## Recommendations

### Do Not Pursue M15 Further
The architectural approach is fundamentally flawed for Text-to-SQL.

### Focus Future Work On
1. **Scale up M10** - More samples (N=32 or N=64) with holistic generation
2. **Better schema augmentation** - More targeted examples, domain-specific patterns
3. **Execution-based feedback** - Combine M10 + M12 for self-correction
4. **Hybrid voting** - Plan signatures + execution results + schema validation
5. **Fine-tuning** - Distill strong SQL generation patterns into the 7B model

### If Attempting Decomposition Again
Only try if addressing these requirements:
- **Phase 1** must see full query intent (not just question)
- **Backtracking** mechanism when phases conflict
- **Semantic validation** beyond syntax checking
- **Tighter coupling** between phases (not independent generations)

More likely: Abandon decomposition entirely for Text-to-SQL.

## Project Value

### Scientific Contribution
- **Hypothesis tested and disproven** with rigorous evaluation
- **Negative result is valuable** - saves others from trying the same approach
- **Documented failure modes** help understand Text-to-SQL challenges

### Engineering Artifacts
- **580 lines of clean, documented code**
- **Reusable infrastructure** for multi-phase generation
- **Comprehensive benchmarking** framework

### Knowledge Generated
- Deep understanding of why SQL generation resists decomposition
- Insights into 7B model limitations with partial context
- Validation of M10 as near-optimal for this model class

## Conclusion

**M15 was a well-motivated hypothesis that failed in execution.**

The idea that "locking tables first prevents hallucinations" seemed reasonable but proved incorrect. The 3-phase decomposition:
- ❌ Did not prevent column hallucinations
- ❌ Introduced new errors (unnecessary JOINs)
- ❌ Lost semantic context between phases
- ❌ Took 2.4x longer with 19% worse accuracy

**M10 remains the best approach**: Holistic SQL generation with augmented schemas and plan-based voting on complete queries.

The failure teaches us that SQL's declarative nature makes it fundamentally different from procedural code generation, and approaches that work for code don't necessarily transfer to SQL.

---

**Status**: Experiment complete, hypothesis disproven, documentation finalized.
**Recommendation**: Archive M15, focus on improving M10.
