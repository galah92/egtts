# Thesis Proposal: Non-Destructive Execution-Guided Decoding for Text-to-SQL

**Project Status:** Concept Validation & Environment Setup
**Target Hardware:** NVIDIA L4 (24GB VRAM)
**Primary Model:** Qwen2.5-Coder-7B-Instruct

---

## 1. Executive Summary

This thesis proposes a novel decoding strategy for Text-to-SQL generation that adapts **Execution-Guided Context-Free Grammar (EG-CFG)** to the database domain.

Current State-of-the-Art (SOTA) systems on the BIRD benchmark rely on "Post-Hoc Execution," where fully generated queries are executed against the database to verify correctness. While effective, this is **unsafe** (risk of data mutation), **slow** (execution latency on large datasets), and **inefficient** (wasted tokens).

Our proposed method replaces full execution with the database engine’s **`EXPLAIN` / `QUERY PLAN`** mechanism. This provides a real-time, non-destructive signal that verifies schema validity and estimates query cost without executing the SQL. We hypothesize that this approach will eliminate schema hallucinations and improve query efficiency (VES) while maintaining safety.

---

## 2. Problem Statement

Text-to-SQL models face two primary failure modes:
1.  **Schema Hallucination:** Generating columns or tables that do not exist or hallucinating relationships (e.g., `JOIN`ing tables that have no foreign keys).
2.  **Semantic Invalidity:** Writing SQL that is syntactically correct but logically impossible given the database structure.

Current solutions often utilize **Execution-Guided Decoding**, which pauses generation to run the code and check for errors. In the context of SQL, this is problematic because:
*   **Destructive Operations:** A generated `DROP`, `UPDATE`, or `DELETE` could corrupt the database.
*   **Resource Intensity:** Running a complex `JOIN` on a massive database (common in the BIRD benchmark) takes significant time, slowing down inference.

## 3. Proposed Solution: Plan-Guided Decoding

We propose a **Non-Destructive Execution-Guided** approach. Instead of executing the SQL for results, we invoke the database optimizer’s planning engine (`EXPLAIN QUERY PLAN`) during the generation process.

### Core Mechanics
1.  **Syntactic Constraint (CFG):** We employ a Context-Free Grammar (via libraries like `outlines` or `lm-format-enforcer`) to ensure the LLM can only output syntactically valid SQL structure.
2.  **Semantic Constraint (The "Plan" Signal):** At critical "stop tokens" (e.g., after a `FROM` clause or a full query generation), we invoke `EXPLAIN`.
    *   If the database returns a **Schema Error** (e.g., "no such column"), the generation path is pruned or penalized.
    *   If the database returns a **Plan** (e.g., "SCAN TABLE"), the generation is accepted.
3.  **Efficiency Guidance:** In advanced stages, we analyze the cost returned by the plan. If the plan indicates a full table scan when an index is available, we can prompt the model to retry or prefer an alternative beam that utilizes indices.

---

## 4. Technical Architecture

### Hardware & Environment
*   **GPU:** NVIDIA L4 (24GB VRAM). This provides ample memory for a 7B parameter model (~14GB FP16) with ~10GB remaining for large schema contexts and beam search overhead.
*   **Database:** SQLite (initially) for fast, local feedback loops using the Spider and BIRD datasets.

### The Model
We will utilize **Qwen2.5-Coder-7B-Instruct**.
*   **Rationale:** This model achieves SOTA performance among similarly sized models on coding benchmarks (HumanEval, MBPP) and fits entirely in GPU memory, allowing for high-throughput experimentation without quantization artifacts.

### The Datasets
1.  **Spider (Dev Split):** Used for the initial development cycle due to its lower complexity and standard schema structures.
2.  **BIRD (Dev Split):** Used for final benchmarking, specifically to test the **Valid Efficiency Score (VES)**, where our optimization-guided approach should excel.

---

## 5. Milestone 1: Feasibility & Signal Validation

Before building the complex decoding pipeline, we must validate the core premise: **Is the `EXPLAIN` signal rich enough to catch the errors LLMs actually make?**

### Goal
Prove that the database optimizer explicitly identifies schema hallucinations in generated SQL, distinct from general syntax errors.

### Implementation Plan
We will construct a "Fail Fast" feedback loop:
1.  **Data Preparation:** Load the **Spider** dataset and its corresponding physical SQLite databases.
2.  **Baseline Generation:** Use Qwen2.5-Coder to generate valid SQL for a set of questions.
3.  **Adversarial Generation (The "Dirty" Set):** Force the model to hallucinate by injecting prompt instructions (e.g., *"Filter by the column 'confidence_score'"*—a column known not to exist).
4.  **Signal Extraction:** Run `EXPLAIN QUERY PLAN` on both sets.

### Expected Deliverables
1.  **The Dataset:** A verifiable setup where Python can access the raw `.sqlite` files for Spider.
2.  **The Verify Function:** A Python wrapper that accepts a SQL string and returns either a success object (Plan) or a specific error message.
3.  **The Report:** A quantitative analysis answering:
    *   *What percentage of hallucinated queries result in an `OperationalError`?* (Target: >95%)
    *   *Does `EXPLAIN` catch these errors instantly (<100ms)?*
    *   *Does the error message specifically name the missing entity?* (e.g., "no such column: confidence_score").

### Success Criteria
If `EXPLAIN` consistently returns specific error messages for hallucinated columns without executing the query, the thesis hypothesis is validated, and we proceed to building the constrained beam search decoder (Milestone 2).
