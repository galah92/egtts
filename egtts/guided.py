"""EXPLAIN-guided SQL generation with plan-based voting."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import sqlglot
from sqlglot import exp

from .database import ExplainSuccess, PlanRow, explain_query
from .model import (
    create_arctic_prompt,
    create_sql_prompt,
    extract_sql_from_arctic_response,
    generate_sql,
)
from .schema import SchemaIndex, build_schema_index, get_augmented_schema

if TYPE_CHECKING:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )

    Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


@dataclass
class GenerationResult:
    """Result from SQL generation with refinement."""

    sql: str
    valid: bool
    iterations: int
    error_history: list[str]
    latency_ms: float
    generation_times_ms: list[float]
    explain_times_ms: list[float]


class ExplainGuidedGenerator:
    """
    SQL generator with EXPLAIN-based validation and plan voting.

    Core strategies:
    - Baseline: Greedy decoding
    - M4: Cost-aware beam selection
    - M7: Plan-based majority voting
    - M8: Massive diversity (32 samples)
    - M10: Schema augmentation + plan voting (best)
    - M12: Execution-based self-correction
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        max_iterations: int = 3,
        is_arctic: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.is_arctic = is_arctic

    def generate(self, question: str, schema: str) -> str:
        """Generate SQL query (greedy decoding)."""
        prompt = create_sql_prompt(question, schema, self.tokenizer)
        sql = generate_sql(
            self.model, self.tokenizer, prompt, max_new_tokens=256, do_sample=False
        )
        return sql

    def calculate_plan_cost(self, explain_output: list[PlanRow]) -> int:
        """Calculate heuristic cost score from EXPLAIN QUERY PLAN output."""
        cost = 0
        plan_str = " ".join(str(row) for row in explain_output).upper()

        if "'SCAN " in plan_str and "USING INDEX" not in plan_str:
            cost += 100  # Table scan penalty
        if "USE TEMP B-TREE" in plan_str:
            cost += 50  # Temporary structure penalty

        return cost

    def _validate_schema_references(
        self, sql: str, schema_index: SchemaIndex
    ) -> tuple[bool, str]:
        """Validate that all table and column references exist in schema."""
        try:
            parsed = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return True, ""  # If parsing fails, let EXPLAIN catch it

        for table in parsed.find_all(exp.Table):
            if not schema_index.has_table(table.name):
                return False, f"Table '{table.name}' not found in schema"

        for column in parsed.find_all(exp.Column):
            if column.name != "*" and not schema_index.has_column(column.name):
                return False, f"Column '{column.name}' not found in schema"

        return True, ""

    def _decode_candidates(self, outputs: torch.Tensor, input_length: int) -> list[str]:
        """Decode model outputs into SQL strings."""
        candidates = []
        for output in outputs:
            generated_ids = output[input_length:]
            raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Use Arctic-specific extraction if needed
            if self.is_arctic:
                sql = extract_sql_from_arctic_response(raw_text)
            else:
                sql = raw_text
                # Clean up SQL
                if "```sql" in sql.lower():
                    sql = sql.split("```sql", 1)[1].split("```")[0].strip()
                elif "```" in sql:
                    parts = sql.split("```")
                    if len(parts) >= 2:
                        sql = parts[1].strip()

                lines = sql.split("\n")
                sql_lines = []
                for line in lines:
                    line = line.strip()
                    if (
                        line
                        and not line.startswith("--")
                        and not line.lower().startswith(
                            ("this query", "the query", "note:")
                        )
                    ):
                        sql_lines.append(line)
                    elif sql_lines:
                        break

                sql = " ".join(sql_lines).strip()
            candidates.append(sql)

        return candidates

    def _generate_diverse_samples(
        self,
        inputs: dict[str, torch.Tensor],
        input_length: int,
        num_samples: int,
        temperature: float = 0.7,
    ) -> tuple[list[str], float]:
        """Generate diverse SQL samples with OOM-safe batching."""
        gen_start = time.perf_counter()
        # Arctic needs more tokens for <think> + <answer> format
        max_tokens = 2048 if self.is_arctic else 256

        try:
            with torch.no_grad():
                outputs = self.model.generate(  # type: ignore[operator]
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=num_samples,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_time = (time.perf_counter() - gen_start) * 1000
            return self._decode_candidates(outputs, input_length), gen_time

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                # Fallback: generate in smaller batches
                batch_size = max(1, num_samples // 3)  # Split into 3 batches
                all_candidates = []

                remaining = num_samples
                while remaining > 0:
                    current_batch = min(batch_size, remaining)
                    with torch.no_grad():
                        outputs = self.model.generate(  # type: ignore[operator]
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            num_return_sequences=current_batch,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    # Decode each batch individually to avoid torch.stack issues
                    batch_candidates = self._decode_candidates(outputs, input_length)
                    all_candidates.extend(batch_candidates)
                    remaining -= current_batch
                    torch.cuda.empty_cache()

                gen_time = (time.perf_counter() - gen_start) * 1000
                return all_candidates, gen_time
            raise

    # =========================================================================
    # M4: Cost-Aware Selection
    # =========================================================================

    def generate_with_cost_guidance(
        self, question: str, schema: str, db_path: str, num_beams: int = 5
    ) -> GenerationResult:
        """Generate SQL with cost-aware beam selection (M4)."""
        start_time = time.perf_counter()
        schema_index = build_schema_index(db_path)

        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=256,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_time = (time.perf_counter() - gen_start) * 1000

        candidates = self._decode_candidates(outputs, input_length)

        # Validate and collect costs
        explain_times = []
        error_history = []
        valid_candidates = []

        for idx, sql in enumerate(candidates):
            schema_valid, schema_error = self._validate_schema_references(
                sql, schema_index
            )
            if not schema_valid:
                error_history.append(f"Beam {idx}: Schema error: {schema_error}")
                continue

            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_times.append((time.perf_counter() - explain_start) * 1000)

            if isinstance(result, ExplainSuccess):
                cost = self.calculate_plan_cost(result.plan)
                valid_candidates.append((sql, cost, idx))
            else:
                error_history.append(
                    f"Beam {idx}: EXPLAIN error: {result.error_message}"
                )

        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[1])
            best_sql, best_cost, best_idx = valid_candidates[0]

            return GenerationResult(
                sql=best_sql,
                valid=True,
                iterations=best_idx,
                error_history=error_history,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                generation_times_ms=[gen_time],
                explain_times_ms=explain_times,
            )

        return GenerationResult(
            sql=candidates[0] if candidates else "",
            valid=False,
            iterations=num_beams - 1,
            error_history=error_history,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            generation_times_ms=[gen_time],
            explain_times_ms=explain_times,
        )

    # =========================================================================
    # M7: Plan-Based Majority Voting
    # =========================================================================

    def generate_with_plan_voting(
        self, question: str, schema: str, db_path: str, num_samples: int = 10
    ) -> GenerationResult:
        """Generate SQL with plan-based majority voting (M7)."""
        from .plans import normalize_plan, vote_by_plan

        start_time = time.perf_counter()

        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        candidates, gen_time = self._generate_diverse_samples(
            inputs, input_length, num_samples
        )

        explain_times = []
        error_history = []
        candidates_with_plans = []

        for idx, sql in enumerate(candidates):
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_times.append((time.perf_counter() - explain_start) * 1000)

            if isinstance(result, ExplainSuccess):
                signature = normalize_plan(result.plan)
                cost = self.calculate_plan_cost(result.plan)
                candidates_with_plans.append((sql, signature, cost))
            else:
                error_history.append(f"Candidate {idx}: {result.error_message}")
                candidates_with_plans.append((sql, "ERROR", -1))

        winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)
        error_history.append(f"Vote stats: {vote_stats}")

        valid = winning_sig != "ERROR" and vote_stats.get("valid_candidates", 0) > 0

        return GenerationResult(
            sql=winning_sql,
            valid=valid,
            iterations=vote_stats.get("winning_votes", 0),
            error_history=error_history,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            generation_times_ms=[gen_time],
            explain_times_ms=explain_times,
        )

    # =========================================================================
    # M8: Massive Diversity Plan-Bagging
    # =========================================================================

    def generate_with_massive_diversity(
        self, question: str, schema: str, db_path: str, num_samples: int = 32
    ) -> GenerationResult:
        """Generate SQL with massive diversity plan-bagging (M8)."""
        from .plans import normalize_plan, vote_by_plan

        start_time = time.perf_counter()

        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        candidates, gen_time = self._generate_diverse_samples(
            inputs, input_length, num_samples, temperature=0.7
        )

        explain_times = []
        error_history = []
        candidates_with_plans = []
        syntax_errors = 0
        schema_errors = 0

        for idx, sql in enumerate(candidates):
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_times.append((time.perf_counter() - explain_start) * 1000)

            if isinstance(result, ExplainSuccess):
                signature = normalize_plan(result.plan)
                cost = self.calculate_plan_cost(result.plan)
                candidates_with_plans.append((sql, signature, cost))
            else:
                error_msg = result.error_message.lower()
                if "no such table" in error_msg or "no such column" in error_msg:
                    schema_errors += 1
                else:
                    syntax_errors += 1
                candidates_with_plans.append((sql, "ERROR", -1))

        winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)

        vote_stats["syntax_errors"] = syntax_errors
        vote_stats["schema_errors"] = schema_errors
        error_history.append(f"Vote stats: {vote_stats}")

        valid = winning_sig != "ERROR" and vote_stats.get("valid_candidates", 0) > 0

        return GenerationResult(
            sql=winning_sql,
            valid=valid,
            iterations=vote_stats.get("winning_votes", 0),
            error_history=error_history,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            generation_times_ms=[gen_time],
            explain_times_ms=explain_times,
        )

    # =========================================================================
    # M10: Schema Augmentation + Plan Voting (Best Strategy)
    # =========================================================================

    def generate_with_augmented_schema(
        self, question: str, db_path: str, num_samples: int = 15
    ) -> GenerationResult:
        """Generate SQL with schema augmentation and plan voting (M10)."""
        from .plans import normalize_plan, vote_by_plan

        start_time = time.perf_counter()

        schema_start = time.perf_counter()
        augmented_schema = get_augmented_schema(db_path, max_rows_per_table=3)
        schema_time = (time.perf_counter() - schema_start) * 1000

        # Use Arctic-specific prompt format if needed
        if self.is_arctic:
            prompt = create_arctic_prompt(question, augmented_schema, self.tokenizer)
            # Reduce samples for Arctic due to memory constraints (2048 tokens per sample)
            num_samples = min(num_samples, 8)
        else:
            prompt = create_sql_prompt(question, augmented_schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        candidates, gen_time = self._generate_diverse_samples(
            inputs, input_length, num_samples, temperature=0.7
        )

        explain_times = []
        error_history = []
        candidates_with_plans = []

        for idx, sql in enumerate(candidates):
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_times.append((time.perf_counter() - explain_start) * 1000)

            if isinstance(result, ExplainSuccess):
                signature = normalize_plan(result.plan)
                cost = self.calculate_plan_cost(result.plan)
                candidates_with_plans.append((sql, signature, cost))
            else:
                candidates_with_plans.append((sql, "ERROR", -1))

        winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)

        vote_stats["strategy"] = "M10_AugmentedSchema"
        vote_stats["schema_augmentation_time_ms"] = schema_time
        vote_stats["num_samples"] = num_samples
        error_history.append(f"Vote stats: {vote_stats}")

        valid = winning_sig != "ERROR" and vote_stats.get("valid_candidates", 0) > 0

        return GenerationResult(
            sql=winning_sql,
            valid=valid,
            iterations=vote_stats.get("winning_votes", 0),
            error_history=error_history,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            generation_times_ms=[gen_time, schema_time],
            explain_times_ms=explain_times,
        )

    # =========================================================================
    # M12: Execution-Based Self-Correction
    # =========================================================================

    def generate_with_execution_correction(
        self,
        question: str,
        db_path: str,
        hint: str | None = None,
        num_samples: int = 15,
        max_corrections: int = 2,
    ) -> GenerationResult:
        """Generate SQL with execution-based self-correction (M12)."""
        from .plans import normalize_plan, vote_by_plan

        start_time = time.perf_counter()
        error_history = []
        all_generation_times = []
        all_explain_times = []
        correction_count = 0

        full_question = f"{question}\n\nHint: {hint}" if hint else question

        schema_start = time.perf_counter()
        augmented_schema = get_augmented_schema(db_path, max_rows_per_table=3)
        schema_time = (time.perf_counter() - schema_start) * 1000
        all_generation_times.append(schema_time)

        previous_attempts = []
        winning_sql = ""
        vote_stats = {}

        while correction_count <= max_corrections:
            if previous_attempts:
                feedback = (
                    "\n\nPREVIOUS ATTEMPTS (returned empty - try different approach):\n"
                )
                for i, (sql, reason) in enumerate(previous_attempts[-2:], 1):
                    feedback += f"Attempt {i}: {sql}\nProblem: {reason}\n"
                prompt_question = full_question + feedback
            else:
                prompt_question = full_question

            prompt = create_sql_prompt(
                prompt_question, augmented_schema, self.tokenizer
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs["input_ids"].shape[1]

            candidates, gen_time = self._generate_diverse_samples(
                inputs, input_length, num_samples, temperature=0.7
            )
            all_generation_times.append(gen_time)

            explain_times = []
            candidates_with_plans = []

            for sql in candidates:
                explain_start = time.perf_counter()
                result = explain_query(sql, db_path)
                explain_times.append((time.perf_counter() - explain_start) * 1000)

                if isinstance(result, ExplainSuccess):
                    signature = normalize_plan(result.plan)
                    cost = self.calculate_plan_cost(result.plan)
                    candidates_with_plans.append((sql, signature, cost))
                else:
                    candidates_with_plans.append((sql, "ERROR", -1))

            all_explain_times.extend(explain_times)

            winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)

            if winning_sig == "ERROR" or vote_stats.get("valid_candidates", 0) == 0:
                error_history.append(
                    f"Correction {correction_count}: No valid candidates"
                )
                correction_count += 1
                previous_attempts.append((winning_sql, "Syntax or schema error"))
                continue

            # Execute to check results
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(winning_sql)
                rows = cursor.fetchall()
                conn.close()

                if len(rows) == 0:
                    error_history.append(
                        f"Correction {correction_count}: 0 rows returned"
                    )
                    previous_attempts.append((winning_sql, "Returned 0 rows"))
                    correction_count += 1
                    continue

                if len(rows) == 1 and all(v is None for v in rows[0]):
                    error_history.append(f"Correction {correction_count}: All NULL")
                    previous_attempts.append((winning_sql, "Returned NULL"))
                    correction_count += 1
                    continue

                # Success
                vote_stats["corrections_needed"] = correction_count
                vote_stats["result_rows"] = len(rows)
                error_history.append(f"Vote stats: {vote_stats}")

                return GenerationResult(
                    sql=winning_sql,
                    valid=True,
                    iterations=vote_stats.get("winning_votes", 0),
                    error_history=error_history,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    generation_times_ms=all_generation_times,
                    explain_times_ms=all_explain_times,
                )

            except Exception as e:
                error_history.append(f"Correction {correction_count}: {e!s}")
                previous_attempts.append((winning_sql, f"Error: {e!s}"))
                correction_count += 1

        # Max corrections reached
        vote_stats["corrections_needed"] = correction_count
        vote_stats["result_rows"] = 0
        error_history.append(f"Vote stats: {vote_stats}")

        return GenerationResult(
            sql=winning_sql,
            valid=True,
            iterations=vote_stats.get("winning_votes", 0),
            error_history=error_history,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            generation_times_ms=all_generation_times,
            explain_times_ms=all_explain_times,
        )
