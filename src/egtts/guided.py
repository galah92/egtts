"""EXPLAIN-guided SQL generation with iterative refinement."""

import re
import time
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .database import ExplainSuccess, explain_query
from .model import create_sql_prompt, generate_sql
from .schema import SchemaIndex, build_schema_index
from .prompts import format_few_shot_prompt, detect_singular_intent


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
    SQL generator with EXPLAIN-based iterative refinement.

    Uses a feedback loop to fix errors:
    1. Generate SQL from question
    2. Verify with EXPLAIN
    3. If error: provide error message as feedback and regenerate
    4. Repeat until valid or max iterations reached
    """

    def __init__(self, model, tokenizer, max_iterations: int = 3):
        """
        Initialize guided generator.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            max_iterations: Maximum refinement iterations (default: 3)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations

    def generate(self, question: str, schema: str) -> str:
        """
        Generate initial SQL query without refinement.

        Args:
            question: Natural language question
            schema: Database schema (CREATE TABLE statements)

        Returns:
            Generated SQL query string
        """
        prompt = create_sql_prompt(question, schema, self.tokenizer)
        sql = generate_sql(self.model, self.tokenizer, prompt, max_new_tokens=256, do_sample=False)
        return sql

    def refine(self, question: str, schema: str, previous_sql: str, error_message: str) -> str:
        """
        Refine SQL query based on error feedback.

        Creates a prompt that includes the error and asks for correction.

        Args:
            question: Original natural language question
            schema: Database schema
            previous_sql: SQL that failed
            error_message: Error message from EXPLAIN

        Returns:
            Refined SQL query
        """
        # Create refinement prompt with error feedback
        refinement_message = f"""Given the following database schema:

{schema}

You previously generated this SQL query to answer the question "{question}":
{previous_sql}

However, it failed with this error:
{error_message}

Generate a corrected SQL query that fixes this error. Return only the SQL query, without explanation."""

        messages = [{"role": "user", "content": refinement_message}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sql = generate_sql(self.model, self.tokenizer, prompt, max_new_tokens=256, do_sample=False)
        return sql

    def generate_with_feedback(
        self, question: str, schema: str, db_path: str
    ) -> GenerationResult:
        """
        Generate SQL with iterative EXPLAIN-based refinement.

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Path to SQLite database for EXPLAIN verification

        Returns:
            GenerationResult with final SQL and iteration metadata
        """
        start_time = time.perf_counter()

        error_history = []
        generation_times = []
        explain_times = []

        # Initial generation (iteration 0)
        gen_start = time.perf_counter()
        sql = self.generate(question, schema)
        generation_times.append((time.perf_counter() - gen_start) * 1000)

        # Verify with EXPLAIN
        explain_start = time.perf_counter()
        result = explain_query(sql, db_path)
        explain_times.append((time.perf_counter() - explain_start) * 1000)

        # If valid on first try, return immediately
        if isinstance(result, ExplainSuccess):
            total_time = (time.perf_counter() - start_time) * 1000
            return GenerationResult(
                sql=sql,
                valid=True,
                iterations=0,
                error_history=[],
                latency_ms=total_time,
                generation_times_ms=generation_times,
                explain_times_ms=explain_times,
            )

        # Iterative refinement
        error_history.append(result.error_message)
        current_sql = sql

        for iteration in range(1, self.max_iterations + 1):
            # Refine based on error
            gen_start = time.perf_counter()
            current_sql = self.refine(question, schema, current_sql, result.error_message)
            generation_times.append((time.perf_counter() - gen_start) * 1000)

            # Verify refined query
            explain_start = time.perf_counter()
            result = explain_query(current_sql, db_path)
            explain_times.append((time.perf_counter() - explain_start) * 1000)

            # Check if refined query is valid
            if isinstance(result, ExplainSuccess):
                total_time = (time.perf_counter() - start_time) * 1000
                return GenerationResult(
                    sql=current_sql,
                    valid=True,
                    iterations=iteration,
                    error_history=error_history,
                    latency_ms=total_time,
                    generation_times_ms=generation_times,
                    explain_times_ms=explain_times,
                )

            # Still invalid, record error and continue
            error_history.append(result.error_message)

        # Failed to fix after max iterations
        total_time = (time.perf_counter() - start_time) * 1000
        return GenerationResult(
            sql=current_sql,
            valid=False,
            iterations=self.max_iterations,
            error_history=error_history,
            latency_ms=total_time,
            generation_times_ms=generation_times,
            explain_times_ms=explain_times,
        )

    def generate_with_beam_search(
        self, question: str, schema: str, db_path: str, num_beams: int = 5
    ) -> GenerationResult:
        """
        Generate SQL using beam search and select first valid candidate.

        Instead of iterative refinement, this generates multiple candidates
        and picks the first one that passes EXPLAIN validation.

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Path to SQLite database for EXPLAIN verification
            num_beams: Number of beams to generate (default: 5)

        Returns:
            GenerationResult with selected SQL and metadata
        """
        import torch

        start_time = time.perf_counter()

        # Create prompt
        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate multiple candidates with beam search
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_time = (time.perf_counter() - gen_start) * 1000

        # Decode all candidates
        candidates = []
        for output in outputs:
            generated_ids = output[input_length:]
            sql = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
                if line and not line.startswith("--") and not line.lower().startswith(
                    ("this query", "the query", "note:")
                ):
                    sql_lines.append(line)
                elif sql_lines:
                    break

            sql = " ".join(sql_lines).strip()
            candidates.append(sql)

        # Test each candidate with EXPLAIN
        explain_times = []
        error_history = []

        for idx, sql in enumerate(candidates):
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_time = (time.perf_counter() - explain_start) * 1000
            explain_times.append(explain_time)

            if isinstance(result, ExplainSuccess):
                # Found valid candidate
                total_time = (time.perf_counter() - start_time) * 1000
                return GenerationResult(
                    sql=sql,
                    valid=True,
                    iterations=idx,  # Use beam index as "iteration"
                    error_history=error_history,
                    latency_ms=total_time,
                    generation_times_ms=[gen_time],
                    explain_times_ms=explain_times,
                )
            else:
                error_history.append(f"Beam {idx}: {result.error_message}")

        # All candidates failed - return first one
        total_time = (time.perf_counter() - start_time) * 1000
        return GenerationResult(
            sql=candidates[0],
            valid=False,
            iterations=num_beams - 1,
            error_history=error_history,
            latency_ms=total_time,
            generation_times_ms=[gen_time],
            explain_times_ms=explain_times,
        )

    def calculate_plan_cost(self, explain_output: list[dict]) -> int:
        """
        Calculate heuristic cost score from EXPLAIN QUERY PLAN output.

        Lower scores are better (more efficient queries).
        This targets the BIRD benchmark's VES (Valid Efficiency Score) metric.

        Args:
            explain_output: List of plan rows from EXPLAIN QUERY PLAN

        Returns:
            Integer cost score (lower is better)
        """
        cost = 0

        # Convert plan rows to string for pattern matching
        plan_str = " ".join(str(row) for row in explain_output).upper()

        # Penalties for inefficient operations
        # SQLite EXPLAIN uses "SCAN <table>" not "SCAN TABLE <table>"
        if "'SCAN " in plan_str and "USING INDEX" not in plan_str:
            cost += 100  # Table scan is expensive
        if "USE TEMP B-TREE" in plan_str:
            cost += 50  # Temporary structures add overhead
        # No penalty for index usage (efficient)
        # "USING INDEX" or "COVERING INDEX" = 0 additional cost

        return cost

    def _validate_schema_references(self, sql: str, schema_index: SchemaIndex) -> tuple[bool, str]:
        """
        Validate that all table and column references in SQL exist in schema.

        Uses sqlglot for robust SQL parsing instead of regex.

        Args:
            sql: SQL query to validate
            schema_index: Schema index for validation

        Returns:
            Tuple of (valid, error_message)
        """
        try:
            # Parse SQL with sqlglot
            parsed = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception as e:
            # If parsing fails, fall back to assuming valid (will be caught by EXPLAIN)
            return True, ""

        # Extract and validate table references
        for table in parsed.find_all(exp.Table):
            table_name = table.name
            if not schema_index.has_table(table_name):
                return False, f"Table '{table_name}' not found in schema"

        # Extract and validate column references
        for column in parsed.find_all(exp.Column):
            column_name = column.name
            # Skip * and check if column exists in schema
            if column_name != "*" and not schema_index.has_column(column_name):
                return False, f"Column '{column_name}' not found in schema"

        return True, ""

    def generate_with_schema_guidance(
        self, question: str, schema: str, db_path: str, num_beams: int = 5
    ) -> GenerationResult:
        """
        Generate SQL with post-generation schema validation (Strategy A).

        This implements the "Schema Lookup" approach from the guidance:
        - Generate multiple beam candidates normally (fast)
        - Validate each candidate against schema index (very fast <1ms)
        - Filter candidates with invalid schema references
        - From remaining candidates, pick first EXPLAIN-valid one

        This is simpler and faster than token-level validation, while still
        preventing schema hallucinations from being selected.

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Path to SQLite database
            num_beams: Number of beams to generate (default: 5)

        Returns:
            GenerationResult with selected SQL and metadata
        """
        import torch

        start_time = time.perf_counter()

        # Build schema index
        index_start = time.perf_counter()
        schema_index = build_schema_index(db_path)
        index_time = (time.perf_counter() - index_start) * 1000

        # Create prompt
        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate beams normally (no logits processor - fast and correct)
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_time = (time.perf_counter() - gen_start) * 1000

        # Decode all candidates
        candidates = []
        for output in outputs:
            generated_ids = output[input_length:]
            sql = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
                if line and not line.startswith("--") and not line.lower().startswith(
                    ("this query", "the query", "note:")
                ):
                    sql_lines.append(line)
                elif sql_lines:
                    break

            sql = " ".join(sql_lines).strip()
            candidates.append(sql)

        # Validate and test each candidate
        explain_times = []
        error_history = []
        schema_validation_count = 0

        for idx, sql in enumerate(candidates):
            # First check schema (very fast)
            schema_valid, schema_error = self._validate_schema_references(sql, schema_index)

            if not schema_valid:
                error_history.append(f"Beam {idx}: Schema validation failed: {schema_error}")
                schema_validation_count += 1
                continue  # Skip EXPLAIN if schema is invalid

            # Then check with EXPLAIN
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_time = (time.perf_counter() - explain_start) * 1000
            explain_times.append(explain_time)

            if isinstance(result, ExplainSuccess):
                # Found valid candidate
                total_time = (time.perf_counter() - start_time) * 1000
                return GenerationResult(
                    sql=sql,
                    valid=True,
                    iterations=idx,  # Use beam index as "iteration"
                    error_history=error_history,
                    latency_ms=total_time,
                    generation_times_ms=[gen_time, index_time],
                    explain_times_ms=explain_times,
                )
            else:
                error_history.append(f"Beam {idx}: EXPLAIN failed: {result.error_message}")

        # All candidates failed - return first one
        total_time = (time.perf_counter() - start_time) * 1000
        return GenerationResult(
            sql=candidates[0],
            valid=False,
            iterations=num_beams - 1,
            error_history=error_history,
            latency_ms=total_time,
            generation_times_ms=[gen_time, index_time],
            explain_times_ms=explain_times,
        )

    def generate_with_cost_guidance(
        self, question: str, schema: str, db_path: str, num_beams: int = 5
    ) -> GenerationResult:
        """
        Generate SQL with cost-aware beam selection (Milestone 4).

        This extends schema guidance to also consider query efficiency.
        Instead of picking the first valid candidate, we:
        1. Filter candidates by schema validity
        2. Get EXPLAIN plan for all schema-valid candidates
        3. Rank by plan cost (lower is better)
        4. Return the most efficient valid query

        This targets the BIRD benchmark's VES (Valid Efficiency Score) metric.

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Path to SQLite database
            num_beams: Number of beams to generate (default: 5)

        Returns:
            GenerationResult with most efficient valid SQL
        """
        import torch

        start_time = time.perf_counter()

        # Build schema index
        index_start = time.perf_counter()
        schema_index = build_schema_index(db_path)
        index_time = (time.perf_counter() - index_start) * 1000

        # Create prompt
        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate beams normally
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_time = (time.perf_counter() - gen_start) * 1000

        # Decode all candidates
        candidates = []
        for output in outputs:
            generated_ids = output[input_length:]
            sql = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
                if line and not line.startswith("--") and not line.lower().startswith(
                    ("this query", "the query", "note:")
                ):
                    sql_lines.append(line)
                elif sql_lines:
                    break

            sql = " ".join(sql_lines).strip()
            candidates.append(sql)

        # Validate all candidates and collect costs
        explain_times = []
        error_history = []
        valid_candidates = []  # List of (sql, cost, beam_idx)

        for idx, sql in enumerate(candidates):
            # Check schema first
            schema_valid, schema_error = self._validate_schema_references(sql, schema_index)

            if not schema_valid:
                error_history.append(f"Beam {idx}: Schema validation failed: {schema_error}")
                continue

            # Check with EXPLAIN
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_time = (time.perf_counter() - explain_start) * 1000
            explain_times.append(explain_time)

            if isinstance(result, ExplainSuccess):
                # Calculate cost
                cost = self.calculate_plan_cost(result.plan)
                valid_candidates.append((sql, cost, idx))
            else:
                error_history.append(f"Beam {idx}: EXPLAIN failed: {result.error_message}")

        # If we have valid candidates, pick the one with lowest cost
        if valid_candidates:
            # Sort by cost (ascending)
            valid_candidates.sort(key=lambda x: x[1])
            best_sql, best_cost, best_idx = valid_candidates[0]

            total_time = (time.perf_counter() - start_time) * 1000
            return GenerationResult(
                sql=best_sql,
                valid=True,
                iterations=best_idx,
                error_history=error_history,
                latency_ms=total_time,
                generation_times_ms=[gen_time, index_time],
                explain_times_ms=explain_times,
            )

        # All candidates failed - return first one
        total_time = (time.perf_counter() - start_time) * 1000
        return GenerationResult(
            sql=candidates[0] if candidates else "",
            valid=False,
            iterations=num_beams - 1,
            error_history=error_history,
            latency_ms=total_time,
            generation_times_ms=[gen_time, index_time],
            explain_times_ms=explain_times,
        )

    def generate_with_confidence_aware_reranking(
        self, question: str, schema: str, db_path: str, num_beams: int = 5,
        confidence_threshold: float = -0.22  # log(0.8) ≈ -0.223
    ) -> GenerationResult:
        """
        Generate SQL with confidence-aware re-ranking (Milestone 5 / Strategy M5).

        This balances accuracy and efficiency using model confidence:
        1. Generate beams and capture sequence scores (log-probs)
        2. Filter candidates by schema + EXPLAIN validation
        3. Find Most Probable Valid Beam (highest score)
        4. Find Most Efficient Valid Beam (lowest cost)
        5. Decision logic:
           - Default to Most Probable
           - Only switch to Efficient if (Score_Probable - Score_Efficient) < THRESHOLD

        This prevents M4's problem of sacrificing high-probability correct answers
        for low-probability efficient answers.

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Path to SQLite database
            num_beams: Number of beams to generate (default: 5)
            confidence_threshold: Maximum allowed confidence gap (default: log(0.8))

        Returns:
            GenerationResult with confidence-aware selected SQL
        """
        import torch

        start_time = time.perf_counter()

        # Build schema index
        index_start = time.perf_counter()
        schema_index = build_schema_index(db_path)
        index_time = (time.perf_counter() - index_start) * 1000

        # Create prompt
        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate beams with scores
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen_time = (time.perf_counter() - gen_start) * 1000

        # Extract sequences and scores
        sequences = outputs.sequences
        scores = outputs.sequences_scores  # Log-probabilities for each beam

        # Decode all candidates
        candidates = []
        for i, seq in enumerate(sequences):
            generated_ids = seq[input_length:]
            sql = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
                if line and not line.startswith("--") and not line.lower().startswith(
                    ("this query", "the query", "note:")
                ):
                    sql_lines.append(line)
                elif sql_lines:
                    break

            sql = " ".join(sql_lines).strip()
            candidates.append(sql)

        # Validate schema and get costs for all candidates
        explain_times = []
        error_history = []
        valid_candidates = []  # (idx, sql, cost, score)

        for idx, sql in enumerate(candidates):
            # Schema validation first
            schema_valid, schema_error = self._validate_schema_references(sql, schema_index)

            if not schema_valid:
                error_history.append(f"Beam {idx}: Schema validation failed: {schema_error}")
                continue

            # EXPLAIN validation and cost calculation
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_time = (time.perf_counter() - explain_start) * 1000
            explain_times.append(explain_time)

            if isinstance(result, ExplainSuccess):
                # Calculate cost from plan
                cost = self.calculate_plan_cost(result.plan)
                # Get beam score (log-probability)
                score = scores[idx].item()
                valid_candidates.append((idx, sql, cost, score))
            else:
                error_history.append(f"Beam {idx}: EXPLAIN failed: {result.error_message}")

        # Confidence-aware re-ranking logic
        if valid_candidates:
            # Find most probable valid beam (highest score)
            most_probable = max(valid_candidates, key=lambda x: x[3])
            prob_idx, prob_sql, prob_cost, prob_score = most_probable

            # Find most efficient valid beam (lowest cost)
            most_efficient = min(valid_candidates, key=lambda x: x[2])
            eff_idx, eff_sql, eff_cost, eff_score = most_efficient

            # Decision logic: Default to most probable
            # Only switch to efficient if confidence gap is small
            confidence_gap = prob_score - eff_score

            if confidence_gap < confidence_threshold:
                # Confidence gap is small enough - use efficient beam
                selected_idx = eff_idx
                selected_sql = eff_sql
                selected_cost = eff_cost
                decision_reason = f"Efficient (gap={confidence_gap:.3f} < {confidence_threshold:.3f})"
            else:
                # Confidence gap too large - stick with most probable
                selected_idx = prob_idx
                selected_sql = prob_sql
                selected_cost = prob_cost
                decision_reason = f"Probable (gap={confidence_gap:.3f} >= {confidence_threshold:.3f})"

            total_time = (time.perf_counter() - start_time) * 1000
            return GenerationResult(
                sql=selected_sql,
                valid=True,
                iterations=selected_idx,  # Which beam was selected
                error_history=error_history + [f"M5 Decision: {decision_reason}"],
                latency_ms=total_time,
                generation_times_ms=[gen_time, index_time],
                explain_times_ms=explain_times,
            )

        # All candidates failed - return first one
        total_time = (time.perf_counter() - start_time) * 1000
        return GenerationResult(
            sql=candidates[0],
            valid=False,
            iterations=num_beams - 1,
            error_history=error_history,
            latency_ms=total_time,
            generation_times_ms=[gen_time, index_time],
            explain_times_ms=explain_times,
        )

    def generate_with_explain_feedback(
        self, question: str, schema: str, db_path: str, max_iterations: int = 2
    ) -> GenerationResult:
        """
        Generate SQL with EXPLAIN feedback in the prompt.

        This strategy:
        1. Generate initial SQL
        2. Run EXPLAIN QUERY PLAN to get the execution plan
        3. If the plan shows inefficiencies (table scans), ask the model to optimize
        4. Repeat until efficient or max iterations reached

        The key insight is that the model can understand EXPLAIN output and
        suggest optimizations like adding appropriate WHERE clauses or using
        indexed columns.

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Path to SQLite database
            max_iterations: Maximum optimization iterations (default: 2)

        Returns:
            GenerationResult with optimized SQL
        """
        start_time = time.perf_counter()

        error_history = []
        generation_times = []
        explain_times = []

        # Initial generation
        gen_start = time.perf_counter()
        prompt = create_sql_prompt(question, schema, self.tokenizer)
        current_sql = generate_sql(self.model, self.tokenizer, prompt, max_new_tokens=256, do_sample=False)
        generation_times.append((time.perf_counter() - gen_start) * 1000)

        # Get initial EXPLAIN
        explain_start = time.perf_counter()
        result = explain_query(current_sql, db_path)
        explain_times.append((time.perf_counter() - explain_start) * 1000)

        # If initial query has errors, try to fix them first
        if not isinstance(result, ExplainSuccess):
            error_history.append(f"Initial error: {result.error_message}")
            # Use the existing refine method to fix errors
            gen_start = time.perf_counter()
            current_sql = self.refine(question, schema, current_sql, result.error_message)
            generation_times.append((time.perf_counter() - gen_start) * 1000)

            # Re-check
            explain_start = time.perf_counter()
            result = explain_query(current_sql, db_path)
            explain_times.append((time.perf_counter() - explain_start) * 1000)

            if not isinstance(result, ExplainSuccess):
                # Still invalid, return what we have
                total_time = (time.perf_counter() - start_time) * 1000
                return GenerationResult(
                    sql=current_sql,
                    valid=False,
                    iterations=1,
                    error_history=error_history,
                    latency_ms=total_time,
                    generation_times_ms=generation_times,
                    explain_times_ms=explain_times,
                )

        # Now we have a valid query - check if it needs optimization
        for iteration in range(max_iterations):
            # Calculate cost from current plan
            cost = self.calculate_plan_cost(result.plan)

            # Format the EXPLAIN output for the prompt
            plan_text = self._format_explain_plan(result.plan)

            # If query is already efficient (no table scans), we're done
            if cost == 0:
                error_history.append(f"Iteration {iteration}: Query already efficient (cost=0)")
                break

            error_history.append(f"Iteration {iteration}: Cost={cost}, attempting optimization")

            # Ask model to optimize based on EXPLAIN output
            gen_start = time.perf_counter()
            optimized_sql = self._optimize_with_explain(
                question, schema, current_sql, plan_text, cost
            )
            generation_times.append((time.perf_counter() - gen_start) * 1000)

            # Verify the optimized query
            explain_start = time.perf_counter()
            new_result = explain_query(optimized_sql, db_path)
            explain_times.append((time.perf_counter() - explain_start) * 1000)

            if isinstance(new_result, ExplainSuccess):
                new_cost = self.calculate_plan_cost(new_result.plan)
                if new_cost < cost:
                    # Optimization improved efficiency
                    error_history.append(f"Optimization successful: cost {cost} -> {new_cost}")
                    current_sql = optimized_sql
                    result = new_result
                    cost = new_cost
                else:
                    # No improvement, keep original
                    error_history.append(f"Optimization did not improve (new cost={new_cost})")
            else:
                # Optimized query is invalid, keep original
                error_history.append(f"Optimized query invalid: {new_result.error_message}")

        total_time = (time.perf_counter() - start_time) * 1000
        return GenerationResult(
            sql=current_sql,
            valid=True,
            iterations=len(generation_times) - 1,
            error_history=error_history,
            latency_ms=total_time,
            generation_times_ms=generation_times,
            explain_times_ms=explain_times,
        )

    def _format_explain_plan(self, plan: list[dict]) -> str:
        """Format EXPLAIN QUERY PLAN output as readable text."""
        lines = []
        for row in plan:
            detail = row.get("detail", "")
            lines.append(f"  {detail}")
        return "\n".join(lines)

    def _optimize_with_explain(
        self, question: str, schema: str, current_sql: str, plan_text: str, cost: int
    ) -> str:
        """
        Ask the model to optimize SQL based on EXPLAIN output.

        Args:
            question: Original question
            schema: Database schema
            current_sql: Current SQL query
            plan_text: Formatted EXPLAIN output
            cost: Current cost score

        Returns:
            Optimized SQL query
        """
        # Build optimization prompt
        optimization_prompt = f"""Given the following database schema:

{schema}

You generated this SQL query to answer "{question}":
{current_sql}

The EXPLAIN QUERY PLAN output shows:
{plan_text}

This query has efficiency issues:
- SCAN operations without index usage indicate full table scans
- These are slow on large tables

Please rewrite the query to be more efficient. Consider:
1. Using indexed columns in WHERE clauses
2. Adding appropriate filters to reduce scanned rows
3. Using JOINs more efficiently

Return only the optimized SQL query, without explanation."""

        messages = [{"role": "user", "content": optimization_prompt}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sql = generate_sql(self.model, self.tokenizer, prompt, max_new_tokens=256, do_sample=False)
        return sql

    def _generate_diverse_samples(
        self, inputs, input_length: int, num_samples: int, temperature: float = 0.7
    ) -> tuple[list[str], float]:
        """
        Generate diverse SQL samples with OOM-safe batching.

        If generating all samples at once causes OOM, falls back to batching.

        Args:
            inputs: Tokenized inputs
            input_length: Length of input tokens
            num_samples: Number of samples to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (list of SQL candidates, generation time in ms)
        """
        import torch

        gen_start = time.perf_counter()

        try:
            # Try to generate all at once
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_return_sequences=num_samples,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            # OOM - fall back to batching
            torch.cuda.empty_cache()
            outputs_list = []
            batch_size = num_samples // 2  # Half the samples per batch

            for _ in range(2):
                with torch.no_grad():
                    batch_outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        num_return_sequences=batch_size,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                outputs_list.append(batch_outputs)
                torch.cuda.empty_cache()

            # Concatenate batches
            outputs = torch.cat(outputs_list, dim=0)

        gen_time = (time.perf_counter() - gen_start) * 1000

        # Decode all candidates
        candidates = []
        for output in outputs:
            generated_ids = output[input_length:]
            sql = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
                if line and not line.startswith("--") and not line.lower().startswith(
                    ("this query", "the query", "note:")
                ):
                    sql_lines.append(line)
                elif sql_lines:
                    break

            sql = " ".join(sql_lines).strip()
            candidates.append(sql)

        return candidates, gen_time

    def generate_with_plan_voting(
        self, question: str, schema: str, db_path: str, num_samples: int = 10
    ) -> GenerationResult:
        """
        Generate SQL with Plan-Based Majority Voting (M7).

        This strategy uses EXPLAIN signatures for semantic self-consistency:
        1. Generate K diverse SQL candidates (using sampling)
        2. Get EXPLAIN QUERY PLAN for each valid candidate
        3. Normalize plans into abstract signatures
        4. Vote: find the most common plan signature
        5. Select: from candidates with winning signature, pick lowest cost

        Key insight: Different SQL syntax can produce identical execution plans.
        By voting on plans instead of SQL strings, we achieve semantic consensus
        that filters out hallucinations and syntax variations.

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Path to SQLite database
            num_samples: Number of diverse samples to generate (default: 10)

        Returns:
            GenerationResult with consensus-selected SQL
        """
        import torch
        from .plans import normalize_plan, vote_by_plan

        start_time = time.perf_counter()

        # Create prompt
        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate diverse candidates using sampling (not beam search)
        candidates, gen_time = self._generate_diverse_samples(inputs, input_length, num_samples)

        # Get plan signatures for all candidates
        explain_times = []
        error_history = []
        candidates_with_plans = []  # (sql, signature, cost)

        for idx, sql in enumerate(candidates):
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_time = (time.perf_counter() - explain_start) * 1000
            explain_times.append(explain_time)

            if isinstance(result, ExplainSuccess):
                signature = normalize_plan(result.plan)
                cost = self.calculate_plan_cost(result.plan)
                candidates_with_plans.append((sql, signature, cost))
            else:
                error_history.append(f"Candidate {idx}: {result.error_message}")
                candidates_with_plans.append((sql, "ERROR", -1))

        # Perform plan-based majority voting
        winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)

        error_history.append(f"Vote stats: {vote_stats}")

        total_time = (time.perf_counter() - start_time) * 1000

        # Determine if we got a valid result
        valid = winning_sig != "ERROR" and vote_stats.get("valid_candidates", 0) > 0

        return GenerationResult(
            sql=winning_sql,
            valid=valid,
            iterations=vote_stats.get("winning_votes", 0),  # Use votes as "iterations"
            error_history=error_history,
            latency_ms=total_time,
            generation_times_ms=[gen_time],
            explain_times_ms=explain_times,
        )

    def generate_with_massive_diversity(
        self, question: str, schema: str, db_path: str, num_samples: int = 32
    ) -> GenerationResult:
        """
        Generate SQL with Massive Diversity Plan-Bagging (M8).

        This strategy scales up M7 with more diverse samples:
        1. Generate N=32 diverse SQL candidates using temperature sampling
        2. Validate all candidates with EXPLAIN (filter syntax/schema errors)
        3. Cluster valid candidates by Plan Signature
        4. Vote: identify the largest cluster (most common plan)
        5. Select: from winning cluster, pick lowest cost query

        Key insights:
        - Hallucinations are "fragile" - they generate unique, weird plans
        - Correct answers are "stable" - multiple samples converge on same plan
        - More samples = higher chance of finding the correct "stable" cluster

        This is similar to Pass@K scaling but uses plan consensus instead of
        oracle/ground-truth comparison.

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Path to SQLite database
            num_samples: Number of diverse samples (default: 32)

        Returns:
            GenerationResult with consensus-selected SQL
        """
        import torch
        from .plans import normalize_plan, vote_by_plan

        start_time = time.perf_counter()

        # Create prompt
        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate diverse candidates with OOM-safe batching
        # Use temperature=0.7 for good diversity without being too random
        candidates, gen_time = self._generate_diverse_samples(
            inputs, input_length, num_samples, temperature=0.7
        )

        # Validate and get plan signatures for all candidates
        explain_times = []
        error_history = []
        candidates_with_plans = []  # (sql, signature, cost)
        syntax_errors = 0
        schema_errors = 0

        for idx, sql in enumerate(candidates):
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_time = (time.perf_counter() - explain_start) * 1000
            explain_times.append(explain_time)

            if isinstance(result, ExplainSuccess):
                signature = normalize_plan(result.plan)
                cost = self.calculate_plan_cost(result.plan)
                candidates_with_plans.append((sql, signature, cost))
            else:
                # Track error types for analysis
                error_msg = result.error_message.lower()
                if "syntax" in error_msg:
                    syntax_errors += 1
                elif "no such table" in error_msg or "no such column" in error_msg:
                    schema_errors += 1
                candidates_with_plans.append((sql, "ERROR", -1))

        # Perform plan-based majority voting
        winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)

        # Enhanced stats for M8
        vote_stats["syntax_errors"] = syntax_errors
        vote_stats["schema_errors"] = schema_errors
        vote_stats["num_samples"] = num_samples

        error_history.append(f"Vote stats: {vote_stats}")

        total_time = (time.perf_counter() - start_time) * 1000

        # Determine if we got a valid result
        valid = winning_sig != "ERROR" and vote_stats.get("valid_candidates", 0) > 0

        return GenerationResult(
            sql=winning_sql,
            valid=valid,
            iterations=vote_stats.get("winning_votes", 0),
            error_history=error_history,
            latency_ms=total_time,
            generation_times_ms=[gen_time],
            explain_times_ms=explain_times,
        )

    def generate_with_few_shot_and_simulation(
        self, question: str, schema: str, db_path: str, hint: str = None, num_samples: int = 32
    ) -> GenerationResult:
        """
        Generate SQL with Few-Shot Prompting + Simulation Filter (M9).

        This is the "Final Squeeze" strategy combining:
        1. Few-Shot Prompting: Domain-specific examples for BIRD patterns
        2. Diverse Generation: Temperature sampling for N candidates
        3. EXPLAIN Validation: Filter invalid queries
        4. Simulation Filter: Reject "chatty" queries that return wrong column count
        5. Plan-Based Voting: Select from consensus cluster

        Addresses two failure modes:
        - Generation Failures (80%): Few-shot teaches SUBSTR, GROUP BY patterns
        - Selection Failures (20%): Simulation filter kills chatty beams

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Path to SQLite database
            hint: Optional evidence/hint
            num_samples: Number of candidates to generate

        Returns:
            GenerationResult with selected SQL
        """
        import torch
        import sqlite3
        from .plans import normalize_plan, vote_by_plan

        start_time = time.perf_counter()

        # Use few-shot prompt instead of standard prompt
        prompt = format_few_shot_prompt(
            question, schema, self.tokenizer, hint=hint, use_few_shot=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate diverse candidates
        candidates, gen_time = self._generate_diverse_samples(
            inputs, input_length, num_samples, temperature=0.7
        )

        # Detect if question expects singular output (for simulation filter)
        expects_singular = detect_singular_intent(question)

        # Validate candidates with EXPLAIN + Simulation Filter
        explain_times = []
        error_history = []
        candidates_with_plans = []
        filtered_by_simulation = 0

        for idx, sql in enumerate(candidates):
            explain_start = time.perf_counter()
            result = explain_query(sql, db_path)
            explain_time = (time.perf_counter() - explain_start) * 1000
            explain_times.append(explain_time)

            if isinstance(result, ExplainSuccess):
                # Simulation Filter: Check column count
                if expects_singular:
                    try:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        # Run on empty result set to check column structure
                        cursor.execute(f"SELECT * FROM ({sql}) LIMIT 0")
                        num_cols = len(cursor.description) if cursor.description else 0
                        conn.close()

                        # Filter: If singular expected but >1 column, kill the beam
                        if num_cols > 1:
                            filtered_by_simulation += 1
                            candidates_with_plans.append((sql, "FILTERED_CHATTY", -1))
                            continue
                    except Exception:
                        pass  # If we can't check, don't filter

                signature = normalize_plan(result.plan)
                cost = self.calculate_plan_cost(result.plan)
                candidates_with_plans.append((sql, signature, cost))
            else:
                candidates_with_plans.append((sql, "ERROR", -1))

        # Perform plan-based voting (excluding filtered candidates)
        winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)

        # Enhanced stats
        vote_stats["filtered_by_simulation"] = filtered_by_simulation
        vote_stats["expects_singular"] = expects_singular
        vote_stats["num_samples"] = num_samples
        vote_stats["strategy"] = "M9_FewShot_Simulation"

        error_history.append(f"Vote stats: {vote_stats}")

        total_time = (time.perf_counter() - start_time) * 1000
        valid = winning_sig not in ("ERROR", "FILTERED_CHATTY") and vote_stats.get("valid_candidates", 0) > 0

        return GenerationResult(
            sql=winning_sql,
            valid=valid,
            iterations=vote_stats.get("winning_votes", 0),
            error_history=error_history,
            latency_ms=total_time,
            generation_times_ms=[gen_time],
            explain_times_ms=explain_times,
        )