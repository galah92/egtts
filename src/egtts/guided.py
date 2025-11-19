"""EXPLAIN-guided SQL generation with iterative refinement."""

import re
import time
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .database import ExplainSuccess, explain_query
from .model import create_sql_prompt, generate_sql
from .schema import SchemaIndex, build_schema_index


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

        # Validate schema and get costs for all candidates
        explain_times = []
        error_history = []
        valid_candidates = []  # (idx, sql, cost)

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
                valid_candidates.append((idx, sql, cost))
            else:
                error_history.append(f"Beam {idx}: EXPLAIN failed: {result.error_message}")

        # If we have valid candidates, pick the one with lowest cost
        if valid_candidates:
            # Sort by cost (lower is better)
            valid_candidates.sort(key=lambda x: x[2])
            best_idx, best_sql, best_cost = valid_candidates[0]

            total_time = (time.perf_counter() - start_time) * 1000
            return GenerationResult(
                sql=best_sql,
                valid=True,
                iterations=best_idx,  # Which beam was selected
                error_history=error_history,
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
