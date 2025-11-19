"""EXPLAIN-guided SQL generation with iterative refinement."""

import time
from dataclasses import dataclass

from .database import ExplainSuccess, explain_query
from .model import create_sql_prompt, generate_sql


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
