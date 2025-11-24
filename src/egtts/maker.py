"""MAKER: Incremental Consensus Generation for Text-to-SQL (M15).

This strategy decomposes SQL generation into 3 atomic phases, applying
majority voting at each phase before proceeding to the next:

Phase 1: Scope Consensus (FROM/JOIN) - Identify required tables
Phase 2: Filter Consensus (WHERE) - Generate filter conditions
Phase 3: Projection Consensus (SELECT/GROUP BY) - Complete the query

Key insight: By locking in table choices via consensus first, we prevent
downstream column hallucinations that occur when the model hasn't committed
to a specific table scope.

Inspired by "Solving a Million-Step Task" - using consensus to verify each
atomic decision before building on it.
"""

import re
import time
import sqlite3
from collections import Counter
from dataclasses import dataclass

import torch

from .database import explain_query, ExplainSuccess
from .schema import get_augmented_schema


@dataclass
class Phase1Result:
    """Result from Phase 1: Table Selection."""
    tables: list[str]  # List of table names
    join_conditions: str  # JOIN conditions if any
    votes: int  # Number of samples that agreed
    total_samples: int  # Total samples generated
    valid: bool  # Whether we got a valid join path


@dataclass
class Phase2Result:
    """Result from Phase 2: Filter Generation."""
    where_clause: str  # WHERE clause (without "WHERE" keyword)
    votes: int
    total_samples: int
    valid: bool


@dataclass
class Phase3Result:
    """Result from Phase 3: Final Query."""
    sql: str  # Complete SQL query
    votes: int
    total_samples: int
    valid: bool


@dataclass
class M15Result:
    """Complete result from M15 strategy."""
    sql: str
    valid: bool
    phase1: Phase1Result
    phase2: Phase2Result
    phase3: Phase3Result
    total_latency_ms: float
    fallback_used: bool = False


class IncrementalGenerator:
    """
    SQL Generator using Incremental Consensus (MAKER approach).

    Decomposes SQL generation into 3 phases with voting at each stage:
    1. Table Selection (FROM/JOIN) with EXPLAIN validation
    2. Filter Generation (WHERE) with string voting
    3. Projection (SELECT/GROUP BY) with plan voting
    """

    def __init__(self, model, tokenizer, temperature: float = 0.7):
        """
        Initialize incremental generator.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            temperature: Sampling temperature for diversity (default: 0.7)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def _generate_diverse_samples(
        self, prompt: str, num_samples: int, max_new_tokens: int = 256
    ) -> tuple[list[str], float]:
        """
        Generate diverse samples with temperature sampling.

        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate
            max_new_tokens: Max tokens per sample

        Returns:
            Tuple of (list of responses, generation_time_ms)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        gen_start = time.perf_counter()

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_samples,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            # Fallback to batching
            torch.cuda.empty_cache()
            outputs_list = []
            batch_size = num_samples // 2

            for _ in range(2):
                with torch.no_grad():
                    batch = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=batch_size,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                outputs_list.append(batch)
                torch.cuda.empty_cache()

            outputs = torch.cat(outputs_list, dim=0)

        gen_time = (time.perf_counter() - gen_start) * 1000

        # Decode responses
        responses = []
        for output in outputs:
            generated_ids = output[input_length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            responses.append(text)

        return responses, gen_time

    def step_1_tables(
        self, question: str, augmented_schema: str, db_path: str, num_samples: int = 12
    ) -> Phase1Result:
        """
        Phase 1: Generate and vote on table selection with EXPLAIN validation.

        Prompt asks ONLY for tables and join conditions. We validate that the
        join path is valid using EXPLAIN on a dummy SELECT 1.

        Args:
            question: Natural language question
            augmented_schema: Schema with sample data
            db_path: Database path for validation
            num_samples: Number of candidates to generate

        Returns:
            Phase1Result with consensus table set
        """
        # Create Phase 1 prompt
        prompt_text = f"""Given this database schema with sample data:

{augmented_schema}

For the question: "{question}"

List ONLY the tables needed and their join conditions. Do not write SELECT, WHERE, GROUP BY, or ORDER BY.

Format your answer as:
Tables: table1, table2, ...
Joins: table1.col = table2.col AND ...

If only one table is needed, just list it without joins."""

        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate diverse table proposals
        responses, _ = self._generate_diverse_samples(prompt, num_samples, max_new_tokens=256)

        # Parse and vote on table sets
        table_proposals = []
        for resp in responses:
            tables, joins = self._parse_table_response(resp)
            if tables:
                table_proposals.append((tables, joins))

        if not table_proposals:
            # Fallback: couldn't parse any proposals
            return Phase1Result(
                tables=[], join_conditions="", votes=0, total_samples=num_samples, valid=False
            )

        # Vote by set similarity (order doesn't matter)
        table_sets = [frozenset(tables) for tables, _ in table_proposals]
        set_counts = Counter(table_sets)

        # Try clusters by vote count, validate with EXPLAIN
        for table_set, vote_count in set_counts.most_common():
            # Get all proposals with this table set
            matching_proposals = [
                (tables, joins) for tables, joins in table_proposals
                if frozenset(tables) == table_set
            ]

            # Pick most common join structure among this cluster
            if len(table_set) > 1:
                join_counts = Counter(joins for _, joins in matching_proposals if joins)
                if join_counts:
                    consensus_joins = join_counts.most_common(1)[0][0]
                else:
                    consensus_joins = ""
            else:
                consensus_joins = ""

            # Validate with EXPLAIN
            tables_list = list(table_set)
            if self._validate_join_path(tables_list, consensus_joins, db_path):
                return Phase1Result(
                    tables=tables_list,
                    join_conditions=consensus_joins,
                    votes=vote_count,
                    total_samples=num_samples,
                    valid=True
                )

        # No valid join path found - return most voted anyway (will fail in Phase 3)
        most_common_set = set_counts.most_common(1)[0][0]
        tables_list = list(most_common_set)
        return Phase1Result(
            tables=tables_list,
            join_conditions="",
            votes=set_counts[most_common_set],
            total_samples=num_samples,
            valid=False
        )

    def _parse_table_response(self, response: str) -> tuple[list[str], str]:
        """
        Parse Phase 1 response to extract tables and joins.

        Returns:
            Tuple of (table_list, join_conditions)
        """
        tables = []
        joins = ""

        lines = response.split("\n")
        for line in lines:
            line = line.strip()

            # Look for "Tables: ..." pattern
            if line.lower().startswith("tables:"):
                table_str = line.split(":", 1)[1].strip()
                # Extract table names (handle commas, "and", etc.)
                table_names = re.findall(r'\b[A-Za-z_]\w*\b', table_str)
                # Filter out common words
                tables = [t for t in table_names if t.lower() not in ('and', 'or', 'the')]

            # Look for "Joins: ..." pattern
            elif line.lower().startswith("join"):
                joins = line.split(":", 1)[1].strip() if ":" in line else ""

        return tables, joins

    def _validate_join_path(self, tables: list[str], join_conditions: str, db_path: str) -> bool:
        """
        Validate that the join path is valid using EXPLAIN.

        Uses a dummy SELECT 1 to test the FROM/JOIN structure.

        Args:
            tables: List of table names
            join_conditions: JOIN conditions string
            db_path: Database path

        Returns:
            True if join path is valid
        """
        if not tables:
            return False

        # Build FROM clause
        if len(tables) == 1:
            from_clause = f"FROM {tables[0]}"
        else:
            # Build JOIN structure
            from_clause = f"FROM {tables[0]}"
            for table in tables[1:]:
                from_clause += f" JOIN {table}"
                if join_conditions:
                    # Simple heuristic: add ON clause if joins specified
                    from_clause += f" ON {join_conditions}"

        # Create dummy query
        test_query = f"SELECT 1 {from_clause} LIMIT 1"

        # Validate with EXPLAIN
        result = explain_query(test_query, db_path)
        return isinstance(result, ExplainSuccess)

    def step_2_filters(
        self, question: str, augmented_schema: str, tables: list[str],
        join_conditions: str, num_samples: int = 12
    ) -> Phase2Result:
        """
        Phase 2: Generate WHERE clauses given locked tables.

        Prompt provides the tables from Phase 1 and asks ONLY for filters.

        Args:
            question: Natural language question
            augmented_schema: Schema with sample data
            tables: Locked table list from Phase 1
            join_conditions: Join conditions from Phase 1
            num_samples: Number of candidates to generate

        Returns:
            Phase2Result with consensus WHERE clause
        """
        # Create Phase 2 prompt
        tables_str = ", ".join(tables)
        prompt_text = f"""Given this database schema:

{augmented_schema}

For the question: "{question}"

The required tables are: {tables_str}
{f"Join conditions: {join_conditions}" if join_conditions else ""}

Write ONLY the WHERE clause conditions (without the "WHERE" keyword).
Do not write SELECT, FROM, JOIN, GROUP BY, or ORDER BY.

If no filters are needed, write "No filters needed"."""

        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate diverse filter proposals
        responses, _ = self._generate_diverse_samples(prompt, num_samples, max_new_tokens=256)

        # Parse and normalize WHERE clauses
        where_clauses = []
        for resp in responses:
            where = self._parse_where_response(resp)
            if where:
                where_clauses.append(where)

        if not where_clauses:
            # No filters needed
            return Phase2Result(
                where_clause="",
                votes=num_samples,
                total_samples=num_samples,
                valid=True
            )

        # Vote by normalized WHERE clause
        # Normalize: lowercase, remove extra spaces
        normalized = [self._normalize_where(w) for w in where_clauses]
        where_counts = Counter(normalized)

        # Pick most common
        consensus_where, votes = where_counts.most_common(1)[0]

        # Map back to original (non-normalized) version
        original_where = where_clauses[normalized.index(consensus_where)]

        return Phase2Result(
            where_clause=original_where,
            votes=votes,
            total_samples=num_samples,
            valid=True
        )

    def _parse_where_response(self, response: str) -> str:
        """
        Parse Phase 2 response to extract WHERE clause.

        Returns:
            WHERE clause string (without "WHERE" keyword)
        """
        # Clean response
        response = response.strip()

        # Check for "no filters" signals
        if "no filter" in response.lower() or "no where" in response.lower():
            return ""

        # Remove "WHERE" keyword if present
        if response.lower().startswith("where"):
            response = response[5:].strip()

        # Take first line (often multi-line explanations)
        first_line = response.split("\n")[0].strip()

        # If it looks like a condition, use it
        if any(op in first_line for op in ["=", ">", "<", "LIKE", "IN", "BETWEEN"]):
            return first_line

        # Otherwise return cleaned response
        return response if response else ""

    def _normalize_where(self, where_clause: str) -> str:
        """Normalize WHERE clause for voting (lowercase, remove extra spaces)."""
        normalized = where_clause.lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        return normalized

    def step_3_final(
        self, question: str, augmented_schema: str, tables: list[str],
        join_conditions: str, where_clause: str, db_path: str, num_samples: int = 12
    ) -> Phase3Result:
        """
        Phase 3: Generate complete SQL given locked tables and filters.

        Prompt provides tables and WHERE clause, asks for SELECT/GROUP BY/ORDER BY.
        We use plan-based voting to select the winner.

        Args:
            question: Natural language question
            augmented_schema: Schema with sample data
            tables: Locked tables from Phase 1
            join_conditions: Join conditions from Phase 1
            where_clause: Locked WHERE clause from Phase 2
            db_path: Database path for EXPLAIN validation
            num_samples: Number of candidates to generate

        Returns:
            Phase3Result with final SQL query
        """
        from .plans import normalize_plan, vote_by_plan

        # Create Phase 3 prompt
        tables_str = ", ".join(tables)
        prompt_text = f"""Given this database schema:

{augmented_schema}

For the question: "{question}"

You must use these tables: {tables_str}
{f"Join conditions: {join_conditions}" if join_conditions else ""}
{f"Filter conditions: {where_clause}" if where_clause else "No filters needed."}

Complete the SQL query by writing the SELECT clause and any GROUP BY, ORDER BY, or LIMIT needed.
Write the COMPLETE SQL query including FROM and WHERE."""

        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate diverse completions
        responses, _ = self._generate_diverse_samples(prompt, num_samples, max_new_tokens=512)

        # Extract SQL from responses
        candidates = []
        for resp in responses:
            sql = self._extract_sql(resp)
            if sql:
                candidates.append(sql)

        if not candidates:
            return Phase3Result(
                sql="",
                votes=0,
                total_samples=num_samples,
                valid=False
            )

        # Validate with EXPLAIN and get plan signatures
        candidates_with_plans = []
        for sql in candidates:
            result = explain_query(sql, db_path)
            if isinstance(result, ExplainSuccess):
                signature = normalize_plan(result.plan)
                cost = self._calculate_cost(result.plan)
                candidates_with_plans.append((sql, signature, cost))
            else:
                candidates_with_plans.append((sql, "ERROR", -1))

        # Vote by plan signature
        winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)

        return Phase3Result(
            sql=winning_sql,
            votes=vote_stats.get("winning_votes", 0),
            total_samples=num_samples,
            valid=winning_sig != "ERROR"
        )

    def _extract_sql(self, response: str) -> str:
        """
        Extract SQL from response text.

        Handles markdown code blocks and explanatory text.
        """
        sql = response.strip()

        # Remove markdown
        if "```sql" in sql.lower():
            sql = sql.split("```sql", 1)[1].split("```")[0].strip()
        elif "```" in sql:
            parts = sql.split("```")
            if len(parts) >= 2:
                sql = parts[1].strip()

        # Clean up multi-line and comments
        lines = sql.split("\n")
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("--") and not line.lower().startswith(
                ("this query", "the query", "note:", "explanation:")
            ):
                sql_lines.append(line)
            elif sql_lines:
                break

        sql = " ".join(sql_lines).strip()
        return sql

    def _calculate_cost(self, plan: list[dict]) -> int:
        """Calculate heuristic cost from EXPLAIN plan."""
        cost = 0
        plan_str = " ".join(str(row) for row in plan).upper()

        if "'SCAN " in plan_str and "USING INDEX" not in plan_str:
            cost += 100
        if "USE TEMP B-TREE" in plan_str:
            cost += 50

        return cost

    def generate(
        self, question: str, db_path: str, num_samples_per_phase: int = 12
    ) -> M15Result:
        """
        Generate SQL using 3-phase incremental consensus.

        Args:
            question: Natural language question
            db_path: Database path
            num_samples_per_phase: Number of samples per phase (default: 12)

        Returns:
            M15Result with complete generation trace
        """
        start_time = time.perf_counter()

        # Get augmented schema
        augmented_schema = get_augmented_schema(db_path, max_rows_per_table=3)

        # Phase 1: Table Selection
        phase1 = self.step_1_tables(question, augmented_schema, db_path, num_samples_per_phase)

        if not phase1.valid or not phase1.tables:
            # Phase 1 failed - use fallback (standard M10)
            total_time = (time.perf_counter() - start_time) * 1000
            return M15Result(
                sql="",
                valid=False,
                phase1=phase1,
                phase2=Phase2Result("", 0, 0, False),
                phase3=Phase3Result("", 0, 0, False),
                total_latency_ms=total_time,
                fallback_used=True
            )

        # Phase 2: Filter Generation
        phase2 = self.step_2_filters(
            question, augmented_schema, phase1.tables, phase1.join_conditions, num_samples_per_phase
        )

        # Phase 3: Final Query
        phase3 = self.step_3_final(
            question, augmented_schema, phase1.tables, phase1.join_conditions,
            phase2.where_clause, db_path, num_samples_per_phase
        )

        total_time = (time.perf_counter() - start_time) * 1000

        return M15Result(
            sql=phase3.sql,
            valid=phase3.valid,
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            total_latency_ms=total_time,
            fallback_used=False
        )
