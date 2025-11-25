"""M18: Validated Decomposition for Text-to-SQL.

This strategy applies PAC-reasoning principles and MGDebugger-style validation
to SQL generation. Key insight: decomposition without validation compounds errors.

Each phase has:
1. Generation: Generate candidates for that phase
2. Validation: Check semantic correctness with Example Validators
3. Feedback: If validation fails, regenerate with error context
4. Consensus: Vote among validated candidates

Phases:
1. Table Selection - Validate columns exist for question entities
2. Filter Generation - Validate non-empty results
3. Aggregation Detection - Validate GROUP BY matches superlatives
4. Final Assembly - Standard plan-based voting

Inspired by:
- PAC-reasoning: Error-bounded guarantees through validation
- MGDebugger: Hierarchical debugging with test cases
"""

import re
import time
import sqlite3
from collections import Counter
from dataclasses import dataclass, field

import torch

from .database import explain_query, ExplainSuccess
from .schema import get_augmented_schema
from .plans import normalize_plan, vote_by_plan


@dataclass
class ValidationResult:
    """Result from a phase validator."""
    valid: bool
    error_message: str = ""
    suggestions: list[str] = field(default_factory=list)


@dataclass
class PhaseResult:
    """Generic result from a decomposition phase."""
    output: str  # Phase output (tables, where clause, etc.)
    votes: int
    total_samples: int
    valid: bool
    validation_attempts: int = 1
    error_history: list[str] = field(default_factory=list)


@dataclass
class M18Result:
    """Complete result from M18 strategy."""
    sql: str
    valid: bool
    phases: dict  # Phase name -> PhaseResult
    total_latency_ms: float
    validation_stats: dict = field(default_factory=dict)


class SQLValidator:
    """
    Validators for each decomposition phase.

    These are the "Example Validators" from PAC-reasoning - they empirically
    verify correctness at each step before proceeding.
    """

    def __init__(self, db_path: str, schema_info: dict = None):
        """
        Initialize validator with database connection.

        Args:
            db_path: Path to SQLite database
            schema_info: Optional pre-computed schema information
        """
        self.db_path = db_path
        self.schema_info = schema_info or self._extract_schema_info()

    def _extract_schema_info(self) -> dict:
        """Extract schema information from database."""
        info = {
            "tables": {},
            "columns": {},
            "foreign_keys": [],
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                # Get columns for each table
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                info["tables"][table.lower()] = columns
                for col in columns:
                    info["columns"][col.lower()] = info["columns"].get(col.lower(), []) + [table]

            # Get foreign keys
            for table in tables:
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                for row in cursor.fetchall():
                    info["foreign_keys"].append({
                        "from_table": table,
                        "from_col": row[3],
                        "to_table": row[2],
                        "to_col": row[4],
                    })

            conn.close()
        except Exception as e:
            pass  # Return partial info on error

        return info

    def validate_tables(self, tables: list[str], question: str, hint: str = "") -> ValidationResult:
        """
        Validate that selected tables can answer the question.

        Checks:
        1. Tables exist in schema
        2. Tables contain columns that could relate to question entities
        3. Tables have a valid join path (if multiple)

        Args:
            tables: List of selected table names
            question: Original question
            hint: Optional hint from BIRD

        Returns:
            ValidationResult with validity and suggestions
        """
        if not tables:
            return ValidationResult(
                valid=False,
                error_message="No tables selected",
                suggestions=list(self.schema_info["tables"].keys())[:5]
            )

        # Check tables exist
        invalid_tables = []
        for table in tables:
            if table.lower() not in self.schema_info["tables"]:
                invalid_tables.append(table)

        if invalid_tables:
            return ValidationResult(
                valid=False,
                error_message=f"Tables not found: {invalid_tables}",
                suggestions=list(self.schema_info["tables"].keys())
            )

        # Extract key entities from question
        question_lower = question.lower()
        hint_lower = (hint or "").lower()
        combined_text = question_lower + " " + hint_lower

        # Check if selected tables have relevant columns
        selected_columns = set()
        for table in tables:
            cols = self.schema_info["tables"].get(table.lower(), [])
            selected_columns.update(col.lower() for col in cols)

        # Heuristic: look for common BIRD patterns in question
        column_keywords = []
        if "consumption" in combined_text:
            column_keywords.append("consumption")
        if "customer" in combined_text:
            column_keywords.append("customerid")
        if "segment" in combined_text:
            column_keywords.append("segment")
        if "currency" in combined_text:
            column_keywords.append("currency")
        if "date" in combined_text or "year" in combined_text or "month" in combined_text:
            column_keywords.append("date")
        if "country" in combined_text:
            column_keywords.append("country")

        # Check if keywords have matching columns
        missing_keywords = []
        for kw in column_keywords:
            found = False
            for col in selected_columns:
                if kw in col or col in kw:
                    found = True
                    break
            if not found:
                missing_keywords.append(kw)

        if missing_keywords and len(missing_keywords) > len(column_keywords) // 2:
            # More than half of keywords missing - likely wrong tables
            # Find tables that have these columns
            suggested_tables = set()
            for kw in missing_keywords:
                for col, tables_with_col in self.schema_info["columns"].items():
                    if kw in col or col in kw:
                        suggested_tables.update(tables_with_col)

            return ValidationResult(
                valid=False,
                error_message=f"Selected tables may be missing columns for: {missing_keywords}",
                suggestions=list(suggested_tables)[:5]
            )

        return ValidationResult(valid=True)

    def validate_where_clause(
        self, tables: list[str], where_clause: str, join_conditions: str = ""
    ) -> ValidationResult:
        """
        Validate WHERE clause by executing a test query.

        Checks:
        1. Syntax is valid (EXPLAIN succeeds)
        2. Query returns at least one row (not empty result)
        3. Columns referenced exist in selected tables

        Args:
            tables: Selected tables
            where_clause: WHERE clause (without "WHERE" keyword)
            join_conditions: JOIN conditions

        Returns:
            ValidationResult
        """
        if not where_clause:
            return ValidationResult(valid=True)  # No WHERE is valid

        # Build test query
        if len(tables) == 1:
            from_clause = f"FROM {tables[0]}"
        else:
            from_clause = f"FROM {tables[0]}"
            for table in tables[1:]:
                if join_conditions:
                    from_clause += f" INNER JOIN {table} ON {join_conditions}"
                else:
                    from_clause += f", {table}"

        test_query = f"SELECT 1 {from_clause} WHERE {where_clause} LIMIT 1"

        # Test with EXPLAIN first (syntax check)
        result = explain_query(test_query, self.db_path)
        if not isinstance(result, ExplainSuccess):
            return ValidationResult(
                valid=False,
                error_message=f"WHERE clause syntax error: {result}",
                suggestions=["Check column names", "Check operators", "Check string quotes"]
            )

        # Execute to check for empty results
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(test_query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return ValidationResult(
                    valid=False,
                    error_message="WHERE clause produces empty result",
                    suggestions=["Relax filter conditions", "Check value formats (e.g., date strings)"]
                )
        except Exception as e:
            return ValidationResult(
                valid=False,
                error_message=f"Execution error: {str(e)[:100]}",
                suggestions=["Check column names exist in selected tables"]
            )

        return ValidationResult(valid=True)

    def validate_aggregation(
        self, question: str, has_group_by: bool, has_order_by: bool, has_limit: bool
    ) -> ValidationResult:
        """
        Validate that aggregation structure matches question intent.

        Checks for common patterns:
        - "highest/most X" needs GROUP BY + ORDER BY DESC + LIMIT 1
        - "lowest/least X" needs GROUP BY + ORDER BY ASC + LIMIT 1
        - "total/sum/count" needs aggregation function

        Args:
            question: Original question
            has_group_by: Whether query has GROUP BY
            has_order_by: Whether query has ORDER BY
            has_limit: Whether query has LIMIT

        Returns:
            ValidationResult
        """
        q_lower = question.lower()
        suggestions = []

        # Check superlative patterns
        superlative_words = ["highest", "lowest", "most", "least", "peak", "maximum", "minimum", "top", "bottom"]
        has_superlative = any(word in q_lower for word in superlative_words)

        if has_superlative:
            if not has_group_by:
                suggestions.append("Superlative questions usually need GROUP BY")
            if not has_order_by:
                suggestions.append("Superlative questions need ORDER BY with SUM/COUNT")
            if not has_limit:
                suggestions.append("Add LIMIT 1 for superlative questions")

            if suggestions:
                return ValidationResult(
                    valid=False,
                    error_message="Query structure may not match superlative question",
                    suggestions=suggestions
                )

        return ValidationResult(valid=True)


class ValidatedDecompositionGenerator:
    """
    SQL Generator using Validated Decomposition (M18).

    Key improvements over M15:
    1. Semantic validators at each phase
    2. Feedback loop when validation fails
    3. Configurable retry limits (PAC-style error control)
    """

    def __init__(
        self, model, tokenizer, temperature: float = 0.7, max_retries: int = 2
    ):
        """
        Initialize generator.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            temperature: Sampling temperature
            max_retries: Max validation retries per phase (controls error probability)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_retries = max_retries

    def _generate_samples(
        self, prompt: str, num_samples: int, max_new_tokens: int = 256
    ) -> list[str]:
        """Generate diverse samples with temperature sampling."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

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
            torch.cuda.empty_cache()
            # Fallback to smaller batches
            outputs_list = []
            batch_size = max(1, num_samples // 2)
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

        responses = []
        for output in outputs:
            generated_ids = output[input_length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            responses.append(text)

        return responses

    def _create_prompt(self, messages: list[dict]) -> str:
        """Create prompt using chat template."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def phase1_tables(
        self, question: str, hint: str, schema: str, validator: SQLValidator,
        num_samples: int = 8
    ) -> PhaseResult:
        """
        Phase 1: Table Selection with validation.

        Validates that selected tables contain columns relevant to the question.
        """
        error_history = []
        feedback = ""

        for attempt in range(self.max_retries + 1):
            # Build prompt with feedback if available
            prompt_text = f"""Given this database schema:

{schema}

Question: {question}
{f"Hint: {hint}" if hint else ""}
{feedback}

List ONLY the tables needed to answer this question.
Format: Tables: table1, table2, ...

If tables need to be joined, also specify:
Joins: table1.column = table2.column"""

            messages = [{"role": "user", "content": prompt_text}]
            prompt = self._create_prompt(messages)

            # Generate candidates
            responses = self._generate_samples(prompt, num_samples, max_new_tokens=200)

            # Parse table proposals
            proposals = []
            for resp in responses:
                tables = self._parse_tables(resp)
                if tables:
                    proposals.append(tables)

            if not proposals:
                error_history.append(f"Attempt {attempt + 1}: No valid table proposals")
                feedback = "\n\nPREVIOUS ATTEMPT FAILED: Could not parse table names. List them clearly as 'Tables: name1, name2'"
                continue

            # Vote on table sets
            table_sets = [frozenset(t) for t in proposals]
            set_counts = Counter(table_sets)

            # Try each cluster by vote count
            for table_set, vote_count in set_counts.most_common():
                tables_list = list(table_set)

                # Validate
                validation = validator.validate_tables(tables_list, question, hint)

                if validation.valid:
                    return PhaseResult(
                        output=",".join(tables_list),
                        votes=vote_count,
                        total_samples=num_samples,
                        valid=True,
                        validation_attempts=attempt + 1,
                        error_history=error_history
                    )
                else:
                    error_history.append(
                        f"Attempt {attempt + 1}: Tables {tables_list} failed validation: {validation.error_message}"
                    )
                    if validation.suggestions:
                        feedback = f"\n\nPREVIOUS ATTEMPT FAILED: {validation.error_message}\nConsider using tables: {', '.join(validation.suggestions)}"

        # Return best effort after all retries
        best_tables = list(set_counts.most_common(1)[0][0]) if set_counts else []
        return PhaseResult(
            output=",".join(best_tables),
            votes=set_counts.get(frozenset(best_tables), 0) if best_tables else 0,
            total_samples=num_samples,
            valid=False,
            validation_attempts=self.max_retries + 1,
            error_history=error_history
        )

    def _parse_tables(self, response: str) -> list[str]:
        """Parse table names from response."""
        tables = []
        for line in response.split("\n"):
            line = line.strip()
            if line.lower().startswith("tables:"):
                table_str = line.split(":", 1)[1].strip()
                # Extract table names
                names = re.findall(r'\b[A-Za-z_]\w*\b', table_str)
                tables = [n for n in names if n.lower() not in ('and', 'or', 'the', 'table', 'tables')]
                break
        return tables

    def phase2_filters(
        self, question: str, hint: str, schema: str, tables: list[str],
        validator: SQLValidator, num_samples: int = 8
    ) -> PhaseResult:
        """
        Phase 2: WHERE clause generation with validation.

        Validates that WHERE clause produces non-empty results.
        """
        error_history = []
        feedback = ""

        tables_str = ", ".join(tables)

        for attempt in range(self.max_retries + 1):
            prompt_text = f"""Given this database schema:

{schema}

Question: {question}
{f"Hint: {hint}" if hint else ""}

You are using tables: {tables_str}
{feedback}

Write ONLY the WHERE clause conditions (without the "WHERE" keyword).
If no filters are needed, write: No filters needed"""

            messages = [{"role": "user", "content": prompt_text}]
            prompt = self._create_prompt(messages)

            responses = self._generate_samples(prompt, num_samples, max_new_tokens=200)

            # Parse WHERE clauses
            where_clauses = []
            for resp in responses:
                where = self._parse_where(resp)
                where_clauses.append(where)

            # Vote on WHERE clauses
            where_counts = Counter(where_clauses)

            for where_clause, vote_count in where_counts.most_common():
                # Validate
                validation = validator.validate_where_clause(tables, where_clause)

                if validation.valid:
                    return PhaseResult(
                        output=where_clause,
                        votes=vote_count,
                        total_samples=num_samples,
                        valid=True,
                        validation_attempts=attempt + 1,
                        error_history=error_history
                    )
                else:
                    error_history.append(
                        f"Attempt {attempt + 1}: WHERE '{where_clause[:50]}...' failed: {validation.error_message}"
                    )
                    if validation.suggestions:
                        feedback = f"\n\nPREVIOUS ATTEMPT FAILED: {validation.error_message}\nSuggestions: {', '.join(validation.suggestions)}"

        # Return best effort
        best_where = where_counts.most_common(1)[0][0] if where_counts else ""
        return PhaseResult(
            output=best_where,
            votes=where_counts.get(best_where, 0),
            total_samples=num_samples,
            valid=False,
            validation_attempts=self.max_retries + 1,
            error_history=error_history
        )

    def _parse_where(self, response: str) -> str:
        """Parse WHERE clause from response."""
        response = response.strip()

        if "no filter" in response.lower() or "no where" in response.lower():
            return ""

        # Remove "WHERE" keyword
        if response.lower().startswith("where"):
            response = response[5:].strip()

        # Take first meaningful line
        for line in response.split("\n"):
            line = line.strip()
            if line and any(op in line for op in ["=", ">", "<", "LIKE", "IN", "BETWEEN"]):
                return line

        return response.split("\n")[0].strip() if response else ""

    def phase3_aggregation(
        self, question: str, hint: str, schema: str, tables: list[str],
        where_clause: str, validator: SQLValidator, num_samples: int = 8
    ) -> PhaseResult:
        """
        Phase 3: Detect and validate aggregation structure.

        For superlative questions, ensures GROUP BY + ORDER BY + LIMIT structure.
        """
        error_history = []
        feedback = ""

        tables_str = ", ".join(tables)

        # Check if this is a superlative question
        q_lower = question.lower()
        is_superlative = any(w in q_lower for w in ["highest", "lowest", "most", "least", "peak", "maximum", "minimum"])

        for attempt in range(self.max_retries + 1):
            if is_superlative:
                prompt_text = f"""Given this database schema:

{schema}

Question: {question}
{f"Hint: {hint}" if hint else ""}

Tables: {tables_str}
{f"WHERE: {where_clause}" if where_clause else "No WHERE clause"}
{feedback}

This question asks for a superlative (highest/lowest/most/least).
Write the GROUP BY, ORDER BY, and LIMIT clauses needed.

Format:
GROUP BY: column(s)
ORDER BY: SUM/COUNT/AVG(column) DESC/ASC
LIMIT: 1"""
            else:
                prompt_text = f"""Given this database schema:

{schema}

Question: {question}
{f"Hint: {hint}" if hint else ""}

Tables: {tables_str}
{f"WHERE: {where_clause}" if where_clause else "No WHERE clause"}
{feedback}

Do we need GROUP BY, ORDER BY, or LIMIT?
Write any that are needed, or "None needed" if not."""

            messages = [{"role": "user", "content": prompt_text}]
            prompt = self._create_prompt(messages)

            responses = self._generate_samples(prompt, num_samples, max_new_tokens=200)

            # Parse aggregation structures
            agg_structures = []
            for resp in responses:
                agg = self._parse_aggregation(resp)
                agg_structures.append(agg)

            # Vote
            agg_counts = Counter(tuple(sorted(a.items())) for a in agg_structures)

            for agg_tuple, vote_count in agg_counts.most_common():
                agg = dict(agg_tuple)

                # Validate
                validation = validator.validate_aggregation(
                    question,
                    has_group_by=bool(agg.get("group_by")),
                    has_order_by=bool(agg.get("order_by")),
                    has_limit=bool(agg.get("limit"))
                )

                if validation.valid:
                    # Convert back to string
                    agg_str = self._format_aggregation(agg)
                    return PhaseResult(
                        output=agg_str,
                        votes=vote_count,
                        total_samples=num_samples,
                        valid=True,
                        validation_attempts=attempt + 1,
                        error_history=error_history
                    )
                else:
                    error_history.append(
                        f"Attempt {attempt + 1}: Aggregation {agg} failed: {validation.error_message}"
                    )
                    feedback = f"\n\nPREVIOUS ATTEMPT FAILED: {validation.error_message}\nSuggestions: {', '.join(validation.suggestions)}"

        # Return best effort
        best_agg = dict(agg_counts.most_common(1)[0][0]) if agg_counts else {}
        return PhaseResult(
            output=self._format_aggregation(best_agg),
            votes=agg_counts.get(tuple(sorted(best_agg.items())), 0) if best_agg else 0,
            total_samples=num_samples,
            valid=False,
            validation_attempts=self.max_retries + 1,
            error_history=error_history
        )

    def _parse_aggregation(self, response: str) -> dict:
        """Parse aggregation clauses from response."""
        agg = {"group_by": "", "order_by": "", "limit": ""}

        for line in response.split("\n"):
            line = line.strip().lower()
            if line.startswith("group by:"):
                agg["group_by"] = line.split(":", 1)[1].strip()
            elif line.startswith("order by:"):
                agg["order_by"] = line.split(":", 1)[1].strip()
            elif line.startswith("limit:"):
                agg["limit"] = line.split(":", 1)[1].strip()

        return agg

    def _format_aggregation(self, agg: dict) -> str:
        """Format aggregation dict as SQL clauses."""
        parts = []
        if agg.get("group_by"):
            parts.append(f"GROUP BY {agg['group_by']}")
        if agg.get("order_by"):
            parts.append(f"ORDER BY {agg['order_by']}")
        if agg.get("limit"):
            parts.append(f"LIMIT {agg['limit']}")
        return " ".join(parts)

    def phase4_final(
        self, question: str, hint: str, schema: str, tables: list[str],
        where_clause: str, aggregation: str, db_path: str, num_samples: int = 10
    ) -> PhaseResult:
        """
        Phase 4: Assemble final SQL with plan-based voting.
        """
        tables_str = ", ".join(tables)

        prompt_text = f"""Given this database schema:

{schema}

Question: {question}
{f"Hint: {hint}" if hint else ""}

You MUST use these components:
- Tables: {tables_str}
{f"- WHERE: {where_clause}" if where_clause else "- No WHERE clause"}
{f"- {aggregation}" if aggregation else "- No GROUP BY/ORDER BY/LIMIT"}

Write the COMPLETE SQL query. Include the SELECT clause with exactly the columns asked for.
Return ONLY the SQL, no explanation."""

        messages = [{"role": "user", "content": prompt_text}]
        prompt = self._create_prompt(messages)

        responses = self._generate_samples(prompt, num_samples, max_new_tokens=400)

        # Extract SQL and validate with EXPLAIN
        candidates_with_plans = []
        for resp in responses:
            sql = self._extract_sql(resp)
            if sql:
                result = explain_query(sql, db_path)
                if isinstance(result, ExplainSuccess):
                    signature = normalize_plan(result.plan)
                    cost = self._calculate_cost(result.plan)
                    candidates_with_plans.append((sql, signature, cost))
                else:
                    candidates_with_plans.append((sql, "ERROR", -1))

        if not candidates_with_plans:
            return PhaseResult(
                output="",
                votes=0,
                total_samples=num_samples,
                valid=False,
                error_history=["No valid SQL candidates generated"]
            )

        # Plan-based voting
        winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)

        return PhaseResult(
            output=winning_sql,
            votes=vote_stats.get("winning_votes", 0),
            total_samples=num_samples,
            valid=winning_sig != "ERROR"
        )

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from response."""
        sql = response.strip()

        # Remove markdown
        if "```sql" in sql.lower():
            sql = sql.split("```sql", 1)[1].split("```")[0].strip()
        elif "```" in sql:
            parts = sql.split("```")
            if len(parts) >= 2:
                sql = parts[1].strip()

        # Clean up
        lines = []
        for line in sql.split("\n"):
            line = line.strip()
            if line and not line.startswith("--"):
                lines.append(line)
            elif lines:
                break

        return " ".join(lines).strip().rstrip(";")

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
        self, question: str, db_path: str, hint: str = "",
        samples_per_phase: int = 8
    ) -> M18Result:
        """
        Generate SQL using 4-phase validated decomposition.

        Args:
            question: Natural language question
            db_path: Database path
            hint: Optional hint from BIRD
            samples_per_phase: Number of samples per phase

        Returns:
            M18Result with complete trace
        """
        start_time = time.perf_counter()

        # Initialize
        schema = get_augmented_schema(db_path, max_rows_per_table=3)
        validator = SQLValidator(db_path)

        phases = {}
        validation_stats = {
            "total_validation_attempts": 0,
            "phases_passed_first_try": 0,
            "phases_needed_retry": 0,
        }

        # Phase 1: Table Selection
        phase1 = self.phase1_tables(question, hint, schema, validator, samples_per_phase)
        phases["tables"] = phase1
        validation_stats["total_validation_attempts"] += phase1.validation_attempts
        if phase1.validation_attempts == 1 and phase1.valid:
            validation_stats["phases_passed_first_try"] += 1
        elif phase1.validation_attempts > 1:
            validation_stats["phases_needed_retry"] += 1

        tables = phase1.output.split(",") if phase1.output else []
        tables = [t.strip() for t in tables if t.strip()]

        if not tables:
            # Fallback: try to generate complete SQL directly
            total_time = (time.perf_counter() - start_time) * 1000
            return M18Result(
                sql="",
                valid=False,
                phases=phases,
                total_latency_ms=total_time,
                validation_stats=validation_stats
            )

        # Phase 2: Filter Generation
        phase2 = self.phase2_filters(question, hint, schema, tables, validator, samples_per_phase)
        phases["filters"] = phase2
        validation_stats["total_validation_attempts"] += phase2.validation_attempts
        if phase2.validation_attempts == 1 and phase2.valid:
            validation_stats["phases_passed_first_try"] += 1
        elif phase2.validation_attempts > 1:
            validation_stats["phases_needed_retry"] += 1

        where_clause = phase2.output

        # Phase 3: Aggregation Detection
        phase3 = self.phase3_aggregation(
            question, hint, schema, tables, where_clause, validator, samples_per_phase
        )
        phases["aggregation"] = phase3
        validation_stats["total_validation_attempts"] += phase3.validation_attempts
        if phase3.validation_attempts == 1 and phase3.valid:
            validation_stats["phases_passed_first_try"] += 1
        elif phase3.validation_attempts > 1:
            validation_stats["phases_needed_retry"] += 1

        aggregation = phase3.output

        # Phase 4: Final Assembly
        phase4 = self.phase4_final(
            question, hint, schema, tables, where_clause, aggregation, db_path, samples_per_phase + 2
        )
        phases["final"] = phase4

        total_time = (time.perf_counter() - start_time) * 1000

        return M18Result(
            sql=phase4.output,
            valid=phase4.valid,
            phases=phases,
            total_latency_ms=total_time,
            validation_stats=validation_stats
        )
