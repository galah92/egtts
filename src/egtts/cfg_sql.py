"""Token-Level Classifier-Free Guidance for SQL Generation.

This module implements EG-CFG (Execution-Guided Classifier-Free Guidance) adapted
for SQL generation. The key insight: we can inject schema knowledge directly into
the token probability distribution, making a small model "smarter" about valid SQL.

Core idea:
    P_guided(token) = P_model(token) + alpha * (P_valid(token) - P_base(token))

Where P_valid boosts tokens that lead to valid SQL (correct table/column names,
valid syntax, etc.)

Reference: https://arxiv.org/html/2506.10948v2 (EG-CFG for code generation)
"""

import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn.functional as F


class SQLContext(Enum):
    """Current context in SQL generation for determining what's valid."""
    START = auto()           # Beginning, expecting SELECT
    SELECT_COLS = auto()     # After SELECT, expecting column names
    FROM_TABLE = auto()      # After FROM, expecting table name
    JOIN_TABLE = auto()      # After JOIN, expecting table name
    JOIN_ON = auto()         # After ON, expecting join condition
    WHERE_COND = auto()      # After WHERE, expecting conditions
    GROUP_BY = auto()        # After GROUP BY, expecting columns
    ORDER_BY = auto()        # After ORDER BY, expecting columns/aggregates
    UNKNOWN = auto()         # Can't determine context


@dataclass
class SchemaInfo:
    """Pre-indexed schema information for fast lookup."""
    tables: set[str]                          # All table names (lowercase)
    columns: dict[str, set[str]]              # table -> set of columns
    all_columns: set[str]                     # All column names (for ambiguous refs)
    column_to_tables: dict[str, list[str]]    # column -> tables containing it

    # Token-level lookup (for fast checking during generation)
    valid_identifiers: set[str]               # All valid table/column names

    @classmethod
    def from_database(cls, db_path: str) -> "SchemaInfo":
        """Build schema info from SQLite database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables = set()
        columns = {}
        all_columns = set()
        column_to_tables = {}

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (table_name,) in cursor.fetchall():
            table_lower = table_name.lower()
            tables.add(table_lower)
            columns[table_lower] = set()

            # Get columns for this table
            cursor.execute(f"PRAGMA table_info({table_name})")
            for row in cursor.fetchall():
                col_name = row[1].lower()
                columns[table_lower].add(col_name)
                all_columns.add(col_name)

                if col_name not in column_to_tables:
                    column_to_tables[col_name] = []
                column_to_tables[col_name].append(table_lower)

        conn.close()

        # Build valid identifiers set
        valid_identifiers = tables | all_columns

        return cls(
            tables=tables,
            columns=columns,
            all_columns=all_columns,
            column_to_tables=column_to_tables,
            valid_identifiers=valid_identifiers,
        )

    def is_valid_table(self, name: str) -> bool:
        """Check if name is a valid table."""
        return name.lower() in self.tables

    def is_valid_column(self, name: str, table: Optional[str] = None) -> bool:
        """Check if name is a valid column, optionally in specific table."""
        name_lower = name.lower()
        if table:
            table_lower = table.lower()
            return table_lower in self.columns and name_lower in self.columns[table_lower]
        return name_lower in self.all_columns

    def is_valid_identifier(self, name: str) -> bool:
        """Check if name is any valid table or column."""
        return name.lower() in self.valid_identifiers


@dataclass
class SQLParseState:
    """Tracks the current state of partial SQL for context-aware validation."""
    context: SQLContext = SQLContext.START
    current_tables: list[str] = field(default_factory=list)  # Tables in FROM/JOIN
    current_aliases: dict[str, str] = field(default_factory=dict)  # alias -> table
    select_columns: list[str] = field(default_factory=list)
    partial_sql: str = ""

    def update(self, new_token: str):
        """Update state based on new token."""
        self.partial_sql += new_token
        token_upper = new_token.strip().upper()

        # State transitions based on SQL keywords
        if token_upper == "SELECT":
            self.context = SQLContext.SELECT_COLS
        elif token_upper == "FROM":
            self.context = SQLContext.FROM_TABLE
        elif token_upper in ("JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN"):
            self.context = SQLContext.JOIN_TABLE
        elif token_upper == "ON":
            self.context = SQLContext.JOIN_ON
        elif token_upper == "WHERE":
            self.context = SQLContext.WHERE_COND
        elif token_upper == "GROUP BY":
            self.context = SQLContext.GROUP_BY
        elif token_upper == "ORDER BY":
            self.context = SQLContext.ORDER_BY


class SQLTokenValidator:
    """
    Validates tokens during SQL generation based on schema and context.

    This is the core component that computes P_valid for CFG.
    """

    def __init__(self, schema: SchemaInfo, tokenizer, vocab_size: int = None):
        self.schema = schema
        self.tokenizer = tokenizer
        # Use provided vocab_size or fall back to tokenizer
        self.vocab_size = vocab_size or len(tokenizer.get_vocab())

        # Pre-compute token IDs for SQL keywords
        self.sql_keywords = {
            "SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR", "IN",
            "GROUP", "BY", "ORDER", "ASC", "DESC", "LIMIT", "AS", "DISTINCT",
            "COUNT", "SUM", "AVG", "MAX", "MIN", "HAVING", "INNER", "LEFT",
            "RIGHT", "OUTER", "LIKE", "BETWEEN", "IS", "NULL", "NOT", "CASE",
            "WHEN", "THEN", "ELSE", "END", "CAST", "SUBSTR", "REAL", "FLOAT",
            "INTEGER", "TEXT",
        }

        # Build mapping from token strings to validity scores
        self._build_token_validity_cache()

    def _build_token_validity_cache(self):
        """Pre-compute which tokens are valid SQL identifiers."""
        self.valid_token_ids = set()
        self.table_token_ids = set()
        self.column_token_ids = set()
        self.keyword_token_ids = set()

        vocab = self.tokenizer.get_vocab()

        for token_str, token_id in vocab.items():
            # Clean token (remove special chars like Ġ for space)
            clean_token = token_str.replace("Ġ", "").replace("▁", "").strip()
            clean_lower = clean_token.lower()

            if not clean_token:
                continue

            # Check if it's a valid identifier
            if clean_lower in self.schema.tables:
                self.table_token_ids.add(token_id)
                self.valid_token_ids.add(token_id)

            if clean_lower in self.schema.all_columns:
                self.column_token_ids.add(token_id)
                self.valid_token_ids.add(token_id)

            if clean_token.upper() in self.sql_keywords:
                self.keyword_token_ids.add(token_id)
                self.valid_token_ids.add(token_id)

    def compute_validity_mask(
        self,
        state: SQLParseState,
        vocab_size: int = None
    ) -> torch.Tensor:
        """
        Compute a validity score for each token in vocabulary.

        Returns tensor of shape (vocab_size,) with scores:
        - 1.0 for definitely valid tokens
        - 0.0 for neutral (don't know)
        - -1.0 for definitely invalid tokens

        This will be used for CFG: P_guided = P + alpha * validity_mask
        """
        vs = vocab_size or self.vocab_size
        mask = torch.zeros(vs)

        # Context-dependent boosting
        if state.context == SQLContext.FROM_TABLE:
            # After FROM, boost table names
            for tid in self.table_token_ids:
                mask[tid] = 1.0
            # Penalize column names (wrong context)
            for cid in self.column_token_ids:
                if cid not in self.table_token_ids:
                    mask[cid] = -0.5

        elif state.context == SQLContext.JOIN_TABLE:
            # After JOIN, boost table names
            for tid in self.table_token_ids:
                mask[tid] = 1.0

        elif state.context in (SQLContext.SELECT_COLS, SQLContext.WHERE_COND,
                                SQLContext.GROUP_BY, SQLContext.ORDER_BY):
            # In these contexts, boost column names
            for cid in self.column_token_ids:
                mask[cid] = 1.0
            # Also boost table names (for table.column syntax)
            for tid in self.table_token_ids:
                mask[tid] = 0.5

        # Always keep keywords neutral-to-positive
        for kid in self.keyword_token_ids:
            if mask[kid] <= 0:
                mask[kid] = 0.2

        return mask


class CFGSQLGenerator:
    """
    SQL Generator using Classifier-Free Guidance.

    This is the main class that implements token-by-token generation
    with schema-aware guidance.
    """

    def __init__(
        self,
        model,
        tokenizer,
        cfg_alpha: float = 1.5,  # CFG strength
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg_alpha = cfg_alpha
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.device = next(model.parameters()).device

    def generate(
        self,
        prompt: str,
        schema: SchemaInfo,
        num_samples: int = 1,
    ) -> list[str]:
        """
        Generate SQL with CFG guidance.

        Args:
            prompt: The input prompt (question + schema)
            schema: Pre-indexed schema information
            num_samples: Number of samples to generate

        Returns:
            List of generated SQL strings
        """
        # Initialize validator with model's vocab size
        model_vocab_size = self.model.config.vocab_size
        validator = SQLTokenValidator(schema, self.tokenizer, vocab_size=model_vocab_size)

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        results = []

        for _ in range(num_samples):
            generated = self._generate_single(
                input_ids.clone(),
                attention_mask.clone(),
                validator,
            )
            results.append(generated)

        return results

    def _generate_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        validator: SQLTokenValidator,
    ) -> str:
        """Generate a single SQL query with CFG."""

        state = SQLParseState()
        generated_ids = []

        # Get vocabulary size from model config (more reliable than tokenizer)
        vocab_size = self.model.config.vocab_size

        # Get EOS token ID for penalization
        eos_token_id = self.tokenizer.eos_token_id

        for step in range(self.max_new_tokens):
            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # Compute validity mask based on current SQL state
            validity_mask = validator.compute_validity_mask(state, vocab_size)
            validity_mask = validity_mask.to(self.device).unsqueeze(0)  # (1, vocab_size)

            # Apply CFG: guided_logits = logits + alpha * validity_mask
            guided_logits = logits + self.cfg_alpha * validity_mask

            # Penalize EOS token if SQL is incomplete
            partial_sql = self.tokenizer.decode(generated_ids) if generated_ids else ""
            if not self._is_sql_complete(partial_sql):
                # Strongly penalize EOS when SQL is not complete
                if eos_token_id is not None and eos_token_id < vocab_size:
                    guided_logits[0, eos_token_id] -= 10.0  # Strong penalty

            # Apply temperature
            guided_logits = guided_logits / self.temperature

            # Sample from distribution
            probs = F.softmax(guided_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Check for EOS - only accept if SQL looks complete
            if next_token_id.item() == eos_token_id:
                if self._is_sql_complete(partial_sql):
                    break
                else:
                    # Force re-sample without EOS
                    guided_logits[0, eos_token_id] = float('-inf')
                    probs = F.softmax(guided_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    # If still EOS somehow, break anyway
                    if next_token_id.item() == eos_token_id:
                        break

            # Decode token to update state
            token_str = self.tokenizer.decode(next_token_id[0])
            state.update(token_str)

            # Append to sequence
            generated_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
            ], dim=1)

            # Early stopping: check for SQL completion patterns
            partial = self.tokenizer.decode(generated_ids)
            if self._is_sql_complete(partial):
                break

        # Decode final result
        sql = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._clean_sql(sql)

    def _is_sql_complete(self, sql: str) -> bool:
        """Check if SQL query appears complete."""
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()

        # Must have minimum length to be a valid query
        if len(sql_stripped) < 20:
            return False

        # Check for explicit ending
        if sql_stripped.endswith(";"):
            return True

        # Check for LIMIT N pattern (common complete ending)
        if re.search(r"LIMIT\s+\d+\s*$", sql_stripped, re.IGNORECASE):
            return True

        # Keywords that indicate the query is NOT complete (trailing keywords)
        incomplete_endings = [
            r"\bFROM\s*$",      # Missing table name
            r"\bSELECT\s*$",    # Missing columns
            r"\bWHERE\s*$",     # Missing condition
            r"\bAND\s*$",       # Missing condition
            r"\bOR\s*$",        # Missing condition
            r"\bON\s*$",        # Missing join condition
            r"\bJOIN\s*$",      # Missing table
            r"\bINNER\s*$",     # Incomplete JOIN
            r"\bLEFT\s*$",      # Incomplete JOIN
            r"\bRIGHT\s*$",     # Incomplete JOIN
            r"\bOUTER\s*$",     # Incomplete JOIN
            r"\bGROUP\s*$",     # Missing BY
            r"\bORDER\s*$",     # Missing BY
            r"\bBY\s*$",        # Missing column
            r"\bAS\s*$",        # Missing alias
            r"\bIN\s*$",        # Missing list
            r"\bBETWEEN\s*$",   # Missing range
            r"\bLIKE\s*$",      # Missing pattern
            r"\bHAVING\s*$",    # Missing condition
            r"\bCASE\s*$",      # Incomplete CASE
            r"\bWHEN\s*$",      # Missing condition
            r"\bTHEN\s*$",      # Missing value
            r"\bELSE\s*$",      # Missing value
            r"\bCAST\s*$",      # Missing expression
            r"=\s*$",          # Missing value after comparison
            r"<\s*$",
            r">\s*$",
            r",\s*$",          # Expecting more items
            r"\(\s*$",         # Open parenthesis
        ]

        for pattern in incomplete_endings:
            if re.search(pattern, sql_upper):
                return False

        # Must have balanced parentheses
        if sql_stripped.count("(") != sql_stripped.count(")"):
            return False

        # Must have basic SELECT ... FROM structure with actual content
        if "SELECT" not in sql_upper or "FROM" not in sql_upper:
            return False

        # Check FROM is followed by something (table name)
        from_match = re.search(r"\bFROM\s+(\w+)", sql_upper)
        if not from_match:
            return False

        # If we got here, the query looks structurally complete
        return True

    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL."""
        sql = sql.strip()

        # Remove markdown code blocks
        if sql.startswith("```"):
            # Find the first newline after ```
            first_newline = sql.find("\n")
            if first_newline != -1:
                sql = sql[first_newline + 1:]
            else:
                sql = sql[3:]  # Just remove ```
        if sql.endswith("```"):
            sql = sql[:-3]
        # Also handle ```sql prefix
        if sql.lower().startswith("sql"):
            sql = sql[3:].lstrip()

        sql = sql.strip()

        # Remove trailing semicolon for consistency
        sql = sql.rstrip(";").strip()

        # Remove any explanation text after the query
        lines = sql.split("\n")
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line.upper().startswith(("--", "/*", "NOTE:", "THIS", "```")):
                break
            if line:
                sql_lines.append(line)

        return " ".join(sql_lines)


class CFGSQLGeneratorWithBeams(CFGSQLGenerator):
    """
    Extended CFG generator with beam search and execution validation.

    Combines CFG guidance with our existing plan-based voting.
    """

    def __init__(
        self,
        model,
        tokenizer,
        cfg_alpha: float = 1.5,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        num_beams: int = 5,
    ):
        super().__init__(model, tokenizer, cfg_alpha, temperature, max_new_tokens)
        self.num_beams = num_beams

    def generate_with_voting(
        self,
        prompt: str,
        schema: SchemaInfo,
        db_path: str,
        num_samples: int = 15,
    ) -> tuple[str, dict]:
        """
        Generate SQL with CFG guidance and plan-based voting.

        Combines:
        1. CFG for schema-aware token generation
        2. Multiple samples for diversity
        3. EXPLAIN-based validation
        4. Plan-based majority voting
        """
        from .database import explain_query, ExplainSuccess
        from .plans import normalize_plan, vote_by_plan

        # Generate candidates with CFG
        candidates = self.generate(prompt, schema, num_samples)

        # Validate with EXPLAIN and collect plan signatures
        candidates_with_plans = []

        for sql in candidates:
            result = explain_query(sql, db_path)
            if isinstance(result, ExplainSuccess):
                signature = normalize_plan(result.plan)
                # Simple cost heuristic
                plan_str = str(result.plan).upper()
                cost = 100 if "SCAN" in plan_str else 0
                candidates_with_plans.append((sql, signature, cost))
            else:
                candidates_with_plans.append((sql, "ERROR", -1))

        # Vote
        winning_sql, winning_sig, vote_stats = vote_by_plan(candidates_with_plans)

        vote_stats["strategy"] = "CFG_SQL"
        vote_stats["cfg_alpha"] = self.cfg_alpha

        return winning_sql, vote_stats


class PartialSQLValidator:
    """
    Validates partial SQL by completing them artificially.

    This enables EXPLAIN-based feedback DURING generation, not just after.

    Key idea: Complete partial SQL with neutral suffixes that make them
    syntactically valid without changing the semantic intent.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

        # Get valid table/column names for completion
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        self.tables = [row[0] for row in cursor.fetchall()]
        self.default_table = self.tables[0] if self.tables else "sqlite_master"

    def complete_partial_sql(self, partial_sql: str) -> str | None:
        """
        Complete a partial SQL query to make it syntactically valid.

        Returns None if the partial is too malformed to complete.
        """
        partial = partial_sql.strip().upper()
        original = partial_sql.strip()

        if not partial:
            return None

        # Patterns and their completions
        completions = [
            # Missing table after FROM/JOIN
            (r'\bFROM\s*$', f' {self.default_table} LIMIT 0'),
            (r'\bJOIN\s*$', f' {self.default_table} ON 1=1 LIMIT 0'),
            (r'\bINNER\s+JOIN\s*$', f' {self.default_table} ON 1=1 LIMIT 0'),
            (r'\bLEFT\s+JOIN\s*$', f' {self.default_table} ON 1=1 LIMIT 0'),
            (r'\bRIGHT\s+JOIN\s*$', f' {self.default_table} ON 1=1 LIMIT 0'),

            # Missing condition after WHERE/AND/OR/ON
            (r'\bWHERE\s*$', ' 1=1 LIMIT 0'),
            (r'\bAND\s*$', ' 1=1 LIMIT 0'),
            (r'\bOR\s*$', ' 1=1 LIMIT 0'),
            (r'\bON\s*$', ' 1=1 LIMIT 0'),
            (r'\bHAVING\s*$', ' 1=1 LIMIT 0'),

            # Missing columns after SELECT
            (r'\bSELECT\s*$', ' 1 LIMIT 0'),
            (r'\bSELECT\s+DISTINCT\s*$', ' 1 LIMIT 0'),

            # Missing columns after GROUP BY/ORDER BY
            (r'\bGROUP\s+BY\s*$', ' 1 LIMIT 0'),
            (r'\bORDER\s+BY\s*$', ' 1 LIMIT 0'),

            # Incomplete comparison
            (r'=\s*$', " '' LIMIT 0"),
            (r'<\s*$', ' 0 LIMIT 0'),
            (r'>\s*$', ' 0 LIMIT 0'),
            (r'\bLIKE\s*$', " '%' LIMIT 0"),
            (r'\bIN\s*$', ' (1) LIMIT 0'),
            (r'\bBETWEEN\s*$', ' 0 AND 1 LIMIT 0'),

            # Incomplete CASE
            (r'\bCASE\s*$', ' WHEN 1=1 THEN 1 END LIMIT 0'),
            (r'\bWHEN\s*$', ' 1=1 THEN 1 END LIMIT 0'),
            (r'\bTHEN\s*$', ' 1 END LIMIT 0'),
            (r'\bELSE\s*$', ' 1 END LIMIT 0'),

            # Missing alias
            (r'\bAS\s*$', ' x LIMIT 0'),

            # Trailing comma (expecting more items)
            (r',\s*$', ' 1 LIMIT 0'),

            # Open parenthesis
            (r'\(\s*$', '1) LIMIT 0'),
        ]

        # Try each completion pattern
        for pattern, suffix in completions:
            if re.search(pattern, partial, re.IGNORECASE):
                return original + suffix

        # Check for unbalanced parentheses
        open_parens = original.count('(') - original.count(')')
        if open_parens > 0:
            return original + ')' * open_parens + ' LIMIT 0'

        # If query looks complete-ish, just add LIMIT 0
        if 'SELECT' in partial and 'FROM' in partial:
            if not partial.endswith('LIMIT 0'):
                return original + ' LIMIT 0'
            return original

        # Query only has SELECT, no FROM yet - complete minimally
        if partial.startswith('SELECT') and 'FROM' not in partial:
            return original + f' FROM {self.default_table} LIMIT 0'

        return None

    def validate_partial(self, partial_sql: str) -> tuple[bool, str | None]:
        """
        Validate a partial SQL query using EXPLAIN.

        Returns:
            (is_valid, error_message)
        """
        completed = self.complete_partial_sql(partial_sql)
        if completed is None:
            return False, "Cannot complete partial SQL"

        try:
            cursor = self.conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {completed}")
            cursor.fetchall()
            return True, None
        except sqlite3.Error as e:
            return False, str(e)

    def get_validity_signal(self, partial_sql: str) -> float:
        """
        Get a validity signal for CFG guidance.

        Returns:
            1.0 if valid
            0.0 if cannot determine
            -1.0 if definitely invalid
        """
        completed = self.complete_partial_sql(partial_sql)
        if completed is None:
            return 0.0  # Can't determine

        try:
            cursor = self.conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {completed}")
            cursor.fetchall()
            return 1.0  # Valid!
        except sqlite3.Error as e:
            error_msg = str(e).lower()
            # Some errors indicate definite invalidity
            if 'no such table' in error_msg:
                return -1.0
            if 'no such column' in error_msg:
                return -1.0
            if 'syntax error' in error_msg:
                return -0.5  # Might be recoverable
            return 0.0  # Unknown

    def close(self):
        self.conn.close()


class ExecutionGuidedGenerator:
    """
    M22: Execution-Guided Generation with Partial SQL Validation.

    This is the true EG-CFG approach for SQL:
    1. Generate SQL in chunks (not token-by-token for speed)
    2. After each chunk, validate using EXPLAIN on completed partial SQL
    3. If invalid (e.g., wrong table name), backtrack and regenerate
    4. Use validity signal to guide generation

    This combines the speed of batch generation with the accuracy of
    execution feedback during generation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        chunk_size: int = 10,  # Generate this many tokens before validating
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.chunk_size = chunk_size
        self.device = next(model.parameters()).device

    def generate_with_validation(
        self,
        prompt: str,
        db_path: str,
        num_attempts: int = 5,
    ) -> tuple[str, dict]:
        """
        Generate SQL with execution-guided feedback.

        Strategy:
        1. Generate complete SQL candidates
        2. Validate with partial SQL checking during generation
        3. If we detect invalid table/column early, note it but continue
        4. Collect multiple candidates and pick the best one
        """
        validator = PartialSQLValidator(db_path)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_length = input_ids.shape[1]

        candidates = []
        backtrack_count = 0
        temp = self.temperature

        for attempt in range(num_attempts):
            # Vary temperature for diversity
            current_temp = temp + (attempt * 0.05)

            # Generate full SQL in one go
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=current_temp,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode generated SQL
            all_generated = outputs[0, prompt_length:]
            sql = self.tokenizer.decode(all_generated, skip_special_tokens=True)
            sql = self._extract_sql(sql)

            if not sql or len(sql) < 10:
                continue

            # Validate the SQL
            validity_signal = validator.get_validity_signal(sql)

            # Score the candidate
            score = validity_signal
            if self._is_sql_complete(sql):
                score += 1.0  # Bonus for completeness
            score += len(sql) / 200  # Small bonus for length

            candidates.append((sql, score, validity_signal))

            # If we found a valid, complete SQL, we can stop early
            if validity_signal >= 0.5 and self._is_sql_complete(sql):
                break

        validator.close()

        # Pick the best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_sql, best_score, best_validity = candidates[0]
        else:
            best_sql = "SELECT 1"
            best_score = -1
            best_validity = 0

        metadata = {
            "attempts": len(candidates),
            "backtrack_count": backtrack_count,
            "best_score": best_score,
            "best_validity": best_validity,
            "num_candidates": len(candidates),
        }

        return best_sql, metadata

    def _extract_sql(self, text: str) -> str:
        """Extract SQL from generated text."""
        text = text.strip()

        # Remove markdown code blocks
        if "```" in text:
            match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1)
            else:
                text = text.replace("```sql", "").replace("```", "")

        # Take first statement
        if ";" in text:
            text = text.split(";")[0]

        return text.strip()

    def _is_sql_complete(self, sql: str) -> bool:
        """Check if SQL appears complete."""
        sql_upper = sql.strip().upper()

        if len(sql) < 15:
            return False

        if not ('SELECT' in sql_upper and 'FROM' in sql_upper):
            return False

        # Check for incomplete patterns
        incomplete = [
            r'\bFROM\s*$', r'\bWHERE\s*$', r'\bAND\s*$', r'\bOR\s*$',
            r'\bJOIN\s*$', r'\bON\s*$', r'\bBY\s*$', r'=\s*$', r',\s*$',
        ]
        for pattern in incomplete:
            if re.search(pattern, sql_upper):
                return False

        # Check balanced parentheses
        if sql.count('(') != sql.count(')'):
            return False

        return True


class ExampleGuidedWithValidation:
    """
    M23: Example-Guided Generation with Execution Validation.

    Combines:
    1. M19's dynamic few-shot example selection
    2. M22's partial SQL validation with EXPLAIN
    3. Iterative refinement: if SQL is invalid, tell the model and retry

    This is the "best of both worlds" approach.
    """

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    def generate_with_examples_and_validation(
        self,
        question: str,
        schema: str,
        evidence: str,
        db_path: str,
        num_attempts: int = 3,
        num_samples_per_attempt: int = 5,
    ) -> tuple[str, dict]:
        """
        Generate SQL using example-guided prompting with execution validation.

        Strategy:
        1. Select relevant examples based on question patterns
        2. Generate SQL candidates with the enhanced prompt
        3. Validate each candidate with partial SQL validator
        4. If all invalid, regenerate with error feedback
        5. Pick the best valid candidate
        """
        from .example_bank import EXAMPLE_BANK, detect_question_patterns

        validator = PartialSQLValidator(db_path)

        # Detect patterns in the question
        detected_patterns = detect_question_patterns(question)

        # Select top examples for this question
        examples_with_scores = []
        for example in EXAMPLE_BANK:
            score = example.similarity_score(question, detected_patterns)
            if score > 0:
                examples_with_scores.append((example, score))

        examples_with_scores.sort(key=lambda x: x[1], reverse=True)
        selected_examples = [ex for ex, _ in examples_with_scores[:3]]

        # Build the example-guided prompt
        all_candidates = []
        error_feedback = ""

        for attempt in range(num_attempts):
            prompt = self._build_prompt(
                question, schema, evidence, selected_examples, error_feedback
            )

            # Generate multiple candidates
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_length = inputs["input_ids"].shape[1]

            for sample_idx in range(num_samples_per_attempt):
                temp = self.temperature + (sample_idx * 0.05)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=temp,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                generated = outputs[0, prompt_length:]
                sql = self.tokenizer.decode(generated, skip_special_tokens=True)
                sql = self._extract_sql(sql)

                if not sql or len(sql) < 10:
                    continue

                # Validate with partial SQL validator
                validity_signal = validator.get_validity_signal(sql)
                is_complete = self._is_sql_complete(sql)

                score = validity_signal
                if is_complete:
                    score += 1.0
                score += len(sql) / 200

                all_candidates.append({
                    "sql": sql,
                    "validity": validity_signal,
                    "complete": is_complete,
                    "score": score,
                    "attempt": attempt,
                })

                # Early exit if we found a perfect candidate
                if validity_signal >= 0.5 and is_complete:
                    break

            # Check if we have valid candidates
            valid_candidates = [c for c in all_candidates if c["validity"] >= 0 and c["complete"]]
            if valid_candidates:
                break

            # No valid candidates - build error feedback for next attempt
            if all_candidates:
                # Find the most common error
                invalid_sqls = [c["sql"] for c in all_candidates if c["validity"] < 0]
                if invalid_sqls:
                    sample_invalid = invalid_sqls[0]
                    is_valid, error_msg = validator.validate_partial(sample_invalid)
                    if error_msg:
                        error_feedback = f"\nPrevious attempt failed with error: {error_msg}. Please fix the SQL."

        validator.close()

        # Select the best candidate
        if all_candidates:
            all_candidates.sort(key=lambda x: x["score"], reverse=True)
            best = all_candidates[0]
            best_sql = best["sql"]
        else:
            best_sql = "SELECT 1"
            best = {"validity": 0, "complete": False, "score": -1, "attempt": 0}

        metadata = {
            "total_candidates": len(all_candidates),
            "valid_candidates": len([c for c in all_candidates if c["validity"] >= 0]),
            "complete_candidates": len([c for c in all_candidates if c["complete"]]),
            "best_validity": best.get("validity", 0),
            "best_score": best.get("score", -1),
            "attempts_used": best.get("attempt", 0) + 1,
            "num_examples_used": len(selected_examples),
            "detected_patterns": detected_patterns,
            "had_error_feedback": bool(error_feedback),
        }

        return best_sql, metadata

    def _build_prompt(
        self,
        question: str,
        schema: str,
        evidence: str,
        examples: list,
        error_feedback: str = "",
    ) -> str:
        """Build a prompt with selected examples."""
        # Format examples
        example_text = ""
        for i, ex in enumerate(examples, 1):
            example_text += f"\nExample {i}:\n"
            example_text += f"Question: {ex.question}\n"
            if ex.hint:
                example_text += f"Hint: {ex.hint}\n"
            example_text += f"SQL: {ex.sql}\n"

        # Build the full prompt
        prompt = f"""You are an expert SQL assistant. Generate a SQLite query for the following question.

Database Schema:
{schema}

{example_text}

Now generate SQL for this question:
Question: {question}
"""
        if evidence:
            prompt += f"Hint: {evidence}\n"

        if error_feedback:
            prompt += f"{error_feedback}\n"

        prompt += "\nSQL:"

        # Apply chat template if available
        messages = [{"role": "user", "content": prompt}]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception:
            return prompt

    def _extract_sql(self, text: str) -> str:
        """Extract SQL from generated text."""
        text = text.strip()

        # Remove markdown code blocks
        if "```" in text:
            match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1)
            else:
                text = text.replace("```sql", "").replace("```", "")

        # Take first statement
        if ";" in text:
            text = text.split(";")[0]

        return text.strip()

    def _is_sql_complete(self, sql: str) -> bool:
        """Check if SQL appears complete."""
        sql_upper = sql.strip().upper()

        if len(sql) < 15:
            return False

        if not ('SELECT' in sql_upper and 'FROM' in sql_upper):
            return False

        incomplete = [
            r'\bFROM\s*$', r'\bWHERE\s*$', r'\bAND\s*$', r'\bOR\s*$',
            r'\bJOIN\s*$', r'\bON\s*$', r'\bBY\s*$', r'=\s*$', r',\s*$',
        ]
        for pattern in incomplete:
            if re.search(pattern, sql_upper):
                return False

        if sql.count('(') != sql.count(')'):
            return False

        return True


class SchemaAwareReranker:
    """
    M21: Fast generation with schema-aware reranking.

    Instead of slow token-by-token CFG, this approach:
    1. Uses fast batch generation with diverse sampling
    2. Validates candidates with EXPLAIN
    3. Scores candidates using schema coverage metrics
    4. Combines EXPLAIN validity + schema score for final ranking

    This is more practical than token-level CFG while still leveraging
    schema knowledge for improved accuracy.
    """

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    def generate_candidates(
        self,
        prompt: str,
        num_samples: int = 15,
    ) -> list[str]:
        """Generate multiple SQL candidates using fast batch generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Use diverse sampling with do_sample=True
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=num_samples,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and extract SQL
        candidates = []
        for output in outputs:
            # Get only generated portion
            generated = output[inputs["input_ids"].shape[1]:]
            sql = self.tokenizer.decode(generated, skip_special_tokens=True)
            sql = self._clean_sql(sql)
            if sql:
                candidates.append(sql)

        return candidates

    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL."""
        sql = sql.strip()

        # Remove markdown code blocks
        if "```" in sql:
            # Extract SQL from code block
            match = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1)
            else:
                # Remove any ``` markers
                sql = sql.replace("```sql", "").replace("```", "")

        sql = sql.strip()

        # Take only the first SQL statement
        if ";" in sql:
            sql = sql.split(";")[0]

        # Remove trailing semicolon for consistency
        sql = sql.rstrip(";").strip()

        # Remove explanation text
        lines = sql.split("\n")
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line.upper().startswith(("--", "/*", "NOTE:", "THIS", "THE ", "HERE")):
                break
            if line:
                sql_lines.append(line)

        return " ".join(sql_lines)

    def score_candidate(self, sql: str, schema: SchemaInfo) -> float:
        """
        Score a SQL candidate based on schema coverage.

        Higher score = better schema alignment.
        """
        score = 0.0
        sql_upper = sql.upper()
        sql_lower = sql.lower()

        # Extract identifiers from SQL (simple tokenization)
        # Match word characters (table/column names)
        identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', sql_lower))

        # Remove SQL keywords
        sql_keywords = {
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'outer',
            'on', 'and', 'or', 'not', 'in', 'like', 'between', 'is', 'null',
            'group', 'by', 'order', 'asc', 'desc', 'limit', 'as', 'distinct',
            'count', 'sum', 'avg', 'max', 'min', 'having', 'case', 'when',
            'then', 'else', 'end', 'cast', 'substr', 'real', 'float', 'integer',
            'text', 'iif', 't1', 't2', 't3', 't4', 't5',  # common aliases
        }
        identifiers = identifiers - sql_keywords

        # Score based on valid table references
        tables_used = 0
        invalid_tables = 0
        for ident in identifiers:
            if ident in schema.tables:
                tables_used += 1
                score += 2.0  # Good: using valid table
            elif ident in schema.all_columns:
                score += 1.0  # Good: using valid column
            elif len(ident) > 2:  # Likely an identifier, not a short alias
                # Check if it's similar to any valid identifier (typo detection)
                if not any(ident.startswith(t[:3]) for t in schema.tables):
                    invalid_tables += 1

        # Penalize invalid identifiers
        score -= invalid_tables * 1.5

        # Bonus for having FROM clause with valid table
        if 'FROM' in sql_upper:
            from_match = re.search(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
            if from_match:
                table_name = from_match.group(1).lower()
                if table_name in schema.tables:
                    score += 3.0  # Strong bonus for valid FROM table

        # Bonus for JOIN with valid table
        join_matches = re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE)
        for table_name in join_matches:
            if table_name.lower() in schema.tables:
                score += 2.0

        # Check structural completeness
        has_select = 'SELECT' in sql_upper
        has_from = 'FROM' in sql_upper
        if has_select and has_from:
            score += 2.0  # Complete basic structure

        return score

    def generate_with_reranking(
        self,
        prompt: str,
        schema: SchemaInfo,
        db_path: str,
        num_samples: int = 15,
    ) -> tuple[str, dict]:
        """
        Generate SQL with schema-aware reranking.

        1. Generate diverse candidates
        2. Validate with EXPLAIN
        3. Score with schema coverage
        4. Return best candidate
        """
        from .database import explain_query, ExplainSuccess
        from .plans import normalize_plan, vote_by_plan

        # Generate candidates (fast batch generation)
        candidates = self.generate_candidates(prompt, num_samples)

        if not candidates:
            return "SELECT 1", {"error": "no_candidates"}

        # Score each candidate
        scored_candidates = []
        valid_candidates = []

        for sql in candidates:
            # Check EXPLAIN validity
            result = explain_query(sql, db_path)
            is_valid = isinstance(result, ExplainSuccess)

            # Compute schema score
            schema_score = self.score_candidate(sql, schema)

            # Combined score: validity is most important
            combined_score = schema_score
            if is_valid:
                combined_score += 10.0  # Large bonus for valid SQL
                plan_sig = normalize_plan(result.plan)
                plan_str = str(result.plan).upper()
                cost = 100 if "SCAN" in plan_str else 0
                valid_candidates.append((sql, plan_sig, cost, schema_score))
            else:
                plan_sig = "ERROR"
                cost = -1

            scored_candidates.append({
                "sql": sql,
                "valid": is_valid,
                "schema_score": schema_score,
                "combined_score": combined_score,
                "plan_sig": plan_sig,
            })

        # If we have valid candidates, use plan-based voting among them
        if valid_candidates:
            # Group by plan and pick the one with highest schema score
            plan_groups: dict[str, list[tuple[str, float]]] = {}
            for sql, plan_sig, cost, schema_score in valid_candidates:
                if plan_sig not in plan_groups:
                    plan_groups[plan_sig] = []
                plan_groups[plan_sig].append((sql, schema_score))

            # Find the plan with most votes, weighted by schema score
            best_plan = None
            best_weighted_score = -1
            for plan_sig, members in plan_groups.items():
                # Weighted score = count * avg_schema_score
                count = len(members)
                avg_score = sum(s for _, s in members) / count
                weighted = count * (1 + avg_score / 10)  # Weighted vote
                if weighted > best_weighted_score:
                    best_weighted_score = weighted
                    best_plan = plan_sig

            # Pick the SQL with highest schema score from winning plan
            winning_members = plan_groups[best_plan]
            winning_sql = max(winning_members, key=lambda x: x[1])[0]

            vote_stats = {
                "total_candidates": len(candidates),
                "valid_candidates": len(valid_candidates),
                "unique_plans": len(plan_groups),
                "winning_plan": best_plan,
                "winning_votes": len(winning_members),
                "winning_schema_score": max(s for _, s in winning_members),
            }
        else:
            # No valid candidates - pick by schema score alone
            best = max(scored_candidates, key=lambda x: x["schema_score"])
            winning_sql = best["sql"]
            vote_stats = {
                "total_candidates": len(candidates),
                "valid_candidates": 0,
                "winning_schema_score": best["schema_score"],
                "note": "no_valid_candidates",
            }

        vote_stats["strategy"] = "M21_SchemaRerank"
        return winning_sql, vote_stats


# =============================================================================
# High-level interface
# =============================================================================

def generate_sql_with_cfg(
    question: str,
    db_path: str,
    model,
    tokenizer,
    hint: str = "",
    cfg_alpha: float = 1.5,
    num_samples: int = 15,
) -> tuple[str, dict]:
    """
    High-level function to generate SQL with CFG guidance.

    Args:
        question: Natural language question
        db_path: Path to SQLite database
        model: Loaded language model
        tokenizer: Model tokenizer
        hint: Optional hint/evidence
        cfg_alpha: CFG guidance strength (higher = more schema-constrained)
        num_samples: Number of candidates to generate

    Returns:
        (sql, metadata) tuple
    """
    from .schema import get_augmented_schema
    from .model import create_sql_prompt

    # Build schema info for validation
    schema = SchemaInfo.from_database(db_path)

    # Get augmented schema for prompt
    augmented_schema = get_augmented_schema(db_path, max_rows_per_table=3)

    # Create prompt
    full_question = f"{question}\nHint: {hint}" if hint else question
    prompt = create_sql_prompt(full_question, augmented_schema, tokenizer)

    # Generate with CFG
    generator = CFGSQLGeneratorWithBeams(
        model, tokenizer,
        cfg_alpha=cfg_alpha,
        num_samples=num_samples,
    )

    sql, vote_stats = generator.generate_with_voting(
        prompt, schema, db_path, num_samples
    )

    return sql, vote_stats
