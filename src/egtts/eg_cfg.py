"""M25: EG-CFG - Execution-Guided Classifier-Free Guidance for SQL.

Based on the paper "Execution Guided Line-by-Line Code Generation" (arxiv:2506.10948).

Key insight: Use CFG to interpolate between:
- Unconditional: P(token | schema, question)
- Conditional: P(token | schema, question, execution_trace)

The CFG formula:
    log P_cfg = log P_uncond + γ * (log P_cond - log P_uncond)

For SQL, we adapt:
- Line boundaries → Clause boundaries (SELECT, FROM, WHERE, JOIN, etc.)
- Execution traces → EXPLAIN output + schema validation
"""

import re
import sqlite3
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SQLExecutionTrace:
    """Execution trace for SQL."""
    partial_sql: str
    is_valid: bool
    error_message: str | None
    explain_output: str | None
    tables_referenced: list[str]
    columns_referenced: list[str]


class EGCFG:
    """
    M25: Proper EG-CFG for SQL generation.

    Key differences from our previous attempts:
    1. Two forward passes per clause for proper CFG interpolation
    2. Rich execution traces (not just pass/fail)
    3. Clause-level boundaries for SQL (not token-level)
    4. Guidance scale γ for controlling execution influence
    """

    # SQL clause keywords that mark boundaries
    CLAUSE_KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT',
        'OUTER', 'ON', 'AND', 'OR', 'GROUP', 'ORDER', 'HAVING', 'LIMIT',
        'UNION', 'INTERSECT', 'EXCEPT', 'AS',
    }

    def __init__(
        self,
        model,
        tokenizer,
        db_path: str,
        gamma: float = 1.0,  # CFG guidance scale
        num_candidates: int = 3,  # Beam size per clause
        max_clauses: int = 10,  # Max SQL clauses
        temperature: float = 0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.db_path = db_path
        self.gamma = gamma
        self.num_candidates = num_candidates
        self.max_clauses = max_clauses
        self.temperature = temperature
        self.device = next(model.parameters()).device

        # Get schema info
        self.schema_info = self._get_schema_info()

    def _get_schema_info(self) -> dict:
        """Extract schema information from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        info = {"tables": {}, "all_tables": [], "all_columns": []}

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        info["all_tables"] = [t.lower() for t in tables]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            info["tables"][table.lower()] = [c.lower() for c in columns]
            info["all_columns"].extend([c.lower() for c in columns])

        conn.close()
        return info

    def _get_execution_trace(self, partial_sql: str) -> SQLExecutionTrace:
        """Get rich execution trace for partial SQL."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Extract referenced tables and columns
        tables = re.findall(r'\bFROM\s+(\w+)', partial_sql, re.IGNORECASE)
        tables += re.findall(r'\bJOIN\s+(\w+)', partial_sql, re.IGNORECASE)
        columns = re.findall(r'(?:SELECT|WHERE|ON|AND|OR|GROUP BY|ORDER BY)\s+(\w+)',
                            partial_sql, re.IGNORECASE)

        # Try to validate/explain
        is_valid = False
        error_message = None
        explain_output = None

        # Complete partial SQL for validation
        completed_sql = self._complete_for_validation(partial_sql)

        if completed_sql:
            try:
                cursor.execute(f"EXPLAIN QUERY PLAN {completed_sql}")
                explain_rows = cursor.fetchall()
                explain_output = "\n".join([str(row) for row in explain_rows])
                is_valid = True
            except sqlite3.Error as e:
                error_message = str(e)
                is_valid = False

        conn.close()

        return SQLExecutionTrace(
            partial_sql=partial_sql,
            is_valid=is_valid,
            error_message=error_message,
            explain_output=explain_output,
            tables_referenced=tables,
            columns_referenced=columns,
        )

    def _complete_for_validation(self, partial_sql: str) -> str | None:
        """Complete partial SQL to make it validatable."""
        partial_sql = partial_sql.strip()
        if not partial_sql:
            return None

        partial_upper = partial_sql.upper()

        # If we have a complete-looking query, return as-is
        if 'SELECT' in partial_upper and 'FROM' in partial_upper:
            # Add LIMIT 0 if not present
            if 'LIMIT' not in partial_upper:
                return partial_sql + " LIMIT 0"
            return partial_sql

        # Incomplete SELECT
        if partial_upper.startswith('SELECT') and 'FROM' not in partial_upper:
            # Get first table from schema
            if self.schema_info["all_tables"]:
                return partial_sql + f" FROM {self.schema_info['all_tables'][0]} LIMIT 0"

        return None

    def _build_execution_prompt(self, base_prompt: str, trace: SQLExecutionTrace) -> str:
        """Build prompt with execution trace feedback."""
        trace_text = f"""
[Execution Feedback]
Current SQL: {trace.partial_sql}
Valid: {trace.is_valid}
"""
        if trace.error_message:
            trace_text += f"Error: {trace.error_message}\n"
        if trace.explain_output:
            trace_text += f"Query Plan: {trace.explain_output[:200]}\n"
        if trace.tables_referenced:
            trace_text += f"Tables used: {', '.join(trace.tables_referenced)}\n"

        return base_prompt + trace_text

    def _cfg_interpolate(
        self,
        logits_uncond: torch.Tensor,
        logits_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Apply CFG interpolation: log P = log P_uncond + γ * (log P_cond - log P_uncond)."""
        # Convert to log probabilities
        log_probs_uncond = F.log_softmax(logits_uncond, dim=-1)
        log_probs_cond = F.log_softmax(logits_cond, dim=-1)

        # CFG formula
        log_probs_cfg = log_probs_uncond + self.gamma * (log_probs_cond - log_probs_uncond)

        return log_probs_cfg

    def _is_clause_boundary(self, token_text: str, current_sql: str) -> bool:
        """Check if we're at a clause boundary."""
        # Check if token starts a new clause
        token_upper = token_text.strip().upper()
        if token_upper in self.CLAUSE_KEYWORDS:
            return True

        # Check for newlines (common boundary)
        if '\n' in token_text:
            return True

        return False

    def generate(
        self,
        question: str,
        schema: str,
        evidence: str = "",
    ) -> tuple[str, dict]:
        """Generate SQL using EG-CFG."""
        # Build base (unconditional) prompt
        base_prompt = f"""Given the database schema below, write a SQL query to answer the question.

Database Schema:
{schema}

Question: {question}
{f'Hint: {evidence}' if evidence else ''}

SQL:"""

        # Format for model
        messages = [{"role": "user", "content": base_prompt}]
        try:
            formatted_uncond = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_uncond = base_prompt

        # Initialize generation
        current_sql = ""
        current_trace = None
        all_traces = []

        # Tokenize the prompt
        inputs = self.tokenizer(formatted_uncond, return_tensors="pt").to(self.device)
        generated_ids = inputs["input_ids"].clone()

        # Generate clause by clause
        for clause_idx in range(self.max_clauses):
            # Get execution trace for current partial SQL
            if current_sql.strip():
                current_trace = self._get_execution_trace(current_sql)
                all_traces.append(current_trace)

                # Build conditional prompt with execution feedback
                cond_prompt = self._build_execution_prompt(base_prompt, current_trace)
                messages_cond = [{"role": "user", "content": cond_prompt}]
                try:
                    formatted_cond = self.tokenizer.apply_chat_template(
                        messages_cond, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    formatted_cond = cond_prompt

                # Add current SQL to conditional prompt
                formatted_cond = formatted_cond + current_sql
            else:
                formatted_cond = formatted_uncond

            # Generate next clause (multiple tokens until boundary)
            clause_tokens = []
            clause_text = ""

            for _ in range(100):  # Max tokens per clause
                with torch.no_grad():
                    # Unconditional forward pass
                    uncond_input = self.tokenizer(
                        formatted_uncond + current_sql + clause_text,
                        return_tensors="pt"
                    ).to(self.device)
                    uncond_outputs = self.model(**uncond_input)
                    logits_uncond = uncond_outputs.logits[:, -1, :]

                    # Conditional forward pass (with execution trace)
                    if current_trace and current_trace.is_valid is not None:
                        cond_input = self.tokenizer(
                            formatted_cond + clause_text,
                            return_tensors="pt"
                        ).to(self.device)
                        cond_outputs = self.model(**cond_input)
                        logits_cond = cond_outputs.logits[:, -1, :]

                        # Apply CFG interpolation
                        log_probs = self._cfg_interpolate(logits_uncond, logits_cond)
                    else:
                        # No trace yet, use unconditional
                        log_probs = F.log_softmax(logits_uncond, dim=-1)

                    # Sample next token
                    if self.temperature > 0:
                        probs = torch.exp(log_probs / self.temperature)
                        probs = probs / probs.sum()
                        next_token = torch.multinomial(probs[0], num_samples=1)
                    else:
                        next_token = torch.argmax(log_probs, dim=-1)

                    # Decode token
                    token_text = self.tokenizer.decode(next_token[0])
                    clause_tokens.append(next_token.item())
                    clause_text += token_text

                    # Check for EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    # Check for clause boundary
                    if self._is_clause_boundary(token_text, current_sql + clause_text):
                        break

            # Add clause to current SQL
            current_sql += clause_text

            # Check if query is complete
            if self._is_sql_complete(current_sql):
                break

            # Check for EOS
            if clause_tokens and clause_tokens[-1] == self.tokenizer.eos_token_id:
                break

        # Clean up SQL
        final_sql = self._clean_sql(current_sql)

        metadata = {
            "raw_sql": current_sql,
            "num_clauses": len(all_traces),
            "traces": [
                {
                    "partial_sql": t.partial_sql,
                    "is_valid": t.is_valid,
                    "error": t.error_message,
                }
                for t in all_traces
            ],
            "gamma": self.gamma,
        }

        return final_sql, metadata

    def _is_sql_complete(self, sql: str) -> bool:
        """Check if SQL query is complete."""
        sql = sql.strip()
        sql_upper = sql.upper()

        if len(sql) < 15:
            return False

        if not ('SELECT' in sql_upper and 'FROM' in sql_upper):
            return False

        # Check for incomplete patterns
        incomplete = [
            r'\bFROM\s*$', r'\bWHERE\s*$', r'\bAND\s*$', r'\bOR\s*$',
            r'\bJOIN\s*$', r'\bON\s*$', r'\bBY\s*$', r'=\s*$', r',\s*$',
            r'\bSELECT\s*$', r'\bINNER\s*$', r'\bLEFT\s*$', r'\bGROUP\s*$',
        ]
        for pattern in incomplete:
            if re.search(pattern, sql_upper):
                return False

        # Check balanced parentheses
        if sql.count('(') != sql.count(')'):
            return False

        return True

    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL."""
        sql = sql.strip()

        # Remove markdown
        if "```" in sql:
            match = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1)
            else:
                sql = sql.replace("```sql", "").replace("```", "")

        # Take first statement
        if ";" in sql:
            sql = sql.split(";")[0]

        # Remove explanatory text
        lines = sql.split("\n")
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line.upper().startswith(("--", "/*", "NOTE:", "THIS", "THE ", "HERE")):
                continue
            if line:
                sql_lines.append(line)

        return " ".join(sql_lines).strip()


class EGCFGFast:
    """
    Faster EG-CFG variant using candidate reranking instead of token-by-token CFG.

    This is closer to what we can do efficiently:
    1. Generate multiple candidate SQLs
    2. Score each with execution trace quality
    3. Use CFG-style reranking
    """

    def __init__(
        self,
        model,
        tokenizer,
        db_path: str,
        num_candidates: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.db_path = db_path
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

        # Get schema info
        self.schema_info = self._get_schema_info()

    def _get_schema_info(self) -> dict:
        """Extract schema information from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        info = {"tables": {}, "all_tables": set(), "all_columns": set()}

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            info["all_tables"].add(table.lower())
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            info["tables"][table.lower()] = set(c.lower() for c in columns)
            for c in columns:
                info["all_columns"].add(c.lower())

        conn.close()
        return info

    def _get_execution_score(self, sql: str) -> tuple[float, dict]:
        """Get detailed execution-based score for SQL."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        score = 0.0
        details = {
            "syntax_valid": False,
            "tables_valid": False,
            "columns_valid": False,
            "executes": False,
            "has_results": False,
            "error": None,
        }

        # Check syntax with EXPLAIN
        try:
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            cursor.fetchall()
            details["syntax_valid"] = True
            score += 0.3
        except sqlite3.Error as e:
            details["error"] = str(e)
            conn.close()
            return score, details

        # Check table references
        tables_in_sql = set(re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql = {t.lower() for t in tables_in_sql}

        if tables_in_sql and tables_in_sql.issubset(self.schema_info["all_tables"]):
            details["tables_valid"] = True
            score += 0.2

        # Check column references (simplified)
        # This is imperfect but gives a signal
        columns_in_sql = set()
        for pattern in [r'SELECT\s+([\w,\s.]+)\s+FROM', r'WHERE\s+(\w+)', r'ON\s+\w+\.(\w+)']:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            for m in matches:
                if isinstance(m, str):
                    for col in m.split(','):
                        col = col.strip().split('.')[-1].strip()
                        if col.lower() not in {'*', 'and', 'or', 'as'}:
                            columns_in_sql.add(col.lower())

        # Check at least some columns exist
        if columns_in_sql:
            valid_cols = columns_in_sql.intersection(self.schema_info["all_columns"])
            if len(valid_cols) >= len(columns_in_sql) * 0.5:
                details["columns_valid"] = True
                score += 0.2

        # Try actual execution
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            details["executes"] = True
            score += 0.2

            if results:
                details["has_results"] = True
                score += 0.1
        except sqlite3.Error as e:
            details["error"] = str(e)

        conn.close()
        return score, details

    def generate(
        self,
        question: str,
        schema: str,
        evidence: str = "",
    ) -> tuple[str, dict]:
        """Generate SQL using fast EG-CFG variant."""
        prompt = f"""Given the database schema below, write a SQL query to answer the question.

Database Schema:
{schema}

Question: {question}
{f'Hint: {evidence}' if evidence else ''}

SQL:"""

        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        # Generate multiple candidates
        candidates = []

        with torch.no_grad():
            for i in range(self.num_candidates):
                # Vary temperature for diversity
                temp = self.temperature + (i * 0.1)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                generated = outputs[0, inputs["input_ids"].shape[1]:]
                sql = self.tokenizer.decode(generated, skip_special_tokens=True)
                sql = self._clean_sql(sql)

                # Score with execution
                exec_score, exec_details = self._get_execution_score(sql)

                candidates.append({
                    "sql": sql,
                    "exec_score": exec_score,
                    "exec_details": exec_details,
                    "temperature": temp,
                })

        # Rank by execution score
        candidates.sort(key=lambda x: x["exec_score"], reverse=True)
        best = candidates[0]

        metadata = {
            "num_candidates": len(candidates),
            "best_exec_score": best["exec_score"],
            "best_exec_details": best["exec_details"],
            "all_scores": [c["exec_score"] for c in candidates],
        }

        return best["sql"], metadata

    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL."""
        sql = sql.strip()

        if "```" in sql:
            match = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1)
            else:
                sql = sql.replace("```sql", "").replace("```", "")

        if ";" in sql:
            sql = sql.split(";")[0]

        lines = sql.split("\n")
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line.upper().startswith(("--", "/*", "NOTE:", "THIS", "THE ", "HERE")):
                continue
            if line:
                sql_lines.append(line)

        return " ".join(sql_lines).strip()
