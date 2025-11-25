"""M24: Task-Aligned SQL Generation (TA-SQL inspired).

Based on the paper "Before Generation, Align it!" (ACL Findings 2024).

Key insight: LLMs hallucinate less when approaching unfamiliar tasks from
familiar perspectives. For Text-to-SQL:

1. TASL (Task-Aligned Schema Linking): Instead of directly asking for schema
   linking (unfamiliar), ask the model to generate a "dummy SQL" (familiar),
   then extract schema entities from it.

2. TALOG (Task-Aligned Logical synthesis): Frame SQL generation as step-by-step
   data analysis using pandas-like operations (familiar), then convert to SQL.

Combined with our partial SQL validation, this should reduce hallucinations.
"""

import re
from dataclasses import dataclass

import torch


@dataclass
class TaskAlignedResult:
    """Result from task-aligned generation."""
    sql: str
    dummy_sql: str  # From TASL
    schema_entities: list[str]  # Extracted from dummy SQL
    reasoning_steps: list[str]  # From TALOG
    validation_score: float


class TaskAlignedGenerator:
    """
    M24: Task-Aligned SQL Generation.

    Combines TA-SQL's task alignment with our execution validation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 0.3,  # Lower for more deterministic
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    def generate(
        self,
        question: str,
        schema: str,
        evidence: str,
        db_path: str,
    ) -> tuple[str, dict]:
        """
        Generate SQL using task alignment strategy.

        Steps:
        1. TASL: Generate dummy SQL for schema linking
        2. Extract schema entities from dummy SQL
        3. TALOG: Generate step-by-step reasoning
        4. Generate final SQL based on reasoning
        5. Validate with partial SQL validator
        """
        from .cfg_sql import PartialSQLValidator

        validator = PartialSQLValidator(db_path)

        # Step 1: TASL - Generate dummy SQL for schema linking
        dummy_sql, schema_entities = self._tasl_schema_linking(
            question, schema, evidence
        )

        # Step 2: TALOG - Generate reasoning steps
        reasoning_steps = self._talog_reasoning(
            question, schema, evidence, schema_entities
        )

        # Step 3: Generate final SQL based on reasoning
        final_sql = self._generate_final_sql(
            question, schema, evidence, schema_entities, reasoning_steps
        )

        # Step 4: Validate
        validity_score = validator.get_validity_signal(final_sql)
        validator.close()

        metadata = {
            "dummy_sql": dummy_sql,
            "schema_entities": schema_entities,
            "reasoning_steps": reasoning_steps,
            "validity_score": validity_score,
        }

        return final_sql, metadata

    def _tasl_schema_linking(
        self,
        question: str,
        schema: str,
        evidence: str,
    ) -> tuple[str, list[str]]:
        """
        Task-Aligned Schema Linking.

        Instead of asking "which tables/columns are relevant?", we ask
        "write a rough SQL query" - leveraging the model's SQL familiarity.
        Then we extract schema entities from the generated SQL.
        """
        prompt = f"""Given the database schema and question, write a rough SQL query.
Don't worry about exact syntax - just use the tables and columns you think are needed.

Database Schema:
{schema}

Question: {question}
{f'Hint: {evidence}' if evidence else ''}

Rough SQL query (use relevant tables and columns):"""

        dummy_sql = self._generate_text(prompt)

        # Extract schema entities from the dummy SQL
        schema_entities = self._extract_schema_entities(dummy_sql)

        return dummy_sql, schema_entities

    def _extract_schema_entities(self, sql: str) -> list[str]:
        """Extract table and column names from SQL."""
        # Simple extraction - get words after FROM, JOIN, SELECT, WHERE, etc.
        entities = []

        # Remove SQL keywords and extract identifiers
        sql_upper = sql.upper()

        # Find table names after FROM and JOIN
        from_matches = re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
        join_matches = re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE)
        entities.extend(from_matches)
        entities.extend(join_matches)

        # Find all identifiers (potential columns)
        all_identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', sql)

        # Filter out SQL keywords
        sql_keywords = {
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'outer',
            'on', 'and', 'or', 'not', 'in', 'like', 'between', 'is', 'null',
            'group', 'by', 'order', 'asc', 'desc', 'limit', 'as', 'distinct',
            'count', 'sum', 'avg', 'max', 'min', 'having', 'case', 'when',
            'then', 'else', 'end', 'cast', 'substr', 'real', 'float', 'integer',
            'text', 'iif',
        }

        for ident in all_identifiers:
            if ident.lower() not in sql_keywords:
                entities.append(ident)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            e_lower = e.lower()
            if e_lower not in seen:
                seen.add(e_lower)
                unique_entities.append(e)

        return unique_entities

    def _talog_reasoning(
        self,
        question: str,
        schema: str,
        evidence: str,
        schema_entities: list[str],
    ) -> list[str]:
        """
        Task-Aligned Logical Synthesis.

        Frame SQL generation as step-by-step data analysis.
        This is more familiar to the model than direct SQL synthesis.
        """
        entities_str = ", ".join(schema_entities[:10])  # Limit to top 10

        prompt = f"""Analyze this question step by step, as if you were a data analyst.

Relevant tables/columns: {entities_str}

Question: {question}
{f'Hint: {evidence}' if evidence else ''}

Think through the data analysis steps needed:
1. What data do I need to access?
2. Do I need to join tables? Which ones and how?
3. What filtering/conditions are needed?
4. What aggregation or calculation is required?
5. How should results be sorted or limited?

Analysis steps:"""

        reasoning_text = self._generate_text(prompt)

        # Parse steps from the response
        steps = []
        for line in reasoning_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                steps.append(line)

        return steps if steps else [reasoning_text]

    def _generate_final_sql(
        self,
        question: str,
        schema: str,
        evidence: str,
        schema_entities: list[str],
        reasoning_steps: list[str],
    ) -> str:
        """Generate final SQL based on reasoning steps."""
        entities_str = ", ".join(schema_entities[:10])
        steps_str = "\n".join(reasoning_steps[:5])

        prompt = f"""Based on the analysis below, write the final SQL query.

Database Schema:
{schema}

Question: {question}
{f'Hint: {evidence}' if evidence else ''}

Relevant entities: {entities_str}

Analysis:
{steps_str}

Now write the SQL query. Use only valid table and column names from the schema.

SQL:"""

        sql = self._generate_text(prompt)
        return self._clean_sql(sql)

    def _generate_text(self, prompt: str) -> str:
        """Generate text from prompt."""
        messages = [{"role": "user", "content": prompt}]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 0.01),
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0, prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL."""
        sql = sql.strip()

        # Remove markdown code blocks
        if "```" in sql:
            match = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1)
            else:
                sql = sql.replace("```sql", "").replace("```", "")

        # Take first statement
        if ";" in sql:
            sql = sql.split(";")[0]

        # Remove explanation text
        lines = sql.split("\n")
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line.upper().startswith(("--", "/*", "NOTE:", "THIS", "THE ", "HERE")):
                break
            if line:
                sql_lines.append(line)

        return " ".join(sql_lines).strip()


class TaskAlignedWithValidation(TaskAlignedGenerator):
    """
    M24 Extended: Task-Aligned + Execution Validation.

    Combines:
    1. TA-SQL's task alignment (TASL + TALOG)
    2. Our partial SQL validation
    3. Multi-candidate generation with reranking
    """

    def generate_with_validation(
        self,
        question: str,
        schema: str,
        evidence: str,
        db_path: str,
        num_candidates: int = 3,
    ) -> tuple[str, dict]:
        """Generate with validation and candidate selection."""
        from .cfg_sql import PartialSQLValidator

        validator = PartialSQLValidator(db_path)

        # Generate TASL schema linking once
        dummy_sql, schema_entities = self._tasl_schema_linking(
            question, schema, evidence
        )

        # Generate multiple candidates with different reasoning
        candidates = []

        for i in range(num_candidates):
            # Vary temperature for diversity
            self.temperature = 0.3 + (i * 0.1)

            # Generate reasoning and SQL
            reasoning_steps = self._talog_reasoning(
                question, schema, evidence, schema_entities
            )

            final_sql = self._generate_final_sql(
                question, schema, evidence, schema_entities, reasoning_steps
            )

            # Validate
            validity_score = validator.get_validity_signal(final_sql)

            candidates.append({
                "sql": final_sql,
                "reasoning": reasoning_steps,
                "validity": validity_score,
                "score": validity_score + (1.0 if self._is_complete(final_sql) else 0),
            })

        validator.close()

        # Reset temperature
        self.temperature = 0.3

        # Select best candidate
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        metadata = {
            "dummy_sql": dummy_sql,
            "schema_entities": schema_entities,
            "reasoning_steps": best["reasoning"],
            "validity_score": best["validity"],
            "num_candidates": len(candidates),
            "all_validities": [c["validity"] for c in candidates],
        }

        return best["sql"], metadata

    def _is_complete(self, sql: str) -> bool:
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
