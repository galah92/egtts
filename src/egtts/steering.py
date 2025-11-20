"""
Execution-Guided SQL Generation with Clause-Aware Beam Search.

Unlike post-hoc re-ranking (M3/M4/M5), this implements TRUE execution guidance
by validating partial SQL at clause boundaries and pruning invalid beams early.

Key Innovation: Speculative Closure
- Partial SQL like "SELECT * FROM table WHERE" is invalid
- We append dummy suffixes (e.g., "1=1") to make it EXPLAIN-able
- Invalid beams get pruned, valid beams expand to fill beam width
- This allows the model to explore more valid logic paths
"""

import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .database import explain_query
from .model import create_sql_prompt
from .schema import build_schema_index, SchemaIndex


@dataclass
class BeamState:
    """State of a single beam during generation."""
    tokens: List[int]  # Token IDs generated so far
    score: float  # Cumulative log probability
    text: str  # Decoded SQL text
    alive: bool  # Whether beam is still active


# Constants
DEFAULT_CHECKPOINT_INTERVAL = 20  # Validate every N tokens
MIN_SQL_LENGTH = 15  # Minimum length to validate ("SELECT * FROM t")

# SQL clause keywords that trigger validation checkpoints
CLAUSE_KEYWORDS = [
    " FROM",
    " WHERE",
    " GROUP BY",
    " HAVING",
    " ORDER BY",
    " LIMIT",
    " JOIN",
    " LEFT JOIN",
    " RIGHT JOIN",
    " INNER JOIN",
    " OUTER JOIN",
    " ON",
    " UNION",
    " INTERSECT",
    " EXCEPT",
]


def detect_clause_boundary(text: str) -> bool:
    """
    Check if text ends at a SQL clause boundary.

    Returns True if the text ends with a clause keyword (case-insensitive).
    """
    text_upper = text.upper().rstrip()
    for keyword in CLAUSE_KEYWORDS:
        if text_upper.endswith(keyword.strip()):
            return True
    return False


def speculative_closure(partial_sql: str, schema_index: SchemaIndex) -> Optional[str]:
    """
    Apply speculative closure to make partial SQL valid for EXPLAIN.

    Strategy:
    1. Count open parentheses to detect unclosed subqueries
    2. Append appropriate dummy clauses based on SQL state
    3. Close any open parentheses

    Args:
        partial_sql: Partial SQL query
        schema_index: Schema information for picking dummy table

    Returns:
        Closed SQL that can be validated, or None if cannot close
    """
    sql = partial_sql.strip()
    sql_upper = sql.upper()

    # Get a valid table name from schema for dummy completions
    dummy_table = list(schema_index.tables)[0] if schema_index.tables else "sqlite_master"

    # SCENARIO 1: We triggered on " FROM".
    # Text is "SELECT * FROM". We need a table.
    if sql_upper.endswith("FROM"):
        return sql + f" {dummy_table} LIMIT 1"

    # SCENARIO 2: We triggered on " WHERE".
    # Text is "SELECT * FROM table WHERE".
    # Add dummy condition to make it valid
    if sql_upper.endswith("WHERE"):
        return sql + " 1=1 LIMIT 1"

    # SCENARIO 3: Triggered on " GROUP BY" or " ORDER BY"
    if sql_upper.endswith("GROUP BY") or sql_upper.endswith("ORDER BY"):
        # We need a column
        if schema_index.columns:
            # Get first column from any table
            first_table = list(schema_index.columns.keys())[0]
            first_col = list(schema_index.columns[first_table])[0]
            return sql + f" {first_col} LIMIT 1"
        else:
            return sql + " 1 LIMIT 1"

    # SCENARIO 4: Triggered on JOIN keywords - need condition
    if sql_upper.endswith("ON"):
        return sql + " 1=1 LIMIT 1"

    # SCENARIO 5: EOS or other - standard closure
    # Count open parentheses
    open_parens = sql.count('(') - sql.count(')')

    # Close any open parentheses
    if open_parens > 0:
        sql += ')' * open_parens

    # Add LIMIT if not present
    if "LIMIT" not in sql_upper:
        sql += " LIMIT 1"

    return sql


def validate_partial_sql(
    partial_sql: str,
    schema_index: SchemaIndex,
    db_path: str
) -> Tuple[bool, str]:
    """
    Validate partial SQL using speculative closure.

    Args:
        partial_sql: Partial SQL to validate
        schema_index: Schema information
        db_path: Database path

    Returns:
        (is_valid, error_message)
    """
    # Apply speculative closure
    closed_sql = speculative_closure(partial_sql, schema_index)

    if closed_sql is None:
        return False, "Cannot close partial SQL"

    # Try EXPLAIN on closed SQL
    result = explain_query(closed_sql, db_path)

    if isinstance(result, Exception):
        return False, str(result)

    return True, ""


class ClauseAwareBeamSearch:
    """
    Clause-aware beam search with execution guidance.

    Generates SQL step-by-step, validating at clause boundaries.
    Invalid beams are pruned early, allowing valid beams to expand.
    """

    def __init__(self, model, tokenizer, num_beams: int = 5):
        """
        Initialize beam search.

        Args:
            model: Language model
            tokenizer: Tokenizer
            num_beams: Number of beams (default: 5)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_beams = num_beams
        self.device = model.device

        # Keywords that imply the PREVIOUS clause is finished
        self.trigger_keywords = [
            " FROM", " WHERE", " GROUP BY", " ORDER BY", " LIMIT",
            " JOIN", " LEFT JOIN", " RIGHT JOIN", " INNER JOIN",
            " ON", " UNION", " INTERSECT", " EXCEPT", ";"
        ]

    def generate(
        self,
        question: str,
        schema: str,
        db_path: str,
        max_tokens: int = 256,
        checkpoint_interval: int = None  # Deprecated, ignored
    ) -> Tuple[str, dict]:
        """
        Generate SQL with clause-aware beam search.

        Args:
            question: Natural language question
            schema: Database schema
            db_path: Database path
            max_tokens: Maximum tokens to generate
            checkpoint_interval: Deprecated - validation now triggers on keywords

        Returns:
            (best_sql, metadata)
        """
        start_time = time.perf_counter()
        schema_index = build_schema_index(db_path)

        prompt = create_sql_prompt(question, schema, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Expand input for beams
        input_ids = input_ids.repeat(self.num_beams, 1)

        beams = [
            BeamState(tokens=input_ids[i].tolist(), score=0.0, text="", alive=True)
            for i in range(self.num_beams)
        ]

        checkpoints = []
        pruned_count = 0
        tokens_generated = 0

        while tokens_generated < max_tokens:
            # 1. Prepare Batch
            alive_indices = [i for i, b in enumerate(beams) if b.alive]
            if not alive_indices:
                break

            current_input_ids = [beams[i].tokens for i in alive_indices]

            # Pad
            max_len = max(len(t) for t in current_input_ids)
            batch_input = []
            attention_mask = []

            for seq in current_input_ids:
                padding = [self.tokenizer.pad_token_id] * (max_len - len(seq))
                batch_input.append(seq + padding)
                attention_mask.append([1] * len(seq) + [0] * len(padding))

            input_tensor = torch.tensor(batch_input, device=self.device)
            mask_tensor = torch.tensor(attention_mask, device=self.device)

            # 2. Model Forward (One Pass)
            with torch.no_grad():
                outputs = self.model(input_ids=input_tensor, attention_mask=mask_tensor, use_cache=False)
                next_token_logits = outputs.logits[:, -1, :]

            # 3. Top-K Selection
            next_log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # Expand candidates
            candidates = []
            for i, beam_idx in enumerate(alive_indices):
                # Get top beams*2 to ensure we have fallbacks if some are pruned
                top_probs, top_ids = torch.topk(next_log_probs[i], self.num_beams * 2)

                for step in range(len(top_ids)):
                    token_id = top_ids[step].item()
                    score = top_probs[step].item()
                    beam = beams[beam_idx]

                    # Decode new token
                    new_text = beam.text + self.tokenizer.decode([token_id], skip_special_tokens=True)

                    candidates.append({
                        "parent_idx": beam_idx,
                        "token": token_id,
                        "score": beam.score + score,
                        "text": new_text,
                        "is_eos": token_id == self.tokenizer.eos_token_id
                    })

            # Sort candidates globally
            candidates.sort(key=lambda x: x["score"], reverse=True)

            # 4. Select Top Beams & Check Triggers
            new_beams = []
            kept_count = 0

            # We verify candidates in order until we fill num_beams
            for cand in candidates:
                if kept_count >= self.num_beams:
                    break

                # TRIGGER CHECK: Did we just finish a keyword?
                should_validate = False
                # Check if we hit EOS
                if cand["is_eos"]:
                    should_validate = True
                else:
                    # Check if we just typed a keyword like " FROM"
                    upper_text = cand["text"].upper()
                    for kw in self.trigger_keywords:
                        if upper_text.endswith(kw):
                            should_validate = True
                            break

                is_valid = True  # Default
                if should_validate:
                    # Validate the query UP TO this point
                    is_valid, _ = validate_partial_sql(cand["text"], schema_index, db_path)

                    # RECORD CHECKPOINT
                    checkpoints.append({
                        "sql": cand["text"],
                        "trigger": "EOS" if cand["is_eos"] else "KEYWORD",
                        "valid": is_valid
                    })

                    if not is_valid:
                        pruned_count += 1

                if is_valid:
                    new_beams.append(BeamState(
                        tokens=beams[cand["parent_idx"]].tokens + [cand["token"]],
                        score=cand["score"],
                        text=cand["text"],
                        alive=not cand["is_eos"]
                    ))
                    kept_count += 1
                # If invalid, we simply SKIP adding it to new_beams (Effective Pruning)
                # The loop continues to the next best candidate

            if not new_beams:
                # All candidates were invalid - fallback to best candidate anyway
                best_cand = candidates[0]
                new_beams.append(BeamState(
                    tokens=beams[best_cand["parent_idx"]].tokens + [best_cand["token"]],
                    score=best_cand["score"],
                    text=best_cand["text"],
                    alive=not best_cand["is_eos"]
                ))

            beams = new_beams
            tokens_generated += 1

            # Global Stop
            if all(not b.alive for b in beams):
                break

        # Select best
        if not beams:
            return "SELECT 1", {"checkpoints": [], "pruned_beams": 0}

        best_beam = beams[0]  # Already sorted

        # Clean up SQL - join multi-line into single line
        sql = best_beam.text.strip()
        if "```sql" in sql.lower():
            sql = sql.split("```sql")[-1].split("```")[0].strip()
        elif "```" in sql:
            parts = sql.split("```")
            sql = parts[1].strip() if len(parts) > 1 else sql

        # Join multi-line SQL into single line for evaluation
        sql = " ".join(line.strip() for line in sql.split("\n") if line.strip())

        total_time = (time.perf_counter() - start_time) * 1000

        metadata = {
            "tokens_generated": tokens_generated,
            "checkpoints": len(checkpoints),
            "pruned_beams": pruned_count,
            "final_score": best_beam.score,
            "latency_ms": total_time,
            "checkpoint_history": checkpoints
        }

        return sql, metadata



def generate_with_steering(
    model,
    tokenizer,
    question: str,
    schema: str,
    db_path: str,
    num_beams: int = 5
) -> Tuple[str, dict]:
    """
    Generate SQL with clause-aware steering.

    Args:
        model: Language model
        tokenizer: Tokenizer
        question: Natural language question
        schema: Database schema
        db_path: Database path
        num_beams: Number of beams

    Returns:
        (sql, metadata)
    """
    search = ClauseAwareBeamSearch(model, tokenizer, num_beams)
    return search.generate(question, schema, db_path)
