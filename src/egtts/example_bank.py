"""M19: Example-Guided Generation with Dynamic Few-Shot Selection.

This module provides a bank of SQL pattern examples and similarity-based
retrieval to provide the most relevant few-shot examples for each question.

Key insight: The model struggles with superlative queries because it doesn't
see enough examples of the GROUP BY + ORDER BY + LIMIT 1 pattern. By
dynamically selecting examples that match the question's SQL pattern needs,
we can dramatically improve accuracy on hard cases.

Pattern categories:
1. Superlative (highest/lowest/most/least) - GROUP BY X ORDER BY SUM(Y) LIMIT 1
2. Ratio/Difference - Conditional aggregation with CASE WHEN
3. Date extraction - SUBSTR patterns for YYYYMMDD strings
4. Multi-table joins - Complex JOIN paths
5. Simple aggregates - COUNT, SUM, AVG without grouping
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SQLExample:
    """A SQL example with metadata for similarity matching."""
    pattern_type: str  # superlative, ratio, date, join, aggregate
    question: str
    hint: str
    sql: str
    # Pattern characteristics for matching
    has_group_by: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    has_join: bool = False
    has_case_when: bool = False
    has_substr: bool = False
    aggregation_type: str = ""  # SUM, COUNT, AVG, MAX, MIN

    def similarity_score(self, question: str, detected_patterns: list[str]) -> float:
        """Calculate similarity score for a given question."""
        score = 0.0
        q_lower = question.lower()

        # Pattern type matching (strongest signal)
        for pattern in detected_patterns:
            if pattern.startswith('superlative:') and self.pattern_type == 'superlative':
                score += 3.0
            elif pattern.startswith('comparison:') and self.pattern_type == 'ratio':
                score += 3.0
            elif pattern.startswith('rate_change:') and self.pattern_type == 'ratio':
                score += 2.5

        # Keyword matching
        superlative_words = ['highest', 'lowest', 'most', 'least', 'peak', 'maximum', 'minimum']
        if any(w in q_lower for w in superlative_words) and self.has_order_by and self.has_limit:
            score += 2.0

        # Date-related matching
        date_words = ['year', 'month', 'date', '2012', '2013', '2014']
        if any(w in q_lower for w in date_words) and self.has_substr:
            score += 1.5

        # Ratio/difference matching
        if ('ratio' in q_lower or 'difference' in q_lower) and self.has_case_when:
            score += 2.0

        # Aggregation matching
        if 'total' in q_lower or 'sum' in q_lower:
            if self.aggregation_type == 'SUM':
                score += 1.0
        if 'how many' in q_lower or 'count' in q_lower:
            if self.aggregation_type == 'COUNT':
                score += 1.0
        if 'average' in q_lower:
            if self.aggregation_type == 'AVG':
                score += 1.0

        return score


# =============================================================================
# EXAMPLE BANK: High-quality examples covering key SQL patterns
# =============================================================================

EXAMPLE_BANK = [
    # =========================================================================
    # SUPERLATIVE PATTERNS - The hardest category (42% accuracy without help)
    # Key: GROUP BY entity, ORDER BY aggregate, LIMIT 1
    # =========================================================================

    SQLExample(
        pattern_type="superlative",
        question="Which customer had the highest total consumption in 2012?",
        hint="Year 2012 can be presented as Between 201201 And 201212.",
        sql="SELECT CustomerID FROM yearmonth WHERE SUBSTR(Date, 1, 4) = '2012' GROUP BY CustomerID ORDER BY SUM(Consumption) DESC LIMIT 1",
        has_group_by=True,
        has_order_by=True,
        has_limit=True,
        has_substr=True,
        aggregation_type="SUM",
    ),

    SQLExample(
        pattern_type="superlative",
        question="Which year recorded the most consumption of gas paid in CZK?",
        hint="The first 4 strings of the Date values in the yearmonth table can represent year.",
        sql="SELECT SUBSTR(T2.Date, 1, 4) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Currency = 'CZK' GROUP BY SUBSTR(T2.Date, 1, 4) ORDER BY SUM(T2.Consumption) DESC LIMIT 1",
        has_group_by=True,
        has_order_by=True,
        has_limit=True,
        has_join=True,
        has_substr=True,
        aggregation_type="SUM",
    ),

    SQLExample(
        pattern_type="superlative",
        question="In 2012, who had the least consumption in LAM?",
        hint="Year 2012 can be presented as Between 201201 And 201212; The first 4 strings of the Date values in the yearmonth table can represent year.",
        sql="SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = 'LAM' AND SUBSTR(T2.Date, 1, 4) = '2012' GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) ASC LIMIT 1",
        has_group_by=True,
        has_order_by=True,
        has_limit=True,
        has_join=True,
        has_substr=True,
        aggregation_type="SUM",
    ),

    SQLExample(
        pattern_type="superlative",
        question="What was the gas consumption peak month for SME customers in 2013?",
        hint="Year 2013 can be presented as Between 201301 And 201312; The 5th and 6th string of the date can refer to month.",
        sql="SELECT SUBSTR(T2.Date, 5, 2) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE SUBSTR(T2.Date, 1, 4) = '2013' AND T1.Segment = 'SME' GROUP BY SUBSTR(T2.Date, 5, 2) ORDER BY SUM(T2.Consumption) DESC LIMIT 1",
        has_group_by=True,
        has_order_by=True,
        has_limit=True,
        has_join=True,
        has_substr=True,
        aggregation_type="SUM",
    ),

    # =========================================================================
    # RATIO / DIFFERENCE PATTERNS - Conditional aggregation
    # Key: CASE WHEN or IIF for conditional counting/summing
    # =========================================================================

    SQLExample(
        pattern_type="ratio",
        question="What is the ratio of customers who pay in EUR against customers who pay in CZK?",
        hint="ratio of customers who pay in EUR against customers who pay in CZK = count(Currency = 'EUR') / count(Currency = 'CZK').",
        sql="SELECT CAST(SUM(CASE WHEN Currency = 'EUR' THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN Currency = 'CZK' THEN 1 ELSE 0 END) FROM customers",
        has_case_when=True,
        aggregation_type="SUM",
    ),

    SQLExample(
        pattern_type="ratio",
        question="What was the difference in gas consumption between CZK-paying customers and EUR-paying customers in 2012?",
        hint="Difference in Consumption = CZK customers consumption in 2012 - EUR customers consumption in 2012",
        sql="SELECT SUM(CASE WHEN T1.Currency = 'CZK' THEN T2.Consumption ELSE 0 END) - SUM(CASE WHEN T1.Currency = 'EUR' THEN T2.Consumption ELSE 0 END) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE SUBSTR(T2.Date, 1, 4) = '2012'",
        has_case_when=True,
        has_join=True,
        has_substr=True,
        aggregation_type="SUM",
    ),

    SQLExample(
        pattern_type="ratio",
        question="How many more discount gas stations does the Czech Republic have compared to Slovakia?",
        hint="Czech Republic = 'CZE'; Slovakia = 'SVK'",
        sql="SELECT SUM(CASE WHEN Country = 'CZE' THEN 1 ELSE 0 END) - SUM(CASE WHEN Country = 'SVK' THEN 1 ELSE 0 END) FROM gasstations WHERE Segment = 'Discount'",
        has_case_when=True,
        aggregation_type="SUM",
    ),

    # =========================================================================
    # DATE EXTRACTION PATTERNS - SUBSTR for YYYYMMDD strings
    # =========================================================================

    SQLExample(
        pattern_type="date",
        question="What is the total consumption for each month in 2013?",
        hint="Year 2013 can be presented as Between 201301 And 201312; The 5th and 6th string of the date can refer to month.",
        sql="SELECT SUBSTR(Date, 5, 2) AS Month, SUM(Consumption) FROM yearmonth WHERE SUBSTR(Date, 1, 4) = '2013' GROUP BY SUBSTR(Date, 5, 2)",
        has_group_by=True,
        has_substr=True,
        aggregation_type="SUM",
    ),

    SQLExample(
        pattern_type="date",
        question="How much did customer 6 consume in total between August and November 2013?",
        hint="Between August And November 2013 refers to Between 201308 And 201311.",
        sql="SELECT SUM(Consumption) FROM yearmonth WHERE CustomerID = 6 AND Date BETWEEN '201308' AND '201311'",
        aggregation_type="SUM",
    ),

    # =========================================================================
    # SIMPLE AGGREGATION PATTERNS - Good baseline examples
    # =========================================================================

    SQLExample(
        pattern_type="aggregate",
        question="What was the average monthly consumption of customers in SME for the year 2013?",
        hint="Average Monthly consumption = AVG(Consumption) / 12",
        sql="SELECT AVG(T2.Consumption) / 12 FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE SUBSTR(T2.Date, 1, 4) = '2013' AND T1.Segment = 'SME'",
        has_join=True,
        has_substr=True,
        aggregation_type="AVG",
    ),

    SQLExample(
        pattern_type="aggregate",
        question="How many customers in the SME segment paid in CZK?",
        hint="SME is a segment value; CZK is a currency value.",
        sql="SELECT COUNT(*) FROM customers WHERE Segment = 'SME' AND Currency = 'CZK'",
        aggregation_type="COUNT",
    ),
]


def detect_question_patterns(question: str, hint: str = "") -> list[str]:
    """
    Detect SQL patterns needed for a question.

    Returns list of pattern identifiers like:
    - superlative:highest
    - comparison:difference
    - date:year
    """
    patterns = []
    q_lower = question.lower()
    hint_lower = (hint or "").lower()
    combined = q_lower + " " + hint_lower

    # Superlative patterns
    superlative_words = [
        'highest', 'lowest', 'most', 'least', 'peak', 'maximum', 'minimum',
        'top', 'bottom', 'largest', 'smallest', 'greatest', 'best', 'worst',
    ]
    for word in superlative_words:
        if word in q_lower:
            patterns.append(f"superlative:{word}")

    # Comparison/ratio patterns
    if 'ratio' in q_lower:
        patterns.append("comparison:ratio")
    if 'difference' in q_lower or 'more than' in q_lower or 'less than' in q_lower:
        patterns.append("comparison:difference")
    if 'compare' in q_lower or 'versus' in q_lower:
        patterns.append("comparison:compare")

    # Date patterns
    if any(year in combined for year in ['2012', '2013', '2014']):
        patterns.append("date:year")
    if 'month' in combined:
        patterns.append("date:month")

    # Aggregation patterns
    if 'total' in q_lower or 'sum' in q_lower:
        patterns.append("aggregate:sum")
    if 'how many' in q_lower or 'count' in q_lower:
        patterns.append("aggregate:count")
    if 'average' in q_lower:
        patterns.append("aggregate:avg")

    return patterns


def select_examples(
    question: str,
    hint: str = "",
    n_examples: int = 3,
    exclude_similar: bool = True,
) -> list[SQLExample]:
    """
    Select the most relevant examples from the bank for a given question.

    Uses pattern matching and similarity scoring to find examples that
    demonstrate the SQL patterns most likely needed for this question.

    Args:
        question: Natural language question
        hint: Optional hint/evidence
        n_examples: Number of examples to return
        exclude_similar: If True, avoid examples too similar to the question

    Returns:
        List of SQLExample objects, sorted by relevance
    """
    detected_patterns = detect_question_patterns(question, hint)

    # Score all examples
    scored_examples = []
    for example in EXAMPLE_BANK:
        # Skip if too similar to the question (would be data leakage)
        if exclude_similar:
            q_words = set(question.lower().split())
            ex_words = set(example.question.lower().split())
            overlap = len(q_words & ex_words) / len(q_words) if q_words else 0
            if overlap > 0.7:  # More than 70% word overlap = too similar
                continue

        score = example.similarity_score(question, detected_patterns)
        if score > 0:
            scored_examples.append((score, example))

    # Sort by score descending and take top n
    scored_examples.sort(key=lambda x: x[0], reverse=True)

    # If we don't have enough scored examples, add some defaults
    selected = [ex for _, ex in scored_examples[:n_examples]]

    # Ensure diversity: try to include different pattern types
    if len(selected) < n_examples:
        pattern_types_included = {ex.pattern_type for ex in selected}
        for example in EXAMPLE_BANK:
            if example not in selected and example.pattern_type not in pattern_types_included:
                selected.append(example)
                pattern_types_included.add(example.pattern_type)
                if len(selected) >= n_examples:
                    break

    # Still need more? Just add from the bank
    if len(selected) < n_examples:
        for example in EXAMPLE_BANK:
            if example not in selected:
                selected.append(example)
                if len(selected) >= n_examples:
                    break

    return selected[:n_examples]


def format_example_for_prompt(example: SQLExample) -> str:
    """Format an example for inclusion in a prompt."""
    parts = [f"Question: {example.question}"]
    if example.hint:
        parts.append(f"Hint: {example.hint}")
    parts.append(f"SQL: {example.sql}")
    return "\n".join(parts)


# =============================================================================
# M19 PROMPT TEMPLATE
# =============================================================================

M19_SYSTEM_PROMPT = """You are a SQL expert. Generate SQLite queries based on the database schema and question.

CRITICAL SQLite patterns for this database:
- Dates are stored as strings in YYYYMMDD format (e.g., '201309')
- Use SUBSTR(Date, 1, 4) to extract year, SUBSTR(Date, 5, 2) for month
- For "which X has most/least Y": GROUP BY X, ORDER BY SUM(Y) DESC/ASC, LIMIT 1
  * Do NOT use MAX(SUM(...)) - that's invalid SQL
  * GROUP first, then ORDER BY the aggregate, then LIMIT 1
- For ratios/differences: use CASE WHEN inside SUM() or COUNT()
- Return ONLY the requested columns, no extra columns

Study the examples below carefully - they show the exact patterns you need."""


def format_m19_prompt(
    question: str,
    schema: str,
    tokenizer=None,
    hint: str = "",
    n_examples: int = 3,
) -> str:
    """
    Create an M19 example-guided prompt with dynamically selected few-shot examples.

    Args:
        question: Natural language question
        schema: Database schema (augmented with sample rows)
        tokenizer: Tokenizer for chat template
        hint: Optional hint/evidence
        n_examples: Number of few-shot examples to include

    Returns:
        Formatted prompt string
    """
    # Select relevant examples
    examples = select_examples(question, hint, n_examples)

    # Build messages
    messages = []

    # System prompt
    messages.append({"role": "system", "content": M19_SYSTEM_PROMPT})

    # Add selected examples as few-shot turns
    for example in examples:
        # User turn
        user_content = f"Question: {example.question}"
        if example.hint:
            user_content += f"\nHint: {example.hint}"
        messages.append({"role": "user", "content": user_content})

        # Assistant turn (just the SQL)
        messages.append({"role": "assistant", "content": example.sql})

    # Final question with schema
    final_user = f"""Database schema and sample data:

{schema}

Question: {question}"""
    if hint:
        final_user += f"\nHint: {hint}"

    messages.append({"role": "user", "content": final_user})

    # Apply chat template
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return str(messages)


def get_pattern_explanation(question: str, hint: str = "") -> str:
    """
    Generate a brief explanation of what SQL pattern is likely needed.

    This can be prepended to the prompt to prime the model.
    """
    patterns = detect_question_patterns(question, hint)

    explanations = []

    if any('superlative' in p for p in patterns):
        explanations.append(
            "This is a superlative question (highest/lowest/most/least). "
            "Use: GROUP BY [entity], ORDER BY SUM/COUNT([measure]) DESC/ASC, LIMIT 1"
        )

    if any('comparison' in p for p in patterns):
        explanations.append(
            "This involves comparing values. "
            "Use CASE WHEN inside SUM() to compute each side, then subtract."
        )

    if any('date' in p for p in patterns):
        explanations.append(
            "Dates are YYYYMMDD strings. "
            "Use SUBSTR(Date, 1, 4) for year, SUBSTR(Date, 5, 2) for month."
        )

    return " ".join(explanations) if explanations else ""
