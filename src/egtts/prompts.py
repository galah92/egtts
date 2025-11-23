"""
Few-Shot Prompting for BIRD Benchmark
======================================
Domain-specific examples that teach critical SQLite patterns:
1. SUBSTR for date extraction (BIRD uses YYYYMMDD strings)
2. GROUP BY with aggregation and ORDER BY
3. Complex JOIN with conditional aggregation
"""

# High-quality few-shot examples from BIRD patterns
# These demonstrate critical patterns the model struggles with

BIRD_FEW_SHOT_EXAMPLES = [
    # Example 1: Date extraction with SUBSTR
    # Teaches: SUBSTR(Date, 1, 4) for year, SUBSTR(Date, 5, 2) for month
    {
        "question": "What is the total consumption for each month in 2013?",
        "hint": "Year 2013 can be presented as Between 201301 And 201312; The first 4 strings of the Date values can represent year; The 5th and 6th string of the date can refer to month.",
        "sql": "SELECT SUBSTR(Date, 5, 2) AS Month, SUM(Consumption) AS TotalConsumption FROM yearmonth WHERE SUBSTR(Date, 1, 4) = '2013' GROUP BY SUBSTR(Date, 5, 2) ORDER BY Month"
    },
    # Example 2: Aggregation with GROUP BY and proper ordering
    # Teaches: GROUP BY entity, ORDER BY SUM() for "which X has most/least Y"
    {
        "question": "Which customer had the highest total consumption in 2012?",
        "hint": "Year 2012 can be presented as Between 201201 And 201212.",
        "sql": "SELECT CustomerID FROM yearmonth WHERE SUBSTR(Date, 1, 4) = '2012' GROUP BY CustomerID ORDER BY SUM(Consumption) DESC LIMIT 1"
    },
    # Example 3: JOIN with conditional counting
    # Teaches: Counting with conditions across joined tables
    {
        "question": "How many customers in the SME segment paid in CZK?",
        "hint": "SME is a segment value; CZK is a currency value.",
        "sql": "SELECT COUNT(*) FROM customers WHERE Segment = 'SME' AND Currency = 'CZK'"
    },
]


def format_few_shot_prompt(
    question: str,
    schema: str,
    tokenizer=None,
    hint: str = None,
    use_few_shot: bool = True,
) -> str:
    """
    Create a few-shot prompt for SQL generation.

    Args:
        question: Natural language question
        schema: Database schema (CREATE TABLE statements)
        tokenizer: Tokenizer for chat template
        hint: Optional hint/evidence
        use_few_shot: Whether to include few-shot examples

    Returns:
        Formatted prompt string
    """
    messages = []

    # System message with instructions
    system_content = """You are a SQL expert. Generate SQLite queries based on the database schema and question.

IMPORTANT SQLite patterns for this database:
- Dates are stored as strings in YYYYMMDD format (e.g., '201309')
- Use SUBSTR(Date, 1, 4) to extract year, SUBSTR(Date, 5, 2) for month
- For "which X has most/least Y": use GROUP BY X ORDER BY SUM(Y) DESC/ASC LIMIT 1
- Return ONLY the requested columns, no extra columns

Return only the SQL query, without explanation or markdown."""

    messages.append({"role": "system", "content": system_content})

    if use_few_shot:
        # Add few-shot examples as user/assistant turns
        for ex in BIRD_FEW_SHOT_EXAMPLES:
            # User turn with question
            user_msg = f"Question: {ex['question']}"
            if ex.get('hint'):
                user_msg += f"\nHint: {ex['hint']}"
            messages.append({"role": "user", "content": user_msg})

            # Assistant turn with SQL
            messages.append({"role": "assistant", "content": ex['sql']})

    # Final user question with schema
    final_user = f"""Given the following database schema:

{schema}

Question: {question}"""

    if hint:
        final_user += f"\nHint: {hint}"

    messages.append({"role": "user", "content": final_user})

    # Apply chat template
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    return messages


def detect_singular_intent(question: str) -> bool:
    """
    Detect if the question expects a single-value answer.

    Used by the Simulation Filter to reject "chatty" beams that
    return extra columns when a single value is expected.

    Args:
        question: Natural language question

    Returns:
        True if question expects singular output
    """
    question_lower = question.lower()

    # Patterns indicating singular output expected
    singular_patterns = [
        "which ",      # "Which year...", "Which customer..."
        "what is the ",  # "What is the total...", "What is the ratio..."
        "who ",        # "Who had the least..."
        "how many ",   # "How many customers..."
        "how much ",   # "How much did..."
        "what was the ",  # "What was the peak month..."
    ]

    for pattern in singular_patterns:
        if pattern in question_lower:
            return True

    return False


def count_expected_columns(question: str) -> int:
    """
    Estimate expected number of columns from question phrasing.

    Args:
        question: Natural language question

    Returns:
        Expected column count (1 for singular, -1 for unknown)
    """
    question_lower = question.lower()

    # Multi-column patterns
    if " and " in question_lower and ("what" in question_lower or "how" in question_lower):
        # "What is X and Y?" -> likely 2+ columns
        return -1  # Unknown, don't filter

    if detect_singular_intent(question):
        return 1

    return -1  # Unknown
