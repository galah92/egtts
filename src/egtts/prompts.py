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


# Chain-of-Thought few-shot examples with reasoning
COT_FEW_SHOT_EXAMPLES = [
    # Example 1: "Highest monthly" requires GROUP BY + SUM + ORDER
    {
        "question": "What is the highest monthly consumption in the year 2012?",
        "hint": "The first 4 strings of the Date values can represent year; The 5th and 6th string can refer to month.",
        "reasoning": """1. Tables needed: yearmonth (contains Date and Consumption)
2. "Monthly consumption" means I need to SUM consumption grouped by month
3. "Highest monthly" means ORDER BY the SUM descending and LIMIT 1
4. Year 2012 filter: SUBSTR(Date, 1, 4) = '2012'
5. Month extraction: SUBSTR(Date, 5, 2)""",
        "sql": "SELECT SUM(Consumption) FROM yearmonth WHERE SUBSTR(Date, 1, 4) = '2012' GROUP BY SUBSTR(Date, 5, 2) ORDER BY SUM(Consumption) DESC LIMIT 1"
    },
    # Example 2: Missing JOIN pattern - need to link through intermediate table
    {
        "question": "Which customers in the LAM segment had transactions at gas stations in CZE?",
        "hint": "LAM is a segment; CZE is a country code.",
        "reasoning": """1. Tables needed: customers (for Segment), transactions_1k (links customer to gas station), gasstations (for Country)
2. JOIN path: customers -> transactions_1k (on CustomerID) -> gasstations (on GasStationID)
3. Filter: Segment = 'LAM' AND Country = 'CZE'
4. Return: DISTINCT CustomerID to avoid duplicates""",
        "sql": "SELECT DISTINCT T1.CustomerID FROM customers AS T1 INNER JOIN transactions_1k AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN gasstations AS T3 ON T2.GasStationID = T3.GasStationID WHERE T1.Segment = 'LAM' AND T3.Country = 'CZE'"
    },
    # Example 3: COUNT(*) vs COUNT(DISTINCT) clarity
    {
        "question": "How many transactions were made by customers who pay in EUR?",
        "hint": "EUR is a Currency value.",
        "reasoning": """1. Tables needed: customers (for Currency), transactions_1k (for counting transactions)
2. JOIN: customers -> transactions_1k on CustomerID
3. "How many transactions" = COUNT(*) of transaction records (not COUNT(DISTINCT CustomerID))
4. Filter: Currency = 'EUR'""",
        "sql": "SELECT COUNT(*) FROM transactions_1k AS T1 INNER JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.Currency = 'EUR'"
    },
]


def format_cot_prompt(
    question: str,
    schema: str,
    tokenizer=None,
    hint: str = None,
) -> str:
    """
    Create a Chain-of-Thought prompt that forces reasoning before SQL.

    The model must output:
    -- Reasoning:
    1. Tables needed: ...
    2. JOIN path: ...
    3. Aggregation logic: ...
    -- SQL:
    SELECT ...

    Args:
        question: Natural language question
        schema: Database schema with sample data (augmented schema)
        tokenizer: Tokenizer for chat template
        hint: Optional hint/evidence

    Returns:
        Formatted prompt string
    """
    messages = []

    # System message enforcing CoT structure
    system_content = """You are a SQL expert. Generate SQLite queries based on the database schema and question.

IMPORTANT: You MUST think step-by-step before writing SQL.

Output format (REQUIRED):
-- Reasoning:
1. Tables needed: [list tables and why]
2. JOIN path: [how tables connect]
3. Aggregation: [GROUP BY? SUM/COUNT/MAX? ORDER BY?]
4. Key insight: [what makes this query tricky]
-- SQL:
[Your SQL query here]

Critical SQLite patterns:
- Dates are YYYYMMDD strings: use SUBSTR(Date, 1, 4) for year, SUBSTR(Date, 5, 2) for month
- "Highest/lowest monthly X" = GROUP BY month, ORDER BY SUM(X), LIMIT 1
- "How many transactions" = COUNT(*), "How many customers" = COUNT(DISTINCT CustomerID)
- Always check if you need intermediate JOINs to connect tables"""

    messages.append({"role": "system", "content": system_content})

    # Add CoT few-shot examples
    for ex in COT_FEW_SHOT_EXAMPLES:
        user_msg = f"Question: {ex['question']}"
        if ex.get('hint'):
            user_msg += f"\nHint: {ex['hint']}"
        messages.append({"role": "user", "content": user_msg})

        # Assistant shows reasoning then SQL
        assistant_msg = f"""-- Reasoning:
{ex['reasoning']}
-- SQL:
{ex['sql']}"""
        messages.append({"role": "assistant", "content": assistant_msg})

    # Final user question with augmented schema
    final_user = f"""Given the database schema and sample data below:

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


def extract_sql_from_cot(response: str) -> str:
    """
    Extract SQL from a Chain-of-Thought response.

    The response format is:
    -- Reasoning:
    ...
    -- SQL:
    SELECT ...

    Args:
        response: Full model response with reasoning and SQL

    Returns:
        Extracted SQL query
    """
    import re

    # Try to find SQL after "-- SQL:" marker
    sql_marker_match = re.search(r'--\s*SQL:\s*\n?(.*)', response, re.IGNORECASE | re.DOTALL)
    if sql_marker_match:
        sql = sql_marker_match.group(1).strip()
        # Clean up: remove trailing comments or extra content
        # Take first complete SQL statement
        lines = []
        for line in sql.split('\n'):
            line = line.strip()
            if line.startswith('--') and not line.upper().startswith('-- SQL'):
                continue  # Skip comment lines after SQL
            if line:
                lines.append(line)
            # Stop if we hit another section marker
            if line.startswith('--') and 'reasoning' in line.lower():
                break
        sql = ' '.join(lines)
        # Remove any trailing semicolons and whitespace
        sql = sql.rstrip(';').strip()
        if sql:
            return sql

    # Fallback: look for SELECT statement
    select_match = re.search(r'(SELECT\s+.+?)(?:;|\Z)', response, re.IGNORECASE | re.DOTALL)
    if select_match:
        return select_match.group(1).strip()

    # Last resort: return the whole response
    return response.strip()


# Data Flow examples - showing execution order (FROM → WHERE → GROUP BY → SELECT)
DATAFLOW_FEW_SHOT_EXAMPLES = [
    {
        "question": "Which customer had the highest total consumption in 2012?",
        "hint": "Year 2012 can be presented as Between 201201 And 201212.",
        "plan": """-- 1. Source Tables: yearmonth (contains CustomerID, Date, Consumption)
-- 2. Filter Conditions: SUBSTR(Date, 1, 4) = '2012' (year 2012)
-- 3. Aggregation: GROUP BY CustomerID, ORDER BY SUM(Consumption) DESC, LIMIT 1
-- 4. Final Projection: CustomerID (the customer with highest total)""",
        "sql": "SELECT CustomerID FROM yearmonth WHERE SUBSTR(Date, 1, 4) = '2012' GROUP BY CustomerID ORDER BY SUM(Consumption) DESC LIMIT 1"
    },
    {
        "question": "What is the total gas consumption for LAM segment customers?",
        "hint": "LAM is a Segment value in customers table.",
        "plan": """-- 1. Source Tables: customers (Segment), yearmonth (Consumption) - JOIN on CustomerID
-- 2. Filter Conditions: Segment = 'LAM'
-- 3. Aggregation: SUM(Consumption) across all matching rows
-- 4. Final Projection: Total consumption (single value)""",
        "sql": "SELECT SUM(T2.Consumption) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = 'LAM'"
    },
]


def format_dataflow_prompt(
    question: str,
    schema: str,
    tokenizer=None,
    hint: str = None,
) -> str:
    """
    Create a Data Flow prompt that forces execution-order planning.

    The key insight: SQL syntax (SELECT-first) conflicts with logical order
    (FROM-first). By forcing the model to plan in execution order, we align
    autoregressive generation with the dependency graph.

    Output format:
    -- 1. Source Tables: [FROM/JOIN - which tables?]
    -- 2. Filter Conditions: [WHERE - what rows to keep?]
    -- 3. Aggregation: [GROUP BY/ORDER BY - how to process?]
    -- 4. Final Projection: [SELECT - what columns to return?]
    SELECT ...

    Args:
        question: Natural language question
        schema: Database schema with sample data (augmented schema)
        tokenizer: Tokenizer for chat template
        hint: Optional hint/evidence

    Returns:
        Formatted prompt string
    """
    messages = []

    # System message enforcing Data Flow structure
    system_content = """You are a SQL expert. Plan queries in EXECUTION ORDER before writing SQL.

OUTPUT FORMAT (all 4 steps required as SQL comments, then the query):
-- 1. Source Tables: [Which tables contain the data? How do they JOIN?]
-- 2. Filter Conditions: [What rows to keep? WHERE clause conditions]
-- 3. Aggregation: [GROUP BY? ORDER BY? LIMIT? SUM/COUNT/MAX?]
-- 4. Final Projection: [What columns to SELECT? Just what's asked, no extras]
SELECT ...

CRITICAL RULES:
- Dates are YYYYMMDD strings: SUBSTR(Date, 1, 4) for year, SUBSTR(Date, 5, 2) for month
- "Which X has most Y" = GROUP BY X, ORDER BY SUM(Y) DESC, LIMIT 1
- Return ONLY requested columns, never add extra "helpful" columns
- Think about JOINs FIRST, then filters, then what to return"""

    messages.append({"role": "system", "content": system_content})

    # Add Data Flow few-shot examples
    for ex in DATAFLOW_FEW_SHOT_EXAMPLES:
        user_msg = f"Question: {ex['question']}"
        if ex.get('hint'):
            user_msg += f"\nHint: {ex['hint']}"
        messages.append({"role": "user", "content": user_msg})

        # Assistant shows plan then SQL
        assistant_msg = f"""{ex['plan']}
{ex['sql']}"""
        messages.append({"role": "assistant", "content": assistant_msg})

    # Final user question with augmented schema
    final_user = f"""Database schema and sample data:

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


def extract_sql_from_dataflow(response: str) -> str:
    """
    Extract SQL from a Data Flow response.

    The response contains SQL comments (-- 1. Source...) followed by the query.
    SQLite ignores comments, so we can keep them, but we extract clean SQL
    for consistency.

    Args:
        response: Full model response with plan and SQL

    Returns:
        Extracted SQL query (may include comments, which SQLite ignores)
    """
    import re

    # The response should have comment lines followed by SELECT
    # We keep the whole thing since SQLite ignores comments
    lines = response.strip().split('\n')

    # Find the SELECT statement
    sql_started = False
    sql_lines = []

    for line in lines:
        line_stripped = line.strip()

        # Skip empty lines before SQL
        if not line_stripped and not sql_started:
            continue

        # Check if this is the start of SQL
        if line_stripped.upper().startswith('SELECT'):
            sql_started = True

        if sql_started:
            # Stop if we hit a new planning section or explanation
            if line_stripped.startswith('--') and any(
                marker in line_stripped.lower()
                for marker in ['source', 'filter', 'aggregation', 'projection', 'explanation', 'note:']
            ):
                break
            sql_lines.append(line_stripped)

    if sql_lines:
        sql = ' '.join(sql_lines)
        # Clean up
        sql = sql.rstrip(';').strip()
        return sql

    # Fallback: look for SELECT anywhere
    select_match = re.search(r'(SELECT\s+.+?)(?:;|\Z)', response, re.IGNORECASE | re.DOTALL)
    if select_match:
        return select_match.group(1).strip()

    # Last resort
    return response.strip()


# =============================================================================
# M16: Selective Reasoning - Question Classification
# =============================================================================

# Patterns that indicate reasoning is likely to help (based on M10 vs M11 analysis)
REASONING_BENEFICIAL_PATTERNS = {
    # "Highest/lowest/most/least" questions - 42% accuracy without reasoning
    # These need GROUP BY + aggregate + ORDER BY logic
    'superlative': [
        'highest', 'lowest', 'most', 'least', 'peak', 'maximum', 'minimum',
        'top', 'bottom', 'largest', 'smallest', 'greatest', 'best', 'worst',
        'fewest', 'biggest',
    ],
    # Multi-step comparisons often confused
    'comparison': [
        'more than', 'less than', 'greater than', 'fewer than',
        'compared to', 'versus', 'vs',
    ],
    # Rate/change calculations need careful thought
    'rate_change': [
        'rate', 'growth', 'decline', 'decrease', 'increase',
        'change from', 'change between',
    ],
}

# Patterns where direct generation works well (63%+ accuracy)
# Reasoning may introduce errors through overthinking
DIRECT_GENERATION_PATTERNS = {
    'simple_count': ['how many', 'count of', 'number of', 'total number'],
    'simple_list': ['list all', 'show all', 'what are the', 'give me'],
    'simple_lookup': ['what is the name', 'who is', 'which is'],
}


def classify_question_for_reasoning(question: str, hint: str = None) -> dict:
    """
    Classify whether a question would benefit from Chain-of-Thought reasoning.

    Based on empirical analysis:
    - Superlative questions (highest/lowest/most/least): 42% accuracy without CoT
    - Simple queries: 63% accuracy - reasoning can hurt here
    - Multi-table joins: Lower accuracy, reasoning helps with JOIN planning

    Args:
        question: Natural language question
        hint: Optional hint/evidence (may indicate complexity)

    Returns:
        dict with:
            - needs_reasoning: bool
            - reason: str explaining the classification
            - patterns_matched: list of matched pattern categories
            - confidence: float (0-1) indicating classification confidence
    """
    q_lower = question.lower()
    hint_lower = (hint or '').lower()

    patterns_matched = []
    reasoning_score = 0  # Positive = needs reasoning, negative = direct generation

    # Check for reasoning-beneficial patterns
    for category, keywords in REASONING_BENEFICIAL_PATTERNS.items():
        for kw in keywords:
            if kw in q_lower:
                patterns_matched.append(f"{category}:{kw}")
                reasoning_score += 2  # Strong signal for reasoning

    # Check for direct-generation patterns
    for category, keywords in DIRECT_GENERATION_PATTERNS.items():
        for kw in keywords:
            if kw in q_lower:
                patterns_matched.append(f"direct:{kw}")
                reasoning_score -= 1  # Weak signal against reasoning

    # Additional heuristics
    # 1. Questions with "for each" likely need GROUP BY reasoning
    if 'for each' in q_lower or 'per ' in q_lower:
        reasoning_score += 1
        patterns_matched.append("grouping:for_each")

    # 2. Complex hints suggest complex queries
    if hint_lower:
        # Long hints often indicate tricky queries
        if len(hint_lower) > 100:
            reasoning_score += 1
            patterns_matched.append("hint:complex")
        # Hints mentioning calculations
        if any(calc in hint_lower for calc in ['calculate', 'formula', 'ratio', 'percentage']):
            reasoning_score += 1
            patterns_matched.append("hint:calculation")

    # 3. Multiple question marks or "and" conjunctions suggest multi-part
    if question.count('?') > 1 or ' and ' in q_lower:
        reasoning_score += 1
        patterns_matched.append("structure:multi_part")

    # Decision threshold
    needs_reasoning = reasoning_score >= 2

    # Calculate confidence
    if reasoning_score >= 3:
        confidence = 0.9
        reason = "Strong reasoning signals: superlative/comparison pattern"
    elif reasoning_score >= 2:
        confidence = 0.7
        reason = "Moderate reasoning signals: likely needs careful planning"
    elif reasoning_score <= -1:
        confidence = 0.8
        reason = "Direct generation preferred: simple query pattern"
    else:
        confidence = 0.5
        reason = "Ambiguous: defaulting to direct generation"
        needs_reasoning = False

    return {
        'needs_reasoning': needs_reasoning,
        'reason': reason,
        'patterns_matched': patterns_matched,
        'confidence': confidence,
        'reasoning_score': reasoning_score,
    }


# Focused CoT prompt for selective reasoning - shorter and more structured
SELECTIVE_COT_TEMPLATE = """You are a SQL expert. This query requires careful reasoning.

THINK STEP BY STEP:
1. Tables needed and JOIN conditions
2. WHERE filters
3. Aggregation: GROUP BY? ORDER BY? LIMIT?
4. SELECT: Return ONLY what's asked

CRITICAL for "{pattern_type}" questions:
{pattern_specific_guidance}

**OUTPUT RULE**: Return EXACTLY the columns asked for. NO extra columns like "TotalConsumption" or "Count" - just the answer.

Schema:
{schema}

Question: {question}
{hint_section}

Think step by step, then write SQL:
"""

PATTERN_GUIDANCE = {
    'superlative': """- "highest/most X" = GROUP BY entity, ORDER BY SUM(X) DESC, LIMIT 1
- "lowest/least X" = GROUP BY entity, ORDER BY SUM(X) ASC, LIMIT 1
- Do NOT use MAX/MIN on aggregated values - use ORDER BY + LIMIT instead
- Return ONLY the entity (e.g., CustomerID, Year), NOT the aggregated value""",

    'comparison': """- Identify BOTH sides of the comparison clearly
- Use CASE WHEN or separate subqueries for each side
- Double-check the order of subtraction (A - B vs B - A)""",

    'rate_change': """- Rate = (new - old) / old
- Ensure you're comparing the same entity across time periods
- Watch for division by zero""",

    'default': """- Identify all required tables and their relationships
- Consider whether GROUP BY is needed for aggregation
- Verify column names exist in the schema""",
}


def format_selective_cot_prompt(
    question: str,
    schema: str,
    classification: dict,
    tokenizer=None,
    hint: str = None,
) -> str:
    """
    Create a focused CoT prompt based on the question classification.

    Unlike the generic CoT prompt, this provides pattern-specific guidance
    to help the model avoid common errors for that question type.

    Args:
        question: Natural language question
        schema: Augmented schema with sample data
        classification: Result from classify_question_for_reasoning()
        tokenizer: Tokenizer for chat template
        hint: Optional hint/evidence

    Returns:
        Formatted prompt string
    """
    # Determine primary pattern type
    patterns = classification.get('patterns_matched', [])
    pattern_type = 'default'
    for p in patterns:
        if p.startswith('superlative:'):
            pattern_type = 'superlative'
            break
        elif p.startswith('comparison:'):
            pattern_type = 'comparison'
            break
        elif p.startswith('rate_change:'):
            pattern_type = 'rate_change'
            break

    guidance = PATTERN_GUIDANCE.get(pattern_type, PATTERN_GUIDANCE['default'])
    hint_section = f"Hint: {hint}" if hint else ""

    prompt_text = SELECTIVE_COT_TEMPLATE.format(
        pattern_type=pattern_type,
        pattern_specific_guidance=guidance,
        schema=schema,
        question=question,
        hint_section=hint_section,
    )

    messages = [{"role": "user", "content": prompt_text}]

    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return prompt_text
