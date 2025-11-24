"""Plan abstraction for semantic SQL comparison.

This module provides utilities to normalize SQLite EXPLAIN QUERY PLAN output
into abstract "plan signatures" that can be compared for semantic equivalence.

Key insight: Different SQL syntax can produce identical execution plans.
By comparing plans instead of SQL strings, we can group semantically
equivalent queries for majority voting.
"""

import re
from collections import Counter


def normalize_plan(explain_output: list[dict]) -> str:
    """
    Normalize EXPLAIN QUERY PLAN output into a hashable signature.

    This extracts the semantic structure of the query execution while
    stripping out variable elements like IDs and cost estimates.

    Args:
        explain_output: List of plan rows from EXPLAIN QUERY PLAN
                       Each row has: id, parent, notused, detail

    Returns:
        A normalized string signature representing the plan structure.
        Semantically equivalent queries will have the same signature.

    Example:
        Input: [{"detail": "SCAN customers"}, {"detail": "SEARCH yearmonth USING INDEX idx_customer"}]
        Output: "SCAN:customers|SEARCH:yearmonth:USING_INDEX"
    """
    if not explain_output:
        return ""

    signature_parts = []

    for row in explain_output:
        detail = row.get("detail", "")
        normalized = _normalize_detail(detail)
        if normalized:
            signature_parts.append(normalized)

    # Sort for consistency (order shouldn't matter for semantic comparison)
    # Actually, keep order - it reflects query structure
    return "|".join(signature_parts)


def _normalize_detail(detail: str) -> str:
    """
    Normalize a single EXPLAIN detail line.

    Examples:
        "SCAN customers" -> "SCAN:customers"
        "SEARCH yearmonth USING INDEX idx_cust (CustomerID=?)" -> "SEARCH:yearmonth:USING_INDEX"
        "USE TEMP B-TREE FOR ORDER BY" -> "TEMP_BTREE:ORDER_BY"
        "COMPOUND SUBQUERIES 2 AND 3 USING TEMP B-TREE (UNION)" -> "COMPOUND:UNION"
    """
    detail = detail.strip()

    # SCAN table (full table scan)
    if detail.startswith("SCAN "):
        match = re.match(r"SCAN\s+(\w+)", detail)
        if match:
            table = match.group(1)
            if "USING" in detail:
                return f"SCAN:{table}:USING_INDEX"
            return f"SCAN:{table}"

    # SEARCH table USING INDEX (index lookup)
    if detail.startswith("SEARCH "):
        match = re.match(r"SEARCH\s+(\w+)", detail)
        if match:
            table = match.group(1)
            if "USING COVERING INDEX" in detail:
                return f"SEARCH:{table}:COVERING_INDEX"
            elif "USING INDEX" in detail:
                return f"SEARCH:{table}:USING_INDEX"
            elif "USING INTEGER PRIMARY KEY" in detail:
                return f"SEARCH:{table}:PRIMARY_KEY"
            return f"SEARCH:{table}"

    # Temp B-tree operations
    if "USE TEMP B-TREE" in detail:
        if "ORDER BY" in detail:
            return "TEMP_BTREE:ORDER_BY"
        if "GROUP BY" in detail:
            return "TEMP_BTREE:GROUP_BY"
        if "DISTINCT" in detail:
            return "TEMP_BTREE:DISTINCT"
        return "TEMP_BTREE"

    # Compound queries (UNION, INTERSECT, EXCEPT)
    if detail.startswith("COMPOUND"):
        if "UNION ALL" in detail:
            return "COMPOUND:UNION_ALL"
        if "UNION" in detail:
            return "COMPOUND:UNION"
        if "INTERSECT" in detail:
            return "COMPOUND:INTERSECT"
        if "EXCEPT" in detail:
            return "COMPOUND:EXCEPT"
        return "COMPOUND"

    # Subqueries
    if "CORRELATED SCALAR SUBQUERY" in detail:
        return "CORRELATED_SUBQUERY"
    if "SCALAR SUBQUERY" in detail:
        return "SCALAR_SUBQUERY"
    if "LIST SUBQUERY" in detail:
        return "LIST_SUBQUERY"

    # Co-routine (for CTEs and subqueries)
    if detail.startswith("CO-ROUTINE"):
        return "CO_ROUTINE"

    # Materialization
    if "MATERIALIZE" in detail:
        return "MATERIALIZE"

    # If we don't recognize it, return a simplified version
    # Remove numbers and specific identifiers
    simplified = re.sub(r'\d+', '', detail)
    simplified = re.sub(r'\s+', '_', simplified.strip())
    return simplified[:50] if simplified else ""


def get_plan_signature(sql: str, db_path: str) -> tuple[str, int]:
    """
    Get plan signature and cost for a SQL query.

    Args:
        sql: SQL query string
        db_path: Path to SQLite database

    Returns:
        Tuple of (signature, cost) or ("ERROR", -1) if query is invalid
    """
    from .database import explain_query, ExplainSuccess

    result = explain_query(sql, db_path)

    if not isinstance(result, ExplainSuccess):
        return "ERROR", -1

    signature = normalize_plan(result.plan)
    cost = _calculate_cost(result.plan)

    return signature, cost


def _calculate_cost(plan: list[dict]) -> int:
    """Calculate heuristic cost from plan (lower is better)."""
    cost = 0
    plan_str = " ".join(str(row) for row in plan).upper()

    # Penalties
    if "SCAN " in plan_str and "USING INDEX" not in plan_str:
        cost += 100  # Full table scan
    if "USE TEMP B-TREE" in plan_str:
        cost += 50   # Temp structure overhead
    if "CORRELATED" in plan_str:
        cost += 75   # Correlated subquery (expensive)

    return cost


def vote_by_plan(candidates: list[tuple[str, str, int]]) -> tuple[str, str, dict]:
    """
    Perform majority voting based on plan signatures.

    Args:
        candidates: List of (sql, signature, cost) tuples

    Returns:
        Tuple of (winning_sql, winning_signature, vote_stats)
    """
    # Filter out errors
    valid = [(sql, sig, cost) for sql, sig, cost in candidates if sig != "ERROR"]

    if not valid:
        # All failed - return first candidate
        if candidates:
            return candidates[0][0], "ERROR", {"error": "all_invalid"}
        return "", "ERROR", {"error": "no_candidates"}

    # Count votes by signature
    signature_counts = Counter(sig for _, sig, _ in valid)

    # Find winning signature
    winning_sig, vote_count = signature_counts.most_common(1)[0]

    # Get all candidates matching winning signature
    winners = [(sql, sig, cost) for sql, sig, cost in valid if sig == winning_sig]

    # Tie-breaker: pick lowest cost among winners
    winners.sort(key=lambda x: x[2])
    best_sql = winners[0][0]

    vote_stats = {
        "total_candidates": len(candidates),
        "valid_candidates": len(valid),
        "unique_signatures": len(signature_counts),
        "winning_votes": vote_count,
        "vote_ratio": vote_count / len(valid) if valid else 0,
        "signature_distribution": dict(signature_counts),
    }

    return best_sql, winning_sig, vote_stats


def extract_string_literals(sql: str) -> list[tuple[str, str]]:
    """
    Extract string literals and their likely column context from SQL.

    Returns:
        List of (literal_value, context_hint) tuples.
        Context hint is the word before the literal (often column name).

    Example:
        "WHERE Segment = 'SME' AND Country = 'France'"
        -> [('SME', 'Segment'), ('France', 'Country')]
    """
    # Pattern: word = 'value' or word LIKE 'value'
    pattern = r"(\w+)\s*(?:=|LIKE|IN\s*\()\s*'([^']+)'"
    matches = re.findall(pattern, sql, re.IGNORECASE)

    # Also catch standalone literals without clear column context
    all_literals = re.findall(r"'([^']+)'", sql)

    results = []
    matched_values = set(m[1] for m in matches)

    for col, val in matches:
        results.append((val, col))

    # Add any literals we didn't catch with context
    for lit in all_literals:
        if lit not in matched_values:
            results.append((lit, None))

    return results


def probe_literal(db_path: str, table: str, column: str, value: str) -> bool:
    """
    Probe database to check if a literal value exists in a column.

    Args:
        db_path: Path to SQLite database
        table: Table name
        column: Column name
        value: Literal value to check

    Returns:
        True if value exists, False otherwise
    """
    import sqlite3

    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        cursor = conn.cursor()
        # Quick existence check with LIMIT 1
        cursor.execute(
            f"SELECT 1 FROM `{table}` WHERE `{column}` = ? LIMIT 1",
            (value,)
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except Exception:
        return True  # On error, assume valid (don't penalize)


def get_schema_columns(db_path: str) -> dict[str, list[str]]:
    """
    Get all columns for each table in the database.

    Returns:
        Dict mapping table_name -> list of column_names
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info(`{table}`)")
        columns = [row[1] for row in cursor.fetchall()]
        schema[table] = columns

    conn.close()
    return schema


def probe_sql_literals(sql: str, db_path: str) -> tuple[int, int, list[str]]:
    """
    Probe all string literals in SQL to check if they exist in the database.

    This implements the "Data Probing" idea: hallucinated values like
    'NonExistentCountry' will fail the probe, allowing us to demote
    candidates that contain non-existent literals.

    Args:
        sql: SQL query to analyze
        db_path: Path to SQLite database

    Returns:
        Tuple of (probes_passed, probes_total, failed_probes_info)
    """
    literals = extract_string_literals(sql)

    if not literals:
        return 0, 0, []

    schema = get_schema_columns(db_path)

    probes_passed = 0
    probes_total = 0
    failed_probes = []

    for value, hint_column in literals:
        # Skip obvious non-data literals (dates, wildcards, etc.)
        if '%' in value or value.isdigit():
            continue

        probed = False

        # If we have a column hint, try to find it in schema
        if hint_column:
            hint_lower = hint_column.lower()
            for table, columns in schema.items():
                for col in columns:
                    if col.lower() == hint_lower:
                        probes_total += 1
                        if probe_literal(db_path, table, col, value):
                            probes_passed += 1
                        else:
                            failed_probes.append(f"{table}.{col}='{value}'")
                        probed = True
                        break
                if probed:
                    break

        # If no hint or hint not found, try all text-like columns
        if not probed:
            # Skip probing without context to avoid false positives
            pass

    return probes_passed, probes_total, failed_probes


def vote_by_plan_with_probing(
    candidates: list[tuple[str, str, int]],
    db_path: str
) -> tuple[str, str, dict]:
    """
    Perform majority voting with Data Probing as tie-breaker (M13).

    This extends vote_by_plan by:
    1. Grouping candidates by plan signature (M7)
    2. For tied clusters, probing literals to detect hallucinations
    3. Demoting clusters with failed probes

    Args:
        candidates: List of (sql, signature, cost) tuples
        db_path: Path to SQLite database for probing

    Returns:
        Tuple of (winning_sql, winning_signature, vote_stats)
    """
    # Filter out errors and special signatures
    valid = [
        (sql, sig, cost) for sql, sig, cost in candidates
        if sig not in ("ERROR", "FILTERED_CHATTY")
    ]

    if not valid:
        if candidates:
            return candidates[0][0], "ERROR", {"error": "all_invalid"}
        return "", "ERROR", {"error": "no_candidates"}

    # Count votes by signature
    signature_counts = Counter(sig for _, sig, _ in valid)

    # Get top clusters (those with significant votes)
    top_clusters = signature_counts.most_common(3)  # Top 3

    # For each cluster, compute probe score
    cluster_scores = []
    probing_stats = {}

    for sig, vote_count in top_clusters:
        cluster_candidates = [(sql, cost) for sql, s, cost in valid if s == sig]

        # Probe the best (lowest cost) candidate from each cluster
        cluster_candidates.sort(key=lambda x: x[1])
        best_sql = cluster_candidates[0][0]

        passed, total, failed = probe_sql_literals(best_sql, db_path)

        # Score: votes + probe_bonus
        # If all probes pass, no penalty. If some fail, penalty.
        probe_ratio = passed / total if total > 0 else 1.0
        probe_penalty = (1.0 - probe_ratio) * 2  # Up to 2 vote penalty

        adjusted_votes = vote_count - probe_penalty

        cluster_scores.append((sig, adjusted_votes, vote_count, best_sql))
        probing_stats[sig] = {
            "original_votes": vote_count,
            "probes_passed": passed,
            "probes_total": total,
            "failed_probes": failed,
            "adjusted_votes": adjusted_votes,
        }

    # Sort by adjusted votes
    cluster_scores.sort(key=lambda x: -x[1])

    winning_sig = cluster_scores[0][0]
    winning_sql = cluster_scores[0][3]
    winning_votes = cluster_scores[0][2]

    vote_stats = {
        "total_candidates": len(candidates),
        "valid_candidates": len(valid),
        "unique_signatures": len(signature_counts),
        "winning_votes": winning_votes,
        "vote_ratio": winning_votes / len(valid) if valid else 0,
        "probing_applied": True,
        "probing_stats": probing_stats,
    }

    return winning_sql, winning_sig, vote_stats
