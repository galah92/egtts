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
