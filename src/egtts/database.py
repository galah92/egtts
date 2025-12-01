"""Database utilities for non-destructive SQL verification using EXPLAIN."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Literal, TypedDict


class PlanRow(TypedDict):
    """A single row from EXPLAIN QUERY PLAN output."""

    id: int
    parent: int
    notused: int
    detail: str


@dataclass
class ExplainSuccess:
    """Successful EXPLAIN result with query plan."""

    plan: list[PlanRow]
    execution_time_ms: float
    status: Literal["success"] = "success"


@dataclass
class ExplainError:
    """Error result from EXPLAIN with specific error details."""

    error_type: str
    error_message: str
    execution_time_ms: float
    status: Literal["error"] = "error"


ExplainResult = ExplainSuccess | ExplainError


def explain_query(sql: str, db_path: str) -> ExplainResult:
    """
    Execute EXPLAIN QUERY PLAN on SQL without running the actual query.

    This is the core verification function for Milestone 1. It validates
    schema correctness without executing potentially destructive SQL.

    Args:
        sql: SQL query string to verify
        db_path: Path to SQLite database file

    Returns:
        ExplainSuccess with query plan if valid
        ExplainError with specific error details if invalid
    """
    start_time = time.perf_counter()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute EXPLAIN QUERY PLAN - non-destructive verification
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        plan_rows = cursor.fetchall()

        # Parse plan into structured format
        plan: list[PlanRow] = [
            PlanRow(id=row[0], parent=row[1], notused=row[2], detail=row[3])
            for row in plan_rows
        ]

        conn.close()
        execution_time = (time.perf_counter() - start_time) * 1000

        return ExplainSuccess(plan=plan, execution_time_ms=execution_time)

    except sqlite3.OperationalError as e:
        # Schema errors: "no such table", "no such column", etc.
        execution_time = (time.perf_counter() - start_time) * 1000
        return ExplainError(
            error_type="OperationalError",
            error_message=str(e),
            execution_time_ms=execution_time,
        )

    except sqlite3.Error as e:
        # Other database errors
        execution_time = (time.perf_counter() - start_time) * 1000
        return ExplainError(
            error_type=type(e).__name__,
            error_message=str(e),
            execution_time_ms=execution_time,
        )

    finally:
        # Ensure connection is closed even if error occurs
        try:
            conn.close()
        except Exception:
            pass
