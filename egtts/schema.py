"""Schema indexing for fast lookup during generation."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SchemaIndex:
    """Fast lookup structure for database schema validation."""

    tables: set[str]
    columns: dict[str, set[str]]  # table_name -> {column_names}
    all_columns: set[str]  # All column names across all tables

    def has_table(self, table_name: str) -> bool:
        """Check if table exists (case-insensitive)."""
        return table_name.lower() in {t.lower() for t in self.tables}

    def has_column(self, column_name: str, table_name: str | None = None) -> bool:
        """
        Check if column exists.

        Args:
            column_name: Column name to check
            table_name: Optional table name for scoped check

        Returns:
            True if column exists (in specified table if given, anywhere if not)
        """
        column_lower = column_name.lower()

        if table_name is not None:
            # Check in specific table
            table_lower = table_name.lower()
            for table, cols in self.columns.items():
                if table.lower() == table_lower:
                    return column_lower in {c.lower() for c in cols}
            return False

        # Check across all tables
        return column_lower in {c.lower() for c in self.all_columns}


def parse_create_table(create_stmt: str) -> tuple[str, list[str]]:
    """
    Parse CREATE TABLE statement to extract table name and columns.

    Args:
        create_stmt: SQL CREATE TABLE statement

    Returns:
        Tuple of (table_name, column_names)

    Raises:
        ValueError: If the table name cannot be parsed from the statement
    """
    # Extract table name
    table_match = re.search(
        r"CREATE TABLE\s+['\"]?(\w+)['\"]?\s*\(", create_stmt, re.IGNORECASE
    )
    if not table_match:
        raise ValueError(f"Could not parse table name from: {create_stmt[:100]}")

    table_name: str = table_match.group(1)

    # Extract column names - match word followed by type/constraints
    # This matches: column_name TYPE ... or column_name, or column_name )
    column_pattern = r"['\"]?(\w+)['\"]?\s+(?:INTEGER|TEXT|REAL|BLOB|NUMERIC|VARCHAR|INT|CHAR|BOOLEAN|DATE|DATETIME|TIMESTAMP|DECIMAL|FLOAT|DOUBLE)"
    columns: list[str] = re.findall(column_pattern, create_stmt, re.IGNORECASE)

    # Also catch primary key definitions like "id INTEGER PRIMARY KEY"
    # The pattern above should catch these, but let's also check for standalone column definitions
    if not columns:
        # Fallback: match any identifier before a comma or closing paren
        basic_pattern = r"['\"]?(\w+)['\"]?\s*[,)]"
        columns = re.findall(basic_pattern, create_stmt)

    return table_name, columns


def build_schema_index(db_path: str | Path) -> SchemaIndex:
    """
    Build fast lookup index from database schema.

    Args:
        db_path: Path to SQLite database

    Returns:
        SchemaIndex with tables and columns
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    table_data: list[tuple[str, str | None]] = cursor.fetchall()
    conn.close()

    tables: set[str] = set()
    columns_by_table: dict[str, set[str]] = {}
    all_columns: set[str] = set()

    for table_name, create_stmt in table_data:
        if create_stmt is None:
            continue

        tables.add(table_name)

        # Parse columns from CREATE statement
        try:
            _, column_list = parse_create_table(create_stmt)
            columns_by_table[table_name] = set(column_list)
            all_columns.update(column_list)
        except ValueError:
            # If parsing fails, try to get columns via PRAGMA
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            pragma_cols: list[str] = [row[1] for row in cursor.fetchall()]
            conn.close()

            columns_by_table[table_name] = set(pragma_cols)
            all_columns.update(pragma_cols)

    return SchemaIndex(
        tables=tables,
        columns=columns_by_table,
        all_columns=all_columns,
    )


def get_augmented_schema(
    db_path: str | Path, max_rows_per_table: int = 3, max_schema_chars: int = 8000
) -> str:
    """
    Generate augmented schema with sample rows for each table.

    This helps the model understand:
    - Actual data formats (e.g., Date='201301' not '2013-01-01')
    - Column value patterns
    - Data relationships

    Args:
        db_path: Path to SQLite database
        max_rows_per_table: Number of sample rows to include (default: 3)
        max_schema_chars: Maximum schema length before truncating samples (default: 8000)

    Returns:
        Augmented schema string with CREATE statements and sample data
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get all tables with their CREATE statements
    cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables: list[tuple[str, str | None]] = cursor.fetchall()

    schema_parts: list[str] = []

    for table_name, create_stmt in tables:
        if create_stmt is None:
            continue

        # Add CREATE statement
        schema_parts.append(create_stmt)

        # Get column names
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        columns: list[str] = [row[1] for row in cursor.fetchall()]

        if not columns:
            schema_parts.append("")
            continue

        # Get sample rows
        try:
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {max_rows_per_table}")
            rows: list[tuple[Any, ...]] = cursor.fetchall()

            if rows:
                # Format as comment with sample data
                schema_parts.append(f"-- Sample data from {table_name}:")
                col_header = ", ".join(columns)
                schema_parts.append(f"-- Columns: {col_header}")

                for row in rows:
                    # Format values, handling None and strings
                    formatted_values: list[str] = []
                    for val in row:
                        if val is None:
                            formatted_values.append("NULL")
                        elif isinstance(val, str):
                            # Truncate long strings
                            val_str = val[:50] + "..." if len(val) > 50 else val
                            formatted_values.append(f"'{val_str}'")
                        else:
                            formatted_values.append(str(val))
                    schema_parts.append(f"-- Example: ({', '.join(formatted_values)})")
        except Exception as e:
            schema_parts.append(f"-- (Could not fetch samples: {e})")

        schema_parts.append("")  # Blank line between tables

    conn.close()

    result = "\n".join(schema_parts)

    # If schema is too large, fall back to just CREATE statements (no samples)
    if len(result) > max_schema_chars:
        # Rebuild without sample data
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = cursor.fetchall()
        conn.close()

        schema_parts = []
        for table_name, create_stmt in tables:
            if create_stmt is not None:
                schema_parts.append(create_stmt)
                schema_parts.append("")
        result = "\n".join(schema_parts)

    return result
