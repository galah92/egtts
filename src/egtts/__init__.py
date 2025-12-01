"""EGTTS: Execution-Guided Text-to-SQL using plan-based voting."""

from .data import get_database_path, load_spider
from .database import ExplainError, ExplainSuccess, explain_query
from .guided import ExplainGuidedGenerator, GenerationResult
from .model import create_sql_prompt, generate_sql, load_model
from .plans import normalize_plan, vote_by_plan
from .schema import SchemaIndex, build_schema_index, get_augmented_schema

__all__ = [
    # Database utilities
    "explain_query",
    "ExplainSuccess",
    "ExplainError",
    # Data loading
    "load_spider",
    "get_database_path",
    # Model utilities
    "load_model",
    "create_sql_prompt",
    "generate_sql",
    # Schema utilities
    "SchemaIndex",
    "build_schema_index",
    "get_augmented_schema",
    # Plan voting
    "normalize_plan",
    "vote_by_plan",
    # Main generator
    "ExplainGuidedGenerator",
    "GenerationResult",
]
