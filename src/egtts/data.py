"""Dataset loading utilities for Spider and BIRD benchmarks."""

from pathlib import Path

from datasets import load_dataset


def load_spider(split: str = "validation", cache_dir: str | None = None):
    """
    Load Spider dataset from HuggingFace.

    The Spider dataset contains:
    - db_id: Database identifier
    - question: Natural language question
    - query: Gold SQL query
    - db_path: Path to SQLite database file (if downloaded)

    Args:
        split: Dataset split ("train" or "validation")
        cache_dir: Optional cache directory for downloaded data

    Returns:
        HuggingFace Dataset object with Spider examples
    """
    dataset = load_dataset(
        "spider",
        split=split,
        cache_dir=cache_dir
    )
    return dataset


def get_database_path(db_id: str, spider_dir: Path) -> Path:
    """
    Get path to SQLite database file for a given database ID.

    Spider databases are stored in: spider_dir/database/{db_id}/{db_id}.sqlite

    Args:
        db_id: Database identifier (e.g., "concert_singer")
        spider_dir: Root directory of Spider dataset

    Returns:
        Path to the SQLite database file
    """
    db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            f"Expected structure: spider_dir/database/{db_id}/{db_id}.sqlite"
        )

    return db_path
