"""Few-shot example selection for Text-to-SQL generation."""

import json
import random
from pathlib import Path


# Static few-shot examples from BIRD dataset - high quality, diverse SQL patterns
# Selected from different databases to avoid data leakage while maintaining BIRD style
STATIC_FEW_SHOT_EXAMPLES = [
    # Simple COUNT with filter
    (
        "How many schools have a free meal rate below 20%?",
        "Free Meal Count (K-12) / Enrollment (K-12) < 0.2",
        "SELECT COUNT(*) FROM frpm WHERE `Free Meal Count (K-12)` * 1.0 / `Enrollment (K-12)` < 0.2"
    ),
    # JOIN with aggregation
    (
        "What is the total budget of all departments in the College of Engineering?",
        None,
        "SELECT SUM(T1.budget) FROM department AS T1 INNER JOIN college AS T2 ON T1.college_id = T2.college_id WHERE T2.college_name = 'College of Engineering'"
    ),
    # GROUP BY with ORDER BY and LIMIT
    (
        "Which country has the most players? Show the country name.",
        None,
        "SELECT nationality, COUNT(*) AS player_count FROM Player GROUP BY nationality ORDER BY player_count DESC LIMIT 1"
    ),
    # Subquery for MAX
    (
        "What is the name of the player with the highest overall rating?",
        None,
        "SELECT player_name FROM Player WHERE overall_rating = (SELECT MAX(overall_rating) FROM Player)"
    ),
    # Date filtering with LIKE
    (
        "How many transactions happened in August 2012?",
        None,
        "SELECT COUNT(*) FROM transactions WHERE Date LIKE '2012-08%'"
    ),
]


class FewShotProvider:
    """Provides few-shot examples for SQL generation."""

    def __init__(self, dataset_path: str = None, num_examples: int = 3, strategy: str = "static"):
        """
        Initialize few-shot provider.

        Args:
            dataset_path: Path to BIRD dataset JSON (for dynamic selection)
            num_examples: Number of few-shot examples to use
            strategy: "static" (fixed examples), "random" (random from dataset),
                     "same_db" (examples from same database)
        """
        self.num_examples = num_examples
        self.strategy = strategy
        self.dataset = None
        self.db_examples = {}  # Cache of examples by db_id

        if dataset_path and Path(dataset_path).exists():
            with open(dataset_path) as f:
                self.dataset = json.load(f)
            # Index examples by database
            for ex in self.dataset:
                db_id = ex['db_id']
                if db_id not in self.db_examples:
                    self.db_examples[db_id] = []
                self.db_examples[db_id].append(ex)

    def get_examples(self, db_id: str = None, exclude_idx: int = None) -> list:
        """
        Get few-shot examples.

        Args:
            db_id: Database ID for same-db strategy
            exclude_idx: Index to exclude (to avoid using target as example)

        Returns:
            List of (question, evidence, sql) tuples
        """
        if self.strategy == "static":
            return STATIC_FEW_SHOT_EXAMPLES[:self.num_examples]

        elif self.strategy == "random" and self.dataset:
            candidates = [ex for i, ex in enumerate(self.dataset) if i != exclude_idx]
            selected = random.sample(candidates, min(self.num_examples, len(candidates)))
            return [
                (ex['question'], ex.get('evidence', ''), ex['SQL'])
                for ex in selected
            ]

        elif self.strategy == "same_db" and db_id and db_id in self.db_examples:
            # Get examples from the same database
            candidates = [
                ex for ex in self.db_examples[db_id]
                if self.dataset.index(ex) != exclude_idx
            ]
            if len(candidates) < self.num_examples:
                # Fall back to random if not enough same-db examples
                other = [ex for ex in self.dataset if ex['db_id'] != db_id]
                candidates.extend(random.sample(other, min(self.num_examples - len(candidates), len(other))))
            selected = candidates[:self.num_examples]
            return [
                (ex['question'], ex.get('evidence', ''), ex['SQL'])
                for ex in selected
            ]

        # Default fallback to static
        return STATIC_FEW_SHOT_EXAMPLES[:self.num_examples]


def get_few_shot_examples(num_examples: int = 3) -> list:
    """
    Get static few-shot examples.

    Args:
        num_examples: Number of examples to return

    Returns:
        List of (question, evidence, sql) tuples
    """
    return STATIC_FEW_SHOT_EXAMPLES[:num_examples]
