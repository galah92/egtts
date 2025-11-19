"""Quick verification script to test core functionality."""

import sqlite3
import sys
import tempfile
from pathlib import Path

print("=" * 60)
print("EGTTS Verification Script")
print("=" * 60)

# Test 1: Imports
print("\n[1/4] Testing imports...")
try:
    from egtts import ExplainError, ExplainSuccess, explain_query, load_spider
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Database verification with test database
print("\n[2/4] Testing EXPLAIN functionality...")
with tempfile.TemporaryDirectory() as tmpdir:
    # Create a test database
    test_db = Path(tmpdir) / "test.db"
    conn = sqlite3.connect(test_db)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("CREATE TABLE posts (id INTEGER, user_id INTEGER, title TEXT)")
    conn.commit()
    conn.close()

    # Test valid query
    result = explain_query("SELECT * FROM users", str(test_db))
    if isinstance(result, ExplainSuccess):
        print(f"✓ Valid query verified ({result.execution_time_ms:.2f}ms)")
        print(f"  Plan: {result.plan[0]['detail']}")
    else:
        print(f"✗ Unexpected error: {result.error_message}")
        sys.exit(1)

    # Test invalid query (hallucinated column)
    result = explain_query("SELECT nonexistent_column FROM users", str(test_db))
    if isinstance(result, ExplainError):
        print(f"✓ Schema error caught ({result.execution_time_ms:.2f}ms)")
        print(f"  Error: {result.error_message}")
    else:
        print("✗ Should have caught schema error")
        sys.exit(1)

# Test 3: Spider dataset loading
print("\n[3/4] Testing Spider dataset access...")
try:
    print("  Downloading Spider dataset (this may take a moment)...")
    dataset = load_spider(split="validation")
    print(f"✓ Spider loaded: {len(dataset)} validation examples")

    # Show a sample
    sample = dataset[0]
    print(f"  Sample question: {sample['question'][:60]}...")
    print(f"  Database ID: {sample['db_id']}")
except Exception as e:
    print(f"✗ Spider loading failed: {e}")
    print("  Note: This is expected if HuggingFace is unreachable")

# Test 4: Check CUDA availability for model
print("\n[4/4] Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠ CUDA not available (model will run on CPU)")
except Exception as e:
    print(f"✗ Error checking CUDA: {e}")

print("\n" + "=" * 60)
print("Verification complete!")
print("=" * 60)
