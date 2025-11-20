# Git Commit Organization Plan

## Files to Commit (Organized)

### Commit 1: Core Refactoring
**Message**: `refactor: Consolidate benchmark scripts into unified runner`

**Files**:
- `.gitignore` - Updated to ignore temporary files
- `scripts/benchmark.py` - NEW unified benchmark runner
- `scripts/archive/` - Moved deprecated scripts (14 files)
- `scripts/create_gold.py` - Utility (keep)
- `scripts/download_bird.py` - Utility (keep)
- `scripts/format_predictions.py` - Utility (keep)
- Deleted files:
  - `scripts/check.py`
  - `scripts/milestone1_validation.py`
  - `scripts/milestone2_beam_search.py`
  - `scripts/milestone2_evaluation.py`
  - `scripts/milestone3_schema_guidance.py`
  - `scripts/milestone4_efficiency_guidance.py`
  - `scripts/test_pipeline.py`

**Rationale**: Consolidates 17 scripts into 1 unified runner + 3 utilities (76% reduction)

---

### Commit 2: Documentation
**Message**: `docs: Add refactoring summary and M5 analysis`

**Files**:
- `README.md` - Updated with new structure and usage
- `REFACTORING_SUMMARY.md` - Complete refactoring documentation
- `M5_THRESHOLD_ANALYSIS.md` - M5 experimental results

**Rationale**: Comprehensive documentation of changes and research findings

---

### Commit 3: Source Updates
**Message**: `feat: Add M5 strategy to guided generator`

**Files**:
- `src/egtts/guided.py` - Added M5 confidence-aware re-ranking

**Rationale**: New generation strategy (though not recommended based on results)

---

### Commit 4: Dependencies
**Message**: `chore: Update project dependencies`

**Files**:
- `pyproject.toml` - Project configuration
- `uv.lock` - Dependency lock file

**Rationale**: Keep dependencies in sync

---

## Suggested Commit Sequence

### Option A: Single Commit (Simple)
```bash
git add .
git commit -m "refactor: Consolidate codebase and add M5 strategy

- Unified 17 benchmark scripts into 1 benchmark.py
- Added M5 confidence-aware re-ranking strategy
- Moved deprecated scripts to scripts/archive/
- Updated documentation (README, REFACTORING_SUMMARY)
- Added M5 threshold analysis results
- Updated .gitignore to exclude temporary files"
```

### Option B: Logical Commits (Organized)
```bash
# 1. Core refactoring
git add .gitignore scripts/
git commit -m "refactor: Consolidate benchmark scripts into unified runner

- Created scripts/benchmark.py - unified entry point
- Moved 14 deprecated scripts to scripts/archive/
- Kept 3 utility scripts (create_gold, download_bird, format_predictions)
- Updated .gitignore to exclude temporary result files
- 76% reduction in active scripts (17 → 4)"

# 2. Source code updates
git add src/egtts/guided.py
git commit -m "feat: Add M5 confidence-aware re-ranking strategy

- Implemented generate_with_confidence_aware_reranking()
- Uses model sequence scores to gate cost-based decisions
- Configurable confidence threshold parameter
- Note: Results show M3 remains best strategy (see M5_THRESHOLD_ANALYSIS.md)"

# 3. Documentation
git add README.md REFACTORING_SUMMARY.md M5_THRESHOLD_ANALYSIS.md
git commit -m "docs: Update documentation and add analysis results

- Comprehensive README with new structure and examples
- REFACTORING_SUMMARY.md documenting all changes
- M5_THRESHOLD_ANALYSIS.md with experimental results
- Clear migration guide from old scripts to new benchmark.py"

# 4. Dependencies
git add pyproject.toml uv.lock
git commit -m "chore: Update project dependencies"
```

## Current Status Summary

**Total files changed**: 17
- Modified: 4 (.gitignore, README.md, guided.py, pyproject.toml, uv.lock)
- Deleted: 7 (old milestone scripts)
- Added: 6 (benchmark.py, 3 utilities, 2 docs)
- Archived: 14 (scripts/archive/)

**Key Benefits**:
- 76% reduction in script count
- Single unified interface
- Comprehensive documentation
- All old code preserved in archive

## Recommendation

**Use Option B (Logical Commits)** for better git history:
1. Shows clear intent for each change
2. Easier to review individually
3. Better for rollback if needed
4. More professional commit history

Each commit is self-contained and can be reviewed independently.
