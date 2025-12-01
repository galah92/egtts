#!/bin/bash
#
# Run BIRD official evaluation on our predictions
#
# Usage:
#   ./scripts/run_official_evaluation.sh baseline
#   ./scripts/run_official_evaluation.sh M4
#

set -e  # Exit on error

# Configuration
STRATEGY=${1:-baseline}
OFFICIAL_REPO="data/mini_dev_official"
EVAL_DIR="${OFFICIAL_REPO}/evaluation"
DB_ROOT="${OFFICIAL_REPO}/sqlite/dev_databases"
GROUND_TRUTH="${OFFICIAL_REPO}/sqlite/mini_dev_sqlite_gold.sql"
DIFF_JSON="${OFFICIAL_REPO}/sqlite/mini_dev_sqlite.jsonl"

# Paths to our predictions
if [ "$STRATEGY" = "baseline" ]; then
    PRED_PATH="$(pwd)/results/official_baseline_500.json"
elif [ "$STRATEGY" = "M4" ]; then
    PRED_PATH="$(pwd)/results/official_M4_500.json"
else
    echo "Error: Unknown strategy '$STRATEGY'. Use 'baseline' or 'M4'."
    exit 1
fi

# Check if prediction file exists
if [ ! -f "$PRED_PATH" ]; then
    echo "Error: Prediction file not found: $PRED_PATH"
    echo "Run: python3 scripts/convert_to_official_format.py --input results/bird_ves_${STRATEGY}_500.json --output $PRED_PATH"
    exit 1
fi

# Check if official repo exists
if [ ! -d "$OFFICIAL_REPO" ]; then
    echo "Error: Official BIRD mini_dev repo not found at $OFFICIAL_REPO"
    echo "Run: git clone https://github.com/bird-bench/mini_dev.git $OFFICIAL_REPO"
    exit 1
fi

echo "================================"
echo "BIRD Official Evaluation"
echo "================================"
echo "Strategy: $STRATEGY"
echo "Predictions: $PRED_PATH"
echo "Database: $DB_ROOT"
echo ""

# Create output directory
mkdir -p results/official_eval

# Run evaluation
cd "$EVAL_DIR"

echo "Running Execution Accuracy (EX) evaluation..."
python3 -u evaluation_ex.py \
    --db_root_path "$DB_ROOT" \
    --predicted_sql_path "$PRED_PATH" \
    --ground_truth_path "$GROUND_TRUTH" \
    --diff_json_path "$DIFF_JSON" \
    --num_cpus 16 \
    --meta_time_out 30.0 \
    --sql_dialect SQLite \
    --output_log_path "$(pwd)/../../results/official_eval/${STRATEGY}_ex.txt" \
    | tee "$(pwd)/../../results/official_eval/${STRATEGY}_ex_output.log"

echo ""
echo "Running R-VES evaluation..."
python3 -u evaluation_ves.py \
    --db_root_path "$DB_ROOT" \
    --predicted_sql_path "$PRED_PATH" \
    --ground_truth_path "$GROUND_TRUTH" \
    --diff_json_path "$DIFF_JSON" \
    --num_cpus 16 \
    --meta_time_out 30.0 \
    --sql_dialect SQLite \
    --output_log_path "$(pwd)/../../results/official_eval/${STRATEGY}_ves.txt" \
    | tee "$(pwd)/../../results/official_eval/${STRATEGY}_ves_output.log"

echo ""
echo "Running Soft F1 evaluation..."
python3 -u evaluation_f1.py \
    --db_root_path "$DB_ROOT" \
    --predicted_sql_path "$PRED_PATH" \
    --ground_truth_path "$GROUND_TRUTH" \
    --diff_json_path "$DIFF_JSON" \
    --num_cpus 16 \
    --meta_time_out 30.0 \
    --sql_dialect SQLite \
    --output_log_path "$(pwd)/../../results/official_eval/${STRATEGY}_f1.txt" \
    | tee "$(pwd)/../../results/official_eval/${STRATEGY}_f1_output.log"

cd - > /dev/null

echo ""
echo "================================"
echo "Evaluation Complete!"
echo "================================"
echo "Results saved to: results/official_eval/${STRATEGY}_*.txt"
echo ""
