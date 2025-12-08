#!/bin/bash
# ------------------------------------------------------------------------
# Sequential Training Script for All RF-DETR Experiments
# ------------------------------------------------------------------------
# This script runs all experiments sequentially, waiting for each to complete
# before starting the next one. Logs are saved to the results directory.
# ------------------------------------------------------------------------

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="/home/fyb/mydir/rf-detr/experiements/results"
PYTHON_BIN="/home/fyb/envs/torch-cuda/bin/python"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run a single experiment
run_experiment() {
    local script_name=$1
    local experiment_name=$2
    local log_file="${RESULTS_DIR}/${experiment_name}/train.log"
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Starting: ${experiment_name}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Script: ${script_name}"
    echo "Log file: ${log_file}"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Create results directory if it doesn't exist
    mkdir -p "${RESULTS_DIR}/${experiment_name}"
    
    # Run the experiment and save output to log file
    if ${PYTHON_BIN} "${SCRIPT_DIR}/${script_name}" > "${log_file}" 2>&1; then
        echo -e "${GREEN}✓ Completed: ${experiment_name}${NC}"
        echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    else
        echo -e "${RED}✗ Failed: ${experiment_name}${NC}"
        echo "Check log file: ${log_file}"
        echo ""
        exit 1
    fi
}

# Print overall start message
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}RF-DETR Sequential Experiments${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Total experiments: 4"
echo "Overall start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run experiments sequentially
run_experiment "train_baseline_p345.py" "e1-baseline_p345"
run_experiment "train_scale_aware_query_grouping_p345.py" "e1_scale_aware_query_grouping_p345"
run_experiment "train_dynamic_gating_p345.py" "e1_dynamic_gating_p345"
run_experiment "train_query_repulsion_p345.py" "e1_query_repulsion_p345"

# Print overall completion message
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All Experiments Completed Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Overall end time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results saved in: ${RESULTS_DIR}"
echo ""
echo "To view logs:"
echo "  - Baseline: tail -f ${RESULTS_DIR}/e1-baseline_p345/train.log"
echo "  - Scale-Aware Grouping: tail -f ${RESULTS_DIR}/e1_scale_aware_query_grouping_p345/train.log"
echo "  - Dynamic Gating: tail -f ${RESULTS_DIR}/e1_dynamic_gating_p345/train.log"
echo "  - Query Repulsion: tail -f ${RESULTS_DIR}/e1_query_repulsion_p345/train.log"
