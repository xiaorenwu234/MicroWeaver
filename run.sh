#!/bin/bash
set -e

CONDA_ENV="MicroWeaver"
BASE_DIR="$(dirname "$(realpath "$0")")"
GENERATE_INPUT_CMD="python -m microweaver.input_builder.main"
MicroWeaver_CMD="python -m microweaver.microservice_split.main"
EVALUATE_CMD="python -m microweaver.evaluation.main"
VISUALIZE_EMD="python -m microweaver.visualization.main"
WORK_DIR="${BASE_DIR}/src/"

CONDA_PATH="/opt/conda"

run_microweaver() {
    local app_name=$1
    local num_clusters=$2
    local min_size=$3
    local max_size=$4
    local pair_threshold=$5
    local time_limit=$6
    local num_cpu=$7

    echo -e "\n=================================================="
    echo "Starting comparison experiment for ${app_name}"
    echo "Config: clusters=${num_clusters} | min_size=${min_size} | max_size=${max_size} | threshold=${pair_threshold} | time_limit=${time_limit}s | cpu_limit=${num_cpu}"
    echo "=================================================="

    export APP_NAME="${app_name}"
    export NUM_CLUSTERS="${num_clusters}"
    export min_size="${min_size}"
    export max_size="${max_size}"
    export pair_threshold="${pair_threshold}"
    export time_limit="${time_limit}"
    export num_cpu="${num_cpu}"
    export BASE_DIR="${BASE_DIR}"

    cd "$WORK_DIR" || {
        echo "Error: cannot switch to working directory $WORK_DIR"
        exit 1
    }

    conda activate "$CONDA_ENV"
    echo "Conda environment activated: $CONDA_ENV"

    echo -e "\n Running generate input command"
    $GENERATE_INPUT_CMD

    echo -e "\n Running MicroWeaver algorithm"
    $MicroWeaver_CMD

    echo -e "\n Running evaluation command"
    $EVALUATE_CMD
    if [ $? -ne 0 ]; then
        echo "Evaluation code execution failed!"
        exit 1
    fi

    echo -e "\n Running visualization command"
    $VISUALIZE_EMD

    # Deactivate environment
    conda deactivate
    echo -e "\n${app_name} experiment completed!"
}

echo "🔧 Initializing conda environment..."
source "$CONDA_PATH/etc/profile.d/conda.sh" || {
    echo "❌ Error: cannot load conda environment configuration"
    exit 1
}

echo -e "\n Starting experiments..."

run_microweaver "daytrader" 5 5 35 0.5 600 12

run_microweaver "acmeair" 5 5 20 0.5 600 12

run_microweaver "jpetstore" 5 5 20 0.5 600 12

run_microweaver "plants" 2 5 17 0.5 600 12

run_microweaver "trainticket" 10 5 50 0.95 1200 12

echo -e "\n All experiments completed!"
exit 0