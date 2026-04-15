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
    echo "开始执行 ${app_name} 的对比实验"
    echo "配置：聚类数=${num_clusters} | 最小大小=${min_size} | 最大大小=${max_size} | 阈值=${pair_threshold} | 时间限制=${time_limit}s | CPU限制=${num_cpu}"
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
        echo "错误：无法切换到工作目录 $WORK_DIR"
        exit 1
    }

    conda activate "$CONDA_ENV"
    echo "已激活conda环境：$CONDA_ENV"

    echo -e "\n 执行获取输入命令"
    $GENERATE_INPUT_CMD

    echo -e "\n 执行 MicroWeaver 算法"
    $MicroWeaver_CMD

    echo -e "\n 执行评估命令"
    $EVALUATE_CMD
    if [ $? -ne 0 ]; then
        echo "评估代码执行失败！"
        exit 1
    fi

    echo -e "\n 执行可视化命令"
    $VISUALIZE_EMD

    # 退出环境
    conda deactivate
    echo -e "\n${app_name} 实验完成！"
}

echo "🔧 初始化conda环境..."
source "$CONDA_PATH/etc/profile.d/conda.sh" || {
    echo "❌ 错误：无法加载conda环境配置"
    exit 1
}

echo -e "\n 开始执行实验..."

run_microweaver "daytrader" 5 5 35 0.5 600 12

run_microweaver "acmeair" 5 5 20 0.5 600 12

run_microweaver "jpetstore" 5 5 20 0.5 600 12

run_microweaver "plants" 2 5 17 0.5 600 12

run_microweaver "trainticket" 10 5 50 0.95 1200 12

echo -e "\n 实验执行完成！"
exit 0