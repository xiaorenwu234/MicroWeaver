#!/bin/bash

# 动态追踪工具执行脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "     动态追踪工具 - 执行脚本"
echo "=========================================="
echo ""

# 步骤 1: 启动 SkyWalking 服务
echo "【步骤 1】启动 SkyWalking 服务..."
echo "------------------------------------------"
cd "${SCRIPT_DIR}/docker"
bash ./quickstart.sh
cd "${SCRIPT_DIR}"
echo "✓ SkyWalking 服务已启动"
echo ""

# 步骤 2: 静态代码注入
echo "【步骤 2】静态代码注入"
echo "------------------------------------------"
read -p "请输入项目路径: " PROJECT_PATH

if [ ! -d "$PROJECT_PATH" ]; then
    echo "错误: 项目路径不存在: $PROJECT_PATH"
    exit 1
fi

echo "正在注入 SkyWalking 代码到项目: $PROJECT_PATH"
java -jar "${SCRIPT_DIR}/skywalking-injector/target/skywalking-injector.jar" "$PROJECT_PATH"
echo "✓ 代码注入完成"
echo ""

# 步骤 3: 项目打包
echo "【步骤 3】项目打包"
echo "------------------------------------------"
echo "正在执行 Maven 打包..."
cd "$PROJECT_PATH"
mvn clean package
cd "${SCRIPT_DIR}"
echo "✓ 项目打包完成"
echo ""

# 步骤 4: 启动应用
echo "【步骤 4】启动应用（带 SkyWalking 代理）"
echo "------------------------------------------"
read -p "请输入 JAR 包路径: " JAR_PATH

if [ ! -f "$JAR_PATH" ]; then
    echo "错误: JAR 包不存在: $JAR_PATH"
    exit 1
fi

echo "正在启动应用: $JAR_PATH"
echo ""
java -javaagent:"${SCRIPT_DIR}/skywalking-agent/skywalking-agent.jar" \
     -Dskywalking.agent.service_name=train-ticket \
     -Dskywalking.collector.backend_service=localhost:11800 \
     -jar "$JAR_PATH"
