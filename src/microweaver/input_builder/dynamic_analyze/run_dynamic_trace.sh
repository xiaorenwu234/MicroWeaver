#!/bin/bash

# Dynamic tracing tool execution script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "     Dynamic Tracing Tool - Execution Script"
echo "=========================================="
echo ""

# Step 1: Start SkyWalking service
echo "[Step 1] Starting SkyWalking service..."
echo "------------------------------------------"
cd "${SCRIPT_DIR}/docker"
bash ./quickstart.sh
cd "${SCRIPT_DIR}"
echo "✓ SkyWalking service started"
echo ""

# Step 2: Static code injection
echo "[Step 2] Static code injection"
echo "------------------------------------------"
read -p "Please enter project path: " PROJECT_PATH

if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: Project path does not exist: $PROJECT_PATH"
    exit 1
fi

echo "Injecting SkyWalking code into project: $PROJECT_PATH"
java -jar "${SCRIPT_DIR}/skywalking-injector/target/skywalking-injector.jar" "$PROJECT_PATH"
echo "✓ Code injection completed"
echo ""

# Step 3: Project packaging
echo "[Step 3] Project packaging"
echo "------------------------------------------"
echo "Running Maven packaging..."
cd "$PROJECT_PATH"
mvn clean package
cd "${SCRIPT_DIR}"
echo "✓ Project packaging completed"
echo ""

# Step 4: Start application
echo "[Step 4] Start application (with SkyWalking agent)"
echo "------------------------------------------"
read -p "Please enter JAR path: " JAR_PATH

if [ ! -f "$JAR_PATH" ]; then
    echo "Error: JAR file does not exist: $JAR_PATH"
    exit 1
fi

echo "Starting application: $JAR_PATH"
echo ""
java -javaagent:"${SCRIPT_DIR}/skywalking-agent/skywalking-agent.jar" \
     -Dskywalking.agent.service_name=train-ticket \
     -Dskywalking.collector.backend_service=localhost:11800 \
     -jar "$JAR_PATH"
