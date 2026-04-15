# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

SW_STORAGE=

SW_VERSION=${SW_VERSION:-10.3.0}
SW_BANYANDB_VERSION=${SW_BANYANDB_VERSION:-0.9.0}

usage() {
  echo "Usage: quickstart-docker.sh [-f]"
  echo "  -f  Run in the foreground"
  exit 1
}

while getopts "hf" opt; do
  case $opt in
    f)
      DETACHED=false
      ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

echo "Checking if Docker is installed..."
if ! [ -x "$(command -v docker)" ]; then
  echo "Docker is not found. Please make sure Docker is installed and the docker command is available in PATH."
  exit 1
fi

temp_dir=$(mktemp -d)

cp ./docker-compose.yml "$temp_dir/docker-compose.yml"

SW_STORAGE=elasticsearch
export OAP_IMAGE=apache/skywalking-oap-server:${SW_VERSION}
export UI_IMAGE=apache/skywalking-ui:${SW_VERSION}

echo "Installing SkyWalking ${SW_VERSION} with ${SW_STORAGE} storage..."

docker compose -f "$temp_dir/docker-compose.yml" \
  --project-name=skywalking-quickstart \
  --profile=$SW_STORAGE \
  up \
  --detach=${DETACHED:-true} \
  --wait

echo "SkyWalking is now running. You can send telemetry data to localhost:11800 and access the UI at http://localhost:8080."

echo "To find SkyWalking Docs, follow the link to our documentation site https://skywalking.apache.org/docs/."

echo "To stop SkyWalking, run 'docker compose --project-name=skywalking-quickstart down'."
