# MicroWeaver

> Automated monolithic application microservice decomposition tool based on deep learning and AI Agent

MicroWeaver adopts a hybrid analysis approach, combining **static code analysis**, **dynamic dependency tracing**, **graph neural network encoding**, and **LLM Agent optimization** to automatically decompose monolithic Java applications into microservice architectures, providing structured evaluation and interactive visualization.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Phase 1: Input Building](#phase-1-input-building)
  - [Phase 2: Microservice Splitting](#phase-2-microservice-splitting)
  - [Phase 3: Evaluation](#phase-3-evaluation)
  - [Phase 4: Visualization](#phase-4-visualization)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Partition Algorithm Parameters](#partition-algorithm-parameters)
  - [Encoder Configuration](#encoder-configuration)
- [Data Format](#data-format)
- [Project Structure](#project-structure)
- [Built-in Sample Applications](#built-in-sample-applications)

---

## System Architecture

MicroWeaver's workflow consists of four phases:

```
┌─────────────┐    ┌─────────────────┐    ┌────────────┐    ┌─────────────┐
│ Input Build │───▶│ Microservice    │───▶│ Evaluation │───▶│Visualization│
│Input Builder│    │    Split        │    │            │    │             │
│             │    │                 │    │            │    │             │
│ · Static    │    │ · GNN Encode    │    │ · Struct   │    │ · Interactive│
│ · Dynamic   │    │ · Semantic      │    │ · Semantic │    │ · Charts    │
│ · Desc Gen  │    │ · Constraint    │    │            │    │ · Tables    │
│ · Data Merge│    │ · Agent Opt     │    │            │    │             │
└─────────────┘    └─────────────────┘    └────────────┘    └─────────────┘
```

---

## Requirements

| Dependency | Version Requirement |
|------------|---------------------|
| Python | 3.12+ |
| CUDA | 12.6 (recommended, for GPU acceleration) |
| Conda | Miniconda / Anaconda |
| Java | JDK 21+ (required for static/dynamic analysis modules) |

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd MicroWeaver
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate MicroWeaver
```

### 3. Configure LLM API

```bash
export DASHSCOPE_API_KEY=your-dashscope-api-key
```

### 4. Build Java Analysis Tools (Optional, for Input Building phase)

```bash
# Static analyzer
cd src/microweaver/input_builder/static_analyze/dependency-extractor
./mvnw package -DskipTests

# Dynamic analysis injector
cd src/microweaver/input_builder/dynamic_analyze/skywalking-injector
mvn package -DskipTests

# Start Skywalking dynamic analysis backend
bash src/microweaver/input_builder/dynamic_analyze/run_dynamic_trace.sh
```

---

## Quick Start

### Method 1: Using Batch Script (Linux/macOS)

```bash
bash run.sh
```

This script will execute the complete four-phase pipeline for 5 built-in sample applications in sequence.

### Method 2: Step-by-step Single Application Execution

Enter the working directory and set environment variables:

```bash
cd src

# Set application name and partition parameters
export APP_NAME=daytrader
export NUM_CLUSTERS=5
export BASE_DIR=/path/to/MicroWeaver
```

Run the four phases in sequence:

```bash
# Phase 1: Build input data
python -m microweaver.input_builder.main

# Phase 2: Execute microservice splitting
python -m microweaver.microservice_split.main

# Phase 3: Evaluate partition results
python -m microweaver.evaluation.main

# Phase 4: Generate visualization
python -m microweaver.visualization.main
```

### Method 3: Windows PowerShell

```powershell
cd src

$env:APP_NAME = "daytrader"
$env:NUM_CLUSTERS = "5"
$env:BASE_DIR = "C:\path\to\MicroWeaver"

python -m microweaver.input_builder.main
python -m microweaver.microservice_split.main
python -m microweaver.evaluation.main
python -m microweaver.visualization.main
```

---

## Detailed Usage

### Phase 1: Input Building

```bash
python -m microweaver.input_builder.main
```

This phase is responsible for extracting dependency relationships and structural information from source code:

1. **Static Analysis**: Based on Java source code AST parsing, extract dependencies such as classes, methods, inheritance, and calls
2. **Dynamic Analysis**: Collect runtime call chains through SkyWalking Agent to capture dynamic dependencies
3. **Description Generation**: Use LLM to generate natural language functional descriptions for each code element
4. **Data Fusion**: Merge static and dynamic analysis results into a unified code graph JSON file

**Output**: `data/inputs/<app_name>/data.json`

> Note: The project includes 5 pre-built sample datasets. You can skip this phase and proceed directly to splitting.

---

### Phase 2: Microservice Splitting

```bash
python -m microweaver.microservice_split.main
```

This is the core splitting phase. Main steps:

1. **Structural Encoding**: Use multi-relational graph attention network (R-GAT) to encode code dependency structure
2. **Semantic Encoding**: Use `BAAI/bge-m3` pre-trained model to extract code semantic vectors
3. **Feature Fusion**: Merge structural and semantic features through attention fusion mechanism
4. **Constraint Optimization**: Solve constrained multi-objective optimization problems based on OR-Tools
   - Maximize structural cohesion
   - Maximize semantic cohesion
   - Minimize cross-service coupling
5. **Agent Optimization** (Optional): LLM Agent performs semantic review and adjustment of partition results

**Output**: `results/splits/<app_name>/microweaver/result.json`

---

### Phase 3: Evaluation

```bash
python -m microweaver.evaluation.main
```

Perform multi-dimensional quality evaluation of partition results:

- **Structural Evaluation**: Calculate modularity metrics, including cohesion and coupling
- **Semantic Evaluation**: Perform semantic similarity and functional coherence analysis through AI Agent

**Output**: `results/reports/<app_name>/report.json`

---

### Phase 4: Visualization

```bash
python -m microweaver.visualization.main
```

Generate three types of visualization outputs:

| Output | File | Description |
|--------|------|-------------|
| Interactive Architecture Diagram | `graph_html.html` | Two-level visualization, supports zoom, drag, and drill-down |
| Evaluation Charts | `evaluate_chart.png` | Visualization charts for various evaluation metrics |
| Evaluation Table | `evaluate_table.png` | Table view of detailed evaluation data |

**Output Directory**: `results/viz/<app_name>/`

Interactive architecture diagram features:
- **Level 1**: Microservice overview, bubble size indicates class count, connections indicate service dependencies
- **Level 2**: Click to enter microservice internals, view class-level dependency relationships
- Supports search, drag, zoom, and side panel detail display

---

## Configuration

### Environment Variables

All key parameters can be configured through environment variables:

| Environment Variable | Default | Description |
|---------|--------|------|
| `APP_NAME` | `daytrader` | Target application name |
| `NUM_CLUSTERS` | `5` | Number of microservices to partition |
| `BASE_DIR` | Project root | Project base path |
| `alpha` | `5.0` | Structural cohesion weight |
| `beta` | `1.0` | Semantic cohesion weight |
| `gamma` | `3.0` | Cross-service coupling penalty weight |
| `beta_struct` | `1.0` | Structural encoder weight |
| `beta_sem` | `2.0` | Semantic encoder weight |
| `beta_fused` | `1.0` | Fused encoder weight |
| `min_size` | `5` | Minimum classes per microservice |
| `max_size` | `35` | Maximum classes per microservice |
| `pair_threshold` | `0.95` | Pair constraint similarity threshold |
| `time_limit` | `1200` | Solver time limit (seconds) |
| `max_iterations` | `1` | Maximum iterations |
| `num_cpu` | `8` | Parallel computation CPU cores |
| `ENABLE_AGENT_OPTIMIZATION` | `True` | Whether to enable AI Agent optimization |
| `SKIP_MODEL_TRAINING` | `False` | Whether to skip model training |

### Partition Algorithm Parameters

The objective function consists of three weighted terms:

```
Objective = α × Structural Cohesion + β × Semantic Cohesion − γ × Cross-service Coupling
```

- **Increase `alpha`**: Emphasizes code structural aggregation, classes with tight dependencies tend to be assigned to the same service
- **Increase `beta`**: Emphasizes functional semantic relevance, classes with similar functionality tend to aggregate
- **Increase `gamma`**: More strictly penalizes cross-service calls, reducing inter-service dependencies

### Encoder Configuration

The system automatically selects encoder configuration based on code graph scale:

| Configuration | Node Count | Hidden Dim | GNN Layers | Attention Heads | Semantic Encoder Frozen |
|------|--------|---------|---------|---------|------------|
| Small Graph | < 100 | 256 | 3 | 8 | No |
| Medium Graph | 100-1000 | 256 | 2 | 4 | Yes |

---

## Data Format

### Input Format

Input is a JSON array, where each element represents a code node (class/interface):

```json
[
  {
    "id": 0,
    "name": "CustomerService",
    "qualifiedName": "com.example.CustomerService",
    "description": "Responsible for handling customer-related service operations",
    "methods": ["getCustomer", "updateCustomer"],
    "dependencies": [1, 3],
    "edge_types": ["call", "extends"],
    "javaDoc": "",
    "filePath": "src/main/java/com/example/CustomerService.java",
    "typeKind": "class"
  }
]
```

**Field Descriptions:**

| Field | Type | Description |
|------|------|------|
| `id` | int | Globally unique identifier |
| `name` | string | Class/interface short name |
| `qualifiedName` | string | Fully qualified class name |
| `description` | string | Functional description (generated by LLM or manually filled) |
| `methods` | string[] | Method list |
| `dependencies` | int[] | List of `id`s of dependent target nodes |
| `edge_types` | string[] | Dependency types corresponding to `dependencies` (`call`, `extends`, etc.) |
| `javaDoc` | string | JavaDoc comments |
| `filePath` | string | Source file path |
| `typeKind` | string | Type kind (`class`, `interface`, etc.) |

### Output Format

Partition result is a JSON object, where keys are microservice names and values are lists of class names contained in that microservice:

```json
{
  "service-auth": ["AuthServiceImpl", "ConnectionManager", "TokenValidator"],
  "service-booking": ["BookingsREST", "BookingService", "FlightService"],
  "service-customer": ["CustomerService", "CustomerInfo", "CustomerDAO"]
}
```

---

## Project Structure

```
MicroWeaver/
├── data/inputs/                    # Input data
│   ├── acmeair/data.json          # AcmeAir aviation application
│   ├── daytrader/                 # DayTrader trading application
│   │   ├── data.json
│   │   └── model/                 # Pre-trained encoder models
│   ├── jpetstore/data.json        # JPetStore pet store
│   ├── plants/data.json           # Plants plant store
│   └── trainticket/data.json      # TrainTicket train ticket system
├── src/microweaver/
│   ├── config.py                  # Global base configuration
│   ├── input_builder/             # Phase 1: Input Building
│   │   ├── main.py               # Entry point
│   │   ├── static_analyze/        # Static analysis (Java AST parsing)
│   │   ├── dynamic_analyze/       # Dynamic analysis (SkyWalking tracing)
│   │   ├── generate_description.py # LLM description generation
│   │   └── merge.py              # Multi-source data fusion
│   ├── microservice_split/        # Phase 2: Microservice Splitting
│   │   ├── main.py               # Entry point
│   │   ├── config.py             # Algorithm configuration
│   │   ├── model/                # Deep learning encoders
│   │   │   ├── code_graph_encoder.py  # GNN + semantic fusion encoder
│   │   │   ├── train_structural_encoder.py
│   │   │   └── train_full_encoder.py
│   │   └── partition/            # Partition algorithms
│   │       ├── microservice_partition.py  # Constraint optimization solver
│   │       └── agent_optimize.py          # LLM Agent optimization
│   ├── evaluation/                # Phase 3: Evaluation
│   │   ├── main.py               # Entry point
│   │   ├── evaluator.py          # Evaluation scheduler
│   │   ├── structural/           # Structural metrics evaluation
│   │   └── semantic/             # Semantic evaluation (AI Agent)
│   ├── visualization/             # Phase 4: Visualization
│   │   ├── main.py               # Entry point
│   │   ├── graph_visualize/      # Interactive D3.js architecture diagrams
│   │   └── report_visualize/     # Evaluation report charts
│   ├── util/                      # Utility modules
│   │   ├── env.py                # Environment variable utilities
│   │   ├── file_op.py            # File operations
│   │   └── silent_agent.py       # Agent silent execution
│   └── pipeline/                  # Pipeline orchestration
├── environment.yml                # Conda environment definition
└── run.sh                         # Batch execution script
```

---

## Built-in Sample Applications

The project provides pre-processed data for 5 classic Java monolithic applications in the `data/inputs/` directory:

| Application | Description | Recommended Microservices | Size Constraints |
|------|------|-------------|---------|
| **daytrader** | IBM DayTrader financial trading system | 5 | 5 - 35 |
| **acmeair** | AcmeAir aviation booking system | 5 | 5 - 20 |
| **jpetstore** | JPetStore online pet store | 5 | 5 - 20 |
| **plants** | Plants By WebSphere plant store | 2 | 5 - 17 |
| **trainticket** | TrainTicket train ticket booking system | 10 | 5 - 50 |

You can use these datasets directly for microservice splitting experiments without re-executing the input building phase.