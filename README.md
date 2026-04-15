# MicroWeaver

> 基于深度学习与 AI Agent 的单体应用微服务自动拆分工具

MicroWeaver 采用混合分析方法，结合**静态代码分析**、**动态依赖追踪**、**图神经网络编码**和 **LLM Agent 优化**，自动将单体 Java 应用拆分为微服务架构，并提供结构化评估与交互式可视化。

---

## 目录

- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
  - [阶段一：输入构建](#阶段一输入构建)
  - [阶段二：微服务划分](#阶段二微服务划分)
  - [阶段三：评估](#阶段三评估)
  - [阶段四：可视化](#阶段四可视化)
- [配置参数](#配置参数)
  - [环境变量](#环境变量)
  - [划分算法参数](#划分算法参数)
  - [编码器配置](#编码器配置)
- [数据格式](#数据格式)
- [项目结构](#项目结构)
- [内置示例应用](#内置示例应用)

---

## 系统架构

MicroWeaver 的工作流由四个阶段组成：

```
┌─────────────┐    ┌─────────────────┐    ┌────────────┐    ┌─────────────┐
│  输入构建    │───▶│  微服务划分      │───▶│   评估     │───▶│   可视化    │
│ Input Builder│    │ Microservice    │    │ Evaluation │    │Visualization│
│              │    │    Split        │    │            │    │             │
│ · 静态分析   │    │ · GNN 编码      │    │ · 结构评估  │    │ · 交互图形  │
│ · 动态分析   │    │ · 语义编码      │    │ · 语义评估  │    │ · 评估图表  │
│ · 描述生成   │    │ · 约束优化      │    │            │    │ · 数据表格  │
│ · 数据融合   │    │ · Agent 优化    │    │            │    │             │
└─────────────┘    └─────────────────┘    └────────────┘    └─────────────┘
```

---

## 环境要求

| 依赖项 | 版本要求                 |
|--------|----------------------|
| Python | 3.12+                |
| CUDA   | 12.6（推荐，用于 GPU 加速）   |
| Conda  | Miniconda / Anaconda |
| Java   | JDK 21+（静态/动态分析模块需要） |

---

## 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd MicroWeaver
```

### 2. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate MicroWeaver
```

### 3. 配置 LLM API

```bash
export DASHSCOPE_API_KEY=your-dashscope-api-key
```

### 4. 构建 Java 分析工具（可选，用于输入构建阶段）

```bash
# 静态分析器
cd src/microweaver/input_builder/static_analyze/dependency-extractor
./mvnw package -DskipTests

# 动态分析注入器
cd src/microweaver/input_builder/dynamic_analyze/skywalking-injector
mvn package -DskipTests

# 启动Skywalking动态分析后端
bash src/microweaver/input_builder/dynamic_analyze/run_dynamic_trace.sh
```

---

## 快速开始

### 方式一：使用批处理脚本（Linux/macOS）

```bash
bash run.sh
```

该脚本将依次对 5 个内置示例应用执行完整的四阶段流水线。

### 方式二：逐步执行单个应用

进入工作目录并设置环境变量：

```bash
cd src

# 设置应用名称和划分参数
export APP_NAME=daytrader
export NUM_CLUSTERS=5
export BASE_DIR=/path/to/MicroWeaver
```

依次运行四个阶段：

```bash
# 阶段一：构建输入数据
python -m microweaver.input_builder.main

# 阶段二：执行微服务划分
python -m microweaver.microservice_split.main

# 阶段三：评估划分结果
python -m microweaver.evaluation.main

# 阶段四：生成可视化
python -m microweaver.visualization.main
```

### 方式三：Windows PowerShell

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

## 详细使用说明

### 阶段一：输入构建

```bash
python -m microweaver.input_builder.main
```

该阶段负责从源代码中提取依赖关系和结构信息：

1. **静态分析**：基于 Java 源代码 AST 解析，提取类、方法、继承、调用等依赖关系
2. **动态分析**：通过 SkyWalking Agent 采集运行时调用链路，捕获动态依赖
3. **描述生成**：使用 LLM 为每个代码元素生成自然语言功能描述
4. **数据融合**：将静态和动态分析结果合并为统一的代码图 JSON 文件

**输出**：`data/inputs/<app_name>/data.json`

> 注意：项目已包含 5 个预构建的示例数据集，可跳过此阶段直接进行划分。

---

### 阶段二：微服务划分

```bash
python -m microweaver.microservice_split.main
```

这是核心划分阶段，主要步骤：

1. **结构编码**：使用多关系图注意力网络（R-GAT）编码代码依赖结构
2. **语义编码**：使用 `BAAI/bge-m3` 预训练模型提取代码语义向量
3. **特征融合**：通过注意力融合机制合并结构与语义特征
4. **约束优化**：基于 OR-Tools 求解带约束的多目标优化问题
   - 最大化结构内聚度
   - 最大化语义内聚度
   - 最小化跨服务耦合
5. **Agent 优化**（可选）：LLM Agent 对划分结果进行语义审查和调整

**输出**：`results/splits/<app_name>/microweaver/result.json`

---

### 阶段三：评估

```bash
python -m microweaver.evaluation.main
```

对划分结果进行多维度质量评估：

- **结构化评估**：计算模块化指标，包括凝聚度、耦合度等
- **语义评估**：通过 AI Agent 进行语义相似性和功能连贯性分析

**输出**：`results/reports/<app_name>/report.json`

---

### 阶段四：可视化

```bash
python -m microweaver.visualization.main
```

生成三类可视化输出：

| 输出 | 文件 | 说明 |
|------|------|------|
| 交互式架构图 | `graph_html.html` | 双层可视化，支持缩放、拖拽、钻取查看 |
| 评估图表 | `evaluate_chart.png` | 各项评估指标的可视化图表 |
| 评估表格 | `evaluate_table.png` | 详细评估数据的表格视图 |

**输出目录**：`results/viz/<app_name>/`

交互式架构图功能：
- **Level 1**：微服务总览，气泡大小表示类数量，连线表示服务间依赖
- **Level 2**：点击进入微服务内部，查看类级别依赖关系
- 支持搜索、拖拽、缩放、侧边面板详情展示

---

## 配置参数

### 环境变量

所有关键参数均可通过环境变量配置：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `APP_NAME` | `daytrader` | 目标应用名称 |
| `NUM_CLUSTERS` | `5` | 划分的微服务数量 |
| `BASE_DIR` | 项目根目录 | 项目基础路径 |
| `alpha` | `5.0` | 结构内聚权重 |
| `beta` | `1.0` | 语义内聚权重 |
| `gamma` | `3.0` | 跨服务耦合惩罚权重 |
| `beta_struct` | `1.0` | 结构编码器权重 |
| `beta_sem` | `2.0` | 语义编码器权重 |
| `beta_fused` | `1.0` | 融合编码器权重 |
| `min_size` | `5` | 每个微服务最小类数 |
| `max_size` | `35` | 每个微服务最大类数 |
| `pair_threshold` | `0.95` | 配对约束相似度阈值 |
| `time_limit` | `1200` | 求解器时间限制（秒） |
| `max_iterations` | `1` | 最大迭代次数 |
| `num_cpu` | `8` | 并行计算 CPU 核心数 |
| `ENABLE_AGENT_OPTIMIZATION` | `True` | 是否启用 AI Agent 优化 |
| `SKIP_MODEL_TRAINING` | `False` | 是否跳过模型训练 |

### 划分算法参数

目标函数由三个加权项组成：

```
Objective = α × 结构内聚 + β × 语义内聚 − γ × 跨服务耦合
```

- **增大 `alpha`**：更强调代码结构上的聚合，依赖紧密的类倾向于分到同一服务
- **增大 `beta`**：更强调功能语义的相关性，功能相近的类倾向于聚合
- **增大 `gamma`**：更严格地惩罚跨服务调用，减少服务间依赖

### 编码器配置

系统根据代码图规模自动选择编码器配置：

| 配置 | 节点数 | 隐层维度 | GNN 层数 | 注意力头 | 语义编码器冻结 |
|------|--------|---------|---------|---------|------------|
| 小规模图 | < 100 | 256 | 3 | 8 | 否 |
| 中规模图 | 100-1000 | 256 | 2 | 4 | 是 |

---

## 数据格式

### 输入格式

输入为 JSON 数组，每个元素代表一个代码节点（类/接口）：

```json
[
  {
    "id": 0,
    "name": "CustomerService",
    "qualifiedName": "com.example.CustomerService",
    "description": "负责处理与客户相关的服务操作",
    "methods": ["getCustomer", "updateCustomer"],
    "dependencies": [1, 3],
    "edge_types": ["call", "extends"],
    "javaDoc": "",
    "filePath": "src/main/java/com/example/CustomerService.java",
    "typeKind": "class"
  }
]
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 全局唯一标识符 |
| `name` | string | 类/接口简名 |
| `qualifiedName` | string | 完全限定类名 |
| `description` | string | 功能描述（由 LLM 生成或手动填写） |
| `methods` | string[] | 方法列表 |
| `dependencies` | int[] | 依赖目标节点的 `id` 列表 |
| `edge_types` | string[] | 与 `dependencies` 对应的依赖类型（`call`、`extends` 等） |
| `javaDoc` | string | JavaDoc 注释 |
| `filePath` | string | 源文件路径 |
| `typeKind` | string | 类型种类（`class`、`interface` 等） |

### 输出格式

划分结果为 JSON 对象，键为微服务名称，值为该微服务包含的类名列表：

```json
{
  "service-auth": ["AuthServiceImpl", "ConnectionManager", "TokenValidator"],
  "service-booking": ["BookingsREST", "BookingService", "FlightService"],
  "service-customer": ["CustomerService", "CustomerInfo", "CustomerDAO"]
}
```

---

## 项目结构

```
MicroWeaver/
├── data/inputs/                    # 输入数据
│   ├── acmeair/data.json          # AcmeAir 航空应用
│   ├── daytrader/                 # DayTrader 交易应用
│   │   ├── data.json
│   │   └── model/                 # 预训练编码器模型
│   ├── jpetstore/data.json        # JPetStore 宠物商店
│   ├── plants/data.json           # Plants 植物商店
│   └── trainticket/data.json      # TrainTicket 火车票系统
├── src/microweaver/
│   ├── config.py                  # 全局基础配置
│   ├── input_builder/             # 阶段一：输入构建
│   │   ├── main.py               # 入口
│   │   ├── static_analyze/        # 静态分析（Java AST 解析）
│   │   ├── dynamic_analyze/       # 动态分析（SkyWalking 追踪）
│   │   ├── generate_description.py # LLM 描述生成
│   │   └── merge.py              # 多源数据融合
│   ├── microservice_split/        # 阶段二：微服务划分
│   │   ├── main.py               # 入口
│   │   ├── config.py             # 算法配置
│   │   ├── model/                # 深度学习编码器
│   │   │   ├── code_graph_encoder.py  # GNN + 语义融合编码器
│   │   │   ├── train_structural_encoder.py
│   │   │   └── train_full_encoder.py
│   │   └── partition/            # 划分算法
│   │       ├── microservice_partition.py  # 约束优化求解
│   │       └── agent_optimize.py          # LLM Agent 优化
│   ├── evaluation/                # 阶段三：评估
│   │   ├── main.py               # 入口
│   │   ├── evaluator.py          # 评估调度器
│   │   ├── structural/           # 结构化指标评估
│   │   └── semantic/             # 语义评估（AI Agent）
│   ├── visualization/             # 阶段四：可视化
│   │   ├── main.py               # 入口
│   │   ├── graph_visualize/      # 交互式 D3.js 架构图
│   │   └── report_visualize/     # 评估报表图表
│   ├── util/                      # 工具模块
│   │   ├── env.py                # 环境变量工具
│   │   ├── file_op.py            # 文件操作
│   │   └── silent_agent.py       # Agent 静默执行
│   └── pipeline/                  # 流水线编排
├── environment.yml                # Conda 环境定义
└── run.sh                         # 批处理执行脚本
```

---

## 内置示例应用

项目在 `data/inputs/` 目录下提供了 5 个经典 Java 单体应用的预处理数据：

| 应用 | 说明 | 推荐微服务数 | 大小约束 |
|------|------|-------------|---------|
| **daytrader** | IBM DayTrader 金融交易系统 | 5 | 5 - 35 |
| **acmeair** | AcmeAir 航空预订系统 | 5 | 5 - 20 |
| **jpetstore** | JPetStore 在线宠物商店 | 5 | 5 - 20 |
| **plants** | Plants By WebSphere 植物商店 | 2 | 5 - 17 |
| **trainticket** | TrainTicket 火车票预订系统 | 10 | 5 - 50 |

可直接使用这些数据集进行微服务划分实验，无需重新执行输入构建阶段。