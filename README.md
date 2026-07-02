<![CDATA[<div align="center">

# 🚢 Maritime Legal QA System

**基于 RAG + LangGraph 的企业级海事法律智能问答系统**

[English](#english) | [中文](#中文)

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-green?style=flat-square&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1+-purple?style=flat-square)
![Milvus](https://img.shields.io/badge/Milvus-2.x-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

[![RAG](https://img.shields.io/badge/RAG-Architecture-red?style=flat-square)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-blue?style=flat-square)]()
[![BGE-M3](https://img.shields.io/badge/BGE--M3-Embedding-green?style=flat-square)]()

</div>

---

## 中文

### 📖 项目简介

船舶海事法律智能问答系统是一套**全链路、高鲁棒性、可扩展**的企业级智能问答系统，专注于海事法律领域的知识管理和智能咨询。

系统采用 **RAG（检索增强生成）** 架构，结合 **LangGraph** 工作流编排，实现了从文档导入到智能问答的完整流程。

### ✨ 核心亮点

<table>
<tr>
<td width="50%">

#### 🔍 智能检索
- **三路并发召回**：向量检索 + HyDE + 联网搜索
- **RRF 融合排序**：多路结果智能融合
- **Rerank 精排**：BGE-reranker 语义重排序
- **动态截断**：基于分数断崖的智能 TopK

</td>
<td width="50%">

#### 🚀 工程化
- **LangGraph 编排**：图结构流程可控可追溯
- **SSE 流式输出**：实时推送用户体验流畅
- **双链路设计**：离线导入 + 在线查询分离
- **完整评估**：检索/生成/端到端评估体系

</td>
</tr>
</table>

### 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户界面层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  导入页面     │  │  查询页面     │  │  评估工具     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
├─────────┼─────────────────┼─────────────────┼───────────────────┤
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    FastAPI 接口层                        │   │
│  │         /upload  /query  /stream  /history              │   │
│  └─────────────────────────┬───────────────────────────────┘   │
├─────────────────────────────┼───────────────────────────────────┤
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               LangGraph 工作流层                         │   │
│  │  ┌─────────────┐              ┌─────────────┐           │   │
│  │  │  导入链      │              │  查询链      │           │   │
│  │  │  7 个节点    │              │  7 个节点    │           │   │
│  │  └─────────────┘              └─────────────┘           │   │
│  └─────────────────────────┬───────────────────────────────┘   │
├─────────────────────────────┼───────────────────────────────────┤
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   RAG 核心逻辑层                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │ 向量检索  │ │ HyDE检索 │ │ RRF融合  │ │ Rerank   │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └─────────────────────────┬───────────────────────────────┘   │
├─────────────────────────────┼───────────────────────────────────┤
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    基础设施层                             │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐│   │
│  │  │ LLM    │ │ Milvus │ │MongoDB │ │ MinIO  │ │MinerU  ││   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 📊 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| 🖥️ 后端框架 | FastAPI + Uvicorn | 异步接口 + SSE 流式 |
| 🔄 工作流引擎 | LangGraph + LangChain | 图结构编排，状态管理 |
| 🤖 大语言模型 | 通义千问 Qwen-Flash | 文本生成 + 意图识别 |
| 👁️ 视觉模型 | Qwen3-VL-Flash | 图片理解 + PDF 解析 |
| 📐 向量模型 | BGE-M3（1024维） | dense + sparse 混合向量 |
| 🎯 重排序模型 | BGE-reranker-large | 精细化语义排序 |
| 🗄️ 向量数据库 | Milvus | 混合检索 + 向量存储 |
| 📄 文档数据库 | MongoDB | 历史对话 + 元数据 |
| 📦 对象存储 | MinIO | 图片存储 + 文件管理 |
| 📑 PDF 解析 | MinerU | PDF → Markdown 转换 |

### 🚀 快速开始

#### 环境要求

- Python 3.11+
- uv 包管理器（推荐）
- Milvus 2.x
- MongoDB 6.x
- MinIO

#### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-username/maritime-legal-qa-system.git
cd maritime-legal-qa-system

# 2. 安装依赖
uv pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置 API 密钥和服务地址

# 4. 下载模型
python -m app.shared.tool.download_bgem3
python -m app.shared.tool.download_reranker

# 5. 启动服务
python -m app.api.http.import_server  # 导入服务 (端口 8000)
python -m app.api.http.query_server   # 查询服务 (端口 8001)
```

#### 访问地址

| 服务 | 地址 |
|------|------|
| 导入页面 | http://localhost:8000/import/html |
| 查询页面 | http://localhost:8001/html |
| API 文档 | http://localhost:8000/docs |

### 📐 核心算法

#### RRF 融合算法

```python
score = Σ(weight × 1/(k + rank))  # k=60
# 同时在多路中排名靠前的文档获得更高分数
```

#### 动态断崖截断

```python
for pre_index in range(min_topk-1, max_topk-1):
    abs_gap = pre_score - next_score
    ratio = abs_gap / pre_score
    if abs_gap > 0.2 or ratio > 0.2:
        topk = pre_index + 1
        break
```

### 📈 评估体系

系统内置完整的 RAG 评估模块：

```bash
# 检索质量评估
python -m app.rag_eval.run_eval retrieval --max-samples 10

# 生成质量评估
python -m app.rag_eval.run_eval generation --max-samples 10

# 端到端评估
python -m app.rag_eval.run_eval e2e --max-samples 10

# 全量评估
python -m app.rag_eval.run_eval all --max-samples 10 --output eval_results.json
```

| 评估维度 | 指标 |
|---------|------|
| 检索质量 | Precision@K, Recall@K, NDCG@K, MRR, MAP |
| 生成质量 | Faithfulness, Relevancy, Correctness |
| 性能 | 延迟, 吞吐量 |

### 📁 项目结构

```
maritime-legal-qa-system/
├── app/
│   ├── api/                    # API 接口层
│   │   ├── http/              # FastAPI 路由
│   │   └── schemas/           # 数据模型
│   ├── process/               # LangGraph 工作流
│   │   ├── import_/           # 导入流程
│   │   └── query/             # 查询流程
│   ├── rag/                   # 核心业务逻辑
│   │   ├── import_/           # 导入服务
│   │   └── query/             # 查询服务
│   ├── rag_eval/              # 评估模块
│   ├── infra/                 # 基础设施
│   ├── shared/                # 公共工具
│   └── resources/             # 静态资源
│       ├── prompts/           # 提示词模板
│       └── html/              # 前端页面
├── doc/                       # 测试文档
├── output/                    # 输出目录
└── .env.example               # 环境变量模板
```

### 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## English

### 📖 Overview

The Maritime Legal QA System is an **enterprise-grade, highly robust, and scalable** intelligent question-answering system focused on knowledge management and intelligent consulting in the maritime law domain.

The system adopts the **RAG (Retrieval-Augmented Generation)** architecture combined with **LangGraph** workflow orchestration, implementing a complete pipeline from document import to intelligent Q&A.

### ✨ Key Features

<table>
<tr>
<td width="50%">

#### 🔍 Intelligent Retrieval
- **Triple Recall**: Vector search + HyDE + Web search
- **RRF Fusion**: Multi-path result fusion
- **Rerank**: BGE-reranker semantic reranking
- **Dynamic Truncation**: Score-cliff based TopK

</td>
<td width="50%">

#### 🚀 Engineering Excellence
- **LangGraph Orchestration**: Traceable graph workflow
- **SSE Streaming**: Real-time response delivery
- **Dual Pipeline**: Offline import + Online query
- **Full Evaluation**: Retrieval/Generation/E2E metrics

</td>
</tr>
</table>

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Import Page  │  │ Query Page   │  │ Eval Tools   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
├─────────┼─────────────────┼─────────────────┼───────────────────┤
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Layer                         │   │
│  │         /upload  /query  /stream  /history              │   │
│  └─────────────────────────┬───────────────────────────────┘   │
├─────────────────────────────┼───────────────────────────────────┤
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               LangGraph Workflow Layer                   │   │
│  │  ┌─────────────┐              ┌─────────────┐           │   │
│  │  │ Import Chain │              │ Query Chain  │           │   │
│  │  │  7 Nodes    │              │  7 Nodes    │           │   │
│  │  └─────────────┘              └─────────────┘           │   │
│  └─────────────────────────┬───────────────────────────────┘   │
├─────────────────────────────┼───────────────────────────────────┤
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   RAG Core Logic                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │ Vector   │ │ HyDE     │ │ RRF      │ │ Rerank   │   │   │
│  │  │ Search   │ │ Search   │ │ Fusion   │ │ Model    │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └─────────────────────────┬───────────────────────────────┘   │
├─────────────────────────────┼───────────────────────────────────┤
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Infrastructure                         │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐│   │
│  │  │ LLM    │ │ Milvus │ │MongoDB │ │ MinIO  │ │MinerU  ││   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 📊 Tech Stack

| Component | Selection | Purpose |
|-----------|-----------|---------|
| 🖥️ Backend | FastAPI + Uvicorn | Async API + SSE Streaming |
| 🔄 Workflow | LangGraph + LangChain | Graph orchestration, state management |
| 🤖 LLM | Qwen-Flash | Text generation + Intent recognition |
| 👁️ Vision | Qwen3-VL-Flash | Image understanding + PDF parsing |
| 📐 Embedding | BGE-M3 (1024d) | Dense + Sparse hybrid vectors |
| 🎯 Reranker | BGE-reranker-large | Fine-grained semantic ranking |
| 🗄️ Vector DB | Milvus | Hybrid search + Vector storage |
| 📄 Document DB | MongoDB | Chat history + Metadata |
| 📦 Object Storage | MinIO | Image storage + File management |
| 📑 PDF Parser | MinerU | PDF → Markdown conversion |

### 🚀 Quick Start

#### Prerequisites

- Python 3.11+
- uv package manager (recommended)
- Milvus 2.x
- MongoDB 6.x
- MinIO

#### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/maritime-legal-qa-system.git
cd maritime-legal-qa-system

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env file with your API keys and service addresses

# 4. Download models
python -m app.shared.tool.download_bgem3
python -m app.shared.tool.download_reranker

# 5. Start services
python -m app.api.http.import_server  # Import service (port 8000)
python -m app.api.http.query_server   # Query service (port 8001)
```

#### Access Points

| Service | URL |
|---------|-----|
| Import Page | http://localhost:8000/import/html |
| Query Page | http://localhost:8001/html |
| API Docs | http://localhost:8000/docs |

### 📐 Core Algorithms

#### RRF Fusion Algorithm

```python
score = Σ(weight × 1/(k + rank))  # k=60
# Documents ranked high in multiple paths get higher scores
```

#### Dynamic Cliff Truncation

```python
for pre_index in range(min_topk-1, max_topk-1):
    abs_gap = pre_score - next_score
    ratio = abs_gap / pre_score
    if abs_gap > 0.2 or ratio > 0.2:
        topk = pre_index + 1
        break
```

### 📈 Evaluation System

Built-in comprehensive RAG evaluation module:

```bash
# Retrieval quality evaluation
python -m app.rag_eval.run_eval retrieval --max-samples 10

# Generation quality evaluation
python -m app.rag_eval.run_eval generation --max-samples 10

# End-to-end evaluation
python -m app.rag_eval.run_eval e2e --max-samples 10

# Full evaluation
python -m app.rag_eval.run_eval all --max-samples 10 --output eval_results.json
```

| Dimension | Metrics |
|-----------|---------|
| Retrieval Quality | Precision@K, Recall@K, NDCG@K, MRR, MAP |
| Generation Quality | Faithfulness, Relevancy, Correctness |
| Performance | Latency, Throughput |

### 📁 Project Structure

```
maritime-legal-qa-system/
├── app/
│   ├── api/                    # API Layer
│   │   ├── http/              # FastAPI routes
│   │   └── schemas/           # Data models
│   ├── process/               # LangGraph Workflows
│   │   ├── import_/           # Import pipeline
│   │   └── query/             # Query pipeline
│   ├── rag/                   # Core Business Logic
│   │   ├── import_/           # Import services
│   │   └── query/             # Query services
│   ├── rag_eval/              # Evaluation Module
│   ├── infra/                 # Infrastructure
│   ├── shared/                # Common Utilities
│   └── resources/             # Static Resources
│       ├── prompts/           # Prompt Templates
│       └── html/              # Frontend Pages
├── doc/                       # Test Documents
├── output/                    # Output Directory
└── .env.example               # Environment Template
```

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

</div>
]]>