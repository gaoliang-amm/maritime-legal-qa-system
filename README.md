# 船舶海事法律智能问答系统

基于 RAG 架构 + LangGraph 工作流编排的企业级私有知识库智能问答系统

## 项目简介

船舶海事法律智能问答系统是一套全链路、高鲁棒性、可扩展的企业级智能问答系统，专注于海事法律领域的知识管理和智能咨询。系统采用 RAG（检索增强生成）架构，结合 LangGraph 工作流编排，实现了从文档导入到智能问答的完整流程。

### 核心特性

- **双链路设计**：离线导入链 + 在线查询链，流程清晰可控
- **三路召回**：向量检索 + HyDE 检索 + MCP 联网搜索，覆盖全面
- **智能融合**：RRF 排名融合 + Rerank 语义精排，结果精准
- **动态截断**：基于分数断崖的智能 TopK 截取，避免噪声
- **流式输出**：SSE 实时推送，用户体验流畅
- **历史记录**：MongoDB 持久化对话历史，支持上下文追问

## 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| 后端框架 | FastAPI + Uvicorn | 异步接口 + SSE 流式 |
| 工作流引擎 | LangGraph + LangChain | 图结构编排，状态管理 |
| 大语言模型 | 通义千问 Qwen-Flash | 文本生成 + 意图识别 |
| 视觉模型 | Qwen3-VL-Flash | 图片理解 + PDF 解析 |
| 向量模型 | BGE-M3（1024维） | dense + sparse 混合向量 |
| 重排序模型 | BGE-reranker-large | 精细化语义排序 |
| 向量数据库 | Milvus | 混合检索 + 向量存储 |
| 文档数据库 | MongoDB | 历史对话 + 元数据 |
| 对象存储 | MinIO | 图片存储 + 文件管理 |
| PDF 解析 | MinerU | PDF → Markdown 转换 |

## 系统架构

### 分层架构

```
app/
├── api/                    # API 接口层
│   ├── http/              # FastAPI 路由处理
│   └── schemas/           # Pydantic 数据模型
├── process/               # LangGraph 工作流层
│   ├── import_/           # 导入流程编排
│   └── query/             # 查询流程编排
├── rag/                   # 核心业务逻辑层
│   ├── import_/           # 导入业务服务
│   └── query/             # 查询业务服务
├── infra/                 # 基础设施层
│   ├── llm/               # LLM 服务封装
│   ├── vectorstore/       # 向量库服务封装
│   ├── object_storage/    # 对象存储服务封装
│   └── persistence/       # 持久化服务封装
├── shared/                # 公共工具层
│   ├── config/            # 配置管理
│   ├── clients/           # 客户端工具
│   ├── model/             # 模型加载工具
│   ├── runtime/           # 运行时工具
│   └── utils/             # 通用工具函数
└── resources/             # 静态资源
    ├── prompts/           # 提示词模板
    └── html/              # 前端页面
```

### 双链路流程

#### 离线导入链（7 个节点）

```
文档上传 → 文件类型判断 → PDF 解析/Markdown 读取 → 图片增强处理
    → 文档智能切分 → 主体名称识别 → BGE-M3 向量化 → Milvus 入库
```

#### 在线查询链（7 个节点）

```
用户提问 → 主体名称确认 → 三路并发召回:
    ├── 向量检索（基础召回）
    ├── HyDE 检索（弱意图增强）
    └── MCP 联网搜索（边界扩展）
→ RRF 融合排序 → Rerank 精细化排序 → 动态 TopK 截断 → 答案生成
```

## 快速开始

### 环境要求

- Python 3.11+
- uv 包管理器（推荐）
- Milvus 2.x
- MongoDB 6.x
- MinIO

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd maritime-legal-qa-system
```

2. **安装依赖**

```bash
# 使用 uv（推荐）
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

3. **配置环境变量**

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置以下关键参数：
# - OPENAI_API_KEY：大模型 API 密钥
# - MILVUS_URL：Milvus 服务地址
# - MONGO_URL：MongoDB 连接地址
# - MINIO_ENDPOINT：MinIO 服务地址
```

4. **下载模型**

```bash
# 下载 BGE-M3 嵌入模型
python -m app.shared.tool.download_bgem3

# 下载 BGE-reranker 重排序模型
python -m app.shared.tool.download_reranker
```

5. **启动服务**

```bash
# 启动导入服务（端口 8000）
python -m app.api.http.import_server

# 启动查询服务（端口 8001）
python -m app.api.http.query_server
```

### 访问地址

- 导入服务 API 文档：`http://localhost:8000/docs`
- 查询服务 API 文档：`http://localhost:8001/docs`
- 导入页面：`http://localhost:8000/import/html`
- 查询页面：`http://localhost:8001/html`

## API 接口

### 导入服务（端口 8000）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/import/html` | GET | 导入页面 |
| `/upload` | POST | 上传文档文件 |
| `/status/{task_id}` | GET | 查询导入任务状态 |

### 查询服务（端口 8001）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/html` | GET | 查询页面 |
| `/health` | GET | 健康检查 |
| `/query` | POST | 智能问答（支持流式/非流式） |
| `/stream/{session_id}` | GET | SSE 流式推送 |
| `/history/{session_id}` | GET | 获取历史记录 |
| `/history/{session_id}` | DELETE | 清空历史记录 |

## 核心算法

### RRF 融合算法

```python
score = Σ(weight × 1/(k + rank))  # k=60
# 同时在多路中排名靠前的文档获得更高分数
```

### 动态断崖截断

```python
for pre_index in range(min_topk-1, max_topk-1):
    abs_gap = pre_score - next_score
    ratio = abs_gap / pre_score
    if abs_gap > 0.2 or ratio > 0.2:
        topk = pre_index + 1
        break
```

### Rerank 超长压缩

```python
# reranker 上下文 512 token
if q_tokens + a_tokens + 4 > 512:
    limit = max(50, int((512 - 4 - q_tokens) / 1.3))
    answer = llm_compress(answer, limit)  # 只压缩用于打分，原始用于生成
```

## 配置说明

### 环境变量配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `APP_ENV` | 运行环境 | dev |
| `APP_HOST` | 服务监听地址 | 0.0.0.0 |
| `IMPORT_APP_PORT` | 导入服务端口 | 8000 |
| `QUERY_APP_PORT` | 查询服务端口 | 8001 |
| `OPENAI_API_KEY` | 大模型 API 密钥 | - |
| `MILVUS_URL` | Milvus 服务地址 | http://127.0.0.1:19530 |
| `MONGO_URL` | MongoDB 连接地址 | mongodb://127.0.0.1:27017 |
| `MINIO_ENDPOINT` | MinIO 服务地址 | 127.0.0.1:9000 |

### 文档切分参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `CHUNK_SIZE` | 基准切块长度 | 600 |
| `CHUNK_OVERLAP` | 重叠长度 | 50 |
| `CHUNK_MIN` | 短合并阈值 | 400 |
| `CHUNK_MAX_SIZE` | 最大上限 | 1000 |

### Rerank 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `RERANK_MAX_TOPK` | 最大返回数量 | 6 |
| `RERANK_MIN_TOPK` | 最小返回数量 | 2 |
| `RERANK_GAP_RATIO` | 分差比例阈值 | 0.2 |
| `RERANK_GAP_ABS` | 绝对分差阈值 | 0.2 |
| `RERANK_MAX_INPUT_TOKENS` | 最大输入 token | 512 |

## 开发指南

### 项目结构规范

- `process` 层：只做节点调度和 state 管理，不写业务逻辑
- `rag` 层：写核心业务逻辑
- `infra` 层：封装外部依赖调用
- `shared` 层：提供通用工具

### 日志使用

```python
from app.shared.runtime.logger import logger

# 节点日志装饰器
@node_log("node_name")
def my_node(state):
    pass

# 步骤日志装饰器
@step_log("step_name")
def my_step():
    pass
```

### 提示词管理

```python
from app.shared.runtime.load_prompt import load_prompt

# 加载提示词模板
prompt = load_prompt("answer_out", context="...", question="...")
```

### 任务状态管理

```python
from app.shared.utils.task_utils import add_running_task, add_done_task

# 标记任务开始
add_running_task(task_id, "节点名")

# 标记任务完成
add_done_task(task_id, "节点名")
```

## 常见问题

### 1. 如何调整文档切分大小？

修改 `app/rag/import_/config.py` 中的参数：
- `CHUNK_SIZE`：基准长度，建议 400-800
- `CHUNK_OVERLAP`：重叠比例，建议 8%-12%
- `CHUNK_MIN`：短合并阈值，建议 300-500

### 2. 如何优化检索效果？

- 调整 RRF 权重：`ranker_weights=(0.6, 0.4)` 中 dense 和 sparse 的比例
- 修改 Rerank 参数：调整 `RERANK_GAP_RATIO` 和 `RERANK_GAP_ABS`
- 优化提示词：修改 `app/resources/prompts/` 中的模板

### 3. 如何添加新的文档格式？

1. 在 `app/process/import_/agent/nodes/` 中添加新的解析节点
2. 在 `app/process/import_/agent/main_graph.py` 中注册节点和边
3. 在 `node_entry` 中添加文件类型判断逻辑

### 4. 如何集成其他大模型？

1. 修改 `app/shared/config/lm_config.py` 中的配置
2. 在 `app/shared/model/lm_utils.py` 中添加模型初始化逻辑
3. 确保模型支持 OpenAI 兼容接口

## 部署建议

### 生产环境配置

- 使用 Gunicorn + Uvicorn worker 提升并发性能
- 配置 Nginx 反向代理和负载均衡
- 使用 Docker Compose 编排所有依赖服务
- 配置日志收集和监控告警

### 性能优化

- 启用 BGE-M3 的 FP16 模式减少显存占用
- 调整 Milvus 索引参数优化检索速度
- 使用 Redis 缓存热点查询结果
- 配置 CDN 加速静态资源访问

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 贡献指南

欢迎提交 Issue 和 Pull Request，共同完善项目。

## 联系方式

如有问题，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至：[your-email@example.com]
