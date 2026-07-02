# CLAUDE.md — 船舶海事法律智能问答系统 RAG 项目参考

> 本文件是船舶海事法律智能问答系统项目的架构参考文档，供新 RAG 项目开发时参考。

## 项目概述

船舶海事法律智能问答系统是一套基于 **RAG 架构 + LangGraph 工作流编排** 的企业级私有知识库智能问答系统，5500+ 行代码。

核心定位：全链路、高鲁棒性、可扩展的企业级智能问答系统。

## 技术栈速查

| 组件 | 选型 | 用途 |
|------|------|------|
| 后端 | FastAPI + Uvicorn | 异步接口 + SSE 流式 |
| 工作流 | LangGraph + LangChain | 图结构编排，状态管理 |
| 大模型 | 通义千问 Qwen-Flash / Qwen3-VL-Flash | 文本生成 + 图片理解 |
| 向量模型 | BGE-M3（1024维） | dense + sparse 混合向量 |
| 重排序 | BGE-reranker-large（512 token） | 精细化排序 |
| 向量库 | Milvus | 混合检索 |
| 文档数据库 | MongoDB | 历史对话 |
| 对象存储 | MinIO | 图片存储 |
| PDF解析 | MinerU | PDF→Markdown |

## 架构模式

### 双链路设计

```
离线导入链（7节点）：文档 → 解析 → 图片增强 → 切分 → 主体识别 → 向量化 → 入库
在线查询链（7节点）：提问 → 意图识别 → 三路召回 → RRF融合 → Rerank → 答案生成
```

### 分层架构

```
app/
├── api/http/          # FastAPI 接口层
├── api/schema/        # Pydantic 数据模型
├── process/           # LangGraph 节点编排（轻量，只做调度）
│   ├── import_/agent/ # 导入图节点
│   └── query/agent/   # 查询图节点
├── rag/               # 核心业务逻辑
│   ├── import_/       # 切分、主体识别、向量化
│   └── query/         # RRF、Rerank、答案生成
├── infra/             # 基础设施门面
│   ├── llm/providers.py    # LLMProvider（chat/vision/embedding/reranker）
│   ├── vector_store/       # MilvusGateway
│   ├── object_storage/     # MinioGateway
│   └── persistence/        # HistoryRepository
├── shared/            # 公共工具
│   ├── config/        # 环境变量配置类
│   ├── clients/       # Milvus/MinIO/MongoDB 客户端
│   ├── model/         # 模型加载工具（embedding_utils/lm_utils/reranker_utils）
│   ├── runtime/       # 日志 + 提示词加载
│   └── utils/         # 工具函数（SSE/Task/正则/路径）
└── rag_eval/          # RAG 评估模块
```

**关键原则**：
- `process` 层只做节点调度和 state 管理，不写业务逻辑
- `rag` 层写核心业务逻辑
- `infra` 层封装外部依赖调用（MilvusGateway/LLMProvider）
- `shared` 层提供通用工具

### LangGraph State 设计

```python
# 导入链 State
class ImportGraphState(TypedDict):
    task_id: str
    local_file_path: str
    md_path: str; is_md_read_enabled: bool
    pdf_path: str; is_pdf_read_enabled: bool
    file_title: str; local_dir: str
    md_content: str
    chunks: list          # [{content, title, file_title, parent_title, part}]
    item_name: str
    embeddings_content: list  # [{..., dense_vector, sparse_vector}]

# 查询链 State
class QueryGraphState(TypedDict):
    session_id: str; original_query: str
    embedding_chunks: list; hyde_embedding_chunks: list; web_search_docs: list
    rrf_chunks: list; reranked_docs: list
    prompt: str; answer: str
    item_names: list; rewritten_query: str
    history: list; is_stream: bool; image_urls: list
```

**并发节点注意**：LangGraph 并发节点不能写入相同的 key，否则报 InvalidUpdateError。每个并发节点只返回自己修改的 key。

## 核心设计决策

### 1. 为什么用 LangGraph 不用原生 Agent？

企业要求：流程可控、可追溯、可审计、可复现。原生 Agent 自主决策不可预测。LangGraph 通过显式图结构（节点+边+条件路由）保证确定性执行。

### 2. 为什么选 BGE-M3？

单一模型同时输出 dense + sparse 向量，支持混合检索。1024 维，8192 token 上下文，130+ 语言。本地加载，无 API 调用风险。

### 3. 文档切分参数

```python
CHUNK_SIZE = 600       # 基准长度（通过评估系统调试）
CHUNK_OVERLAP = 50     # 重叠（8%-12%）
CHUNK_MIN = 400        # 短合并阈值
CHUNK_MAX_SIZE = 1000  # 最大上限
```

切割流程：标题粗切 → 递归细切（>600）→ 短合并（<400 同标题）→ 补全属性

### 4. 双层索引

- **kb_item_names**（文档级）：主体名称索引，检索前置过滤
- **kb_chunks**（切片级）：核心知识表，参与检索+排序+生成

逻辑：先圈定主体范围，再精准匹配切片。

### 5. 三路召回 + RRF + Rerank

```
向量检索（保基础）  ─┐
HyDE检索（补弱意图）─┼→ RRF融合(k=60) → Rerank精排 → 动态断崖截断
MCP联网（扩边界）  ─┘（直接进Rerank）
```

- RRF 只融合同源两路（向量+HyDE），基于排名
- Rerank 统一打分所有来源（本地+网络），基于语义
- 动态截断：分差>0.2 或比例>20% 时截断，min=2, max=6

### 6. 主体识别（item_name）

LLM 从 chunks 前 10 个切片识别文档主体 → 回填每个 chunk → 向量化入库。查询时先验证主体置信度（≥0.7 确认 / 0.6-0.7 可选 / <0.6 拒绝）。

### 7. SSE 流式输出

- 前端提问后立即返回，后台异步执行查询图
- 前端连接 SSE 流式接口，通过 Queue.get() 阻塞接收事件
- 事件类型：progress（节点进度）/ delta（增量）/ final（完整答案+图片）/ error

## 关键实现细节

### Milvus 混合检索

```python
# 创建请求
reqs = milvus_gateway.create_requests(
    dense_vector=dense, sparse_vector=sparse,
    expr=f"item_name in {item_names}", limit=10
)
# 执行混合检索
result = milvus_gateway.hybrid_search(
    collection_name="kb_chunks", reqs=reqs,
    ranker_weights=(0.6, 0.4),  # dense 0.6 + sparse 0.4
    norm_score=True, limit=5
)
```

### BGE-M3 向量化

```python
# 归一化 + 混合向量
model = BGEM3EmbeddingFunction(model_name="BAAI/bge-m3", device="cpu", use_fp16=False, normalize_embeddings=True)
embeddings = model.encode_documents(texts)
# dense: list[list[float]], sparse: CSR → dict[{index: weight}]
```

### Rerank 超长压缩

```python
# reranker 上下文 512 token
if q_tokens + a_tokens + 4 > 512:
    limit = max(50, int((512 - 4 - q_tokens) / 1.3))
    answer = llm_compress(answer, limit)  # 只压缩用于打分，原始用于生成
```

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

## 面试高频考点

### 项目相关

1. 切块大小为什么是 600？（评估系统调试 + 模型上下文约束）
2. HyDE 是什么？为什么需要？（弱语义查询增强）
3. RRF 和 Rerank 的区别？（排名融合 vs 语义打分）
4. 动态 TopK 怎么实现的？（断崖检测）
5. 三层保障 LLM 返回 JSON？（提示词+json_mode+JsonOutputParser+key校验）
6. Rerank 超长怎么处理？（LLM 压缩，只用于打分）
7. 并发节点 state 冲突？（每个节点只返回自己修改的 key）
8. SSE 和 WebSocket 区别？（单向 vs 双向）
9. MCP 是什么？（模型上下文协议，共享 tools/resources/prompt）
10. 四层 RAG 评估？（embedding→hyde→rrf→rerank 各层 precision/recall）

### 基础相关

11. Transformer 架构 + QKV + 多头注意力
12. GQA/MQA 区别
13. SFT/DPO/PPO/GRPO/KTO 区别
14. LoRA 原理 + 参数（r, target_modules, lora_alpha）
15. AdamW 优化器
16. DeepSpeed-Zero 三阶段
17. Top P / Top K / Temperature
18. Lost in the Middle 现象

## 开发规范

- Python 3.11+，使用 uv 管理依赖
- 日志：`from app.shared.runtime.logger import logger`
- 提示词：`from app.shared.runtime.load_prompt import load_prompt`，文件在 `app/resources/prompts/`
- 节点装饰器：`@node_log("node_name")` 记录节点执行
- 步骤装饰器：`@step_log("step_name")` 记录业务步骤
- 任务状态：`add_running_task(task_id, "节点名")` / `add_done_task(task_id, "节点名")`
- 环境变量：`.env` 文件，配置类在 `app/shared/config/`
- Frontmatter：`tags: [项目, AI, RAG]`，`created: YYYY-MM-DD`
