"""
@Author: gl
@Date: 2026/6/26
@Desc: 实现 RRF 算法
"""

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import NODE_RRF_LIMIT_TOP, NODE_RRF_K
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state: QueryGraphState):
    """获取并校验参数"""
    # 1. 获取参数
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])

    # 2. 校验
    if not embedding_chunks or not hyde_embedding_chunks:
        logger.error(f"embedding_chunks与hyde_embedding_chunks数据为空，无法继续业务！")
        raise ValueError(f"embedding_chunks与hyde_embedding_chunks数据为空，无法继续业务！")

    # 3. 返回结果
    return embedding_chunks, hyde_embedding_chunks

@step_log("use_by_rrf")
def use_by_rrf(rrf_list: list, top: int = NODE_RRF_LIMIT_TOP, k: int = NODE_RRF_K):
    """使用 RRF 算法排名"""
    # 3.1 定义字典
    score_dict: dict[str, float] = {}  # 存储每个 chunk_id 的总融合分数
    chunk_dict: dict[str, float] = {}  # 存储每个 chunk_id 对应的原始片段信息

    # 3.2 遍历每一路检索结果及其权重
    for weight, chunks_list in rrf_list:
        # 遍历当前路的所有片段
        for rank, chunk in enumerate(chunks_list, 1):
            chunk_id = chunk.get('chunk_id')
            if not chunk_id:
                continue
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (k + rank)) * weight
            chunk_dict.setdefault(chunk_id, chunk)

    # 3.3 组装最终结果：把分数和原文信息合并
    chunk_list = []
    for chunk_id, score in score_dict.items():
        chunk = chunk_dict.get('chunk_id', {}).copy()
        chunk['score'] = score
        chunk_list.append(chunk)

    # 3.4 融合分数从高到低排序
    chunk_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)

    # 3.5 返回 Top K 条最终融合结果
    return chunk_list[: top]

@step_log("fuse_by_rrf")
def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    """
    # 1. 获取并校验参数
    embedding_chunks, hyde_embedding_chunks = get_data_and_validates(state)

    # 2. 构造 RRF 入参：(结果列表, 权重)
    rrf_list = [
        (1.0, embedding_chunks),
        (1.0, hyde_embedding_chunks)
    ]

    # 3. 使用 RRF 算法进行数据处理
    rrf_chunks = use_by_rrf(rrf_list)

    # 4. 更新state
    state['rrf_chunks'] = rrf_chunks

    return state
