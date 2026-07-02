"""
@Author: gl
@Date: 2026/6/26
@Desc: 调用 BGE-M3 模型进行混合检索
"""
from app.infra.llm.providers import llm_providers
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger, step_log


@step_log("get_data_and_validates")
def get_data_and_validates(state: QueryGraphState) -> tuple[list[str], str]:
    """获取并校验参数"""
    # 1.1 获取数据
    item_names = state.get('item_names', [])
    rewritten_query = state.get('rewritten_query')

    # 1.2 校验
    if not item_names or not rewritten_query:
        logger.error(f"item_names或rewritten_query为空，业务无法继续！")
        raise ValueError(f"item_names或rewritten_query为空，业务无法继续！")

    # 1.3 返回结果
    return item_names, rewritten_query


@step_log("search_by_milvus")
def search_by_milvus(item_names: list[str], rewritten_query: str) -> list:
    """混合向量检索"""
    # 2.1 生成向量
    result = llm_providers.generate_embeddings([rewritten_query])

    # 2.2 构造reqs
    reqs = milvus_gateway.create_requests(
        dense_vector=result['dense'][0],
        sparse_vector=result['sparse'][0],
        expr=f"item_name in {item_names}",
        limit=5 * 2
    )

    # 2.3 混合检索
    milvus_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_weights=(0.6, 0.4),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "file_title", "title", "parent_title", "part", "content", "item_name"]
    )

    # 2.4 返回结果
    return milvus_result[0] if milvus_result else []


@step_log("deal_milvus_list")
def deal_milvus_list(milvus_list) -> list[dict]:
    """结果标准化"""
    embedding_chunks = []
    for item in milvus_list:
        entity = item.get("entity", {})
        embedding_chunks.append(
            {
                "chunk_id": entity.get('chunk_id'),
                "score": item.get('distance', 0.0),
                "title": entity.get('title'),
                "file_title": entity.get('file_title'),
                "parent_title": entity.get('parent_title'),
                "part": entity.get('part'),
                "content": entity.get('content'),
                "item_name": entity.get('item_name'),
                "source": "milvus",
                "url": ""
            }
        )

    return embedding_chunks


@step_log("search_by_embedding")
def search_by_embedding(state: QueryGraphState) -> list[dict]:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的法律法规范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """
    # 1. 获取并校验参数
    item_names, rewritten_query = get_data_and_validates(state)

    # 2. 向量检索
    milvus_list = search_by_milvus(item_names, rewritten_query)

    # 3. 格式化
    embedding_chunks = deal_milvus_list(milvus_list)

    return embedding_chunks
