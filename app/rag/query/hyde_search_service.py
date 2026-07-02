"""
@Author: gl
@Date: 2026/6/26
@Desc: 实现 HyDE 策略（生成假设性答案 → 用答案检索）
"""
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_providers
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
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


@step_log("call_llm_answer")
def call_llm_answer(rewritten_query) -> str:
    """调用模型获取答案"""
    # 2.1 获取模型客户端对象
    llm_client = llm_providers.chat()

    # 2.2 加载提示词
    prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)

    # 2.3 封装messages
    messages = [HumanMessage(content=prompt)]

    # 2.4 构建chains
    chains = llm_client | StrOutputParser()

    # 2.5 调用大模型
    answer = chains.invoke(messages)

    return answer


@step_log("search_by_milvus")
def search_by_milvus(item_names: list[str], rewritten_query: str, llm_answer: str):
    """混合向量检索"""
    # 3.1 生成向量
    result = llm_providers.generate_embeddings([rewritten_query + ":" + llm_answer])

    # 3.2 构造reqs
    reqs = milvus_gateway.create_requests(
        dense_vector=result['dense'][0],
        sparse_vector=result['sparse'][0],
        expr=f"item_name in {item_names}",
        limit=5 * 2
    )

    # 3.3 混合检索
    milvus_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_weights=(0.6, 0.4),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "file_title", "title", "parent_title", "part", "content", "item_name"]
    )

    # 3.4 返回结果
    return milvus_result[0] if milvus_result else []


@step_log("deal_milvus_list")
def deal_milvus_list(milvus_list) -> list[dict]:
    """结果标准化"""
    hyde_embedding_chunks = []
    for item in milvus_list:
        entity = item.get("entity", {})
        hyde_embedding_chunks.append(
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

    return hyde_embedding_chunks


@step_log("search_by_hyde")
def search_by_hyde(state: QueryGraphState) -> list[dict]:
    """
    HyDE 检索服务：
    1. 让 LLM 基于问题虚构一个"理想答案"
    2. 对这个假设性答案进行向量化
    3. 用答案向量在 Milvus 中检索真实文档
    4. 回写 hyde_embedding_chunks
    """
    # 1. 获取并校验参数
    item_names, rewritten_query = get_data_and_validates(state)

    # 2. 模型获取答案
    llm_answer = call_llm_answer(rewritten_query)

    # 3. 向量混合检索
    milvus_list = search_by_milvus(item_names, rewritten_query, llm_answer)

    # 3. 格式化
    hyde_embedding_chunks = deal_milvus_list(milvus_list)

    return hyde_embedding_chunks
