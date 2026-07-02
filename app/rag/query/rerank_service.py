"""
@Author: gl
@Date: 2026/6/26
@Desc: 重排节点主入口
    流程：校验输入 → 合并本地+网页 → 模型打分排序 → 动态截断
    输出最终高质量候选文档列表
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from torch.nn.init import normal

from app.infra.llm.providers import llm_providers
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_MIN_SUMMARY_CHARS, RERANK_SUMMARY_CHAR_RATIO, \
    RERANK_MIN_TOPK, RERANK_MAX_TOPK, RERANK_GAP_RATIO, RERANK_GAP_ABS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log


@step_log("get_data_and_validates")
def get_data_and_validates(state: QueryGraphState):
    """获取并校验cnashu"""
    # 1. 先取出外层包裹字典
    rrf_wrap_dict = state.get("rrf_chunks", {})
    # 2. 取出真正的chunk文档列表
    rrf_chunks = rrf_wrap_dict.get("embedding_chunks", [])

    web_search_docs = state.get('web_search_docs', [])

    # 基础非空校验
    if not rrf_chunks or not web_search_docs:
        logger.error(f"rrf检索文档embedding_chunks或联网搜索数据为空，无法重排！")
        raise ValueError("rrf_chunks与web_search_docs数据为空，流程终止")

    # 强校验：列表内必须全是字典，防止脏数据
    for idx, item in enumerate(rrf_chunks):
        if not isinstance(item, dict):
            logger.error(f"rrf_chunks第{idx}项非法，类型{type(item)}，内容：{item}")
            raise TypeError("rrf列表包含非字典脏数据")

    for idx, item in enumerate(web_search_docs):
        if not isinstance(item, dict):
            logger.error(f"web_search_docs第{idx}项非法，类型{type(item)}，内容：{item}")
            raise TypeError("web搜索列表包含非字典脏数据")

    return rrf_chunks, web_search_docs


@step_log("merge_rrf_and_web")
def merge_rrf_and_web(rrf_chunks: list[dict], web_search_docs: list[dict]) -> list[dict]:
    """统一合并本地知识库结果 + 联网搜索结果"""
    final_chunk_list: list[dict] = []
    # 2.1 处理本地RRF融合结果
    for chunk in rrf_chunks or []:
        final_chunk_list.append(
            {
                "title": chunk.get('title'),
                "text": chunk.get('content'),
                "chunk_id": chunk.get('chunk_id'),
                "url": None,
                "type": 'milvus',
                "score": chunk.get('score', 0.0)
            }
        )

    # 2.2 处理联网搜索网页结果
    for doc in web_search_docs or []:
        final_chunk_list.append(
            {
                "title": doc.get('title'),
                "text": doc.get('snippet'),
                "chunk_id": doc.get('chunk_id'),
                "url": doc.get('url'),
                "type": 'web',
                "score": 0.0
            }
        )

    return final_chunk_list


@step_log("_summarize_long_rerank_text")
def _summarize_long_rerank_text(question: str, answer: str, limit: int) -> str:
    """超长文本精简"""
    prompt = load_prompt(
        """rerank_text_refine""",
        question=question,
        answer=answer,
        limit=limit
    )

    messages = [
        SystemMessage(content="你现在是文本精简提炼专家。根据用户发送的文本完成文本精炼要求。"),
        HumanMessage(content=prompt)
    ]

    # 调用大模型
    refined_answer = (llm_providers.chat() | StrOutputParser()).invoke(messages)

    return refined_answer


@step_log("_build_question_pairs")
def _build_question_pairs(query: str, final_chunk_list: list[dict], reranker) -> list[list[str]]:
    """构建重排模型输入对：[问题，文本]"""
    # 3.3.1 获取切词器并编码
    tokenizer = reranker.tokenizer
    query_tokens = tokenizer.encode(query, add_special_tokens=False)

    # 3.3.2 循环获取答案
    question_pairs = []
    for item in final_chunk_list:
        answer = item.get('text') or ""
        answer_for_rerank = answer

        # 答案编码
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        logger.info(f"答案token长度：{len(answer_tokens)}")
        total_tokens = len(query_tokens) + len(answer_tokens) + 4

        # 超过最大输入长度  调用大模型压缩精简
        if total_tokens > RERANK_MAX_INPUT_TOKENS:
            limit = max(
                RERANK_MIN_SUMMARY_CHARS,
                int((RERANK_MAX_INPUT_TOKENS - len(query_tokens) - 4) / RERANK_SUMMARY_CHAR_RATIO)
            )

            # 精简长文本
            answer_for_rerank = _summarize_long_rerank_text(query, answer, limit)

        # 组装答案对
        question_pairs.append([query, answer_for_rerank])

    return question_pairs


@step_log("score_and_sort_chunks")
def score_and_sort_chunks(state: QueryGraphState, final_chunk_list: list[dict]) -> list[dict]:
    """调用重排序模型对所有候选文档打分，并按分数从高到低排序"""
    if not final_chunk_list:
        return []

    # 3.1 获取用户查询问题
    rewritten_query = state.get('rewritten_query') or state.get('original_query') or ""

    # 3.2 获取重排模型实例
    reranker_model = llm_providers.reranker_model()

    # 3.3 构建模型输入对
    question_pairs = _build_question_pairs(rewritten_query, final_chunk_list, reranker_model)

    # 3.4 模型打分（归一化）
    reranker_model.compute_score(question_pairs, normalize=True)

    # 3.5 按分数降序排序
    final_chunk_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)

    return final_chunk_list


@step_log("dynamic_topk")
def dynamic_topk(sorted_docs):
    """根据分数断崖截取 top k"""
    min_topk = RERANK_MIN_TOPK
    max_topk = min(RERANK_MAX_TOPK, len(sorted_docs))
    gap_ratio = RERANK_GAP_RATIO
    max_gap = RERANK_GAP_ABS
    topk = max_topk

    # 遍历寻找分数断崖
    if topk > min_topk:
        for pre_index in range(min_topk - 1, max_topk - 1):
            pre_score = sorted_docs[pre_index].get('score', 0.0)
            next_score = sorted_docs[pre_index + 1].get('score', 0.0)
            # 分差
            abs_score = pre_score - next_score
            ratio = abs_score / (pre_score + 1e-7)

            # 发现断崖，截断
            if abs_score > max_gap or ratio > gap_ratio:
                topk = pre_index + 1
                break

    # 动态获取topk
    final_reranker_docs = sorted_docs[: topk]

    return final_reranker_docs


@step_log("rerank_documents")
def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    # 1. 获取并校验参数
    rrf_chunks, web_search_docs = get_data_and_validates(state)

    # 2. 统一格式合并两路结果
    merged = merge_rrf_and_web(rrf_chunks, web_search_docs)

    # 3. 重排模型打分 + 排序
    sorted_docs = score_and_sort_chunks(state, merged)

    # 4. 动态截断数据
    return dynamic_topk(sorted_docs)
