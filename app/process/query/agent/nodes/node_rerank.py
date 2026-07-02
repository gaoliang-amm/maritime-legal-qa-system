"""
@Author: gl
@Date: 2026/6/26
@Desc: 重排序节点
"""

import sys

from app.shared.runtime.logger import node_log
from app.rag.query.rerank_service import rerank_documents
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_rerank")
def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    state['reranked_docs'] = rerank_documents(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "承运人对货物的责任期间，从货物装上船时起至卸下船时止", "title": "第四十七条 承运人责任期间"},
        {"chunk_id": "local_2", "content": "承运人对集装箱货物的责任期间，从装货港接收货物时起至卸货港交付货物时止", "title": "第四十七条 集装箱运输"},
    ]
    mock_web_docs = [
        {"title": "海商法承运人责任解读", "url": "http://web.com/1", "snippet": "海商法规定了承运人对货物运输的基本责任"},
    ]
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "海商法中承运人的责任期间是什么？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False,
    }
    result = node_rerank(mock_state)
    print(result)
