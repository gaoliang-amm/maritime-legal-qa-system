"""
@Author: gl
@Date: 2026/6/26
@Desc: 向量检索节点
"""

import sys

from app.shared.runtime.logger import node_log
from app.rag.query.embedding_search_service import search_by_embedding
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_search_embedding")
def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    embedding_chunks = search_by_embedding(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"embedding_chunks": embedding_chunks}


if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "海商法中承运人的责任是什么？",
        "item_names": ["中华人民共和国海商法"],
        "is_stream": False,
    }
    result = node_search_embedding(test_state)
    print(result)
