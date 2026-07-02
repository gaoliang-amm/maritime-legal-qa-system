"""
@Author: gl
@Date: 2026/6/26
@Desc: 最终答案生成节点
"""

import sys

from app.shared.runtime.logger import node_log
from app.rag.query.answer_service import generate_answer
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_answer_output")
def node_answer_output(state):
    """
    节点功能：生成最终回答并交付给用户（支持流式/非流式）。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    state = generate_answer(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state["is_stream"])
    return state

if __name__ == "__main__":
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "type": "milvus",
            "title": "第四十七条 承运人责任期间",
            "score": 0.95,
            "text": """
            承运人对货物的责任期间，从货物装上船时起至卸下船时止，货物处于承运人掌管之下的全部期间。
            """,
        }
    ]
    mock_history = [
        {"role": "user", "text": "海商法中承运人的责任是什么？", "rewritten_query": "海商法中承运人对货物运输的责任规定是什么？"},
    ]
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "海商法中承运人的责任是什么？",
        "rewritten_query": "海商法中承运人对货物运输的责任规定是什么？",
        "item_names": ["中华人民共和国海商法"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None,
    }
    result = node_answer_output(mock_state)
    print(result)
