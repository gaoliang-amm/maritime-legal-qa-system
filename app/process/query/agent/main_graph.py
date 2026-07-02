"""
@Author: gl
@Date: 2026/6/26
@Desc: 主图
"""

from langgraph.graph import StateGraph, END

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

# 1. 定义状态图对象，并且指定全局的 state
query_graph_builder = StateGraph(QueryGraphState)

# 2. 添加节点信息
query_graph_builder.add_node(node_item_name_confirm)
query_graph_builder.add_node(node_search_embedding)
query_graph_builder.add_node(node_search_embedding_hyde)
query_graph_builder.add_node(node_web_search_mcp)
query_graph_builder.add_node(node_rrf)
query_graph_builder.add_node(node_rerank)
query_graph_builder.add_node(node_answer_output)

# 3. 指定入口节点（有条件边）
query_graph_builder.set_entry_point("node_item_name_confirm")


# 4. 指定条件边，动态边
def node_item_name_confirm_after_router(state: QueryGraphState):
    """
    state answer 进行判定
    None -> 第一个节点识别出 item_names ，提问没问题
    str -> 提问是空 | 有不确定的 item_names | 没有识别对应的 item_name
    如果前置节点已经生成了澄清或兜底大难，直接跳转到输出节点收口
    :param state:
    :return:
    """
    if state.get('answer'):
        logger.warning(f"没有明确的item_name，跳到 node_answer_output 节点！")
        return "node_answer_output"
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"


query_graph_builder.add_conditional_edges(
    "node_item_name_confirm",
    node_item_name_confirm_after_router,
    {
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp",
        "node_answer_output": "node_answer_output",
    }
)

# 5. 指定静态边
query_graph_builder.add_edge("node_search_embedding", "node_rrf")
query_graph_builder.add_edge("node_search_embedding_hyde", "node_rrf")
query_graph_builder.add_edge("node_web_search_mcp", "node_rrf")
query_graph_builder.add_edge("node_rrf", "node_rerank")
query_graph_builder.add_edge("node_rerank", "node_answer_output")
query_graph_builder.add_edge("node_answer_output", END)

# 6. 编译对象
query_app = query_graph_builder.compile()

# 执行图对象
if __name__ == "__main__":
    from app.process.query.agent.state import create_query_default_state

    # 全流程测试：验证查询链完整流程
    logger.info("===== 开始执行查询链全流程测试 =====")

    # 1. 构造测试状态
    test_state = create_query_default_state(
        session_id="test_query_session_001",
        original_query="海商法中承运人的责任是什么？",
        is_stream=False
    )

    try:
        logger.info(f"测试任务启动，原始问题：{test_state.get('original_query')}")
        logger.info("开始执行全流程节点，依次执行：item_name_confirm→三路召回→rrf→rerank→answer")

        # 2. 执行LangGraph全流程（流式执行，打印节点执行进度）
        final_state = None
        for step in query_app.stream(test_state, stream_mode="values"):
            # 打印当前执行完成的节点（流式输出更直观）
            current_node = list(step.keys())[-1] if step else "未知节点"
            logger.info(f"✅ 节点执行完成：{current_node}")
            final_state = step  # 保存最终状态

        # 3. 全流程执行完成，结果预览
        if final_state:
            logger.info("-" * 80)
            logger.info("===== 全流程测试执行成功，核心结果预览 =====")

            # 提取核心结果指标
            answer = final_state.get("answer", "")
            item_names = final_state.get("item_names", [])
            rewritten_query = final_state.get("rewritten_query", "")
            rrf_chunks = final_state.get("rrf_chunks", [])
            reranked_docs = final_state.get("reranked_docs", [])
            image_urls = final_state.get("image_urls", [])

            # 打印核心指标
            logger.info(f"🏷️  识别的主体名称：{item_names}")
            logger.info(f"✏️  改写后的问题：{rewritten_query}")
            logger.info(f"📊 RRF融合结果数量：{len(rrf_chunks)}")
            logger.info(f"📈 Rerank精排结果数量：{len(reranked_docs)}")
            logger.info(f"🖼️  引用图片数量：{len(image_urls)}")
            logger.info(f"💬 最终答案（前200字符）：{answer[:200]}...")
            logger.info(f"📂 最终状态包含的核心键：{list(final_state.keys())}")
            logger.info("-" * 80)

    except Exception as e:
        logger.exception(f"===== 全流程测试运行失败 =====")

    logger.info("===== 查询链全流程测试结束 =====")
