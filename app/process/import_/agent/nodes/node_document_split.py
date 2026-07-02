"""
@Author: gl
@Date: 2026/6/17
@Desc: 文档切分节点，按标题层级粗切+递归细切+短合并
"""

from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.split_service import split_document

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    """
    add_running_task(state["task_id"], "node_document_split")
    state = split_document(state)
    add_done_task(state["task_id"], "node_document_split")
    return state

if __name__ == '__main__':
    from app.shared.utils.path_util import PROJECT_ROOT
    from app.process.import_.agent.nodes.node_md_img import node_md_img
    from app.shared.runtime.logger import logger
    import os

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 先运行 node_pdf_to_md 生成 MD 文件，再测试本节点
    test_md_name = os.path.join(r"output\中华人民共和国海商法_20251028", "中华人民共和国海商法_20251028.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请先运行 node_pdf_to_md 生成 MD 文件")
    else:
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "中华人民共和国海商法_20251028",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        result_state = node_md_img(test_state)
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"测试成功：最终生成{len(final_chunks)}个有效Chunk")
