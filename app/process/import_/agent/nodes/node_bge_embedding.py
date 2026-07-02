"""
@Author: gl
@Date: 2026/6/17
@Desc: BGE-M3向量化节点，生成dense+sparse混合向量
"""
from dotenv import load_dotenv

from app.shared.runtime.logger import node_log,logger
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.embedding_service import generate_chunk_embeddings


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    """
    add_running_task(state["task_id"], "node_bge_embedding")
    state = generate_chunk_embeddings(state)
    add_done_task(state["task_id"], "node_bge_embedding")
    return state


if __name__ == '__main__':
    import os
    # 加载环境变量：定位项目根目录下的.env，读取模型路径/设备等配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "item_name": "中华人民共和国海商法",
        "chunks": [  # 模拟带item_name的文本切片（上游法律法规名称识别节点产出）
            {
                "content": "第一条 为了调整海上运输关系、船舶关系，维护当事人各方的合法权益，促进海上运输和经济贸易的发展，制定本法。",
                "title": "第一章 总则",
                "item_name": "中华人民共和国海商法",
                "file_title": "中华人民共和国海商法.pdf"
            },
            {
                "content": "本法所称船舶，是指海船和其他海上移动式装置，但是用于军事的、政府公务的船舶和20总吨以下的小型船艇除外。",
                "title": "第二章 船舶",
                "item_name": "中华人民共和国海商法",
                "file_title": "中华人民共和国海商法.pdf"
            }
        ]
    })

    # 执行本地测试
    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 打印测试结果统计
        logger.info(f"=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")
        logger.info(f"返回的结果:{result_chunks}")


    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")
