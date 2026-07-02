"""
@Author: gl
@Date: 2026/6/17
@Desc: 导入链入口节点服务，负责文件类型识别与状态分发
"""
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log

# 支持的文件扩展名（小写）
SUPPORTED_EXTENSIONS = {".md", ".pdf"}


@step_log('resolve_input_file')
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
    入口节点服务：验证文件存在性，识别文件类型，设置状态标记
    :param state: 导入链状态
    :return: 更新后的状态
    :raises FileNotFoundError: 文件路径为空或文件不存在
    :raises ValueError: 不支持的文件格式
    """
    # 1. 先获取local_file_path参数state
    local_file_path = state.get("local_file_path")

    # 2. local_file_path进行非空校验
    if not local_file_path:
        logger.error(f"local_file_path为空，无法继续业务，提前终止！")
        raise FileNotFoundError(f"local_file_path为空，无法继续业务，提前终止！")

    # 3. 判断是不是md
    if local_file_path.endswith(".md"):
        state['md_path'] = local_file_path
        state['is_md_read_enabled'] = True
        state['is_pdf_read_enabled'] = False

    # 4. 判断是不是pdf
    elif local_file_path.endswith(".pdf"):
        state['pdf_path'] = local_file_path
        state['is_pdf_read_enabled'] = True
        state['is_md_read_enabled'] = False

    # 5. 都不是，警告
    else:
        logger.warning(f"{local_file_path}对应的文件类型无法解析，只支持md/pdf格式")
        state['is_md_read_enabled'] = False
        state['is_pdf_read_enabled'] = False
        return state

    # 6. 获取file_title参数，更新state
    file_title = Path(local_file_path).stem

    # 7. 返回处理后的state
    state['file_title'] = file_title

    return state
