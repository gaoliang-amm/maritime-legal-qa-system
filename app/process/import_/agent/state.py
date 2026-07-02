"""
@Author: gl
@Date: 2026/6/17
@Desc:
"""
import copy
import json
from typing import TypedDict
from app.shared.runtime.logger import logger


class ImportGraphState(TypedDict):
    # 传入数据
    task_id: str
    local_file_path: str

    # 文件类型标识
    md_path: str
    is_md_read_enabled: bool

    pdf_path: str
    is_pdf_read_enabled: bool

    file_title: str  # 文件名

    # 输出文件地址
    local_dir: str  # 存储生成文件地址

    md_content: str  # md文件内容

    chunks: list  # 切割后的文本块

    item_name: str  # 主体file_title兜底

    embedding_content: list  # 带有向量的切块


# 准备一个state对象
default_state: ImportGraphState = {
    "task_id": "",
    "local_file_path": "",
    "md_path": "",
    "is_md_read_enabled": False,
    "pdf_path": "",
    "is_pdf_read_enabled": False,
    "file_title": "",
    "local_dir": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embedding_content": []
}


# 定义一个可以更新对象属性的函数，并且返回更新后的对象
def create_default_state(**kwargs) -> ImportGraphState:
    """
    创建一个对象！更新指定的属性，形参列表中指定要更新的属性
    :param kwargs:
    :return:
    """

    new_state = copy.deepcopy(default_state)
    new_state.update(kwargs)
    return new_state


def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    :return:
    """
    return copy.deepcopy(default_state)


if __name__ == '__main__':
    state = create_default_state(task_id="123", local_file_path="./..")
    logger.info(f"state:{json.dumps(state, indent=4, ensure_ascii=False)}")
