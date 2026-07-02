"""
@Author: gl
@Date: 2026/6/17
@Desc: 文档切分服务，按标题粗切+递归细切+短合并+属性补全
"""
import json
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path
from app.shared.runtime.logger import logger, step_log
from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN

@step_log("load_markdown_content")
def load_markdown_content(state: ImportGraphState) -> tuple[str, str]:
    """
    获取参数和校验
    :param state:
    :return:
    """
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path = state.get("md_path")

    # md_content校验
    if not md_content:
        if md_path and Path(md_path).exists():
            logger.warning(f"md_content内容为空，从备份地址：{md_path}再次读取数据！")
            md_content = Path(md_path).read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"md_content为空,尝试从md_path读取,依然为空,业务无法继续进行,提前终止!")
            raise ValueError(f"md_content为空,尝试从md_path读取,依然为空,业务无法继续进行,提前终止!")

    # file_title校验
    if not file_title:
        if md_path and Path(md_path).exists():
            file_title = Path(md_path).stem
        if not file_title:
            file_title = "default"
        state['file_title'] = file_title
        logger.warning(f"file_title为空,启动默认值机制,赋值后:{file_title}")

    # 数据清洗，统一换行符
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    state['md_content'] = md_content

    return md_content, file_title

@step_log("split_chunks_document")
def split_chunks_document(md_content, file_title) -> list[dict[str, Any]]:
    """
    根据标题完成语义切割
    :param md_content:
    :param file_title:
    :return:
    """
    # 1. md_content按行切割
    md_content_lines: list[str] = md_content.split('\n')

    # 2. 准备数据（记录当前标题和标题行和代码块 历史chunks）
    chunks: list[dict[str, Any]] = []
    current_title: str = None
    current_title_lines: list[str] = []
    is_code_block = False
    title_reg = re.compile(r"^\s*#{1,6}\s.+")
    empty_line_count = 0

    # 3. 循环行 -> 判断是不是标题 是不是代码块 是不是空行 是不是普通
    for line in md_content_lines:
        # 空行
        line_strip = line.strip()
        if not line_strip:
            empty_line_count += 1
            continue
        # 判断是否代码块
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(line_strip)
            continue
        # 判断是不是有效标题
        if not is_code_block and title_reg.match(line_strip):
            if current_title and len(current_title_lines) > 1:
                chunks.append(
                    {
                        "content": "\n".join(current_title_lines),
                        "title": current_title,
                        "file_title": file_title
                    }
                )

            if not current_title and len(current_title_lines) > 0:
                current_title_lines.append(line_strip)
            else:
                current_title_lines = [line_strip]  # 将标题设置为第一行字符串
            # 开启新的
            current_title = line_strip  # 新的设置为当前处理标签
        else:
            current_title_lines.append(line_strip)

    # 最后一次可能没有被结算
    if current_title and len(current_title_lines) > 1:
        chunks.append(
            {
                "content": "\n".join(current_title_lines),
                "title": current_title,
                "file_title": file_title
            }
        )

    # 整个文档没有标题
    if len(chunks) == 0 and len(current_title_lines) > 0:
        chunks.append(
            {
                "content": "\n".join(current_title_lines),
                "title": "default",
                "file_title": file_title
            }
        )

    logger.info(f"完成语义标题切割，总行数: {len(md_content_lines)}，空行: {empty_line_count}，生成chunks: {len(chunks)}")

    return chunks

@step_log("_split_long_chunk")
def _split_long_chunk(chunk) -> list[dict[str, Any]]:
    """
    拆分【过长的文本块】，保证单个chunk不超过最大长度限制
    :param chunk:
    :return:
    """
    # 1. 清洗原有的content
    content = chunk.get("content")
    title = chunk.get("title")
    file_title = chunk.get("file_title")
    clear_content = content[len(title) + 1:]  # 去掉\n，+1

    # 2. 定义公共前缀
    sub_content_prefix = title + "\n"

    # 3. 定义递归切割器
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";"],
        chunk_size=CHUNK_SIZE - len(sub_content_prefix),
        chunk_overlap=CHUNK_OVERLAP
    )

    # 4. 切割content
    sub_chunk_list = []
    for part_index, split_text in enumerate(splitter.split_text(clear_content), start=1):
        # 5. 拼接每个子chunk内容
        split_text = split_text.strip()
        content_sub_new = sub_content_prefix + split_text
        sub_chunk_list.append(
            {
                "title": f"{title}_第{part_index}部分",
                "content": content_sub_new,
                "file_title": file_title,
                "parent_title": title,
                "part": part_index
            }
        )

    # 6. 返回结果
    logger.info(f"进入标题:{title},完成切割后,切成:{len(sub_chunk_list)}块!")
    return sub_chunk_list

@step_log("_merge_short_chunks_same_parent_title")
def _merge_short_chunks_same_parent_title(refine_list) -> list[dict[str, Any]]:
    """
    合并【过短的文本块】，避免碎片内容
    :param refine_list:
    :return:
    """
    merged_chunk_list = []
    # 1. 定义base_chunk
    base_chunk = None

    # 2. 处理需要合并的元素
    for next_chunk in refine_list:
        # 第一次给base_chunk赋值
        if not base_chunk:
            base_chunk = next_chunk
            logger.info("短合并第一次进入，设置base_chunk内容！")
            continue

        # 3. 合并
        is_short_chunk = len(base_chunk.get("content")) < CHUNK_MIN
        is_same_parent_title = base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get(
            "parent_title")
        if is_short_chunk and is_same_parent_title:
            next_content = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 1:]
            is_not_long = (len(base_chunk.get("content")) + len(next_content)) <= CHUNK_MAX_SIZE
            if is_not_long:
                base_chunk["content"] = base_chunk.get("content") + "\n" + next_content
                continue
            else:
                merged_chunk_list.append(base_chunk)
                base_chunk = next_chunk
                continue
        else:
            merged_chunk_list.append(base_chunk)
            base_chunk = next_chunk
            continue

    # 4. 最后一个base_chunk
    if base_chunk:
        merged_chunk_list.append(base_chunk)

    # 5. 返回结果
    logger.info(f"进行短合并,合并之前:{len(refine_list)},合并之后:{len(merged_chunk_list)}")
    return merged_chunk_list

@step_log("refine_chunks")
def refine_chunks(chunks) -> list[dict[str, Any]]:
    """
    精细切割
    :param chunks:
    :return:
    """
    # 1. 定义接收最终结果
    refine_list = []

    # 2. 循环处理，是否过长
    for chunk in chunks:
        content = chunk.get("content")
        if len(content) > CHUNK_SIZE:
            long_chunk_list = _split_long_chunk(chunk)
            refine_list.extend(long_chunk_list)
        else:
            refine_list.append(chunk)

    # 3. 短合并处理
    refine_list = _merge_short_chunks_same_parent_title(refine_list)

    # 4. 补全属性
    for chunk in refine_list:
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk.get("title", "default_title")
        if "part" not in chunk:
            chunk["part"] = 1

    logger.info(f"完成chunks的精细处理! 进入切块数量:{len(chunks)},处理后:{len(refine_list)}")
    return refine_list

@step_log("backup_chunks_json")
def backup_chunks_json(refine_chunks_list, md_path: str):
    """
    数据备份
    :param refine_chunks_list:
    :param md_path:
    :return:
    """
    # 1. 获取目标的地址
    json_path_obj: Path = Path(md_path).with_name(f"{Path(md_path).stem}.json")

    # 2. 目标位置写入字符串
    json_path_obj.write_text(json.dumps(refine_chunks_list, indent=4, ensure_ascii=False), encoding="utf-8")

@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    # 1. 获取参数的校验
    md_content, file_title = load_markdown_content(state)

    # 2. 确保语义切割，根据标题切割（只保留关联标题）
    chunks: list[dict[str, Any]] = split_chunks_document(md_content, file_title)

    # 3. 精细切割
    refine_chunks_list = refine_chunks(chunks)

    # 4. 更新state
    state['chunks'] = refine_chunks_list

    # 5. 备份
    backup_chunks_json(refine_chunks_list, state['md_path'])

    return state
