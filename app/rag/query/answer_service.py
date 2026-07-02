"""
@Author: gl
@Date: 2026/6/26
@Desc: 答案输出节点
"""
import re

from app.infra.llm.providers import llm_providers
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task, add_running_task, push_to_session, set_task_result
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import step_log
import time


@step_log("exist_answer")
def exist_answer(state):
    """
    如已有answer，直接返回，不调用模型
    :param state:
    :return:
    """
    # 1.1 获取参数
    answer = state.get('answer')
    is_stream = state.get('is_stream', False)
    session_id = state.get('session_id')

    # 1.2 无答案
    if not answer:
        return False

    # 1.3 流式模式
    if is_stream:
        for ch in answer:
            push_to_session(session_id, SSEEvent.DELTA, {'delta': ch})
            time.sleep(0.1)

    # 1.4 答案写入结果
    set_task_result(session_id, 'answer', answer)
    return True

@step_log("validate_generation_input")
def validate_generation_input(state):
    """答案生成前参数校验"""
    # 2.1 参数获取
    history = state.get('history', [])
    reranked_docs = state.get('reranked_docs')
    item_names = state.get('item_names', [])
    rewritten_query = state.get('rewritten_query') or state.get('original_query')

    # 2.2 校验
    if not reranked_docs or not rewritten_query:
        raise ValueError("生成答案需要 reranked_docs 和 rewritten_query/original_query")

    return reranked_docs, item_names, rewritten_query, history

@step_log("build_answer_prompt")
def build_answer_prompt(reranked_docs, rewritten_query, item_names, history):
    """构造最终答案生成的prompt"""
    context_chunk_list = []
    # 3.1 遍历重排后的文档，按照序号拼接参考内容
    for num, chunk in enumerate(reranked_docs, 1):
        context_chunk_list.append(
            f"第{num}块: 标题:{chunk['title']} 匹配度得分:{chunk['score']} 来源:{'网络搜索' if chunk['type'] == 'web' else '向量查询'}\n内容:{chunk['text']}"
        )

    # 3.2 合并所有参考块
    context_chunk_str = "\n\n".join(context_chunk_list)

    # 3.3 格式化历史记录
    history_text = []
    for idx, item in enumerate(history, 1):
        if item.get('role') == 'user':
            content = item.get('rewritten_query', '')
            prefix = "提问"
        else:
            content = item.get('text', '')[:50]
            prefix = "回答"

        item_names = '，'.join(item.get('item_names', []))
        history_text.append(f"序号：{idx}，{prefix}：{content}，关联主体：{item_names}")

    history_text = '\n'.join(history_text)

    # 3.4 格式化关联主体
    item_name_str = "本次关联主体:" + ",".join(item_names) if item_names else "没有关联主体"

    # 3.5 生成prompt
    return load_prompt(
        "answer_out",
        context=context_chunk_str,
        history=history_text,
        item_names=item_name_str,
        question=rewritten_query
    )

@step_log("final_answer")
def final_answer(state, prompt):
    """调用大模型生成最终答案"""
    # 4.1 获取参数
    is_stream = state.get('is_stream', False)
    session_id = state.get('session_id')
    final_result = ""

    # 4.2 获取大模型客户端
    llm_client = llm_providers.chat()

    # 4.3 流式生成
    if is_stream:
        for chunk in llm_client.stream(prompt):
            final_result += chunk.content
            push_to_session(session_id, SSEEvent.DELTA, {"delta": chunk.content})
    # 普通生成，一次性调用
    else:
        response = llm_client.invoke(prompt)
        final_result = response.content

    # 4.4 更新状态
    set_task_result(session_id, 'answer', final_result)
    state['answer'] = final_result

    return final_result

@step_log("extract_image_urls")
def extract_image_urls(reranked_docs):
    """从参考文档获取所有图片 url"""
    image_urls: list[str] = []
    # 5.1 匹配 markdown 图片正则
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")

    # 5.2 循环每个 doc url | text
    for doc in reranked_docs:
        url = doc.get('url')
        text = doc.get('text')

        # 提取直接作为 url 的图片
        if url and url.endswith(SUPPORTED_IMAGE_EXTENSIONS) and url not in image_urls:
            image_urls.append(url)

        # 提取文本中的 markdown 图片
        if text:
            for image_url in reg.findall(text):
                if image_url not in image_urls:
                    image_urls.append(image_url)

    return image_urls

@step_log("save_messages")
def save_messages(state):
    """保存历史聊天记录"""
    history_repository.save_message(
        session_id=state.get('session_id'),
        role="assistant",
        text=state.get('answer'),
        rewritten_query=state.get('rewritten_query') or state.get('original_query'),
        item_names=state.get('item_names', []),
        image_urls=state.get('image_urls', [])
    )

@step_log("generate_answer")
def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    # 1. 判断是否已有答案
    if not exist_answer(state):
        # 2. 校验输入
        reranked_docs, item_names, rewritten_query, history = validate_generation_input(state)

        # 3. 构建提示词
        prompt = build_answer_prompt(reranked_docs, rewritten_query, item_names, history)

        # 4. 生成答案
        final_answer(state, prompt)

        # 5. 提取图片中url
        state['image_urls'] = extract_image_urls(reranked_docs)

    # 6. 保存历史消息
    save_messages(state)

    return state
