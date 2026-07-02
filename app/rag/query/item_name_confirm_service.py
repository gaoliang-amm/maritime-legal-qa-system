"""
@Author: gl
@Date: 2026/6/26
@Desc: 意图识别、法律法规名称提取和问题改写
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_providers
from app.infra.persistence.history_repository import history_repository
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

from app.process.query.agent.state import QueryGraphState

# ====================== 全局配置 ======================
# 拉取历史消息最大条数
QUERY_HISTORY_LIMIT = 10
# 主体名称确认阈值：高于该分数 → 直接确认
ITEM_NAME_CONFIRM_THRESHOLD = 0.65
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.50
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2


@step_log("get_data_and_validates")
def get_data_and_validates(state: QueryGraphState) -> tuple[str, str]:
    """获取`original_query` 和 `session_id` 并校验"""
    # 1. 获取数据
    session_id = state.get('session_id')
    original_query = state.get('original_query')

    # 2. 进行数据校验
    if not session_id or not original_query:
        logger.error(f"session_id或original_query为空，业务无法继续！")
        raise ValueError(f"session_id或original_query为空，业务无法继续！")

    # 返回结果
    return session_id, original_query


@step_log("get_history_messages_and_context")
def get_history_messages_and_context(session_id: str) -> str:
    """获取近期有效的历史聊天记录并且拼接上下文"""
    # 1. 获取近10条历史记录
    message_list: list[dict] = history_repository.list_recent(session_id=session_id, limit=QUERY_HISTORY_LIMIT)
    if not message_list:
        logger.warning(f"当前会话：{session_id}没有历史对话记录！")
        return f"当前会话：{session_id}没有历史对话记录！"

    # 2. 判断聊天记录是否有效
    valid_messages = [
        item for item in message_list
        if item.get('item_names')
    ]

    # 3. 拼接历史正文
    history_text = []
    for idx, item in enumerate(valid_messages, 1):
        if item.get('role') == 'user':
            content = item.get('rewritten_query', '')
            prefix = "提问"
        else:
            content = item.get('text', '')[:50]
            prefix = "回答"

        item_names = '，'.join(item.get('item_names', []))
        history_text.append(f"序号：{idx}，{prefix}：{content}，关联主体：{item_names}")

    return '\n'.join(history_text)


@step_log("call_llm_item_name_and_rewritten")
def call_llm_item_name_and_rewritten(history_text: str, original_query: str) -> dict:
    """调用模型识别"""
    # 1. 加载模型
    chat_client = llm_providers.chat(json_mode=True)

    # 2. 加载提示词
    prompt = load_prompt(
        "rewritten_query_and_itemnames",
        query=original_query,
        history_text=history_text
    )

    # 3. 构造大模型消息
    messages = [
        SystemMessage(content="你是一个专业的海事法律顾问，擅长理解用户关于船舶海事法律的咨询意图和提取关键信息。"),
        HumanMessage(content=prompt),
    ]

    # 4. 构造chains
    chains = chat_client | JsonOutputParser()

    # 5. 调用大模型
    result = chains.invoke(messages)

    # 6. 校验
    if "item_names" not in result:
        logger.warning("模型识别法律法规名称失败,给item_names赋予空列表")
        result['item_names'] = []
    if "rewritten_query" not in result:
        logger.warning(f"模型重写问题失败,给rewritten_query赋予原始问题:{original_query}")
        result['rewritten_query'] = original_query

    # 7. 返回结果
    return result


@step_log("search_item_names")
def search_item_names(item_names: list[str]) -> dict[str, list[dict]]:
    """向量数据库搜索"""
    vector_dict: dict[str, list[dict]] = {}
    # 4.1 批量生成向量
    item_names_vector = llm_providers.generate_embeddings(item_names)

    # 4.2 遍历获取稠密和稀疏向量
    for idx, item_name in enumerate(item_names):
        dense_vector = item_names_vector["dense"][idx]
        sparse_vector = item_names_vector["sparse"][idx]

        # 4.3 创建检索请求
        reqs = milvus_gateway.create_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=5 * 2
        )

        # 4.4 执行混合检索
        results = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_name_collection_name,
            reqs=reqs,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            output_fields=['item_name']
        )

        # 4.5 处理检索结果
        item_name_list = []
        if results:
            for item in results[0]:
                item_name_list.append(
                    {
                        "item_name": item.get('entity', {}).get('item_name', ""),
                        "score": item.get('distance', 0)
                    }
                )
        vector_dict[item_name] = item_name_list

    return vector_dict


@step_log("select_item_names")
def select_item_names(milvus_result: dict[str, list[dict]]) -> dict:
    """根据向量检索分数分类"""
    confirm_item_name_list: list[str] = []
    options_item_name_list: list[str] = []

    for item_name, milvus_result_list_dict in milvus_result.items():
        milvus_result_list_dict.sort(key=lambda x: x['score'], reverse=True)

        # 高分：直接确认
        high_score_list = [item for item in milvus_result_list_dict if
                           item.get('score', 0) >= ITEM_NAME_CONFIRM_THRESHOLD]

        # 中分：候选列表
        mid_score_list = [item for item in milvus_result_list_dict if
                          item.get('score', 0) < ITEM_NAME_CONFIRM_THRESHOLD]

        if high_score_list:
            confirm_item_name_list.append(high_score_list[0]['item_name'])
            logger.info(
                f"模型识别到item_name：{item_name}，向量数据库有对应item_name：{high_score_list[0].get('item_name')}")
            continue

        if mid_score_list:
            options_item_name_list.extend(mid_score_list[: ITEM_NAME_OPTIONS_TOPK])
            logger.info(f"模型识别到item_name：{item_name}，向量数据库没有有对应item_name!"
                        f"但是有可选的：{'，'.join([item.get('item_name') for item in mid_score_list[:ITEM_NAME_OPTIONS_TOPK]])}")
            continue

    return {
        "confirm_item_name_list": confirm_item_name_list,
        "options_item_name_list": options_item_name_list
    }


@step_log("apply_item_name_result")
def apply_item_name_result(state: QueryGraphState, final_result, rewritten_query):
    """根据主体确认结果，更新state"""
    confirmed_list = final_result.get('confirm_item_name_list', [])
    options_list = final_result.get('options_item_name_list', [])

    if confirmed_list:
        state['item_names'] = confirmed_list
        state['rewritten_query'] = rewritten_query
        if "answer" in state:
            state['answer'] = None

        return

    if options_list:
        state[
            'answer'] = f"您想查询的是以下哪部法律法规：{'，'.join(options_list)}？请明确说明。"
        state['rewritten_query'] = rewritten_query,
        state['item_names'] = []
        return

    state['answer'] = "抱歉，未找到相关法律法规。请确认您的问题是否涉及《海商法》或《海上交通安全法》等法律条文。"


@step_log("save_user_messages")
def save_user_messages(state: QueryGraphState):
    """保存用户提问和处理结果到MongoDB"""
    history_repository.save_message(
        session_id=state['session_id'],
        role="user",
        text=state.get('original_query'),
        rewritten_query=state.get('rewritten_query', ""),
        item_names=state.get('item_names', []),
        image_urls=[]
    )


@step_log("confirm_item_name")
def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取法律法规名称
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动匹配法律法规，或生成反问让用户明确
    5. 同步历史记录到 MongoDB
    """
    # 1. 校验 `original_query` 和 `session_id`
    session_id, original_query = get_data_and_validates(state)

    # 2. 获取历史消息，并且拼接上下文
    history_text = get_history_messages_and_context(session_id)

    # 3. 调用模型识别item_names及重写
    llm_result: dict = call_llm_item_name_and_rewritten(history_text, original_query)
    item_names = llm_result['item_names']
    rewritten_query = llm_result['rewritten_query']

    # 4. 如果提取到item_names，去向量库匹配标准名称
    final_result = {"confirm_item_name_list": [], "options_item_name_list": []}
    if item_names:
        milvus_result: dict[str, list[dict]] = search_item_names(item_names)
        final_result = select_item_names(milvus_result)

    # 5. 结果写入state
    apply_item_name_result(state, final_result, rewritten_query)

    # 6. 保存历史记录
    save_user_messages(state)

    return state
