"""
@Author: gl
@Date: 2026/6/26
@Desc: 调用百炼 MCP 联网搜索服务
"""
import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from app.infra.config.providers import infra_config
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state: QueryGraphState) -> str:
    """获取并校验参数"""
    # 1.1 获取参数
    rewritten_query = state.get('rewritten_query')

    # 1.2 校验
    if not rewritten_query:
        logger.error(f"重写的问题为空，无法继续业务！")
        raise ValueError(f"重写的问题为空，无法继续业务！")

    return rewritten_query

@step_log("open_ai_mcp")
async def open_ai_mcp(rewritten_query: str):
    """调用MCP工具"""
    # 2.1 创建MCP服务
    mcp_server = MCPServerStreamableHttp(
        name="search_mcp",
        client_session_timeout_seconds=300,
        params={
            "url": infra_config.mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp_config.api_key}"},
            "timeout": 50,
            "sse_read_timeout": 300,
        },
        cache_tools_list=True,
        max_retry_attempts=3
    )

    try:
        # 2.2 mcp服务连接
        await mcp_server.connect()

        # 2.3 mcp工具调用
        mcp_result = await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": 5
            }
        )
        return mcp_result
    except Exception as e:
        logger.exception(f"mcp调用发生异常{str(e)}")
    finally:
        # 2.4 清空连接
        await mcp_server.cleanup()

@step_log("search_by_web")
def search_by_web(state: QueryGraphState) -> dict:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    """
    # 1. 获取并校验参数
    rewritten_query = get_data_and_validates(state)

    # 2. async 使用openai提供mcp方式进行调用
    mcp_result = asyncio.run(open_ai_mcp(rewritten_query))

    # 3. 结果解析
    text = mcp_result.content[0].text
    text_dict = json.loads(text)
    pages = text_dict.get('pages', [])

    return pages
