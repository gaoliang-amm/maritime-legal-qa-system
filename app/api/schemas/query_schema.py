"""
@Author: gl
@Date: 2026/6/27
@Desc: 查询模块接口请求和响应结构定义
"""
from typing import Any

from pydantic import BaseModel


class HealthResponseSchema(BaseModel):
    """健康检查的响应json"""
    code: int = 200
    message: str | None = None


class QueryRequestSchema(BaseModel):
    """查询接口的请求参数json"""
    query: str
    session_id: str = None
    is_stream: bool = False

class QueryStreamResponseSchema(BaseModel):
    """查询模块流式响应"""
    message: str
    session_id: str

class QueryNotStreamResponseSchema(BaseModel):
    """查询模式非流式响应"""
    message: str
    session_id: str
    answer: str
    done_list: list[str]
    image_urls: list[str]

class HistoryClearResponseSchema(BaseModel):
    """清空历史会话"""
    message: str
    deleted_count: int

class HistoryItemResponseSchema(BaseModel):
    """历史会话结构"""
    id: str
    session_id: str
    role: str
    text: str
    rewritten_query: str = None
    item_names: list[str]
    image_urls: list[str]
    ts: Any

class HistoryListResponseSchema(BaseModel):
    """历史记录"""
    session_id: str
    items: list[HistoryItemResponseSchema]
