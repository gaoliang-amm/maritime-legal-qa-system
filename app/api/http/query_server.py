"""
@Author: gl
@Date: 2026/6/27
@Desc: 查询服务，提供智能问答、流式推送、历史记录管理等接口
"""
import datetime
from mimetypes import guess_type
from pathlib import Path
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from app.infra.persistence.history_repository import history_repository

from app.api.schemas.query_schema import HealthResponseSchema, QueryRequestSchema, QueryStreamResponseSchema, \
    QueryNotStreamResponseSchema, HistoryClearResponseSchema, HistoryListResponseSchema, HistoryItemResponseSchema
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_app
from app.process.query.agent.state import create_query_default_state, QueryGraphState
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)

# 定义fastapi对象
app = FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.2.0"
)

# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

# 静态文件路由（CSS、JS）
static_dir = PROJECT_ROOT / "app" / "resources" / "html"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 接口1：返回查询页面
@app.get("/html")
def query_html():
    query_html_obj: Path = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"
    return FileResponse(
        path=str(query_html_obj),
        media_type=guess_type(query_html_obj.name)[0]
    )

# 接口2：健康检查
@app.get("/health")
def health():
    logger.info(f"完成健康度检查！{datetime.datetime.now()}")
    return HealthResponseSchema(
        code=200,
        message=f"完成健康度检查！{datetime.datetime.now()}"
    )

# 接口3：查询接口
def invoke_query_graph(session_id, original_query, is_stream):
    """执行图流程"""
    try:
        clear_task(session_id)  # 清空task字典

        # 流式，提取创建队列
        if is_stream:
            create_sse_queue(session_id)
        update_task_status(session_id, TASK_STATUS_PROCESSING, push_queue=is_stream)

        # 封装成state
        query_state = create_query_default_state(
            session_id=session_id,
            original_query=original_query,
            is_stream=is_stream
        )

        # 执行图对象
        result_state = query_app.invoke(query_state)

        # 更新任务状态，完成
        update_task_status(session_id, TASK_STATUS_COMPLETED, push_queue=is_stream)

        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": result_state.get('answer'),
                    "status": "completed",
                    "image_urls": result_state.get('image_urls', [])
                }
            )

        return result_state
    except Exception as e:
        update_task_status(session_id, TASK_STATUS_FAILED, push_queue=is_stream)
        logger.exception(f"执行查询流程报错，错误信息：{str(e)}")



@app.post("/query")
def query(backgroundtasks: BackgroundTasks, query_params: QueryRequestSchema):
    """
    查询数据
    :param backgroundtasks:
    :param query_params:
    :return:
    """
    # 1. 获取参数
    query = query_params.query
    session_id = query_params.session_id or str(uuid.uuid4())
    is_stream = query_params.is_stream

    # 2. 判断是否流式查询
    if is_stream:
        # 3. 是流式，异步执行图方法
        backgroundtasks.add_task(
            invoke_query_graph,
            session_id=session_id,
            original_query=query,
            is_stream=is_stream
        )

        return QueryStreamResponseSchema(
            message=f"开始：{query}查询！",
            session_id=session_id
        )
    else:
        # 4. 非流式，直接调用执行图
        state: QueryGraphState = invoke_query_graph(session_id=session_id, original_query=query, is_stream=is_stream)

        # task_utils
        done_task_list = get_done_task_list(session_id)

        return QueryNotStreamResponseSchema(
            message=f"完成：{query}所有内容检索！",
            session_id=session_id,
            answer=state.get('answer'),
            done_list=done_task_list,
            image_urls=state.get('image_urls', [])
        )


# 接口4：流式查询接口
@app.get("/stream/{session_id}")
def stream(session_id: str, request: Request):
    """流式查询"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )

# 接口5：清空历史会话
@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    deleted_count = history_repository.clear_session(session_id=session_id)
    return HistoryClearResponseSchema(
        message=f"session_id: {session_id}历史记录已经清空！",
        deleted_count=deleted_count
    )

# 接口6：获取历史会话
@app.get("/history/{session_id}")
def get_history(session_id: str, limit: int = 10):
    history_list: list[dict] = history_repository.list_recent(session_id=session_id, limit=limit)
    return HistoryListResponseSchema(
        session_id=session_id,
        items=[
            HistoryItemResponseSchema(
                id=str(item.get('_id')),
                session_id=session_id,
                role=item.get('role'),
                text=item.get('text'),
                rewritten_query=item.get('rewritten_query'),
                item_names=item.get('item_names', []),
                image_urls=item.get('image_urls', []),
                ts=item.get('ts')
            )
            for item in history_list
        ]
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)