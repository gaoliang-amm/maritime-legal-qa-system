"""
@Author: gl
@Date: 2026/6/26
@Desc: 导入服务，负责文件上传、LangGraph导入执行与状态查询
"""
import shutil
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schemas.import_schema import UploadResponseSchema, StatusResponseSchema
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.main_graph import import_app
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.infra.config.providers import infra_config
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
    add_running_task,
    add_done_task
)

app = FastAPI(
    title=infra_config.settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0"
)

# 配置 CORS（跨域资源共享）以允许前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(infra_config.settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# 接口1：返回html文件
@app.get("/import/html")
def import_html():
    html_path_obj: Path = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"
    return FileResponse(
        path=str(html_path_obj),
        media_type=guess_type(html_path_obj.name)[0]
    )


# 接口2：导入文件
# 前端 -> 文件 -> 接口 -> langgraph解析 -> 前端返回结果
def invoke_import_graph(task_id: str, local_file_path: str, local_dir: str):
    """
    后台任务：LangGraph全流程执行
    :param task_id:
    :param local_file_path:
    :param local_dir:
    :return:
    """
    try:
        # processing
        update_task_status(task_id, status_name=TASK_STATUS_PROCESSING)
        state: ImportGraphState = create_default_state(
            task_id=task_id,
            local_file_path=local_file_path,
            local_dir=local_dir
        )
        import_app.invoke(state)
        # completed
        update_task_status(task_id, status_name=TASK_STATUS_COMPLETED)
    except Exception as e:
        # failed
        update_task_status(task_id, status_name=TASK_STATUS_FAILED)
        logger.exception(f"导入模块执行发生异常！异常信息：{str(e)}")


@app.post("/upload")
def uploads(backgroundtasks: BackgroundTasks, files: list[UploadFile]):
    """
    异步调用图对象，解析本次上传的文件
    :param backgroundtasks:
    :param files:
    :return:
    """
    # 1. 定义参数
    task_id = str(uuid.uuid4())  # uuid -> 时区 -> 时间戳 -> ip地址 -> mac地址

    time_now_str = datetime.now().strftime("%Y%m%d")
    local_dir_obj: Path = PROJECT_ROOT / "output" / time_now_str / task_id
    local_dir_obj.mkdir(parents=True, exist_ok=True)

    upload_file = files[0]
    local_file_path_obj: Path = local_dir_obj / upload_file.filename

    # 2. 上传文件存储到地址
    add_running_task(task_id, "upload_file")
    with local_file_path_obj.open("wb") as file_buffer:
        # 使用 read() 读取文件内容，然后写入
        content = upload_file.file.read()
        file_buffer.write(content)
    add_done_task(task_id, "upload_file")

    # 3. 异步执行
    backgroundtasks.add_task(
        invoke_import_graph,
        task_id=task_id,
        local_file_path=str(local_file_path_obj),
        local_dir=str(local_dir_obj)
    )

    return UploadResponseSchema(
        code=200,
        message=f"{upload_file.filename}文件上传成功！",
        task_ids=[task_id]
    )


# 接口3：获取请求状态
@app.get("/status/{task_id}")
def task_status(task_id: str):
    done_list = get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)
    status = get_task_status(task_id)

    return StatusResponseSchema(
        code=200,
        task_id=task_id,
        status=status,
        done_list=done_list,
        running_list=running_list
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host=infra_config.settings.app_host,
        port=infra_config.settings.import_app_port
    )
