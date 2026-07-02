"""
@Author: gl
@Date: 2026/6/17
@Desc: PDF解析服务，调用MinerU将PDF转换为Markdown
"""
import shutil
import time

import requests

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path

from app.rag.import_.config import PDF_PARSE_SERVICE_LOCAL_DIR, MINERU_MODEL_VERSION, MINERU_POLL_TIMEOUT_SECONDS, \
    MINERU_POLL_INTERVAL_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log
from app.infra.config.providers import infra_config


@step_log("validate_pdf_paths")
def validate_pdf_paths(state: ImportGraphState) -> tuple[Path, Path]:
    """
    参数获取和校验！完成文件存在性判断和文件夹的创建
    :param state:
    :return:
    """
    # 1.1 state获取 pdf_path 和 local_dir
    pdf_path = state.get('pdf_path')
    local_dir = state.get('local_dir')

    # 1.2 进行 pdf_path 非空校验
    if not pdf_path:
        logger.error(f"pdf_path 参数为空，业务无法继续进行，提前终止！")
        raise ValueError(f"pdf_path 参数为空，业务无法继续进行，提前终止！")

    # 1.3 进行 local_dir 非空校验
    if not local_dir:
        logger.warning(f"local_dir 为空，赋值默认值，默认值为：项目根地址 / output")
        local_dir: Path = PROJECT_ROOT / PDF_PARSE_SERVICE_LOCAL_DIR
        state['local_dir'] = str(local_dir)

    # 1.4 将 pdf_path, local_dir 转成 Path
    pdf_path_obj: Path = Path(pdf_path)
    local_dir_obj: Path = Path(local_dir)

    # 1.5 pdf_path_obj 判断是否存在
    if not pdf_path_obj.exists():
        logger.error(f"存在 pdf_path 地址：{str(pdf_path_obj)}，但是地址没有对应文件，业务无法继续进行，提前终止！")
        raise FileNotFoundError(
            f"存在 pdf_path 地址：{str(pdf_path_obj)}，但是地址没有对应文件，业务无法继续进行，提前终止！")

    # 1.6 local_dir 是否是目录
    if not local_dir_obj.is_dir():
        logger.warning(f"存在 local_dir 地址：{str(local_dir_obj)}，但是没有对应的文件夹，创建对应的文件夹，业务继续！")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj


@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj: Path) -> str:
    """
    进行 minerU 的交互的zip的获取
    :param pdf_path_obj:
    :return:
    """
    # 2.1 向 minerU 服务器发送请求上传地址
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {infra_config.mineru_config.api_key}"
    }

    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": MINERU_MODEL_VERSION
    }

    url = f"{infra_config.mineru_config.base_url}/file-urls/batch"

    try:
        # 禁用代理检测，避免本地代理导致连接超时
        response = requests.post(url, headers=header, json=data, timeout=30, proxies={})
    except requests.exceptions.ConnectionError as e:
        logger.error(f"无法连接到 MinerU 服务，请检查网络连接或 MinerU 服务状态。错误：{str(e)}")
        raise RuntimeError(
            f"无法连接到 MinerU 服务，请检查网络连接或 MinerU 服务状态。如果 MinerU 服务不可用，请直接上传 MD 文件进行导入。"
        )
    except requests.exceptions.Timeout as e:
        logger.error(f"连接 MinerU 服务超时，请检查网络连接。错误：{str(e)}")
        raise RuntimeError(
            f"连接 MinerU 服务超时，请检查网络连接。如果 MinerU 服务不可用，请直接上传 MD 文件进行导入。"
        )

    if response.status_code != 200:
        logger.error(f"向 minerU 服务器申请上传文件解析，但是http状态码为：{response.status_code}，状态错误，无法继续业务！")
        raise RuntimeError(
            f"向 minerU 服务器申请上传文件解析，但是http状态码为：{response.status_code}，状态错误，无法继续业务！")
    response_dict = response.json()
    if response_dict.get('code', -1) != 0:
        logger.error(
            f"向 minerU 服务器申请上传文件解析，网络状态正常，服务业务状态异常，code = {response_dict.get('code', -1)}，错误原因：{response_dict.get('msg')}，无法进行业务！")
        raise RuntimeError(
            f"向 minerU 服务器申请上传文件解析，网络状态正常，服务业务状态异常，code = {response_dict.get('code', -1)}，错误原因：{response_dict.get('msg')}，无法进行业务！")

    batch_id = response_dict.get('data', {}).get('batch_id')
    file_upload_urls = response_dict.get('data', {}).get('file_urls', [])
    file_upload_url = None
    if len(file_upload_urls) > 0:
        file_upload_url = file_upload_urls[0]
    if not batch_id:
        logger.error(f"申请minerU解析文件，返回的batch_id为空，业务无法继续进行，业务中断！")
        raise ValueError(f"申请minerU解析文件，返回的batch_id为空，业务无法继续进行，业务中断！")

    logger.info(f"完成上传文件申请，batch_id={batch_id}，上传文件预签名地址：{file_upload_url}")

    # 2.2 向指定的url地址发起网络请求并且上传pdf文件
    # 注意：禁用代理检测（trust_env=False），避免本地代理导致连接超时
    try:
        with requests.Session() as session:
            session.trust_env = False  # 禁用代理检测
            file_data = pdf_path_obj.read_bytes()
            logger.info(f"开始上传文件，文件大小：{len(file_data)} bytes")
            upload_response = session.put(
                url=file_upload_url,
                data=file_data,
                timeout=120  # 增加超时时间到120秒
            )
            if upload_response.status_code != 200:
                logger.error(
                    f"向：{file_upload_url}上传文件，服务器返回的网络状态码为：{upload_response.status_code}，业务失败，提前终止！")
                raise RuntimeError(
                    f"向：{file_upload_url}上传文件，服务器返回的网络状态码为：{upload_response.status_code}，业务失败，提前终止！")
            logger.info(f"文件上传成功，状态码：{upload_response.status_code}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"上传文件到 MinerU 失败，网络连接错误。错误：{str(e)}")
        raise RuntimeError(f"上传文件到 MinerU 失败，网络连接错误。如果 MinerU 服务不可用，请直接上传 MD 文件进行导入。")
    except requests.exceptions.Timeout as e:
        logger.error(f"上传文件到 MinerU 超时（已等待120秒）。错误：{str(e)}")
        raise RuntimeError(f"上传文件到 MinerU 超时。如果 MinerU 服务不可用，请直接上传 MD 文件进行导入。")

    # 2.3 轮询向minerU获取batch_id解析状态 zip_url
    result_url = f"{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}"
    start_time = time.time()
    while True:
        # 1. 先判断时间，是否超时
        if time.time() - start_time >= MINERU_POLL_TIMEOUT_SECONDS:
            logger.error(f"轮询获取{batch_id}对应的解析结果超时！耗时为：{time.time() - start_time}")
            raise TimeoutError(f"轮询获取{batch_id}对应的解析结果超时！耗时为：{time.time() - start_time}")
        # 2. 没有超时向接口发起请求获取解析结果
        try:
            # 禁用代理检测
            poll_result = requests.get(url=result_url, headers=header, timeout=30, proxies={})
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"轮询 MinerU 结果时网络连接错误：{str(e)}，稍后再试！")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
        except requests.exceptions.Timeout as e:
            logger.warning(f"轮询 MinerU 结果超时：{str(e)}，稍后再试！")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
        except Exception as e:
            logger.warning(f"申请结果出现网络波动{str(e)}，稍后再试！")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
        # 3. 网络状态判定
        if poll_result.status_code != 200:
            if 500 <= poll_result.status_code < 600:
                logger.warning(f"申请结果出现网络状态错误：{poll_result.status_code}，稍后再试！")
                time.sleep(MINERU_POLL_INTERVAL_SECONDS)
                continue
            else:
                logger.error(
                    f"获取：{batch_id}对应的解析结果，服务器访问报错，http的状态码：{poll_result.status_code}，错误无法修复，业务失败，提前终止！")
                raise RuntimeError(
                    f"获取：{batch_id}对应的解析结果，服务器访问报错，http的状态码：{poll_result.status_code}，错误无法修复，业务失败，提前终止！")
        # 4. 业务状态判定
        poll_result_dict = poll_result.json()
        if poll_result_dict.get('code', -1) != 0:
            logger.error(
                f"获取:{batch_id}对应的解析结果,业务状态报错! 业务状态码:{poll_result_dict.get('code', -1)},错误信息:{poll_result_dict.get('msg')},业务失败,提前终止!")
            raise RuntimeError(
                f"获取:{batch_id}对应的解析结果,业务状态报错! 业务状态码:{poll_result_dict.get('code', -1)},错误信息:{poll_result_dict.get('msg')},业务失败,提前终止!")
        # 5. 获取解析结果和状态判定
        extract_result_list = poll_result_dict.get('data', {}).get('extract_result', [])
        if len(extract_result_list) == 0:
            logger.warning(f"解析结果为空，跳过本次，稍后再试！")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
        extract_result = extract_result_list[0]
        state = extract_result.get('state')
        if state == 'done':
            full_zip_url = extract_result.get('full_zip_url')
            if not full_zip_url:
                logger.error(f"获取:{batch_id}对应的解析结果,任务已经完成,但是full_zip_url没有地址!业务失败,提前终止!")
                raise ValueError(
                    f"获取:{batch_id}对应的解析结果,任务已经完成,但是full_zip_url没有地址!业务失败,提前终止!")
            return full_zip_url
        elif state == 'failed':
            logger.error(f"获取:{batch_id}对应的解析结果,任务解析失败!业务失败,提前终止!")
            raise ValueError(f"获取:{batch_id}对应的解析结果,任务解析失败!业务失败,提前终止!")
        else:
            logger.warning(f"本次解析，没有获得结果，继续下一次！")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue

@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_url: str, local_dir_obj: Path, file_name: str) -> Path:
    """
    地址下载、解压、重命名
    :param zip_url:
    :param local_dir_obj:
    :param file_name:
    :return:
    """
    # 1. 下载数据
    try:
        # 禁用代理检测
        response = requests.get(url=zip_url, timeout=MINERU_POLL_TIMEOUT_SECONDS, proxies={})
    except requests.exceptions.ConnectionError as e:
        logger.error(f"下载 MinerU 解析结果失败，网络连接错误。错误：{str(e)}")
        raise RuntimeError(f"下载 MinerU 解析结果失败，网络连接错误。如果 MinerU 服务不可用，请直接上传 MD 文件进行导入。")
    except requests.exceptions.Timeout as e:
        logger.error(f"下载 MinerU 解析结果超时。错误：{str(e)}")
        raise RuntimeError(f"下载 MinerU 解析结果超时。如果 MinerU 服务不可用，请直接上传 MD 文件进行导入。")

    if response.status_code != 200:
        logger.error(f"向指定地址:{zip_url}下载zip文件报错,状态码为:{response.status_code},业务无法继续进行!")
        raise RuntimeError(f"向指定地址:{zip_url}下载zip文件报错,状态码为:{response.status_code},业务无法继续进行!")

    zip_file_obj: Path = local_dir_obj / f"{file_name}.zip"
    zip_file_obj.write_bytes(response.content)

    # 2. 解压数据
    zip_extract_dir: Path = local_dir_obj / file_name
    if zip_extract_dir.is_dir():
        shutil.rmtree(zip_extract_dir)
    zip_extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_file_obj, zip_extract_dir)

    # 3. 重命名
    md_obj_list: list[Path] = list(zip_extract_dir.rglob("*.md"))
    if len(md_obj_list) == 0:
        logger.error(f"向指定地址:{zip_url}下载zip文件,解压后发现没有md文件,业务无法继续进行!!")
        raise ValueError(f"向指定地址:{zip_url}下载zip文件,解压后发现没有md文件,业务无法继续进行!!")

    for current_md_obj in md_obj_list:
        if current_md_obj.stem == file_name:
            logger.info(f"向指定地址:{zip_url}下载zip文件,解压后的文件名,等于原文件名{file_name},直接返回!!")
            return current_md_obj

    md_obj_path: Path = None
    for current_md_obj in md_obj_list:
        if current_md_obj.stem == 'full':
            md_obj_path = current_md_obj
            break

    if not md_obj_path:
        md_obj_path = md_obj_list[0]

    logger.info(f"触发了md文件的重命名机制,原名称:{md_obj_path.stem},目标名称:{file_name}")
    md_obj_path = md_obj_path.rename(md_obj_path.with_name(f"{file_name}.md"))
    return md_obj_path


@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir
    """
    # 1. 获取并校验参数
    pdf_path_obj, local_dir_obj = validate_pdf_paths(state)

    # 2. minerU解析pdf文件并返回zip的下载地址
    zip_url: str = upload_pdf_and_poll(pdf_path_obj)

    # 3. 根据zip_url下载并解压和重命名md文件
    md_path_obj: Path = download_and_extract_markdown(zip_url, local_dir_obj, pdf_path_obj.stem)

    # 4. 更新state md_path
    state['md_path'] = str(md_path_obj)

    return state
