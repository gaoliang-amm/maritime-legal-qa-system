"""
@Author: gl
@Date: 2026/6/17
@Desc: Milvus入库服务，准备集合→删除旧数据→批量插入向量化chunks
"""
from pymilvus import DataType

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log

@step_log("require_embeddings_content")
def require_embeddings_content(state: dict) -> list[dict]:
    """
    校验导入状态中是否已经生成切块结果
    :param state:
    :return: 已通过校验的切块列表
    """

    # 1. 从 state 中获取核心数据
    embedding_content = state.get("embedding_content", [])

    # 2. 校验chunks
    if not embedding_content:
        logger.error("embedding_content为空，无法继续业务！")
        raise ValueError("embedding_content为空，无法继续业务！")

    # 3. 返回结果
    return embedding_content

@step_log("prepare_chunks_collection")
def prepare_chunks_collection() -> None:
    """准备 Milvus 切片集合"""
    # 1. 获取 Milvus 客户端 和 集合名称
    milvus_client = milvus_gateway.milvus_client
    collection_name = milvus_gateway.chunk_collection_name

    # 2. 判断集合是否已经存在
    if milvus_client.has_collection(collection_name=collection_name):
        return

    # 3. 创建schema，添加字段
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 4. 创建索引
    index_params = milvus_client.prepare_index_params()
    # 稠密向量添加索引
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={
            "M": 64,
            "efConstruction": 100
        }
    )
    # 稀疏向量添加索引
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE"
        }
    )

    # 5. 创建集合
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

@step_log("remove_old_chunks")
def remove_old_chunks(file_title: str) -> None:
    """根据主体名称删除已存在的切片记录"""
    # 访问 Milvus 客户端并执行删除操作
    milvus_gateway.milvus_client.delete(
        collection_name=milvus_gateway.chunk_collection_name,
        filter=f"file_title=='{file_title}'"
    )

@step_log("insert_embeddings_content")
def insert_embeddings_content(embeddings_content: list[dict]) -> None:
    """批量插入数据"""
    result = milvus_gateway.milvus_client.insert(
        collection_name=milvus_gateway.chunk_collection_name,
        data=embeddings_content
    )
    # 记录插入结果
    logger.info(f"插入数据成功！总条数：{result.get('insert_count', 0)}")
    logger.info(f"插入数据主键回显：{result.get('ids', [])}")

@step_log("index_chunks")
def index_chunks(state: dict) -> dict:
    """
    入库服务：
    1. 准备集合 schema 和索引
    2. 根据 item_name 删除旧数据
    3. 批量插入新的 chunks
    4. 回写 chunk_id 等入库结果
    """
    # 1. 先校验切片存在，避免把空数据写入向量库
    embeddings_content = require_embeddings_content(state)

    # 2. 集合不存在时先自动创建，保证首次导入能直接跑通
    prepare_chunks_collection()

    # 3. 获取主体名称，用于幂等性清理
    file_title = state.get("file_title", "")

    # 4. 同一主体重复导入时先删除旧数据，保持当前导入结果覆盖旧版本
    if file_title:
        remove_old_chunks(file_title)

    # 5. 批量插入新数据
    insert_embeddings_content(embeddings_content)

    return state
