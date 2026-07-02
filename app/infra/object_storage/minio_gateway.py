"""
@Author: gl
@Date: 2026/6/23
@Desc: 统一封装对象存储客户端与桶配置访问
"""

from minio import Minio

from app.infra.config.providers import infra_config
from app.shared.clients import get_minio_client


class MinioGateway:

    @property
    def bucket_name(self):
        """获取桶名称"""
        return infra_config.minio_config.bucket_name

    @property
    def minio_img_dir(self):
        """获取图片前缀"""
        return infra_config.minio_config.minio_img_dir

    @property
    def minio_client(self):
        return get_minio_client()

    def build_image_url(self, stem: str, image_name: str):
        """
        拼接访问地址
        :param stem:
        :param image_name:
        :return:
        """
        image_url = "https://" if infra_config.minio_config.minio_secure else "http://" + (
            f"{infra_config.minio_config.endpoint}"
            f"/{infra_config.minio_config.bucket_name}{infra_config.minio_config.minio_img_dir}/{stem}/{image_name}"
        )
        return image_url


minio_gateway = MinioGateway()
