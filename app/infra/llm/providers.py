"""
@Author: gl
@Date: 2026/6/23
@Desc: 统一封装聊天模型、嵌入模型、重排序模型
"""

from app.infra.config.providers import infra_config
from app.shared.model import get_bge_m3_ef, generate_embeddings, get_reranker_model
from app.shared.model.lm_utils import get_llm_client


class LLMProvider:
    def chat(self, model_name: str = None, json_mode: bool = None):
        """
        聊天模型
        :param model_name:模型名称，不传入时使用默认：.env
        :param json_mode:json模式参数
        :return:
        """
        return get_llm_client(model=model_name, json_mode=json_mode)

    def vision_chat(self, vision_model_name: str = None):
        """
        视觉模型
        :param vision_model_name:
        :return:
        """
        return get_llm_client(vision_model_name or infra_config.lm_config.lv_model)

    def bge_m3_embedding(self):
        """嵌入模型"""
        return get_bge_m3_ef()

    def generate_embeddings(self, texts: list[str]):
        """生成向量"""
        return generate_embeddings(texts)

    def reranker_model(self):
        """重排序模型"""
        return get_reranker_model()


llm_providers = LLMProvider()
