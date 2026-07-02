"""
检索质量评估模块
评估向量检索、HyDE检索、RRF融合、Rerank各层的效果
"""
import json
from typing import List, Dict, Set, Optional
from pathlib import Path

from app.infra.llm.providers import llm_providers
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.rag.query.embedding_search_service import search_by_embedding
from app.rag.query.hyde_search_service import search_by_hyde
from app.rag.query.rrf_service import fuse_by_rrf
from app.rag.query.rerank_service import rerank_documents
from app.rag_eval.metrics import compute_retrieval_metrics, aggregate_metrics
from app.shared.runtime.logger import logger


class RetrievalEvaluator:
    """检索评估器"""

    def __init__(self, eval_dataset_path: str = None):
        """
        初始化评估器

        Args:
            eval_dataset_path: 评估数据集路径
        """
        if eval_dataset_path is None:
            eval_dataset_path = str(
                Path(__file__).parent / "eval_dataset.json"
            )

        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            self.eval_dataset = json.load(f)

        logger.info(f"加载评估数据集: {len(self.eval_dataset)} 条")

    def extract_chunk_ids(self, chunks: List[Dict]) -> List[str]:
        """
        从 chunks 中提取标识符（chunk_id、title、content 前50字符）

        Args:
            chunks: 检索到的 chunks 列表

        Returns:
            标识符列表
        """
        ids = []
        for chunk in chunks:
            # 构造复合标识符，用于模糊匹配
            chunk_id = str(chunk.get("chunk_id", ""))
            title = chunk.get("title", "")
            content = chunk.get("content", "")[:50]
            # 返回格式: "chunk_id|title|content_prefix"
            identifier = f"{chunk_id}|{title}|{content}"
            ids.append(identifier)
        return ids

    def is_relevant(self, retrieved_id: str, relevant_docs: Set[str]) -> bool:
        """
        判断检索到的文档是否相关（支持模糊匹配）

        Args:
            retrieved_id: 检索到的文档标识符
            relevant_docs: 相关文档集合

        Returns:
            是否相关
        """
        for relevant in relevant_docs:
            # 如果 relevant 是 retrieved_id 的子串，或者反之
            if relevant in retrieved_id or retrieved_id in relevant:
                return True
            # 检查 title 部分
            parts = retrieved_id.split("|")
            if len(parts) >= 2 and relevant in parts[1]:
                return True
        return False

    def eval_single_query_embedding(
        self,
        query: str,
        item_names: List[str],
        relevant_docs: Set[str],
        k: int = 10
    ) -> Dict[str, float]:
        """
        评估单条查询的向量检索效果

        Args:
            query: 查询问题
            item_names: 法律法规名称列表
            relevant_docs: 相关文档ID集合
            k: Top-K

        Returns:
            评估指标
        """
        # 构造 state
        state = {
            "rewritten_query": query,
            "item_names": item_names,
        }

        try:
            # 执行向量检索
            embedding_chunks = search_by_embedding(state)
            retrieved_ids = self.extract_chunk_ids(embedding_chunks)

            # 调试日志
            logger.info(f"[DEBUG] Query: {query}")
            logger.info(f"[DEBUG] Relevant docs: {relevant_docs}")
            logger.info(f"[DEBUG] Retrieved IDs (first 3): {retrieved_ids[:3]}")
            if embedding_chunks:
                logger.info(f"[DEBUG] First chunk keys: {embedding_chunks[0].keys()}")
                logger.info(f"[DEBUG] First chunk title: {embedding_chunks[0].get('title', 'N/A')}")
                logger.info(f"[DEBUG] First chunk content (first 50): {embedding_chunks[0].get('content', 'N/A')[:50]}")

            # 计算指标
            metrics = compute_retrieval_metrics(retrieved_ids, relevant_docs, [1, 3, 5, k])
            metrics["retrieval_count"] = len(retrieved_ids)
            metrics["success"] = True

        except Exception as e:
            logger.error(f"向量检索评估失败: {str(e)}")
            metrics = {"success": False, "error": str(e)}

        return metrics

    def eval_single_query_hyde(
        self,
        query: str,
        item_names: List[str],
        relevant_docs: Set[str],
        k: int = 10
    ) -> Dict[str, float]:
        """
        评估单条查询的 HyDE 检索效果

        Args:
            query: 查询问题
            item_names: 法律法规名称列表
            relevant_docs: 相关文档ID集合
            k: Top-K

        Returns:
            评估指标
        """
        state = {
            "rewritten_query": query,
            "item_names": item_names,
        }

        try:
            hyde_chunks = search_by_hyde(state)
            retrieved_ids = self.extract_chunk_ids(hyde_chunks)

            metrics = compute_retrieval_metrics(retrieved_ids, relevant_docs, [1, 3, 5, k])
            metrics["retrieval_count"] = len(retrieved_ids)
            metrics["success"] = True

        except Exception as e:
            logger.error(f"HyDE 检索评估失败: {str(e)}")
            metrics = {"success": False, "error": str(e)}

        return metrics

    def eval_single_query_rrf(
        self,
        query: str,
        item_names: List[str],
        relevant_docs: Set[str],
        k: int = 10
    ) -> Dict[str, float]:
        """
        评估单条查询的 RRF 融合效果

        Args:
            query: 查询问题
            item_names: 法律法规名称列表
            relevant_docs: 相关文档ID集合
            k: Top-K

        Returns:
            评估指标
        """
        state = {
            "rewritten_query": query,
            "item_names": item_names,
        }

        try:
            # 先执行两路检索
            embedding_chunks = search_by_embedding(state)
            hyde_chunks = search_by_hyde(state)

            state["embedding_chunks"] = embedding_chunks
            state["hyde_embedding_chunks"] = hyde_chunks

            # 执行 RRF 融合
            rrf_result = fuse_by_rrf(state)
            rrf_chunks = rrf_result.get("rrf_chunks", []) if isinstance(rrf_result, dict) else rrf_result
            retrieved_ids = self.extract_chunk_ids(rrf_chunks)

            metrics = compute_retrieval_metrics(retrieved_ids, relevant_docs, [1, 3, 5, k])
            metrics["retrieval_count"] = len(retrieved_ids)
            metrics["success"] = True

        except Exception as e:
            logger.error(f"RRF 融合评估失败: {str(e)}")
            metrics = {"success": False, "error": str(e)}

        return metrics

    def eval_single_query_rerank(
        self,
        query: str,
        item_names: List[str],
        relevant_docs: Set[str],
        k: int = 10
    ) -> Dict[str, float]:
        """
        评估单条查询的 Rerank 效果

        Args:
            query: 查询问题
            item_names: 法律法规名称列表
            relevant_docs: 相关文档ID集合
            k: Top-K

        Returns:
            评估指标
        """
        state = {
            "rewritten_query": query,
            "item_names": item_names,
        }

        try:
            # 执行完整检索链路
            embedding_chunks = search_by_embedding(state)
            hyde_chunks = search_by_hyde(state)

            state["embedding_chunks"] = embedding_chunks
            state["hyde_embedding_chunks"] = hyde_chunks
            # 评估时不使用网络搜索，但需要添加占位文档避免校验失败
            state["web_search_docs"] = [{"title": "placeholder", "snippet": "", "chunk_id": "placeholder", "url": ""}]

            # RRF 融合
            rrf_result = fuse_by_rrf(state)
            # rrf_result 返回的是列表，需要包装成 rerank_service 期望的格式
            rrf_chunks_list = rrf_result.get("rrf_chunks", []) if isinstance(rrf_result, dict) else rrf_result
            state["rrf_chunks"] = {"embedding_chunks": rrf_chunks_list}

            # Rerank
            reranked_docs = rerank_documents(state)
            retrieved_ids = self.extract_chunk_ids(reranked_docs)

            metrics = compute_retrieval_metrics(retrieved_ids, relevant_docs, [1, 3, 5, k])
            metrics["retrieval_count"] = len(retrieved_ids)
            metrics["success"] = True

        except Exception as e:
            logger.error(f"Rerank 评估失败: {str(e)}")
            metrics = {"success": False, "error": str(e)}

        return metrics

    def eval_retrieval_pipeline(
        self,
        sample: Dict,
        stages: List[str] = ["embedding", "hyde", "rrf", "rerank"]
    ) -> Dict[str, Dict[str, float]]:
        """
        评估单条查询的完整检索链路

        Args:
            sample: 评估样本
            stages: 要评估的阶段列表

        Returns:
            各阶段的评估指标
        """
        query = sample["query"]
        relevant_docs = set(sample.get("relevant_docs", []))
        relevant_laws = sample.get("relevant_laws", [])

        # 使用法律法规名称作为 item_names
        item_names = relevant_laws

        results = {}

        if "embedding" in stages:
            results["embedding"] = self.eval_single_query_embedding(
                query, item_names, relevant_docs
            )

        if "hyde" in stages:
            results["hyde"] = self.eval_single_query_hyde(
                query, item_names, relevant_docs
            )

        if "rrf" in stages:
            results["rrf"] = self.eval_single_query_rrf(
                query, item_names, relevant_docs
            )

        if "rerank" in stages:
            results["rerank"] = self.eval_single_query_rerank(
                query, item_names, relevant_docs
            )

        return results

    def run_evaluation(
        self,
        stages: List[str] = ["embedding", "hyde", "rrf", "rerank"],
        max_samples: Optional[int] = None
    ) -> Dict:
        """
        运行完整评估

        Args:
            stages: 要评估的阶段列表
            max_samples: 最大评估样本数（None 表示全部）

        Returns:
            评估结果
        """
        dataset = self.eval_dataset[:max_samples] if max_samples else self.eval_dataset

        logger.info(f"开始评估，样本数: {len(dataset)}，评估阶段: {stages}")

        all_results = []
        stage_metrics = {stage: [] for stage in stages}

        for i, sample in enumerate(dataset):
            logger.info(f"评估样本 {i+1}/{len(dataset)}: {sample['query'][:50]}...")

            # 评估各阶段
            results = self.eval_retrieval_pipeline(sample, stages)
            results["sample_id"] = sample["id"]
            results["query"] = sample["query"]
            all_results.append(results)

            # 收集各阶段指标
            for stage in stages:
                if stage in results and results[stage].get("success"):
                    stage_metrics[stage].append(results[stage])

        # 聚合各阶段指标
        aggregated = {}
        for stage in stages:
            if stage_metrics[stage]:
                aggregated[stage] = aggregate_metrics(stage_metrics[stage])
            else:
                aggregated[stage] = {}

        return {
            "total_samples": len(dataset),
            "stages_evaluated": stages,
            "aggregated_metrics": aggregated,
            "detailed_results": all_results
        }


def print_eval_report(eval_results: Dict):
    """
    打印评估报告

    Args:
        eval_results: 评估结果
    """
    print("\n" + "=" * 80)
    print("检索质量评估报告")
    print("=" * 80)
    print(f"总样本数: {eval_results['total_samples']}")
    print(f"评估阶段: {eval_results['stages_evaluated']}")
    print()

    for stage, metrics in eval_results["aggregated_metrics"].items():
        print(f"\n--- {stage.upper()} 阶段 ---")
        if not metrics:
            print("  无有效数据")
            continue

        # 打印关键指标
        key_metrics = [
            "avg_precision@5", "avg_recall@5", "avg_ndcg@5",
            "avg_mrr", "avg_map", "avg_hit_rate@5"
        ]
        for key in key_metrics:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")

    print("\n" + "=" * 80)
