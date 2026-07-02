"""
端到端评估模块
评估完整 RAG 流程的效果和性能
"""
import json
import time
from typing import List, Dict, Optional
from pathlib import Path

from app.process.query.agent.main_graph import query_app
from app.process.query.agent.state import create_query_default_state
from app.rag_eval.metrics import compute_retrieval_metrics, compute_generation_metrics
from app.shared.runtime.logger import logger


class E2EEvaluator:
    """端到端评估器"""

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

    def eval_single_e2e(self, sample: Dict) -> Dict:
        """
        评估单条查询的端到端效果

        Args:
            sample: 评估样本

        Returns:
            评估结果
        """
        query = sample["query"]
        expected_answer = sample["expected_answer"]
        relevant_docs = set(sample.get("relevant_docs", []))

        logger.info(f"E2E 评估: {query[:50]}...")

        # 构造初始状态
        state = create_query_default_state(
            session_id=f"eval_{sample['id']}",
            original_query=query,
            is_stream=False
        )

        # 记录开始时间
        start_time = time.time()

        try:
            # 执行完整 RAG 流程
            result_state = query_app.invoke(state)

            # 计算耗时
            elapsed_time = time.time() - start_time

            # 提取结果
            generated_answer = result_state.get("answer", "")
            reranked_docs = result_state.get("reranked_docs", [])
            image_urls = result_state.get("image_urls", [])

            # 提取检索到的文档ID
            retrieved_ids = []
            for doc in reranked_docs:
                doc_id = doc.get("chunk_id") or doc.get("title", "")
                retrieved_ids.append(str(doc_id))

            # 计算检索指标
            retrieval_metrics = compute_retrieval_metrics(
                retrieved_ids, relevant_docs, [1, 3, 5]
            )

            # 计算生成指标
            context = "\n".join([doc.get("text", "") for doc in reranked_docs[:3]])
            generation_metrics = compute_generation_metrics(
                generated_answer, expected_answer, context
            )

            return {
                "sample_id": sample["id"],
                "query": query,
                "success": True,
                "elapsed_time": elapsed_time,
                "answer_length": len(generated_answer),
                "retrieved_count": len(retrieved_ids),
                "has_images": len(image_urls) > 0,
                "retrieval_metrics": retrieval_metrics,
                "generation_metrics": generation_metrics,
                "generated_answer": generated_answer[:200] + "..." if len(generated_answer) > 200 else generated_answer
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"E2E 评估失败: {str(e)}")
            return {
                "sample_id": sample["id"],
                "query": query,
                "success": False,
                "elapsed_time": elapsed_time,
                "error": str(e)
            }

    def run_evaluation(
        self,
        max_samples: Optional[int] = None
    ) -> Dict:
        """
        运行端到端评估

        Args:
            max_samples: 最大评估样本数

        Returns:
            评估结果
        """
        dataset = self.eval_dataset[:max_samples] if max_samples else self.eval_dataset

        logger.info(f"开始 E2E 评估，样本数: {len(dataset)}")

        all_results = []
        successful_results = []

        for sample in dataset:
            result = self.eval_single_e2e(sample)
            all_results.append(result)

            if result["success"]:
                successful_results.append(result)

        # 聚合指标
        aggregated = self._aggregate_results(successful_results)

        return {
            "total_samples": len(dataset),
            "successful_samples": len(successful_results),
            "failed_samples": len(dataset) - len(successful_results),
            "aggregated_metrics": aggregated,
            "detailed_results": all_results
        }

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """
        聚合评估结果

        Args:
            results: 成功的评估结果列表

        Returns:
            聚合指标
        """
        if not results:
            return {}

        # 性能指标
        avg_time = sum(r["elapsed_time"] for r in results) / len(results)
        max_time = max(r["elapsed_time"] for r in results)
        min_time = min(r["elapsed_time"] for r in results)

        # 检索指标
        retrieval_keys = ["precision@5", "recall@5", "ndcg@5", "mrr", "hit_rate@5"]
        retrieval_agg = {}
        for key in retrieval_keys:
            values = [r["retrieval_metrics"].get(key, 0) for r in results if "retrieval_metrics" in r]
            retrieval_agg[f"avg_{key}"] = sum(values) / len(values) if values else 0

        # 生成指标
        generation_keys = ["keyword_coverage", "faithfulness"]
        generation_agg = {}
        for key in generation_keys:
            values = [r["generation_metrics"].get(key, 0) for r in results if "generation_metrics" in r]
            generation_agg[f"avg_{key}"] = sum(values) / len(values) if values else 0

        # 其他指标
        avg_answer_length = sum(r["answer_length"] for r in results) / len(results)
        avg_retrieved_count = sum(r["retrieved_count"] for r in results) / len(results)
        image_rate = sum(1 for r in results if r.get("has_images")) / len(results)

        return {
            "performance": {
                "avg_latency": avg_time,
                "max_latency": max_time,
                "min_latency": min_time
            },
            "retrieval": retrieval_agg,
            "generation": generation_agg,
            "other": {
                "avg_answer_length": avg_answer_length,
                "avg_retrieved_count": avg_retrieved_count,
                "image_rate": image_rate
            }
        }


def print_e2e_report(eval_results: Dict):
    """
    打印端到端评估报告

    Args:
        eval_results: 评估结果
    """
    print("\n" + "=" * 80)
    print("端到端评估报告")
    print("=" * 80)
    print(f"总样本数: {eval_results['total_samples']}")
    print(f"成功样本数: {eval_results['successful_samples']}")
    print(f"失败样本数: {eval_results['failed_samples']}")

    metrics = eval_results["aggregated_metrics"]

    if not metrics:
        print("\n无有效评估数据")
        print("=" * 80)
        return

    # 性能指标
    print("\n--- 性能指标 ---")
    perf = metrics.get("performance", {})
    print(f"  平均延迟: {perf.get('avg_latency', 0):.2f}s")
    print(f"  最大延迟: {perf.get('max_latency', 0):.2f}s")
    print(f"  最小延迟: {perf.get('min_latency', 0):.2f}s")

    # 检索指标
    print("\n--- 检索指标 ---")
    retrieval = metrics.get("retrieval", {})
    for key, value in retrieval.items():
        print(f"  {key}: {value:.4f}")

    # 生成指标
    print("\n--- 生成指标 ---")
    generation = metrics.get("generation", {})
    for key, value in generation.items():
        print(f"  {key}: {value:.4f}")

    # 其他指标
    print("\n--- 其他指标 ---")
    other = metrics.get("other", {})
    print(f"  平均答案长度: {other.get('avg_answer_length', 0):.0f} 字符")
    print(f"  平均检索文档数: {other.get('avg_retrieved_count', 0):.1f}")
    print(f"  图片引用率: {other.get('image_rate', 0):.2%}")

    # 显示部分详细结果
    print("\n--- 部分详细结果 ---")
    for result in eval_results["detailed_results"][:3]:
        if result["success"]:
            print(f"\n  样本: {result['sample_id']}")
            print(f"    问题: {result['query'][:50]}...")
            print(f"    耗时: {result['elapsed_time']:.2f}s")
            print(f"    答案: {result['generated_answer'][:100]}...")

    print("\n" + "=" * 80)
