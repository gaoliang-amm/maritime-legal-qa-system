"""
生成质量评估模块
评估答案生成的忠实度、相关性、正确性
"""
import json
from typing import List, Dict, Optional
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_providers
from app.rag.query.answer_service import generate_answer
from app.rag_eval.metrics import compute_generation_metrics, aggregate_metrics
from app.shared.runtime.logger import logger


# LLM 评估提示词
FAITHFULNESS_PROMPT = """你是一个答案质量评估专家。请评估以下答案是否忠实于提供的上下文。

上下文：
{context}

问题：
{query}

答案：
{answer}

请从以下维度评估并返回 JSON：
{{
    "faithfulness_score": 0-1 的分数，表示答案基于上下文的程度
    "hallucination_detected": true/false，是否检测到幻觉
    "hallucination_details": "如果检测到幻觉，说明具体哪些内容是幻觉"
    "explanation": "评估理由"
}}

只返回 JSON，不要其他内容。"""

RELEVANCY_PROMPT = """你是一个答案质量评估专家。请评估以下答案是否回答了用户的问题。

问题：
{query}

答案：
{answer}

请从以下维度评估并返回 JSON：
{{
    "relevancy_score": 0-1 的分数，表示答案与问题的相关程度
    "completeness_score": 0-1 的分数，表示答案的完整性
    "explanation": "评估理由"
}}

只返回 JSON，不要其他内容。"""

CORRECTNESS_PROMPT = """你是一个答案质量评估专家。请评估以下答案是否正确。

标准答案：
{expected_answer}

生成的答案：
{answer}

请从以下维度评估并返回 JSON：
{{
    "correctness_score": 0-1 的分数，表示答案的正确程度
    "key_points_covered": ["覆盖的关键点列表"],
    "key_points_missing": ["缺失的关键点列表"],
    "explanation": "评估理由"
}}

只返回 JSON，不要其他内容。"""


class GenerationEvaluator:
    """生成评估器"""

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

    def call_llm_for_eval(self, prompt: str) -> Dict:
        """
        调用 LLM 进行评估

        Args:
            prompt: 评估提示词

        Returns:
            评估结果字典
        """
        try:
            llm_client = llm_providers.chat(json_mode=True)
            messages = [
                SystemMessage(content="你是一个专业的答案质量评估专家。"),
                HumanMessage(content=prompt)
            ]
            chain = llm_client | JsonOutputParser()
            result = chain.invoke(messages)
            return result
        except Exception as e:
            logger.error(f"LLM 评估调用失败: {str(e)}")
            return {"error": str(e)}

    def eval_faithfulness(
        self,
        query: str,
        answer: str,
        context: str
    ) -> Dict:
        """
        评估答案的忠实度

        Args:
            query: 用户问题
            answer: 生成的答案
            context: 检索到的上下文

        Returns:
            忠实度评估结果
        """
        prompt = FAITHFULNESS_PROMPT.format(
            context=context,
            query=query,
            answer=answer
        )
        return self.call_llm_for_eval(prompt)

    def eval_relevancy(self, query: str, answer: str) -> Dict:
        """
        评估答案的相关性

        Args:
            query: 用户问题
            answer: 生成的答案

        Returns:
            相关性评估结果
        """
        prompt = RELEVANCY_PROMPT.format(
            query=query,
            answer=answer
        )
        return self.call_llm_for_eval(prompt)

    def eval_correctness(
        self,
        answer: str,
        expected_answer: str
    ) -> Dict:
        """
        评估答案的正确性

        Args:
            answer: 生成的答案
            expected_answer: 标准答案

        Returns:
            正确性评估结果
        """
        prompt = CORRECTNESS_PROMPT.format(
            expected_answer=expected_answer,
            answer=answer
        )
        return self.call_llm_for_eval(prompt)

    def eval_single_generation(
        self,
        sample: Dict,
        generated_answer: str,
        context: str
    ) -> Dict:
        """
        评估单条生成结果

        Args:
            sample: 评估样本
            generated_answer: 生成的答案
            context: 检索到的上下文

        Returns:
            评估结果
        """
        query = sample["query"]
        expected_answer = sample["expected_answer"]

        results = {}

        # 1. 忠实度评估
        faithfulness = self.eval_faithfulness(query, generated_answer, context)
        results["faithfulness"] = faithfulness

        # 2. 相关性评估
        relevancy = self.eval_relevancy(query, generated_answer)
        results["relevancy"] = relevancy

        # 3. 正确性评估
        correctness = self.eval_correctness(generated_answer, expected_answer)
        results["correctness"] = correctness

        # 4. 基础指标
        basic_metrics = compute_generation_metrics(
            generated_answer, expected_answer, context
        )
        results["basic_metrics"] = basic_metrics

        return results

    def run_evaluation(
        self,
        generated_answers: List[Dict],
        max_samples: Optional[int] = None
    ) -> Dict:
        """
        运行生成评估

        Args:
            generated_answers: 生成的答案列表，格式为 [{"sample_id": "...", "answer": "...", "context": "..."}]
            max_samples: 最大评估样本数

        Returns:
            评估结果
        """
        dataset = self.eval_dataset[:max_samples] if max_samples else self.eval_dataset

        # 建立 sample_id 到生成答案的映射
        answer_map = {a["sample_id"]: a for a in generated_answers}

        logger.info(f"开始生成评估，样本数: {len(dataset)}")

        all_results = []
        faithfulness_scores = []
        relevancy_scores = []
        correctness_scores = []

        for sample in dataset:
            sample_id = sample["id"]

            if sample_id not in answer_map:
                logger.warning(f"样本 {sample_id} 无生成答案，跳过")
                continue

            gen_answer = answer_map[sample_id]
            answer = gen_answer.get("answer", "")
            context = gen_answer.get("context", "")

            logger.info(f"评估样本 {sample_id}: {sample['query'][:50]}...")

            # 评估
            result = self.eval_single_generation(sample, answer, context)
            result["sample_id"] = sample_id
            all_results.append(result)

            # 收集分数
            if "faithfulness" in result and "faithfulness_score" in result["faithfulness"]:
                faithfulness_scores.append(result["faithfulness"]["faithfulness_score"])
            if "relevancy" in result and "relevancy_score" in result["relevancy"]:
                relevancy_scores.append(result["relevancy"]["relevancy_score"])
            if "correctness" in result and "correctness_score" in result["correctness"]:
                correctness_scores.append(result["correctness"]["correctness_score"])

        # 计算平均分数
        aggregated = {
            "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            "avg_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
            "avg_correctness": sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0,
            "total_evaluated": len(all_results)
        }

        return {
            "total_samples": len(dataset),
            "evaluated_samples": len(all_results),
            "aggregated_metrics": aggregated,
            "detailed_results": all_results
        }


def print_generation_report(eval_results: Dict):
    """
    打印生成评估报告

    Args:
        eval_results: 评估结果
    """
    print("\n" + "=" * 80)
    print("生成质量评估报告")
    print("=" * 80)
    print(f"总样本数: {eval_results['total_samples']}")
    print(f"已评估样本数: {eval_results['evaluated_samples']}")
    print()

    metrics = eval_results["aggregated_metrics"]
    print("--- 聚合指标 ---")
    print(f"  平均忠实度 (Faithfulness): {metrics['avg_faithfulness']:.4f}")
    print(f"  平均相关性 (Relevancy): {metrics['avg_relevancy']:.4f}")
    print(f"  平均正确性 (Correctness): {metrics['avg_correctness']:.4f}")

    print("\n--- 详细结果 ---")
    for result in eval_results["detailed_results"][:5]:  # 只显示前5条
        print(f"\n  样本: {result['sample_id']}")
        if "faithfulness" in result and "faithfulness_score" in result["faithfulness"]:
            print(f"    忠实度: {result['faithfulness']['faithfulness_score']:.2f}")
        if "relevancy" in result and "relevancy_score" in result["relevancy"]:
            print(f"    相关性: {result['relevancy']['relevancy_score']:.2f}")
        if "correctness" in result and "correctness_score" in result["correctness"]:
            print(f"    正确性: {result['correctness']['correctness_score']:.2f}")

    print("\n" + "=" * 80)
