"""
评估指标计算模块
提供检索质量、生成质量等评估指标的计算函数
"""
from typing import List, Dict, Set
import numpy as np


def fuzzy_match(doc_id: str, relevant: Set[str]) -> bool:
    """
    模糊匹配文档ID

    Args:
        doc_id: 检索到的文档标识符（格式: "chunk_id|title|content"）
        relevant: 相关文档集合

    Returns:
        是否匹配
    """
    for rel in relevant:
        # 检查 relevant 是否是 doc_id 的子串
        if rel in doc_id:
            return True
        # 检查 doc_id 的 title 部分是否包含 relevant
        parts = doc_id.split("|")
        if len(parts) >= 2 and rel in parts[1]:
            return True
    return False


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int, use_fuzzy: bool = True) -> float:
    """
    计算 Precision@K
    Top-K 结果中相关文档的比例

    Args:
        retrieved: 检索到的文档ID列表（按相关性排序）
        relevant: 相关文档ID集合
        k: 取前K个结果
        use_fuzzy: 是否使用模糊匹配

    Returns:
        Precision@K 值
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if use_fuzzy:
        hits = sum(1 for doc in top_k if fuzzy_match(doc, relevant))
    else:
        hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int, use_fuzzy: bool = True) -> float:
    """
    计算 Recall@K
    Top-K 结果中检索到的相关文档比例

    Args:
        retrieved: 检索到的文档ID列表
        relevant: 相关文档ID集合
        k: 取前K个结果
        use_fuzzy: 是否使用模糊匹配

    Returns:
        Recall@K 值
    """
    if not relevant:
        return 1.0  # 没有相关文档时，recall 定义为 1
    top_k = retrieved[:k]
    if use_fuzzy:
        hits = sum(1 for doc in top_k if fuzzy_match(doc, relevant))
    else:
        hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str], use_fuzzy: bool = True) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)
    第一个相关文档的排名倒数

    Args:
        retrieved: 检索到的文档ID列表
        relevant: 相关文档ID集合
        use_fuzzy: 是否使用模糊匹配

    Returns:
        MRR 值
    """
    for i, doc in enumerate(retrieved):
        if use_fuzzy:
            if fuzzy_match(doc, relevant):
                return 1.0 / (i + 1)
        else:
            if doc in relevant:
                return 1.0 / (i + 1)
    return 0.0


def hit_rate(retrieved: List[str], relevant: Set[str], k: int = 10, use_fuzzy: bool = True) -> float:
    """
    计算 Hit Rate
    Top-K 中是否至少有一个相关文档

    Args:
        retrieved: 检索到的文档ID列表
        relevant: 相关文档ID集合
        k: 取前K个结果
        use_fuzzy: 是否使用模糊匹配

    Returns:
        1.0 如果命中，否则 0.0
    """
    top_k = retrieved[:k]
    if use_fuzzy:
        return 1.0 if any(fuzzy_match(doc, relevant) for doc in top_k) else 0.0
    else:
        return 1.0 if any(doc in relevant for doc in top_k) else 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int, use_fuzzy: bool = True) -> float:
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)
    考虑排名位置的评分

    Args:
        retrieved: 检索到的文档ID列表
        relevant: 相关文档ID集合
        k: 取前K个结果
        use_fuzzy: 是否使用模糊匹配

    Returns:
        NDCG@K 值
    """
    if not relevant:
        return 1.0

    # 计算 DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        if use_fuzzy:
            if fuzzy_match(doc, relevant):
                dcg += 1.0 / np.log2(i + 2)  # i+2 因为 log2(1) = 0
        else:
            if doc in relevant:
                dcg += 1.0 / np.log2(i + 2)

    # 计算理想 DCG
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def f1_at_k(retrieved: List[str], relevant: Set[str], k: int, use_fuzzy: bool = True) -> float:
    """
    计算 F1@K
    Precision 和 Recall 的调和平均

    Args:
        retrieved: 检索到的文档ID列表
        relevant: 相关文档ID集合
        k: 取前K个结果
        use_fuzzy: 是否使用模糊匹配

    Returns:
        F1@K 值
    """
    p = precision_at_k(retrieved, relevant, k, use_fuzzy)
    r = recall_at_k(retrieved, relevant, k, use_fuzzy)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def map_score(retrieved: List[str], relevant: Set[str], use_fuzzy: bool = True) -> float:
    """
    计算 MAP (Mean Average Precision)
    所有相关文档的平均精度

    Args:
        retrieved: 检索到的文档ID列表
        relevant: 相关文档ID集合
        use_fuzzy: 是否使用模糊匹配

    Returns:
        MAP 值
    """
    if not relevant:
        return 1.0

    hits = 0
    sum_precision = 0.0
    for i, doc in enumerate(retrieved):
        if use_fuzzy:
            if fuzzy_match(doc, relevant):
                hits += 1
                sum_precision += hits / (i + 1)
        else:
            if doc in relevant:
                hits += 1
                sum_precision += hits / (i + 1)

    return sum_precision / len(relevant)


def compute_retrieval_metrics(
    retrieved: List[str],
    relevant: Set[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    计算所有检索指标

    Args:
        retrieved: 检索到的文档ID列表
        relevant: 相关文档ID集合
        k_values: K 值列表

    Returns:
        所有指标的字典
    """
    metrics = {}

    # MRR 和 MAP（与 K 无关）
    metrics["mrr"] = mrr(retrieved, relevant, use_fuzzy=True)
    metrics["map"] = map_score(retrieved, relevant, use_fuzzy=True)

    # 各个 K 值的指标
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k, use_fuzzy=True)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k, use_fuzzy=True)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k, use_fuzzy=True)
        metrics[f"f1@{k}"] = f1_at_k(retrieved, relevant, k, use_fuzzy=True)
        metrics[f"hit_rate@{k}"] = hit_rate(retrieved, relevant, k, use_fuzzy=True)

    return metrics


def compute_generation_metrics(
    generated_answer: str,
    expected_answer: str,
    context: str
) -> Dict[str, float]:
    """
    计算生成质量指标（简化版本，使用字符串匹配）

    Args:
        generated_answer: 生成的答案
        expected_answer: 期望的答案
        context: 检索到的上下文

    Returns:
        生成指标字典
    """
    metrics = {}

    # 答案长度比
    metrics["answer_length"] = len(generated_answer)
    metrics["expected_length"] = len(expected_answer)

    # 关键词覆盖率（简单实现）
    expected_keywords = set(expected_answer.replace("。", " ").replace("，", " ").split())
    generated_keywords = set(generated_answer.replace("。", " ").replace("，", " ").split())

    if expected_keywords:
        keyword_coverage = len(expected_keywords & generated_keywords) / len(expected_keywords)
    else:
        keyword_coverage = 0.0
    metrics["keyword_coverage"] = keyword_coverage

    # 上下文忠实度（答案是否基于上下文）
    context_keywords = set(context.replace("。", " ").replace("，", " ").split())
    if context_keywords:
        faithfulness = len(context_keywords & generated_keywords) / len(generated_keywords) if generated_keywords else 0.0
    else:
        faithfulness = 0.0
    metrics["faithfulness"] = min(faithfulness, 1.0)

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    聚合多条评估结果的平均值

    Args:
        all_metrics: 多条评估结果列表

    Returns:
        平均指标字典
    """
    if not all_metrics:
        return {}

    aggregated = {}
    keys = all_metrics[0].keys()

    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        aggregated[f"avg_{key}"] = np.mean(values) if values else 0.0
        aggregated[f"std_{key}"] = np.std(values) if values else 0.0

    return aggregated
