"""
RAG 评估模块
提供检索质量、生成质量、端到端效果的评估能力
"""
from app.rag_eval.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
    f1_at_k,
    map_score,
    compute_retrieval_metrics,
    compute_generation_metrics,
    aggregate_metrics
)

from app.rag_eval.retrieval_eval import RetrievalEvaluator, print_eval_report
from app.rag_eval.generation_eval import GenerationEvaluator, print_generation_report
from app.rag_eval.e2e_eval import E2EEvaluator, print_e2e_report

__all__ = [
    # 指标计算
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
    "f1_at_k",
    "map_score",
    "compute_retrieval_metrics",
    "compute_generation_metrics",
    "aggregate_metrics",
    # 评估器
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "E2EEvaluator",
    # 报告打印
    "print_eval_report",
    "print_generation_report",
    "print_e2e_report"
]
