"""
评估入口脚本
支持命令行运行不同类型的评估
"""
import argparse
import json
import sys
from pathlib import Path

from app.shared.runtime.logger import logger
from app.rag_eval.retrieval_eval import RetrievalEvaluator, print_eval_report
from app.rag_eval.generation_eval import GenerationEvaluator, print_generation_report
from app.rag_eval.e2e_eval import E2EEvaluator, print_e2e_report


def run_retrieval_eval(args):
    """运行检索评估"""
    logger.info("=== 开始检索质量评估 ===")

    evaluator = RetrievalEvaluator(args.dataset)

    stages = args.stages.split(",") if args.stages else ["embedding", "hyde", "rrf", "rerank"]

    results = evaluator.run_evaluation(
        stages=stages,
        max_samples=args.max_samples
    )

    # 打印报告
    print_eval_report(results)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {output_path}")

    return results


def run_generation_eval(args):
    """运行生成评估"""
    logger.info("=== 开始生成质量评估 ===")

    # 首先需要运行 E2E 评估获取生成的答案
    e2e_evaluator = E2EEvaluator(args.dataset)
    e2e_results = e2e_evaluator.run_evaluation(max_samples=args.max_samples)

    # 提取生成的答案
    generated_answers = []
    for result in e2e_results["detailed_results"]:
        if result["success"]:
            generated_answers.append({
                "sample_id": result["sample_id"],
                "answer": result.get("generated_answer", ""),
                "context": ""  # 可以从检索结果中提取
            })

    # 运行生成评估
    gen_evaluator = GenerationEvaluator(args.dataset)
    results = gen_evaluator.run_evaluation(
        generated_answers=generated_answers,
        max_samples=args.max_samples
    )

    # 打印报告
    print_generation_report(results)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {output_path}")

    return results


def run_e2e_eval(args):
    """运行端到端评估"""
    logger.info("=== 开始端到端评估 ===")

    evaluator = E2EEvaluator(args.dataset)

    results = evaluator.run_evaluation(
        max_samples=args.max_samples
    )

    # 打印报告
    print_e2e_report(results)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {output_path}")

    return results


def run_all_eval(args):
    """运行所有评估"""
    logger.info("=== 开始全量评估 ===")

    # 1. 检索评估
    logger.info("\n--- 阶段1: 检索质量评估 ---")
    retrieval_results = run_retrieval_eval(args)

    # 2. E2E 评估
    logger.info("\n--- 阶段2: 端到端评估 ---")
    e2e_results = run_e2e_eval(args)

    # 3. 生成评估
    logger.info("\n--- 阶段3: 生成质量评估 ---")
    gen_results = run_generation_eval(args)

    # 汇总结果
    all_results = {
        "retrieval": retrieval_results,
        "e2e": e2e_results,
        "generation": gen_results
    }

    # 保存汇总结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"全量评估结果已保存到: {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="RAG 系统评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行检索评估
  python -m app.rag_eval.run_eval retrieval

  # 运行 E2E 评估，最多评估 5 条
  python -m app.rag_eval.run_eval e2e --max-samples 5

  # 运行全量评估，保存结果
  python -m app.rag_eval.run_eval all --output eval_results.json

  # 运行特定阶段的检索评估
  python -m app.rag_eval.run_eval retrieval --stages embedding,rerank
        """
    )

    parser.add_argument(
        "eval_type",
        nargs="?",
        default="retrieval",
        choices=["retrieval", "generation", "e2e", "all"],
        help="评估类型（默认: retrieval）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="评估数据集路径（默认使用内置数据集）"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="最大评估样本数（默认: 3）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出文件路径"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="检索评估阶段（逗号分隔：embedding,hyde,rrf,rerank）"
    )

    args = parser.parse_args()

    # 运行评估
    logger.info(f"运行评估类型: {args.eval_type}")
    if args.eval_type == "retrieval":
        run_retrieval_eval(args)
    elif args.eval_type == "generation":
        run_generation_eval(args)
    elif args.eval_type == "e2e":
        run_e2e_eval(args)
    elif args.eval_type == "all":
        run_all_eval(args)


if __name__ == "__main__":
    main()
