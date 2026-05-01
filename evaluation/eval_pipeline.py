"""End-to-end evaluation pipeline using RAGAS metrics."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import logging

from agent.langchain_agent import AgenticRAGPipeline
from hallucination.hallucination_detector import HallucinationDetector

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    test_dataset_path: str = "./data/eval_dataset.json"
    output_path: str = "./evaluation/results"
    min_faithfulness: float = 0.85
    min_context_precision: float = 0.80
    min_answer_relevancy: float = 0.80


def run_evaluation(config: Optional[EvalConfig] = None) -> dict:
    """Run full RAGAS evaluation suite on the RAG pipeline."""
    config = config or EvalConfig()
    pipeline = AgenticRAGPipeline()
    detector = HallucinationDetector()

    logger.info("Loading evaluation dataset...")
    with open(config.test_dataset_path) as f:
        eval_data = json.load(f)

    results = []
    for item in eval_data:
        result = pipeline.run(query=item["question"])
        hallucination = detector.score(
            question=item["question"],
            answer=result.answer,
            contexts=result.contexts,
        )
        results.append({
            "question": item["question"],
            "answer": result.answer,
            "contexts": result.contexts,
            "ground_truth": item.get("ground_truth", ""),
            "faithfulness": hallucination.faithfulness,
        })

    dataset = Dataset.from_list(results)
    ragas_results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    metrics = ragas_results.to_pandas().mean().to_dict()
    logger.info(f"Evaluation metrics: {metrics}")

    # Save results
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / "eval_results.csv", index=False)
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Check thresholds
    passed = (
        metrics.get("faithfulness", 0) >= config.min_faithfulness
        and metrics.get("context_precision", 0) >= config.min_context_precision
        and metrics.get("answer_relevancy", 0) >= config.min_answer_relevancy
    )
    metrics["evaluation_passed"] = passed
    print(f"\n{'PASSED' if passed else 'FAILED'}: Evaluation {'meets' if passed else 'does not meet'} quality thresholds")
    return metrics


if __name__ == "__main__":
    metrics = run_evaluation()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
