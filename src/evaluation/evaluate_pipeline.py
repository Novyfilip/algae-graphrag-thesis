"""
evaluate_pipeline.py

Runs the RAGAS testset through the full RAG pipeline and scores the results.

Usage:
    python evaluate_pipeline.py
    python evaluate_pipeline.py --testset path/to/testset.csv --output path/to/results.csv
"""

import argparse
import pandas as pd
import time

from pipeline import setup, run_pipeline
from config import OUTPUTS_DIR

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


def run_evaluation(testset_path, output_path, components):
    """Run all testset questions through pipeline and evaluate with RAGAS."""

    print(f"Loading testset from {testset_path}")
    df = pd.read_csv(testset_path)
    print(f"Testset size: {len(df)} questions\n")

    # Run each question through the pipeline
    answers = []
    all_contexts = []

    for i, row in df.iterrows():
        question = row["user_input"]
        print(f"[{i+1}/{len(df)}] {question[:80]}...")

        try:
            answer, contexts, top_chunks = run_pipeline(question, components)
            answers.append(answer)
            all_contexts.append(contexts)
        except Exception as e:
            print(f"  ERROR: {e}")
            answers.append("Error generating answer.")
            all_contexts.append([])

        # Small delay to respect rate limits
        time.sleep(0.5)

    print(f"\nAll {len(df)} questions processed.")

    # Build RAGAS evaluation dataset
    eval_dataset = Dataset.from_dict({
        "question": df["user_input"].tolist(),
        "answer": answers,
        "contexts": all_contexts,
        "ground_truth": df["reference"].tolist(),
    })

    # Run RAGAS scoring
    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n===== EVALUATION RESULTS =====")
    print(result)

    # Save detailed results
    results_df = result.to_pandas()
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS")
    parser.add_argument(
        "--testset",
        type=str,
        default=str(OUTPUTS_DIR / "ragas_testset.csv"),
        help="Path to RAGAS testset CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "evaluation_results.csv"),
        help="Path to save evaluation results"
    )
    args = parser.parse_args()

    components = setup()
    run_evaluation(args.testset, args.output, components)