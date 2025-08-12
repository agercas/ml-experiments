import json
from typing import Any

from src.models.base import BaseModel


def evaluate_model(
    model: BaseModel, chapter_path: str, benchmark_path: str, max_questions: int | None = None, verbose: bool = True
) -> dict[str, Any]:
    """
    Evaluate a model on the benchmark dataset.

    Args:
        model: Model instance that implements the BaseModel interface
        chapter_path: Path to the chapter text file
        benchmark_path: Path to the benchmark JSON file
        max_questions: Maximum number of questions to evaluate (None for all)
        verbose: Whether to print detailed results during evaluation

    Returns:
        Dictionary containing evaluation results and metadata
    """
    # Load chapter text
    with open(chapter_path, encoding="utf-8") as f:
        chapter_text = f.read()

    # Load benchmark data
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    # Extract data from benchmark
    questions: list[str] = [item["question"] for item in benchmark]
    answers_list: list[list[str]] = [item["answers"] for item in benchmark]
    correct_indices: list[int] = [item["correct_index"] for item in benchmark]

    # Store original total for metadata
    original_total = len(questions)

    # Limit questions if specified
    if max_questions is not None:
        questions = questions[:max_questions]
        answers_list = answers_list[:max_questions]
        correct_indices = correct_indices[:max_questions]

    # Get predictions from model
    predicted_indices: list[int] = model.predict(questions, answers_list, chapter_text)

    # Calculate accuracy and collect detailed results
    num_correct = 0
    detailed_results = []

    for i, (question, answers, correct_index, predicted_index) in enumerate(
        zip(questions, answers_list, correct_indices, predicted_indices, strict=False)
    ):
        is_correct = predicted_index == correct_index
        num_correct += is_correct

        # Store detailed result for this question
        question_result = {
            "question_id": i + 1,
            "question": question,
            "answers": answers,
            "correct_index": correct_index,
            "predicted_index": predicted_index,
            "correct_answer": answers[correct_index],
            "predicted_answer": answers[predicted_index] if predicted_index != -1 else "N/A",
            "is_correct": is_correct,
        }
        detailed_results.append(question_result)

        if verbose:
            print(f"Q{i + 1}: {question}")
            print(f"  Correct Answer   [{correct_index}]: {answers[correct_index]}")
            print(
                f"  Predicted Answer [{predicted_index}]: {answers[predicted_index] if predicted_index != -1 else 'N/A'}"
            )
            print("  ✅ Correct!" if is_correct else "  ❌ Incorrect")
            print()

    total = len(questions)
    accuracy = num_correct / total * 100

    if verbose:
        print(f"Model Accuracy: {accuracy:.2f}% ({num_correct}/{total})")

    # Get model configuration
    model_config = {}
    if hasattr(model, "__dict__"):
        model_config = {k: v for k, v in model.__dict__.items() if not k.startswith("_") and not callable(v)}

    # Return comprehensive results dictionary
    return {
        "model_name": model.__class__.__name__,
        "model_config": model_config,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total_questions": total,
        "original_total_questions": original_total,
        "max_questions_limit": max_questions,
        "chapter_path": chapter_path,
        "benchmark_path": benchmark_path,
        "detailed_results": detailed_results,
        "summary_stats": {
            "accuracy_percentage": accuracy,
            "correct_count": num_correct,
            "incorrect_count": total - num_correct,
            "total_evaluated": total,
        },
    }


def aggregate_results_to_markdown(results_list: list[dict[str, Any]]) -> str:
    """
    Convert a list of evaluation results to a markdown report.

    Args:
        results_list: List of result dictionaries from evaluate_model

    Returns:
        Markdown formatted string with aggregated results
    """
    if not results_list:
        return "# Evaluation Results\n\nNo results to display."

    markdown = "# Model Evaluation Results\n\n"

    # Summary table
    markdown += "## Summary\n\n"
    markdown += "| Model | Configuration | Accuracy | Correct/Total |\n"
    markdown += "|-------|---------------|----------|---------------|\n"

    model_config_filter = ["text_chunks", "text", "chunk_embeddings", "collection", "collection_name", "client"]
    for result in results_list:
        model_name = result["model_name"]
        config_str = (
            ", ".join([f"{k}={v}" for k, v in result["model_config"].items() if k not in model_config_filter])
            if result["model_config"]
            else "default"
        )
        accuracy = result["accuracy"]
        correct_total = f"{result['num_correct']}/{result['total_questions']}"
        markdown += f"| {model_name} | {config_str} | {accuracy:.2f}% | {correct_total} |\n"

    # Detailed results for each model
    markdown += "\n## Detailed Results\n\n"

    for i, result in enumerate(results_list, 1):
        markdown += f"### {i}. {result['model_name']}\n\n"

        if result["model_config"]:
            markdown += "**Configuration:**\n"
            for key, value in result["model_config"].items():
                markdown += f"- {key}: {value}\n"
            markdown += "\n"

        markdown += "**Performance:**\n"
        markdown += f"- Accuracy: {result['accuracy']:.2f}%\n"
        markdown += f"- Correct answers: {result['num_correct']}/{result['total_questions']}\n"
        markdown += f"- Dataset: {result['benchmark_path']}\n"
        markdown += f"- Chapter: {result['chapter_path']}\n\n"

        # Show first few incorrect answers for analysis
        incorrect_results = [r for r in result["detailed_results"] if not r["is_correct"]]
        if incorrect_results:
            markdown += "**Sample Incorrect Answers:**\n"
            for wrong in incorrect_results[:3]:  # Show first 3 incorrect
                markdown += f"- Q{wrong['question_id']}: {wrong['question'][:100]}...\n"
                markdown += f"  - Correct: [{wrong['correct_index']}] {wrong['correct_answer']}\n"
                markdown += f"  - Predicted: [{wrong['predicted_index']}] {wrong['predicted_answer']}\n"
            markdown += "\n"

    return markdown
