from src.evaluation import evaluate_model
from src.models.baseline import BaselineModel
from src.models.retrieval_chroma import VectorStoreModel
from src.models.retrieval_st import RetrievalModel


# Example usage functions
def evaluate_baseline(
    chapter_path: str = "chapter.txt",
    benchmark_path: str = "benchmark.json",
    max_questions: int | None = None,
):
    """Evaluate the baseline model."""
    model = BaselineModel()
    evaluate_model(model, chapter_path, benchmark_path, max_questions)


def evaluate_retrieval_model(
    chapter_path: str = "chapter.txt",
    benchmark_path: str = "benchmark.json",
    max_questions: int | None = None,
):
    """Evaluate the retrieval model."""
    model = RetrievalModel()
    evaluate_model(model, chapter_path, benchmark_path, max_questions)


def evaluate_vector_store_model(
    chapter_path: str = "chapter.txt",
    benchmark_path: str = "benchmark.json",
    max_questions: int | None = None,
):
    """Evaluate the vector store model."""
    model = VectorStoreModel()
    evaluate_model(model, chapter_path, benchmark_path, max_questions)


if __name__ == "__main__":
    # Example: Evaluate all models

    print("\n=== Evaluating Retrieval Model ===")
    evaluate_retrieval_model()

    print("\n=== Evaluating Vector Store Model ===")
    evaluate_vector_store_model()
