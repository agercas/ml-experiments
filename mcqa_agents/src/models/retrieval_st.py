import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.models.base import BaseModel


class RetrievalModel(BaseModel):
    """Model that uses sentence transformers for retrieval-augmented generation."""

    def __init__(
        self, retriever_name: str = "all-MiniLM-L6-v2", qa_model_name: str = "google/flan-t5-small", top_k: int = 1
    ):
        self.retriever = SentenceTransformer(retriever_name)
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)
        self.qa_model.eval()
        self.top_k = top_k

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks for retrieval."""
        return text.split("\n\n")

    def retrieve_context(self, question: str, chunks: list[str], top_k: int = 1) -> list[str]:
        """Retrieve most relevant chunks for the question."""
        question_embedding = self.retriever.encode(question, convert_to_tensor=True)
        chunk_embeddings = self.retriever.encode(chunks, convert_to_tensor=True)
        scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        return [chunks[i] for i in top_indices]

    def format_prompt(self, context: str, question: str, answers: list[str]) -> str:
        """Format the prompt with context."""
        return (
            f"Context: {context}\nQuestion: {question}\nOptions: "
            + ", ".join(f"{j + 1}. {ans}" for j, ans in enumerate(answers))
            + "\nAnswer:"
        )

    def decode_answer(self, generated_text: str, answers: list[str]) -> int:
        """Decode the generated text to get the predicted answer index."""
        predicted_index = -1

        # First try to match answer text
        for idx, ans in enumerate(answers):
            if ans.lower() in generated_text.lower():
                predicted_index = idx
                break

        # If no text match, try to match answer number
        if predicted_index == -1:
            for idx, _ans in enumerate(answers):
                if str(idx + 1) in generated_text:
                    predicted_index = idx
                    break

        return predicted_index

    def predict(
        self, questions: str | list[str], answers_list: list[str] | list[list[str]], text: str
    ) -> int | list[int]:
        """Predict answers using retrieval-augmented generation."""
        if isinstance(questions, str):
            questions = [questions]
            answers_list = [answers_list]

        chunks = self.chunk_text(text)
        results = []

        for question, answers in zip(questions, answers_list, strict=False):
            contexts = self.retrieve_context(question, chunks, top_k=self.top_k)
            print(f"Retrieved context for question '{question}':")
            for idx, ctx in enumerate(contexts):
                print(f"  Context {idx + 1}: {ctx[:100]}...")  # Print first 100 chars
            print()

            context = contexts[0] if contexts else ""
            prompt = self.format_prompt(context, question, answers)

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.qa_model.generate(**inputs, max_new_tokens=10)

            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            predicted_index = self.decode_answer(predicted_text, answers)

            results.append(predicted_index)

        return results if len(results) > 1 else results[0]
