import chromadb
import torch
from chromadb.utils import embedding_functions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.models.base import BaseModel


class VectorStoreModel(BaseModel):
    """Model that uses ChromaDB vector store for retrieval-augmented generation."""

    def __init__(
        self, qa_model_name: str = "google/flan-t5-small", top_k: int = 1, collection_name: str = "collection_01"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)
        self.qa_model.eval()
        self.top_k = top_k
        self.collection_name = collection_name

        # Initialize ChromaDB
        self.client = chromadb.Client()

        # Use default embedding function (you can customize this)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = None

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks for storage."""
        return text.split("\n\n")

    def initialize_vector_store(self, text: str):
        """Initialize the vector store with text chunks."""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name, embedding_function=self.sentence_transformer_ef
            )
            print(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            print(f"Error getting collection: {e}")
            # Create new collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=self.collection_name, embedding_function=self.sentence_transformer_ef
            )
            print(f"Created new collection: {self.collection_name}")

            # Add chunks to the collection
            chunks = self.chunk_text(text)
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

            self.collection.add(documents=chunks, ids=chunk_ids)
            print(f"Added {len(chunks)} chunks to vector store")

    def retrieve_context(self, question: str, top_k: int = 1) -> list[str]:
        """Retrieve most relevant chunks for the question from vector store."""
        if self.collection is None:
            raise ValueError("Vector store not initialized. Call initialize_vector_store first.")

        results = self.collection.query(query_texts=[question], n_results=top_k)

        return results["documents"][0] if results["documents"] else []

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
        """Predict answers using vector store retrieval-augmented generation."""
        if isinstance(questions, str):
            questions = [questions]
            answers_list = [answers_list]

        # Initialize vector store if not already done
        if self.collection is None:
            self.initialize_vector_store(text)

        results = []

        for question, answers in zip(questions, answers_list, strict=False):
            contexts = self.retrieve_context(question, top_k=self.top_k)
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
