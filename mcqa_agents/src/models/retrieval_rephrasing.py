import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.models.base import BaseModel


class StatementBasedRAGModel(BaseModel):
    def __init__(self, retriever_name="all-MiniLM-L6-v2", qa_model_name="google/flan-t5-small", top_k=2):
        self.retriever = SentenceTransformer(retriever_name)
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)
        self.qa_model.eval()
        self.top_k = top_k

    def chunk_text(self, text: str) -> list[str]:
        return text.split("\n\n")

    def generate_statements(self, question: str, answers: list[str]) -> list[str]:
        statements = []
        for idx, answer in enumerate(answers):
            prompt = f"Turn the QA pair into a statement.\nQuestion: {question}\nAnswer: {answer}\nStatement:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                output = self.qa_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            statement = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            if not statement.endswith((".", "!", "?")):
                statement += "."

            print(f"Q: {question} | A{idx + 1}: {answer} -> Statement: {statement}")
            statements.append(statement)
        return statements

    def retrieve_context_for_statements(self, statements: list[str], chunks: list[str]) -> list[list[str]]:
        chunk_embeddings = self.retriever.encode(chunks, convert_to_tensor=True)
        all_contexts = []

        for statement in statements:
            stmt_embedding = self.retriever.encode(statement, convert_to_tensor=True)
            scores = util.cos_sim(stmt_embedding, chunk_embeddings)[0]
            top_indices = torch.topk(scores, k=self.top_k).indices.tolist()
            contexts = [chunks[i] for i in top_indices]
            all_contexts.append(contexts)

        return all_contexts

    def evaluate_statement_with_context(self, statement: str, contexts: list[str]) -> float:
        context = " ".join(contexts)
        prompt = (
            f"Given the context and a statement, determine if the statement is supported.\n\n"
            f"Context:\n{context}\n\nStatement:\n{statement}\n\n"
            f"Is the statement supported? Answer only with SUPPORTED or NOT_SUPPORTED."
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.qa_model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip().upper()
        return 0.8 if "SUPPORTED" in response and "NOT_SUPPORTED" not in response else 0.2

    def format_final_prompt(self, question: str, answers: list[str], best_contexts: list[str]) -> str:
        context = " ".join(best_contexts)
        options = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(answers))
        return f"Context:\n{context}\n\nQuestion: {question}\n\nOptions:\n{options}\n\nWhich option is correct?"

    def decode_answer(self, generated_text: str, answers: list[str]) -> int:
        text = generated_text.lower()
        for i, answer in enumerate(answers):
            if answer.lower() in text:
                return i
        for i in range(len(answers)):
            if str(i + 1) in text:
                return i
        return -1

    def predict(
        self, questions: str | list[str], answers_list: list[str] | list[list[str]], text: str
    ) -> int | list[int]:
        if isinstance(questions, str):
            questions, answers_list = [questions], [answers_list]

        chunks = self.chunk_text(text)
        results = []

        for question, answers in zip(questions, answers_list, strict=False):
            statements = self.generate_statements(question, answers)
            contexts = self.retrieve_context_for_statements(statements, chunks)
            scores = [self.evaluate_statement_with_context(s, c) for s, c in zip(statements, contexts, strict=False)]

            best_idx = scores.index(max(scores))
            final_prompt = self.format_final_prompt(question, answers, contexts[best_idx])

            inputs = self.tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = self.qa_model.generate(**inputs, max_new_tokens=10)
            text_output = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

            pred_idx = self.decode_answer(text_output, answers)
            results.append(pred_idx if pred_idx != -1 else best_idx)

        return results if len(results) > 1 else results[0]
