import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.models.base import BaseModel


class BaselineModel(BaseModel):
    """Baseline model using google/flan-t5-small without any context enhancement."""

    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def format_prompt(self, question: str, answers: list[str]) -> str:
        """Format the prompt for the baseline model."""
        return (
            f"Question: {question}\nOptions: "
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
        """Predict answers using the baseline model without context."""
        if isinstance(questions, str):
            questions = [questions]
            answers_list = [answers_list]

        results = []

        for question, answers in zip(questions, answers_list, strict=False):
            prompt = self.format_prompt(question, answers)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10)

            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            predicted_index = self.decode_answer(predicted_text, answers)

            results.append(predicted_index)

        return results if len(results) > 1 else results[0]
