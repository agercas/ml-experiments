import os

import pydantic
from google import genai
from google.colab import userdata

from src.models.base import BaseModel


class GeminiAnswer(pydantic.BaseModel):
    answer: str
    reasoning: str


class GeminiReasoningModel(BaseModel):
    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        api_key = os.getenv("GOOGLE_API_KEY") or userdata.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing gemini key")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def create_prompt(self, text: str, questions: list[str], answers_list: list[list[str]]) -> str:
        prompt = f"""You are given the following passage:\n\n{text}\n\n"""
        prompt += "Based on the passage, answer the following multiple-choice questions. For each question, choose the best answer and explain your reasoning.\n\n"
        for i, (question, options) in enumerate(zip(questions, answers_list, strict=False), start=1):
            prompt += f"Q{i}: {question}\n"
            for j, opt in enumerate(options):
                prompt += f"{chr(65 + j)}. {opt}\n"
            prompt += "\n"
        prompt += "Respond in JSON format as a list of objects, each with 'answer' (A-D) and 'reasoning'."
        return prompt

    def predict(self, questions: list[str], answers_list: list[list[str]], text: str) -> list[int]:
        prompt = self.create_prompt(text, questions, answers_list)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[GeminiAnswer],
            },
        )

        parsed: list[GeminiAnswer] = response.parsed
        print("🧠 Gemini Reasoning Output:")
        for idx, item in enumerate(parsed):
            print(f"Q{idx + 1}: Answer={item.answer}, Reasoning={item.reasoning}")

        indices = [ord(ans.answer.upper()) - ord("A") for ans in parsed]
        return indices
