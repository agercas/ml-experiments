import re

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.models.base import BaseModel


class ChainOfThoughtModel(BaseModel):
    """
    Advanced model that uses Chain-of-Thought reasoning with multiple retrieval passes
    to break down complex questions into reasoning steps.
    """

    def __init__(
        self,
        qa_model_name: str = "google/flan-t5-small",
        retriever_name: str = "all-MiniLM-L6-v2",
        max_reasoning_steps: int = 3,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)
        self.qa_model.eval()
        self.retriever = SentenceTransformer(retriever_name)
        self.max_reasoning_steps = max_reasoning_steps

        # Pre-compute chunks for efficiency
        self.text_chunks = []
        self.chunk_embeddings = None

    def prepare_text(self, text: str):
        """Prepare text by creating chunks and embeddings once."""
        self.text_chunks = self.create_comprehensive_chunks(text)
        if self.text_chunks:
            self.chunk_embeddings = self.retriever.encode(self.text_chunks, convert_to_tensor=True)

    def create_comprehensive_chunks(self, text: str) -> list[str]:
        """Create multiple types of chunks for comprehensive retrieval."""
        chunks = []

        # 1. Sentence-level chunks (for precise facts)
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        chunks.extend(sentences)

        # 2. Paragraph-level chunks (for context)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks.extend(paragraphs)

        # 3. Multi-paragraph chunks (for broader context)
        chunk_size = 3
        for i in range(0, len(paragraphs), chunk_size - 1):
            multi_para = "\n\n".join(paragraphs[i : i + chunk_size])
            if multi_para and multi_para not in chunks:
                chunks.append(multi_para)

        # 4. Character-focused chunks
        character_chunks = []
        for chunk in chunks:
            # Find chunks with multiple character names (likely interactions)
            char_names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", chunk)
            if len(set(char_names)) >= 2:  # At least 2 different characters
                character_chunks.append(chunk)

        chunks.extend(character_chunks)

        # Remove duplicates and very short chunks
        unique_chunks = []
        seen = set()
        for chunk in chunks:
            if chunk not in seen and len(chunk) > 20:
                seen.add(chunk)
                unique_chunks.append(chunk)

        return unique_chunks

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> list[tuple]:
        """Retrieve most relevant chunks with similarity scores."""
        if not self.text_chunks or self.chunk_embeddings is None:
            return []

        query_embedding = self.retriever.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]

        # Get top-k chunks with scores
        top_indices = torch.topk(scores, k=min(top_k, len(self.text_chunks))).indices.tolist()
        top_scores = [scores[i].item() for i in top_indices]

        return [(self.text_chunks[i], score) for i, score in zip(top_indices, top_scores, strict=False)]

    def decompose_question(self, question: str, answers: list[str]) -> list[str]:
        """Break down complex questions into simpler sub-questions."""
        question_lower = question.lower()
        sub_questions = [question]  # Always include original question

        # Identify question type and create sub-questions
        if "react" in question_lower or "response" in question_lower:
            # For reaction questions, need context about the event and the reaction
            who_match = re.search(r"how did (\w+)", question_lower)
            if who_match:
                character = who_match.group(1)
                sub_questions.extend(
                    [f"What happened with {character}?", f"What did {character} do?", f"How did {character} respond?"]
                )

        elif (
            "sister" in question_lower
            or "brother" in question_lower
            or "father" in question_lower
            or "mother" in question_lower
        ):
            # For relationship questions
            relation_match = re.search(r"who is (\w+) the (\w+) of", question_lower)
            if relation_match:
                person, relation = relation_match.groups()
                sub_questions.extend([f"Who is {person}?", f"What is {person}'s family?", f"{person} {relation}"])

        elif "why" in question_lower:
            # For causal questions
            sub_questions.extend(
                [question.replace("Why", "What happened"), "What was the reason?", "What caused this?"]
            )

        elif "what happened" in question_lower:
            # For event questions
            sub_questions.extend(["What occurred?", "What was the outcome?", "What was the result?"])

        return sub_questions

    def reason_step_by_step(self, question: str, answers: list[str], context_chunks: list[str]) -> str:
        """Perform step-by-step reasoning using retrieved context."""

        reasoning_prompt = f"""Let's think step by step to answer this question.

Context information:
{chr(10).join(f"- {chunk}" for chunk in context_chunks)}

Question: {question}

Answer choices:
{chr(10).join(f"{i + 1}. {ans}" for i, ans in enumerate(answers))}

Let me reason through this step by step:

Step 1: What is the question asking?
The question is asking: {question}

Step 2: What relevant information do I have?
From the context, I can see:"""

        # Generate reasoning steps
        inputs = self.tokenizer(reasoning_prompt, return_tensors="pt", truncation=True, max_length=800)

        with torch.no_grad():
            outputs = self.qa_model.generate(
                **inputs, max_new_tokens=150, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )

        reasoning_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the reasoning part
        if "From the context, I can see:" in reasoning_text:
            reasoning_part = reasoning_text.split("From the context, I can see:")[1].strip()
        else:
            reasoning_part = "Unable to extract clear reasoning."

        return reasoning_part

    def final_answer_with_reasoning(self, question: str, answers: list[str], reasoning: str, best_context: str) -> str:
        """Generate final answer using the reasoning and context."""

        final_prompt = f"""Based on my step-by-step reasoning and the context, I need to choose the correct answer.

Context: {best_context}

My reasoning: {reasoning}

Question: {question}

Choices:
{chr(10).join(f"{i + 1}. {ans}" for i, ans in enumerate(answers))}

Based on the context and my reasoning, the answer is number:"""

        inputs = self.tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=900)

        with torch.no_grad():
            outputs = self.qa_model.generate(
                **inputs, max_new_tokens=20, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )

        final_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return final_text

    def multi_pass_retrieval(self, question: str, answers: list[str]) -> list[str]:
        """Perform multiple retrieval passes with different strategies."""
        all_retrieved = []

        # Pass 1: Direct question retrieval
        direct_chunks = self.retrieve_relevant_chunks(question, top_k=3)
        all_retrieved.extend([chunk for chunk, score in direct_chunks if score > 0.3])

        # Pass 2: Answer-guided retrieval
        for answer in answers:
            answer_query = f"{question} {answer}"
            answer_chunks = self.retrieve_relevant_chunks(answer_query, top_k=2)
            all_retrieved.extend([chunk for chunk, score in answer_chunks if score > 0.25])

        # Pass 3: Sub-question retrieval
        sub_questions = self.decompose_question(question, answers)
        for sub_q in sub_questions[1:]:  # Skip original question
            sub_chunks = self.retrieve_relevant_chunks(sub_q, top_k=2)
            all_retrieved.extend([chunk for chunk, score in sub_chunks if score > 0.2])

        # Pass 4: Entity-focused retrieval
        entities = re.findall(r"\b[A-Z][a-z]+\b", question)
        for entity in entities:
            entity_chunks = self.retrieve_relevant_chunks(entity, top_k=2)
            all_retrieved.extend([chunk for chunk, score in entity_chunks if score > 0.2])

        # Remove duplicates while preserving order
        unique_chunks = []
        seen = set()
        for chunk in all_retrieved:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)

        # Limit to top chunks to avoid overwhelming the model
        return unique_chunks[:8]

    def predict(
        self, questions: str | list[str], answers_list: list[str] | list[list[str]], text: str
    ) -> int | list[int]:
        """Predict using Chain-of-Thought reasoning with multiple retrieval passes."""
        if isinstance(questions, str):
            questions = [questions]
            answers_list = [answers_list]

        # Prepare text chunks once
        self.prepare_text(text)

        results = []

        for question, answers in zip(questions, answers_list, strict=False):
            print(f"\n🧠 Chain-of-Thought Analysis for: {question}")

            # Step 1: Multi-pass retrieval
            print("🔍 Performing multi-pass retrieval...")
            retrieved_chunks = self.multi_pass_retrieval(question, answers)
            print(f"📚 Retrieved {len(retrieved_chunks)} relevant chunks")

            # Step 2: Step-by-step reasoning
            print("💭 Performing step-by-step reasoning...")
            reasoning = self.reason_step_by_step(question, answers, retrieved_chunks[:5])
            print(f"🎯 Reasoning: {reasoning[:200]}...")

            # Step 3: Select best context for final answer
            best_context = "\n".join(retrieved_chunks[:3])  # Use top 3 chunks

            # Step 4: Generate final answer with reasoning
            print("✅ Generating final answer...")
            final_response = self.final_answer_with_reasoning(question, answers, reasoning, best_context)

            # Step 5: Decode answer
            predicted_index = self.enhanced_decode_answer(final_response, answers)

            print(
                f"🎲 Final prediction: {predicted_index} ({answers[predicted_index] if predicted_index != -1 else 'N/A'})"
            )

            results.append(predicted_index)

        return results if len(results) > 1 else results[0]

    def enhanced_decode_answer(self, generated_text: str, answers: list[str]) -> int:
        """Enhanced answer decoding with Chain-of-Thought considerations."""
        generated_lower = generated_text.lower().strip()

        # Look for explicit answer numbers first
        # Pattern 1: "the answer is number X" or "answer is X"
        answer_patterns = [
            r"answer is (?:number )?([1-4])",
            r"the answer is ([1-4])",
            r"choice ([1-4])",
            r"option ([1-4])",
            r"\b([1-4])\b(?:\s*\.|\s*\)|\s*$)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, generated_lower)
            if match:
                try:
                    num = int(match.group(1))
                    if 1 <= num <= len(answers):
                        return num - 1
                except Exception as e:
                    print(f"Error decoding answer: {e}")
                    continue

        # Fallback to text matching
        best_match_idx = -1
        best_score = 0

        for idx, answer in enumerate(answers):
            answer_words = set(answer.lower().split())
            gen_words = set(generated_lower.split())

            if answer_words and gen_words:
                intersection = answer_words.intersection(gen_words)
                score = len(intersection) / len(answer_words)

                if score > best_score and score > 0.2:
                    best_score = score
                    best_match_idx = idx

        return best_match_idx
