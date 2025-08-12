import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.models.base import BaseModel


class ReasoningModel(BaseModel):
    """
    Enhanced model that uses multiple retrieval strategies and reasoning techniques
    to improve performance on complex literary comprehension tasks.
    """

    def __init__(
        self, qa_model_name: str = "google/flan-t5-small", retriever_name: str = "all-MiniLM-L6-v2", top_k: int = 3
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)
        self.qa_model.eval()
        self.retriever = SentenceTransformer(retriever_name)
        self.top_k = top_k

    def advanced_chunk_text(self, text: str) -> list[str]:
        """Create overlapping chunks with sentence boundaries for better context."""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        chunk_size = 3  # Number of paragraphs per chunk
        overlap = 1  # Overlap between chunks

        for i in range(0, len(paragraphs), chunk_size - overlap):
            chunk = "\n\n".join(paragraphs[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def extract_entities_and_relationships(self, text: str) -> list[str]:
        """Extract key entities and relationships for better retrieval."""
        # Create focused chunks around key entities and relationships
        entity_chunks = []

        # Look for character relationships
        relationship_patterns = [
            r"(\w+)\s+(?:is|was|are|were)\s+(?:the\s+)?(?:son|daughter|brother|sister|father|mother|wife|husband)\s+of\s+(\w+)",
            r"(\w+),?\s+(?:son|daughter|brother|sister)\s+of\s+(\w+)",
            r"(\w+)\s+and\s+(\w+)\s+(?:are|were)\s+(?:brothers?|sisters?|siblings?)",
        ]

        import re

        sentences = text.split(". ")

        for sentence in sentences:
            # Check for relationship patterns
            for pattern in relationship_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    entity_chunks.append(sentence + ".")

            # Look for character actions and reactions
            action_keywords = [
                "react",
                "response",
                "answer",
                "reply",
                "said",
                "told",
                "spoke",
                "called",
                "sent away",
                "welcomed",
            ]
            if any(keyword in sentence.lower() for keyword in action_keywords):
                entity_chunks.append(sentence + ".")

        return entity_chunks

    def multi_strategy_retrieval(self, question: str, text: str) -> str:
        """Use multiple retrieval strategies to find the most relevant context."""
        # Strategy 1: Standard paragraph-based chunking
        standard_chunks = self.advanced_chunk_text(text)

        # Strategy 2: Entity and relationship extraction
        entity_chunks = self.extract_entities_and_relationships(text)

        # Strategy 3: Keyword-based retrieval
        keyword_chunks = []

        # Extract key entities from the question
        import re

        # Look for proper nouns (capitalized words)
        entities_in_question = re.findall(r"\b[A-Z][a-z]+\b", question)

        sentences = text.split(". ")
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # If sentence contains entities from question, include surrounding context
            if any(entity.lower() in sentence_lower for entity in entities_in_question):
                keyword_chunks.append(sentence + ".")

        # Combine all chunks
        all_chunks = standard_chunks + entity_chunks + keyword_chunks

        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)

        # Retrieve top-k most relevant chunks
        if not unique_chunks:
            return text[:1000]  # Fallback to first 1000 chars

        question_embedding = self.retriever.encode(question, convert_to_tensor=True)
        chunk_embeddings = self.retriever.encode(unique_chunks, convert_to_tensor=True)
        scores = util.cos_sim(question_embedding, chunk_embeddings)[0]

        # Get top chunks
        top_indices = torch.topk(scores, k=min(self.top_k, len(unique_chunks))).indices.tolist()
        top_chunks = [unique_chunks[i] for i in top_indices]

        return "\n\n".join(top_chunks)

    def create_enhanced_prompt(self, context: str, question: str, answers: list[str]) -> str:
        """Create a more detailed prompt that encourages careful reasoning."""
        prompt = f"""Read the following context carefully and answer the question based ONLY on the information provided.

Context: {context}

Question: {question}

Available options:
"""
        for j, ans in enumerate(answers):
            prompt += f"{j + 1}. {ans}\n"

        prompt += """\nInstructions: 
- Read the context carefully
- Look for specific details that directly answer the question
- Choose the option that matches the information in the context
- If unsure, choose the option most supported by the text

Answer (number only):"""

        return prompt

    def decode_answer_enhanced(self, generated_text: str, answers: list[str]) -> int:
        """Enhanced answer decoding with multiple fallback strategies."""
        generated_lower = generated_text.lower().strip()

        # Strategy 1: Look for exact number
        import re

        number_match = re.search(r"\b([1-4])\b", generated_text)
        if number_match:
            try:
                num = int(number_match.group(1))
                if 1 <= num <= len(answers):
                    return num - 1
            except Exception as e:
                print(f"Error decoding answer: {e}")
                pass

        # Strategy 2: Look for answer text (partial and full matches)
        best_match_idx = -1
        best_match_score = 0

        for idx, ans in enumerate(answers):
            ans_lower = ans.lower()

            # Exact match
            if ans_lower in generated_lower:
                return idx

            # Partial match scoring
            ans_words = set(ans_lower.split())
            gen_words = set(generated_lower.split())

            if ans_words and gen_words:
                intersection = ans_words.intersection(gen_words)
                score = len(intersection) / len(ans_words)

                if score > best_match_score and score > 0.3:  # Threshold for partial match
                    best_match_score = score
                    best_match_idx = idx

        if best_match_idx != -1:
            return best_match_idx

        # Strategy 3: Look for ordinal numbers
        ordinals = ["first", "second", "third", "fourth"]
        for idx, ordinal in enumerate(ordinals):
            if ordinal in generated_lower and idx < len(answers):
                return idx

        # Strategy 4: Look for letter choices (A, B, C, D)
        letter_match = re.search(r"\b([ABCD])\b", generated_text.upper())
        if letter_match:
            letter_idx = ord(letter_match.group(1)) - ord("A")
            if 0 <= letter_idx < len(answers):
                return letter_idx

        return -1  # No match found

    def predict(
        self, questions: str | list[str], answers_list: list[str] | list[list[str]], text: str
    ) -> int | list[int]:
        """Predict answers using enhanced reasoning and retrieval."""
        if isinstance(questions, str):
            questions = [questions]
            answers_list = [answers_list]

        results = []

        for question, answers in zip(questions, answers_list, strict=False):
            print(f"\n🔍 Processing question: {question}")

            # Use multi-strategy retrieval
            context = self.multi_strategy_retrieval(question, text)

            print(f"📖 Retrieved context length: {len(context)} characters")
            print(f"📖 Context preview: {context[:200]}...")

            # Create enhanced prompt
            prompt = self.create_enhanced_prompt(context, question, answers)

            # Generate with multiple attempts for robustness
            best_prediction = -1
            attempts = 3

            for attempt in range(attempts):
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

                with torch.no_grad():
                    outputs = self.qa_model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=attempt > 0,  # Sample for later attempts
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                predicted_index = self.decode_answer_enhanced(predicted_text, answers)

                print(f"  Attempt {attempt + 1}: '{predicted_text}' -> Index {predicted_index}")

                if predicted_index != -1:
                    best_prediction = predicted_index
                    break

            results.append(best_prediction)

        return results if len(results) > 1 else results[0]
