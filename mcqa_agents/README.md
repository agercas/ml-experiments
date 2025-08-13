# MCQA Agents (Multiple Choice Question Answering)
A collection of approaches for answering multiple-choice text comprehension questions based on a provided text using LLMs with retrieval and reasoning techniques. Includes implementations with very lightweight LLMs, retrieval-augmented generation (RAG), and reasoning strategies such as chain-of-thought (CoT).

---

# Multiple-Choice QA over Short Texts

This repository explores different approaches for answering multiple-choice comprehension questions when the reference text is provided. The experiments focus on small and efficient LLMs, but also compare against state-of-the-art reasoning models.

---

## **Problem Overview**

* **Task:** Answer multiple-choice questions based on a given text passage.
* **Input:**

  * A short passage (e.g., a book chapter, article, or story excerpt) in plain text.
  * A set of multiple-choice questions designed to test comprehension and reasoning.
* **Output:** Selected answer choice for each question.

---

## **Approach**

Several model variants are implemented and compared:

1. **Baseline LLM** – Directly answer questions using a small instruction-tuned model (e.g., `flan-t5-small` or `flan-t5-large`).
2. **Retrieval-Augmented Model (RAG)** – Retrieve relevant paragraphs before answering.
3. **Vector Store Model** – Store text embeddings and retrieve the most relevant sections for each question.
4. **Reasoning Model** – Retrieve context and explicitly instruct the model to reason before answering.
5. **Chain-of-Thought Model (CoT)** – Encourage step-by-step reasoning.
6. **Statement-Based RAG** – Convert questions into declarative statements for more targeted retrieval.
7. **SOTA Model** – Compare results with a top-tier reasoning-capable LLM.

---

## **Key Observations**

1. Adding relevant context via retrieval consistently improves accuracy.
2. Explicit reasoning steps improve performance for questions that require inference.
3. Statement-based retrieval with a capable LLM yields large gains.
4. State-of-the-art LLMs can nearly solve the task without specialized prompt engineering.

---

## **Results**

| Model                    | Configuration           | Accuracy  | Correct / Total |
| ------------------------ | ----------------------- | --------- | --------------- |
| Baseline (flan-t5-small) | –                       | 20.0%     | 6 / 30          |
| Retrieval Model          | top\_k=1                | 36.7%     | 11 / 30         |
| Vector Store Model       | top\_k=1                | 36.7%     | 11 / 30         |
| Reasoning Model          | top\_k=3                | 50.0%     | 15 / 30         |
| Chain-of-Thought Model   | max\_reasoning\_steps=3 | 50.0%     | 15 / 30         |
| Statement-Based RAG      | top\_k=2                | 63.3%     | 19 / 30         |
| SOTA Model               | –                       | **93.3%** | 28 / 30         |

---

## **Insights**

* **Context + Reasoning = Big Gains** – Even small LLMs benefit significantly from targeted retrieval and explicit reasoning prompts.
* **High Baseline for SOTA Models** – Large reasoning-capable models achieve near-perfect accuracy without much tuning.
* **Limits of Small Models** – Some questions remain difficult due to subtle reasoning or near-identical answer choices.

---

## **Future Work**

* Explore fine-tuning on domain-specific comprehension datasets.
* Extend to open-ended QA instead of multiple choice.
* Investigate hybrid methods combining retrieval, reasoning, and knowledge grounding.
