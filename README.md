# ML Experiments

A colleciton of code samples for training or fine-tuning ML models for hackathons and Kaggle competitions. 

## Classificaiton with Threshold
The problem is to select a subset of samples such that the total value of the selected samples is maximized. This is different from typical classification or regression problems, since the optimization metric is clear and one can optimize the threshold for selecting samples (e.g. due to noise in the data, selecting samples that are expected to be sufficiently greater than zero might yield better results that selecting samples that are predicted to be positive). 
The solution includes several approaches. First option is to train a regression model and then optimize the threshold for selecting samples. Second option is to pick a threshold and train a classification model. Linear and tree models are trained with various reguliarization techniques. Optimized using Optuna. 

## GAIA Agents
Implemented several LLM agents for solving the GAIA benchmark as part of the huggingface agents course.  
[code](https://huggingface.co/spaces/agercas/HF_Agents_Course_GAIA_Agent/tree/main)

## HF Audio Course
Solutions to HF audio course exercises, training and fine-tuning various audio models. 

## Kaggle Semantle
The problem is a variation of the game Semantle, where the goal is to identify a sequence of words. Each guess returns a similarity score indicating how close the guess is to the correct sequence. 
The solution involves an algorithmic search using word embedding models, by iterating over the closest words (based on embeddings) in the direction that improves the similarity score. 

## MCQA Agents (Mutiple Choice Question Answering)
A collection of approaches for answering multiple-choice text comprehension questions based on a provided text using LLMs with retrieval and reasoning techniques. Includes implementations with very lightweight LLMs, retrieval-augmented generation (RAG), and reasoning strategies such as chain-of-thought (CoT).

## Qualification Extraction
The problem is to identify academic qualificaitons mentioned in text snippets. 
The solution includes a baseline based on regex, two fine-tuned models based on BERT variations as well as using a pretrained cross-encoder model.
