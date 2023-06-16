# Multilingual-Named-Entity-Recognition

# Video Demonstration of Working

https://github.com/NityamPareek/Multilingual-Named-Entity-Recognition/assets/97893479/bf19f33d-716a-4fba-a427-83241f17b5bc

# Objective

This repository contains code in which a single transformer model called XLM-RoBERTa has be fine-tuned to perform named entity recognition (NER) across 4 languages - German, French, Italian, and English. NER is a common NLP task that identifies entities like people, organizations, or locations in text. These entities can be used for various applications such as gaining insights from company documents, augmenting the quality of search engines, or simply building a structured database from a corpus.

To simulate the real world, I have assumed that we want to perform NER for a customer based in Switzerland, where there are 4 national languages, with English often serving as a bridge between them. I wanted to build the system for Indian languages but there is a huge lack of data in Indian languages compared to European languages (hopefully that changes!).

# Dataset

I have used a subset of the Cross-lingual TRansfer Evaluation of Multilingual Encoders (XTREME) benchmark called WikiANN or PAN-X. This dataset consists of Wikipedia articles in many languages, including the 4 most commonly spoken languages in Switzerland: German (62.9%), French (22.9%), Italian (8.4%), and English (5.9%). Each article is annotated with LOC (location), PER (person), and ORG (organization) tags in the “inside-outside-beginning” (IOB2) format.

To make a realistic Swiss corpus, I sampled the German (de), French (fr), Italian (it), and English (en) corpora from PAN-X according to their spoken proportions.

# Choice of Transformer

Multilingual transformers involve similar architectures and training procedures as their monolingual counterparts, except that the corpus used for pretraining consists of documents in many languages. Despite receiving no explicit information to differentiate among the languages, the resulting linguistic representations are able to generalize well across languages for a variety of downstream tasks.

For our task, we consider the XLM-RoBERTa model or XLM-R. XLM-R uses only MLM as a pretraining objective for 100 languages, and its pre-training corpus is several orders of magnitude larger than the ones used
in earlier models.

# Tokenizer

XLM-R uses the SentencePiece tokenizer, which is based on a type of subword segmentation called Unigram and encodes each input text as a sequence of Unicode characters. This last feature is especially useful for multilingual corpora since it allows SentencePiece to be agnostic about accents, punctuation, and the fact that many languages, like Japanese, do not have whitespace characters.

# Procedure for Getting to the Final Model

1. Made a corpus of data from the PAN-X dataset
2. Tokenized the input corpus
3. Imported the base pre-trained XLM-R model from Hugging Face
4. Fine tuned the model on the multilingual corpus
5. Deployed the model using Gradio Spaces on Hugging Face

# Results and Findings Along the Way

1. For a small corpus, zero shot cross lingual transfer outperforms fine-tuning. As we increase the corpus size for fine tuning, we see an improvement in performance as compared to zero shot transfer.

<center>![image](https://github.com/NityamPareek/Multilingual-Named-Entity-Recognition/assets/97893479/eee4dc80-1672-42f8-8da3-701374b6a653)</center>

2. An increase in training data for one language improves the performance of the model in other languages as well
Following is a table of F1 scores after various modes of training:

![image](https://github.com/NityamPareek/Multilingual-Named-Entity-Recognition/assets/97893479/0d585931-2612-4c8a-9f52-72eb40fe854c)

# Test Your Own Inputs!

The model has been deployed using Gradio Spaces on Hugging Face, you can find it <a href = "https://huggingface.co/spaces/NityamPareek/NityamPareek-xlm-roberta-base-finetuned-panx-all">here</a>.
