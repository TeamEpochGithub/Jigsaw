# Jigsaw
# About the competition

In this competition, we will be asking you to **score a set of about fourteen thousand comments**. Pairs of comments were presented to expert raters, who marked one of two comments more harmful. When you provide scores for comments, they will be **compared with several hundred thousand rankings**. Your **average agreement** with the raters will determine your individual score.

# Why does this competition matter?

Toxicity online poses a **serious challenge** for platforms and publishers. Online abuse and harassment **silence important voices in conversation**, forcing already marginalized people offline.

We consider this an important matter, especially given that at times, some online environments can get quite hostile and silence individuals who could otherwise play an important part in the conversation. 

Part of Team Epoch's mission is to involve and educate as many people as possible in the field of AI. We believe that by first constructing a toxic-free environment, we have a much better chance to include more people in the conversation and help spread the knowledge with interested individuals.

# Approaches

- Pre-trained models: **RoBERTa** stands for Robustly Optimized Bidirectional Encoder Representations from Transformers. RoBERTa is an extension of [BERT](https://paperswithcode.com/method/bert) with changes to the pre-training procedure.
- Pre-trained Word Embeddings: **GloVe.** Global Vectors for Word Representation.
- Data Augmentation. For example: **Translations as train/test-time augmentation (TTA)**
- Pseudo Labelling. Pseudo labelling is the process of adding confidently predicted test data to your training data. (proved to work in previous years [source](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557))
- Ensemble LMs (language models).

**Inputs**: We are given a set of comments to be ranked according to the severity of their toxicity.

**Validation**: We are given pairs of comments with their relative toxicity (i.e. which of the two is more toxic).

**Output**: Our goal is to predict the overall rankings of comments. We get scored a 1 if our ranking is the same as the Annotators for a pair, and a 0 if not. There shouldn't be any ties. All ties get a score of 0.

**Metric**: Average Agreement with the Annotators
