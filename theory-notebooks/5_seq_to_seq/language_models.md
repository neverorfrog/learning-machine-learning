# Language Models

## Intro
- Language models can be considered as autoregressive models
- The task is to predict the next token given a sequence of tokens

## Types of language models
- Autoencoding models (like BERT)
  - Bidirectional context (it sees the whole sentence at once)
  - Used for sentence or text classification
- Encoder-decoder models (like T5)
  - Used for machine translation, summarization
- Autoregressive models (like GPT)
  - Masked decoder
  - Used for text generation

## ELMo
- First improvement over Word2Vec towards LLMs

### Embedding depending on the context
- Token embeddings are not fixed anymore
- Obtained from the hidden states of a bidirectional LSTM
- It is possible to finetune baseline models to obtain a very good performance

## BERT (Encoder Only)
- Uses only the encoder part of the transformer
  - Goal is to build a more meaningful representation of the input data
  - Then finetune the model on supervised downstream tasks
- First example of large language model

### How does training happen?
- Pre-training using self-supervision
  - The proxy tasks:
    - Word masking
      - Replace a target token with a fixed token (and 10% of the time with a random token)
    - Next sequence prediction
      - Two different sentences are inputted as a single input sequence
      - There is [SEP] token that separates the sentences
      - There is a [CLS] token that indicates if the first sentence can precede the second
    - Shuffling sentence
- Finetuning on downstream (supervised) tasks

### What about the embeddings?
  - Token embeddings (WordPiece is used)
  - Segment embeddings (to which sentence does the token belong to?)
  - Positional embeddings (learned)

### For what tasks do we finetune BERT?
  - Named entity recognition: classification at the level of entities in the sentence
  - Sequence classification: classify the whole sentence (with [CLS] token)
  - Grounded common sense interface: most plausible continuation for a sentence

### How do we evaluate it?
  - On those downstream tasks

## GPT (Decoder Only)
- Autoregressive language model
  - Computes joint probability of having all tokens in the sentence
  - And also the probability of a certain token following the sentence
### How do we train it?
- We want to maximise the log probability of the input sequence
  - We are effectively defining a probability distribution over text sequences
  - In that sense, this decoder is a generative architecture
- Achieved by masked self-attention (causal decoder)
- The goal at each position of the input sequence is to maximize the probability of the following ground truth token in the sequence. 

## What is the pipeline in modern LLMs?
- Pretraining using internet text
- Supervised finetuning for downstream task
  - Using reinforcement learning by human feedback