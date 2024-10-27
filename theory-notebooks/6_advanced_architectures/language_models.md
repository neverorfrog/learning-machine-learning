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

## BERT
- Uses only the encoder part of the transformer

### How does training happen?
- Using some proxy tasks:
  - Word masking
    - Replace a target token with a fixed token (and 10% of the time with a random token)
  - Next sequence prediction
    - Two different sentences are inputted as a single input sequence
    - There is [SEP] token that separates the sentences
    - There is a [CLS] token that indicates if the first sentence can precede the second
  - Shuffling sentence

### What about the embeddings?
  - Token embeddings (WordPiece is used)
  - Segment embeddings (to which sentence does the token belong to?)
  - Positional embeddings (learned)

### Can we finetune BERT?
  - Named entity recognition: classification at the level of entities in the sentence
  - Sequence classification: classify the whole sentence (with [CLS] token)
  - Grounded common sense interface: most plausible continuation for a sentence

### How do we evaluate it?
  - Own downstream tasks