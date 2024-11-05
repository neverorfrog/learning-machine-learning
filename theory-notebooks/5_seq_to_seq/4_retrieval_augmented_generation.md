# Retrieval Augmented Generation

## Why?

### Do we need external knowledge in NLP?
  - Sentiment Analysis: NO
  - Question Answering: YES

### There is a problem connected to knowledge: hallucinations
  - Intrinsic: can be verified against the source (our true problem)
  - Extrinsic: could be voluntary

### RAG is needed to avoid allucinations

## What are the main components of RAG (Seminal Paper)?
- Two components: retriever and generator

### Dense Passage Retriever
- Uses a bi-encoder (both encoders are BERT)
  - Question encoder: encodes a question
  - Passage encoder: encodes every passage in the input corpus
- Applies dot product similarity between question and passages representations
- Learns better representations by using contrastive loss
  - One positive passage is selected for the question
  - Hard negatives are used

#### How do I look for relevant passages for a question?
- Indexing
- Data structure that stores apriori the passages vectors in memory

### Generator
- Seq-to-seq (encoder-decoder)
- Generates a passage from the input
- Uses actual reference passage as label to construct a loss function
- Backpropagation flows back to generator and question encoder

## Challenges
- Source quality
- Time critical information
