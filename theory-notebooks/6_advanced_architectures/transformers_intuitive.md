# Intuitive Transformers (3Blue1Brown Video)

## Context
  - Predict next word in a piece of text
  - Every word is embedded into a high-dimensional vector that lives in a metric space
  - Distance and direction have some meaning in this space

## What exactly is encoded by the embedding?
  - Identity of the word in a lookup-table fashion
  - Position of the word inside the sentence

## Attention

### How could the context of a word be inferred?
  - A word could ask what words are next to it
  - Adjectives could refine the context of the nouns next to them

#### How does a noun search for adjectives?
  - With the query weight matrix $W_Q$
  - $W_Q$ is multiplied to the embedding of each token
  - The result is a vector $Q$ for each token
  - This vector is in the Query/Key space
  - It encompasses a QUERY for other tokens

### What tokens are searched by the Query?
  - The tokens search for keys $k$
  - $K$ result from multiplying $W_K$ by the embeddings

### How are $K$ and $Q$ combined?
  - They go into an attention function (similarity function)
    - It is a sort of dot product
  - The higher the similarity, the more $K$ **attends** to $Q$
  - Attending to means that $K$ updates the meaning of $Q$

### How are these similarity values used?
  - We normalize the values through softmax
  - The result is an **attention pattern**
    - Grid of normalized attentions

#### What if i wanna make the forecasting causal?
  - The prediction of the next token depends only on past tokens
  - Every $Q$ is multiplied only bey preceding $K$
  - The attention pattern becomes an upper triangular matrix
  - This is achieved by **masking**
    - The unnormalized attention pattern has $- \infty$ in the lower triangular part
    - Following, by applying softmax the lower triangular matrix has all zeros

## How is the attention pattern used?
  - We compute a value matrix $V$ by multiplying $W_V$ by the embeddings
  - Each column of $V$ is the value of the corresponding token
  - $V$ is then multiplied by the attention pattern
    - We are weighing each value vector by each query-key
  - The value matrix has as many rows as the embedding dimension
  - In the end, the result is a delta that, added to each embedding, gives it more context

## What is the output of the attention block?
  - There are embedded tokens with added contextual information coming from key, query and value

## Where does this output go?
  - In a multilayer perceptron
  - Each embedded token is fed into linear layers and activation functions

## Why?
  - Facts need to be stored
  - Extracted features need to be queried to make some guess related to the task

## Intuition
  - Weights of the mlp encompass **questions** to the embedded tokens
  - By questions we mean features that indicate a certain pattern

## What happens exactly?
  - The embedded tokens become scaled to the number of rows of the first weight matrix (to the number of questions)
  - The questions are then scaled down to the embedding space by a second matrix
  - This can be used as a direction to add more context to the input or as logits for classification

## What is this fuss about high dimensional embeddings?
  - Apparently the principle of SUPERPOSITION is important here
  - The intuition is that:
    - The higher the dimension of the embedding
    - The more indipendent directions can be represented in this space
  - This comes from the fact that in higher dimensions, to be independent to
    dimensions don't have to be exactly at 90 degrees