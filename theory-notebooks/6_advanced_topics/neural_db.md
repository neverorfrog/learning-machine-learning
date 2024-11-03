# Neural Databases

## What is the neural component?
- A transformer that answers to queries by performing database operations on facts scattered around in the dataset
- It does not produce simply an answer, but performs answering and aggregations (count, avg, etc.)

## Task
- LLM trained on large data of text in a self-supervised way, they can be used on downstream tasks like neural databases
- Given a small query and some relevant facts from the databse, can T5 answer queries?
- We want to substitute typical db operations with natural language answers

## How do we generate data?
- We need supervision: tuple of $(D, Q, A)$
  - $D$ are the facts
  - $Q$ is a query
    - Generated using templates simulating joins
  - $A$ is the correct answer

## Architecture
### Support Set Generator
- Find relevant facts
### Neural SPJ
#### Input
- Which kind of query we have
#### Output
- Output the data to be aggregated
- Answer depends of course on query type
#### Structure
- Transformer
- Each head has a subset of relevant facts
### Aggregation (Not Neural)
