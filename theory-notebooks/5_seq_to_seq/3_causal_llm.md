# Large Language Models

## Transformers
- Contextual
  - Output embedding is aware of surrounding text
- Fast
- Scalable
  - If you increase the number of parameters, the representation potential increases linear with the parameters

## Which is the generative part of transformers? DECODER
- Two steps: embedding and decoding
### Embedding
- Giving meaning to a token (in the context)
- Positional encodings (summed to the embedding)
### Decoding
- Each decoder block gives each token more information
  - We go from grammatical rules in first layers to semantics etc...

## What is the output of the decoder?
- The input embeddings but with more context
- These are multiplied by the possible next tokens from the vocabulary to obtain the logits
- Softmax on the logits
