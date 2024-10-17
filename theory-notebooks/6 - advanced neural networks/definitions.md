# Some definitions related to RNN

## Forecasting

Hi

## Time Series

- Sequence of random variables $x_0, x_1, ..., x_n$
- We see only the empirical realizations
- Observations are acquired at a certain sampling rate

## Regressor

- Statistical technique used to uncover relationships between variables
- Could be used to estimate a dependant variable (future price) depending on past observations (prices, volume, etc...)

## Latent Variable

- Unobserved variables, constructed by inference
- Low-dim representation of input data

## Dealing with text

### Tokenization
- Dividing a text into subwords
- Assigning each subword a unique identifier

### Embedding
- Converting tokens into representations in a vector space
- To be used in CNNs, Transformers and RNNs