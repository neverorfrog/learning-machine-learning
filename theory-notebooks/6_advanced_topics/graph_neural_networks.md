# Graph Neural Networks

## Graph
- A set of $N$ nodes and $M$ edges
### How do we represent it?
- Node feature matrix $X \sim (N,D)$
  - Embedding could be learned from this
  - Important graph properties should be preserved in embedding space
- Edge embedding matrix $E$
- Adjacency matrix $A \sim (N,N)$
### Is there some inductive bias like in images?
- Yes, permutation
- A permutation matrix $P$ has every row and every column with exactly one entry equal to 1
- Graphs are invariant to permutation
  - If we change the names of nodes, the structure of the graph does not change

## How do neural networks process graphs?
- They take as input $X$ and $A$
  - At the beginning each column $X$ contains information about just the node
- Each layer conveys more information to $X$
  - At each layer more context is added to each column

## Which tasks do GNNs solve?
- Graph-level tasks
  - Label assigned to entire graph
  - Molecule is toxic or not
  - $P(y=1|X,A) = sig[\beta_k + mean(W_k H_k, dim=1)]$
- Node-level tasks
  - Label assigned to each node
  - Point of 3D cloud belonging to a certain region
  - $P(y^{(n)}=1|X,A) = sig[\beta_k + W_k H_k^{(n)}]$
- Edge-level tasks
  - Whether there should be an edge between node m and n
  - Social networks
  - $P(y^{(mn)}=1|X,A) = sig[H_k^{(n)} H_k^{(m)} ]$

## GCN
### Why?
- Aggregation from nearby nodes
  - There is a linear transformation of $X$
  - Then a weighted sum, where the weights come from $A$
- Spatial-based because we use original graph structure
- Equivariant to permutations $\rightarrow$ inductive bias
### How?
- Aggregation (at each node with neighboring nodes)
- Linear transformation (to the embedding of the current node and its aggregation)
- $H_{k+1} = \alpha[\beta_k + W_kH_k+W_kH_kA]$
  - First term is bias
  - Second term are embeddings (in columns)
  - Third term are aggregations
### How do we do minibatches?
- Problem is that each graph has a different number of nodes
- Make a single big adjacency matrix divided in blocks

## What is transductive learning?
- Different paradigm where we basically don't have a separate test set
- It does not produce a rule to be applied to new unseen samples
- It considers both labaled and unlabeled data at the same time
- Produces labels for unlabeled data
  - Advantage: uses patterns in unlabeled data
  - Disadvantage: need to be retrained every time unseen data arrives

## Node Classification in a transductive way
- Same as before, but with a final sigmoid on the whole graph (no pooling)
### Drawback to no pooling
- Inefficient
- Don't know how to do SGD
### Solution
- Choose as batch a random subset of labeled nodes
- Each node has a sort of receptive field
### Drawback to batching
- If the graph is densely connected things may actually worsen
- Called graph expansion problem
### Solution
#### Neighborhood sampling
- We sample a fixed number of neighbors
- This way the graph's size increases in a controlled way
#### Graph partitioning 
- Split the graph into subsets of nodes
- Each subgraph can be treated as a batch

## Specific Layers
### Aggregation
- Diagonal Enhancement
  - Each node is premultiplied by a learned parameter
  - Can be generalized with a linear transformation for each node
- Mean Aggregation
  - Taking mean instaed of sum
  - Especially useful if embedding information is important more than structure
  - Efficient with degree matrix
  - With Kipf, information is normalized to discourage nodes with many neighbors having more importance
- Max-Pooling Aggregation
  - Max of the neighbors instaed of mean
- Attention Aggregation
  - Does not exploit graph topology
  - Attention mask is computed between nodes
  - Only those entries that are 1 in adjacency matrix remain valid
### Residual Connection
- First we activate aggregations
- Then we concatenate with current node

## What is the Graph Convolution Framework?
- Patch Functions
  - Define for each node how to aggregate the neighborhood
  - Matrix of size |V| x |V|
- Convolution Filter Weights
  - Apply the filters
  - Computes a score for each node to be passed to next layer
- Merging Functions
  - Merges results of multiple filtering results

## Spectral Graph Convolutional Network
- We basically learn the convolution filters in the spectral domain of the normalized laplacian
- Patch function can be expressed in terms of eigenvectors of the laplacian


