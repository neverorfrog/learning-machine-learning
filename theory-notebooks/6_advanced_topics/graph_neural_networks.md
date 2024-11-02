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
- A permutation matrix $P$ has every row and every with exactly one entry equal to 1
- Graphs are invariant to permutation
  - If we change the names of nodes, the structure of the graph does not change
