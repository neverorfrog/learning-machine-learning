# Decision Trees

- Answers the problem of finding through search the best hypothesis among $H$ that fits at best data $D$ for a target function $x$
- Problem:
  - Target function $f: X \rightarrow C$
  - $X$ discrete set of attributes
  - $H$ the set of decision trees

## Definition of decision tree

- Ingredients:
  - Nodes: attributes (discrete space $X$)
  - Branches: $j$-th value $a_{i,j}$ for attribute $A_i$
  - Leaves: classification $c$ of an instance
- An **instance is a path of the tree**
- In other words, decision trees are disjunction (branches) of conjunctions (paths)

## ID3 Algorithm

- Answers the problem of how to build a tree able to classify a sample
- Structure:
  - Inputs:
    - Set of Examples, where each one has a set of attribute values
    - Set of Attributes
    - Target Attribute
  - Output: decision tree
  - Recursive algorithm
- Pseudocode
  - Initialization
    - Create a root node for the tree
  - Termination conditions:
    - All remaining Examples have the same label $\rightarrow$ return that label
    - Attributes is empty $\rightarrow$ return the most common label in the Examples
  - Expansion step:
    - We choose the attribute A that maximises the information gain as root of new tree
    - For each value of A
      - Add a branch for that value
      - Extract the Examples where that value is present
      - If extracted examples are empty add a leaf node with most common label
      - Else expand the tree further
- But how exactly is the best attribute chosen?
  - Concept of **ENtropy**
    - $E(s)=\Sigma_c (-p_i \log p_i)$
    - $p_i$ is the proportion of examples labeled as $C_i$ wrt total number of examples
    - Indicates the number of bits needed to encode the classification of an example of S
  - We will select the attribute that reduces most entropy
  - How do we measure this reduction? **Information Gain**
    - $G(S,A) = E(S)-\Sigma_v \frac{|S_v|}{|S|}E(S_v)$
- Some comments:
  - Greedy algorithm (we reach only local minimum)
  - Outputs just a single hypothesis
  - It is not an incremental implementation, because we use all data at each step
  - To some extent robust to noisy data

## Overfitting in DT

- Decision trees are sensitive to overfitting because they can be expanded too much. We have three main solutions to this problem:
  - Random forest
  - Stop growing the tree when datasplit is not statistically significant anymore: **reduced-error pruning**
    - Usage of an evaluation dataset
    - Trying a split and see the effect on the validation accuracy
    - If the accuracy does not rise, substitute to the whole subtree just the most recurring label
    - Problematic if the dataset is small
  - Grow full tree and post-prune: **rule post-pruning**
    - Convert decision tree to rule
    - Prune some rules independently of the others
    - Sort final rule
