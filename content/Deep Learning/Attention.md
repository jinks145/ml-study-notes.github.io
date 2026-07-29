### Big Idea
#### Sentence As Sequences

When human beings learn or speak a sentence, we are actually encoding and decoding words as a sequence. We can model this sequence

$$
S=(w_1,\ldots,w_n), \qquad w_i\in\Sigma.
$$

where $\Sigma$ is the vocabulary (word set).

But we also have grammatical rules that determine the relationships among words beyond their positions in the sequence.

Also, as "encoders", we appear to operate through a language-independent semantic representation that relates concepts, images, experiences, and words. For example, when we hear the word *vanilla*, we may recall the taste of vanilla ice cream rather than simply its spelling.

Translation seems to occur through the same idea: instead of translating words one-by-one, we reason about the underlying situation or meaning, and then express that meaning in another language from a common context.

---

### Modeling Assumption

Neural machine translation adopts this intuition as a modeling assumption. Rather than directly learning a mapping between two sequences of words, we assume that both languages can be represented through contextual representations that encode their underlying semantics. Attention is then used to learn the relationships among these representations.

---

### Neural Machine Translation

Why representations?
Remember that $\Sigma$ is a **set of symbols**, not a vector space. Hence it requires a $f:\Sigma^T\rightarrow(\mathbb{R}^d)^T,$ an encoder that represents them. Same for the target sequences. We therefore replace the symbolic tokens with their encoded representations and construct a fully connected weighted biparte graph over these representations.

ex)
![[Pasted image 20260707195524.png]]
From this representation, we model the relationships as

$$
G=(V,E,A),
$$

where

- $V$ represents the encoded contextual representations,
- $E$ represents the possible semantic relationships,
- $A$ is the weighted adjacency (attention) matrix.

Specifically,

$$
A=[\alpha_{ij}],
$$

where weight $\alpha_{ij}$ measures how strongly representation $i$ points to representation $j$. Thus, the attention matrix captures the semantic dependencies learned by the model.

We also define our context representation as vector $c$ and original encoded sequence $H$ and target $S$.
#### Goal

Given the encoder representation sequence

$$
H=(h_1,\ldots,h_{T_x}),
$$

and the decoder representation sequence

$$
S=(s_1,\ldots,s_{T_y}),
$$

our goal is to compute a context representation

$$
c_i,
$$

for every decoder step \(i\). The context vector should summarize the parts of the source sequence that are most relevant for generating the next target representation.

In other words, we seek a mapping/operator

$$
f:(H,S)\longrightarrow c_i.
$$
based on how strongly the encoder representations $H$ relate to the decoder representations $S$.



## Attention Matrix

So, how do we find our $f$?

Using probability, we can formulate this question naturally.

At decoder step $i$, we do have the current decoder representation

$$
s_i.
$$

This representation summarizes everything the decoder currently knows. Given this information, the decoder asks:

> Which encoder representation $h_j$ is most relevant for generating the next output?

Instead of selecting a single encoder representation, we assign a prior(aka assume that h is relevant over $S$) over all encoder representations,

$$
P(h_j\mid s_i).
$$

This conditional probability measures how relevant encoder representation $h_j$ is given the current decoder state.

By Bayes' rule,

$$
P(h_j\mid s_i)
=
\frac{P(s_i,h_j)}
{P(s_i)}
=
\frac{P(s_i,h_j)}
{\sum_k P(s_i,h_k)}.
$$

The difficulty is that the joint distribution

$$
P(s_i,h_j)
$$

is unknown and generally intractable to model directly.

Instead, we learn an **unnormalized compatibility score**

$$
e_{ij}
=
\operatorname{sim}(s_i,h_j),
$$

which measures how compatible the decoder representation is with each encoder representation.

**Remark.**
The matrix

$$
E=[e_{ij}]
$$

is called the compatibility (or score) matrix. When the similarity function is chosen as a dot product,

$$
e_{ij}=s_i^\top h_j,
$$

(or cosine similarity after vector normalization), $E$ can also be interpreted as a similarity matrix from linear algebra.

Since probabilities must be nonnegative, we model the unknown joint distribution up to a proportionality constant by

$$
P(s_i,h_j)
\propto
\exp(e_{ij}).
$$

The exponential guarantees positivity, while the proportionality indicates that the values are still unnormalized.

Substituting this model into Bayes' rule yields

$$
P(h_j\mid s_i)
=
\frac{\exp(e_{ij})}
{\sum_k\exp(e_{ik})},
$$

which is precisely the **softmax** function.

We therefore define

$$
\alpha_{ij}
=
P(h_j\mid s_i),
$$

forming the matrix

$$
A=[\alpha_{ij}].
$$

Since

$$
\sum_j\alpha_{ij}=1,
\qquad
\alpha_{ij}\ge0,
$$

each row of $A$ is a categorical probability distribution over the encoder representations.

Finally, the context vector is obtained as the conditional [[Expectation|expectation]] of the encoder representations,

$$
\begin{aligned}
c_i
&=
\sum_j
\alpha_{ij}h_j \\
&=
\sum_j
P(h_j\mid s_i)h_j \\
&=
\mathbb E[H\mid s_i].
\end{aligned}
$$

Thus, we have found a principled way to model the relationship between the encoder representations $H$ and the decoder representations $S$. The resulting attention matrix simultaneously represents a conditional probability distribution and a linear operator that computes the context representation.

For each decoder state,

$$
c_i
=
\mathbb E[H\mid s_i],
$$

and, in batch form,

$$
C=AH.
$$

Notice that every row of \(A\) is row-stochastic. Therefore, \(A\) admits a **Markov-style interpretation**: each row defines a conditional probability distribution over the encoder representations given the current decoder state. Consequently, \(A\) can also be interpreted as the weighted adjacency matrix of the fully connected encoder-decoder graph.

In this way, attention unifies probability, linear algebra, and graph theory into a single operator that infers context from the encoder representations.

#### But Why the Name "Attention"?

Recall that the context vector is computed as

$$
\begin{aligned}
c_i
&=
\sum_j
\alpha_{ij}h_j \\
&=
\sum_j
P(h_j\mid s_i)h_j \\
&=
\alpha_i^\top H,
\end{aligned}
$$

where $\alpha\_i$ denotes the $i$-th row of the attention matrix $A$.

Rather than treating every encoder representation equally, the probability distribution

$$
\alpha_i=P(H\mid s_i)
$$

assigns larger weights to the encoder representations that are most relevant to the current decoder state. Consequently, the context vector is formed by selectively emphasizing the most informative parts of the source sequence.

This behavior is analogous to **human visual attention**. When viewing a scene, we do not process every region equally; instead, we focus on the regions that are most relevant to the task at hand. Likewise, the attention matrix selectively focuses on the most relevant encoder representations when constructing the context vector.

For this reason, we call

$$
A=[\alpha_{ij}]
$$

the **attention matrix**.

## Self-Attention

Recall that attention measures the relationships between the encoder representations

$$
H=(h_1,\ldots,h_{T_x})
$$

and the decoder representations

$$
S=(s_1,\ldots,s_{T_y}).
$$

The key idea behind self-attention is simple: instead of using a separate decoder sequence, we allow the sequence to attend to itself.

That is, we replace the external query sequence with a learnable transformation of the same sequence to learning the relationships **within** a single sequence rather than between two different sequences,

$$
Q = HW_Q,
$$

while also learning

$$
K = HW_K,
\qquad
V = HW_V.
$$

Now the relationships are learned entirely within the sequence itself.

The similarity matrix becomes

$$
QK^\top,
$$

whose entries measure how strongly one representation relates to another.

Applying the softmax function row-wise gives the attention matrix,

$$
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right),
$$

which can again be interpreted as a weighted adjacency matrix.

Finally,

$$
Z = AV,
$$

produces a new sequence of contextualized embeddings.

So, self-attention performs **representation learning**. Rather than learning relationships between a source and target sequence, it learns the semantic relationships among the representations within a single sequence and embeds them into a richer contextual representation space.

Thus, every output embedding

$$

z_i

$$

contains not only information about token $i$, but also information aggregated from the entire sequence according to the learned attention distribution.