## Intro

![[Pasted image 20260728154644.png]]

Modern Transformer models stack multiple Transformer blocks. In decoder-only LLMs, each block contains causal self-attention followed by a feed-forward network.

For a sequence of length $N$, full self-attention constructs an $N\times N$ attention matrix, giving a computational complexity of

$$
O(N^2).
$$

This becomes an efficiency bottleneck for long sequences.

Recall that [[1911.03584v2.pdf|self-attention can reproduce convolutional operations]]. This connection motivates restricting attention to local neighborhoods. Instead of allowing every token to attend to every previous token, **sliding-window attention** allows each token to attend only to a fixed number of nearby previous tokens.

Because each Transformer layer produces another sequence of contextualized token representations, applying local attention repeatedly across stacked layers gradually propagates information beyond a single window.

---

## Sliding-Window Attention as Local Graph Aggregation

To understand sliding-window attention, consider the input tokens and the first attention layer.

The input tokens are first mapped to embedding vectors. Because the attention matrix specifies which token positions exchange information, the token positions can also be interpreted as nodes in a directed graph, with attention links acting as weighted edges.

Under full causal attention, token $i$ may attend to itself and every preceding token. Its neighborhood is therefore

$$
\mathcal N(i)=\{j:j\le i\}.
$$

Sliding-window attention replaces this full causal neighborhood with a local causal neighborhood:

$$
\mathcal N_w(i)
=
\left\{
j:
\max(0,i-w+1)\le j\le i
\right\},
$$

where $w$ denotes the window size.

Attention is then computed only over the edges permitted by this neighborhood:

$$
h_i'
=
\sum_{j\in\mathcal N_w(i)}
\alpha_{ij}v_j,
$$

where

$$
\alpha_{ij}
=
\operatorname{softmax}_{j\in\mathcal N_w(i)}
\left(
\frac{q_i^\top k_j}{\sqrt{d_k}}
\right).
$$

The sliding-window mask defines the local adjacency structure, while the attention scores $\alpha_{ij}$ define dynamic, content-dependent edge weights within that structure.

For example, consider the token sequence

$$
[\text{in},\text{the},\text{cat},\text{is},\text{on},\text{the},\text{chair}]
$$

with window size $w=3$. The local causal neighborhoods include

$$
\mathcal N_w(\text{cat})
=
\{\text{in},\text{the},\text{cat}\},
$$

$$
\mathcal N_w(\text{is})
=
\{\text{the},\text{cat},\text{is}\},
$$

$$
\mathcal N_w(\text{on})
=
\{\text{cat},\text{is},\text{on}\}.
$$

These windows are overlapping local neighborhoods, not independent chunks of the input sequence. Each token obtains a contextualized hidden representation by aggregating information from the tokens in its own neighborhood.

---

## Comparison with Causal Convolution

Now, let us compare sliding-window attention with causal convolution.

For a scalar-valued sequence, a causal convolution with kernel width $w$ is

$$
y_i
=
\sum_{r=0}^{w-1}
k_r x_{i-r}
+
b.
$$

Position $i$ therefore uses only itself and previous positions.

For vector-valued hidden states

$$
x_i\in\mathbb R^{d_{\mathrm{in}}},
$$

each kernel element is a matrix:

$$
y_i
=
\sum_{r=0}^{w-1}
K_r x_{i-r}
+
b,
$$

where

$$
K_r
\in
\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}.
$$

For a kernel width of $3$,

$$
y_i
=
K_0x_i
+
K_1x_{i-1}
+
K_2x_{i-2}.
$$

The matrices $K_0,K_1,K_2$ are learned during training and reused at every sequence position.

Sliding-window attention has a similar local aggregation form:

$$
h_i'
=
\sum_{j\in\mathcal N_w(i)}
\alpha_{ij}v_j.
$$

However, the aggregation weights are not fixed kernel parameters. They are dynamically computed from the current query and keys:

$$
\alpha_{ij}
=
\operatorname{softmax}_{j\in\mathcal N_w(i)}
\left(
\frac{q_i^\top k_j}{\sqrt{d_k}}
\right).
$$

Thus, the ordinary convolution kernel

$$
K_r
$$

is replaced by an input-dependent attention kernel

$$
\alpha_{ij}(x).
$$

In an ordinary convolution, the same learned kernel weights are reused at every position. In sliding-window attention, the local connectivity pattern is fixed by the window, but the aggregation weights are recomputed from the current hidden states.

Sliding-window attention can consequently be interpreted as a dynamic local convolution whose kernel weights are produced by a softmax over query-key similarities.

The dynamic kernel weights are

$$
\alpha_{ij}
=
\operatorname{softmax}_{j\in\mathcal N_w(i)}
\left(
\frac{q_i^\top k_j}{\sqrt{d_k}}
\right),
$$

and these weights are applied to the local value vectors:

$$
h_i'
=
\sum_{j\in\mathcal N_w(i)}
\alpha_{ij}v_j.
$$

Therefore,

$$
\text{CNN: fixed learned kernel weights}
$$

whereas

$$
\text{local attention: input-dependent dynamic kernel weights}.
$$

---

## Stacking Local Attention Layers

Now, let us apply the same mechanism across all Transformer layers.

Let

$$
H^{(0)}
=
[h_1^{(0)},\ldots,h_N^{(0)}]
$$

denote the initial token embeddings.

The first Transformer layer produces another sequence:

$$
H^{(1)}
=
[h_1^{(1)},\ldots,h_N^{(1)}].
$$

These hidden states remain associated with their original token positions. The next layer treats them as its input sequence and computes new projections:

$$
q_i^{(\ell)}
=
h_i^{(\ell-1)}W_Q^{(\ell)},
$$

$$
k_i^{(\ell)}
=
h_i^{(\ell-1)}W_K^{(\ell)},
$$

$$
v_i^{(\ell)}
=
h_i^{(\ell-1)}W_V^{(\ell)}.
$$

The attention sublayer then performs local aggregation over the current token's neighborhood:

$$
a_i^{(\ell)}
=
\sum_{j\in\mathcal N_w(i)}
\alpha_{ij}^{(\ell)}
v_j^{(\ell)}.
$$

Including the residual connection, the intermediate hidden state can be written schematically as

$$
\tilde h_i^{(\ell)}
=
h_i^{(\ell-1)}
+
a_i^{(\ell)}.
$$

The feed-forward sublayer then transforms this contextualized representation:

$$
h_i^{(\ell)}
=
\tilde h_i^{(\ell)}
+
\operatorname{FFN}^{(\ell)}
\left(
\tilde h_i^{(\ell)}
\right).
$$

Normalization is omitted here for readability.

Thus, each layer operates on a sequence of increasingly contextualized token-position features:

$$
H^{(0)}
\rightarrow
H^{(1)}
\rightarrow
H^{(2)}
\rightarrow
\cdots
\rightarrow
H^{(L)}.
$$

This is not recursion in the strict recurrent-network sense. It is the **repeated application of local aggregation across network depth**.

Although one layer has only a local receptive field, stacking layers allows information to propagate beyond one window. A hidden state at Layer $\ell$ already summarizes information collected by earlier layers from neighboring positions.

In this sense, the hidden-state sequence produced by one Transformer layer acts like the feature map passed into the next layer.

---

## Efficiency Consequences

### 1. Reduced Attention Computation

With full causal attention, each of the $N$ queries may attend to as many as $N$ keys. The attention computation therefore scales as

$$
O(N^2d),
$$

where $d$ denotes the hidden or attention dimension.

With a fixed sliding-window size $w$, each query attends to at most $w$ keys, reducing the attention computation to

$$
O(Nwd).
$$

When

$$
w\ll N,
$$

this is approximately linear in sequence length:

$$
O(Nwd)\approx O(Nd).
$$

Ignoring the feature dimension, this is often summarized as a reduction from

$$
O(N^2)
$$

to

$$
O(Nw).
$$

The attention matrix also becomes a banded lower-triangular matrix rather than a dense lower-triangular matrix.

---

### 2. Rolling [[KV Cache]]

Since stacked localized aggregation preserves access to longer-range dependencies through progressively contextualized hidden states, autoregressive inference only needs to retain the keys and values inside the active local window.

For every layer $\ell$, the current query attends to the cached keys and values inside that window:

$$
h_t^{(\ell)}
=
\operatorname{Attention}
\left(
q_t^{(\ell)},
K_{t-w+1:t}^{(\ell)},
V_{t-w+1:t}^{(\ell)}
\right).
$$

The new key and value are appended to the cache:

$$
K^{(\ell)}
\leftarrow
[K^{(\ell)};k_t^{(\ell)}],
$$

$$
V^{(\ell)}
\leftarrow
[V^{(\ell)};v_t^{(\ell)}].
$$

Once the cache exceeds the window size, the oldest entries are discarded or overwritten:

$$
[k_{t-w},\ldots,k_{t-1}]
\longrightarrow
[k_{t-w+1},\ldots,k_t].
$$

The same update occurs for the value cache:

$$
[v_{t-w},\ldots,v_{t-1}]
\longrightarrow
[v_{t-w+1},\ldots,v_t].
$$

This bounded cache is called a **rolling buffer cache** or **rolling KV cache**.

The two mechanisms have distinct roles:

$$
\boxed{
\text{Sliding-window attention defines the local connectivity pattern}
}
$$

while

$$
\boxed{
\text{The rolling KV cache reuses recent layer-specific keys and values}
}
$$

The sliding window limits which token positions may interact. The rolling KV cache implements this efficiently during autoregressive inference by avoiding the recomputation of previous key and value projections.