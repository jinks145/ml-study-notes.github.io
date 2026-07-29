# **From Self-Attention to Transformers**

The Transformer did not invent attention from nothing.

[[attention2014|Earlier models]] already used attention to select and combine information from recurrent hidden states. An important intermediate step was the **[[Self-Attention2017|2017 self-attentive sentence embedding architecture]]**:

$$  
\text{Tokens}  
\rightarrow  
\text{BiLSTM}  
\rightarrow  
\text{Contextual hidden states}  
\rightarrow  
\text{Self-attention}  
\rightarrow  
\text{Sentence representation}  
$$

The Transformer made one decisive change:

If self-attention can directly model relationships between all positions, perhaps the recurrent layer is no longer necessary.

This changed the architecture into:

$$  
\text{Tokens}  
\rightarrow  
\text{Self-attention}  
\rightarrow  
\text{Contextual representations}  
$$

Thus, attention was promoted from a **readout mechanism applied after an RNN** into the primary **representation-learning operator**.

---

# **1. Before the Transformer: Recurrence Creates Context**

In an RNN or LSTM, a sequence is processed one position at a time.

For a forward LSTM:



$$

\overrightarrow{h_t}=

\overrightarrow{\operatorname{LSTM}}  
\left(  
w_t,\overrightarrow{h_{t-1}}  
\right)  
$$

The representation at time $t$ therefore contains information from the past:


**$$ 

\overrightarrow{h_t}=

f(w_1,w_2,\ldots,w_t)  
$$

A backward LSTM processes the sequence in the opposite direction:


$$ 

\overleftarrow{h_t}=

\overleftarrow{\operatorname{LSTM}}  
\left(  
w_t,\overleftarrow{h_{t+1}}  
\right)  
$$

Its representation contains information from the future:



$$

\overleftarrow{h_t}=

g(w_t,w_{t+1},\ldots,w_n)  
$$

The two directions are concatenated:

**$$** 

**h_t**

\begin{bmatrix}  
\overrightarrow{h_t}\  
\overleftarrow{h_t}  
\end{bmatrix}  
$$

and the sequence of contextual representations is collected into a matrix:

 $$ 

H
=
\begin{bmatrix}  
h_1^\top\  
h_2^\top\  
\vdots\  
h_n^\top  
\end{bmatrix}  
\in \mathbb{R}^{n\times d_h}  
$$

The BiLSTM therefore performs the original contextualization step:

$$  
(w_1,\ldots,w_n)  
\longrightarrow  
(h_1,\ldots,h_n)  
$$

Each $h_t$ already contains information from the surrounding sequence.

---

# **2. The 2017 Step: Self-Attention over BiLSTM States**

The model in [[Self-Attention2017]] retained the BiLSTM but added self-attention on top of its hidden states.

Instead of compressing the entire sequence into only the final hidden state, the model allowed the sentence representation to be constructed from **all** hidden states:

$$  
H=(h_1,h_2,\ldots,h_n)  
$$

The attention scores were computed as:



 **$$ 

A=

\operatorname{softmax}  
\left(  
W_{s_2}  
\tanh  
\left(  
W_{s_1}H^\top  
\right)  
\right)  
$$

where:

- $H$ contains the contextual BiLSTM states.
- $W_{s_1}$ projects each hidden state into an attention space.
- $\tanh$ performs a nonlinear transformation.
- $W_{s_2}$ produces one or more attention score vectors.
- Softmax normalizes each row into attention weights.

The resulting sentence representation is:

$$  
M=AH  
$$

Each row of $A$ specifies how strongly one attention head weights each position in the sequence.

Therefore, each row of $M$ is a different weighted summary of the sentence.

## **What Was New?**

The recurrent network still created the contextual states:

$$  
w_t  
\overset{\text{BiLSTM}}{\longrightarrow}  
h_t  
$$

But self-attention determined how those states should be combined:

$$  
(h_1,\ldots,h_n)  
\overset{\text{self-attention}}{\longrightarrow}  
M  
$$

The complete architecture was therefore:

$$  
X  
\overset{\text{BiLSTM}}{\longrightarrow}  
H  
\overset{\text{Self-Attention}}{\longrightarrow}  
M  
$$

Self-attention was not yet replacing the RNN. It was operating **on top of the RNN output**.

However, it revealed something important:

Once the hidden states had been computed, attention could directly compare and combine every position with every other position.

That observation created the path toward the Transformer.

---

# **3. The Remaining Bottleneck**

Although the attention operation could be computed in parallel, its input matrix $H$ still had to be produced sequentially.

For the forward LSTM:

$$  
h_1  
\rightarrow  
h_2  
\rightarrow  
h_3  
\rightarrow  
\cdots  
\rightarrow  
h_n  
$$

The model cannot calculate $h_t$ until it has calculated $h_{t-1}$.

Therefore:

$$  
h_t=f(h_{t-1},x_t)  
$$

The attention layer was parallel, but the representation-producing layer beneath it was not.

## **Long Dependency Paths**

Suppose the first token must influence the tenth token.

In an RNN, the information must travel through:

$$  
h_1  
\rightarrow  
h_2  
\rightarrow  
\cdots  
\rightarrow  
h_{10}  
$$

The computational path has length proportional to the distance between the tokens:

$$  
O(n)  
$$

Every intermediate transition can transform, weaken, overwrite, or distort the information.

## **Repeated Matrix Multiplication**

For a simplified linear recurrence:

$$  
h_t=W_hh_{t-1}+W_xx_t  
$$

the contribution of an earlier hidden state to a later state includes repeated multiplication:

$$  
h_t  
\supset  
W_h^{,t-k}h_k  
$$

Similarly, gradients contain products of Jacobian matrices:

$$

\frac{\partial h_t}{\partial h_k}=

\prod_{i=k+1}^{t}  
\frac{\partial h_i}{\partial h_{i-1}}  
$$

The behavior of these products depends on their singular values and eigenvalue structure.

Roughly:

- Repeated factors with magnitude below $1$ tend to produce vanishing gradients.
- Repeated factors with magnitude above $1$ tend to produce exploding gradients.
- Even with LSTM gates, sequential computation remains difficult to parallelize.

LSTMs were specifically designed to reduce the forgetting and gradient problems of vanilla RNNs. They improved recurrence, but they did not remove its sequential dependency.

---

# **4. The Transformer’s Leap: Remove the Recurrent Layer**

The Transformer asked:

If attention already creates direct relationships between positions, can attention construct the contextual representations itself?

Instead of first computing:

$$  
X  
\overset{\text{BiLSTM}}{\longrightarrow}  
H  
$$

and then applying attention:

$$  
H  
\overset{\text{Attention}}{\longrightarrow}  
M  
$$

the Transformer applies attention directly to representations derived from the input:

$$  
X  
\overset{\text{Self-Attention}}{\longrightarrow}  
Z  
$$

The recurrent hidden-state chain disappears.

![[Pasted image 20260710134102.png]]

## $\to$

![[Pasted image 20260710131500.png]]

where

$$  
\boxed{  
X  
\rightarrow  
\text{BiLSTM}  
\rightarrow  
H  
\rightarrow  
\text{Self-Attention}  
}  
$$

becoming:

$$  
\boxed{  
X  
\rightarrow  
\text{Self-Attention}  
\rightarrow  
H’  
}  
$$

Attention is no longer merely selecting from contextual states.

It now **creates the contextual states**.

---

# **5. From Hidden-State Transport to Direct Interaction**

## **RNN: Transport Information through a Chain**

An RNN represents a sequence as a directed chain:

$$  
x_1  
\rightarrow  
x_2  
\rightarrow  
x_3  
\rightarrow  
\cdots  
\rightarrow  
x_n  
$$

For $x_i$ to affect $x_j$, information must be transported through every intermediate state.

The path length between positions $i$ and $j$ is approximately:

$$  
|j-i|  
$$

## **Self-Attention: Directly Connect Positions**

In self-attention, every position may compare itself directly with every other permitted position:

$$  
x_i  
\longleftrightarrow  
x_j  
$$

Within one attention layer, the dependency path between any two visible tokens is one edge:

$$  
O(1)  
$$

This is the structural reason recurrence becomes unnecessary.

|**Feature**|**Recurrent Model**|**Self-Attention**|
|---|---|---|
|Representation structure|Chain|Learned dense graph|
|Context transport|Through intermediate states|Direct weighted interaction|
|Dependency path|Proportional to token distance|One attention edge per layer|
|Training computation|Sequential across positions|Parallel across positions|
|Sequence order|Built into recurrence|Must be added explicitly|
|Causality|Built into forward recurrence|Enforced using a mask|

The Transformer does not simply make an LSTM faster.

It replaces the sequence’s computational topology:

$$  
\text{Chain}  
\longrightarrow  
\text{Learned graph}  
$$

---

# **6. Attention as a Learned Graph**

[[Attention|Self-attention can be interpreted as a learned graph]].

For a sequence of $n$ tokens:

- Each token is a node.
- Each possible attention connection is a directed edge.
- Each attention coefficient is an edge weight.
- Each value vector contains the information passed across that edge.

The attention matrix is:

# 

# # **$$** 

**A**

\operatorname{softmax}  
\left(  
\frac{QK^\top}{\sqrt{d_k}}  
\right)  
$$

where:

$$  
A_{ij}  
$$

measures how strongly token $i$ attends to token $j$.

The new representation of token $i$ is:

# 

# # **$$** 

**z_i**

\sum_{j=1}^{n}  
A_{ij}v_j  
$$

This resembles graph aggregation:

# 

# # **$$** 

**h_i’**

\sum_{j\in\mathcal{N}(i)}  
\alpha_{ij}h_j  
$$

The difference is that the graph is not fixed in advance.

Its edge weights are computed dynamically from the current token representations:

# 

# # **$$** 

**\alpha_{ij}**

f(q_i,k_j)  
$$

Self-attention therefore forms a:

- directed graph,
- weighted graph,
- input-dependent graph,
- usually dense graph.

The sequence remains ordered data, but attention changes how information moves through it.

---

# **7. Scaled Dot-Product Attention**

The Transformer constructs three different representations of every token:

$$  
Q=XW_Q  
$$

$$  
K=XW_K  
$$

$$  
V=XW_V  
$$

These are the **queries**, **keys**, and **values**.

## **Query**

The query represents what the current token is looking for:

$$  
q_i=x_iW_Q  
$$

## **Key**

The key represents how each token can be matched or addressed:

$$  
k_j=x_jW_K  
$$

## **Value**

The value contains the information that will be transferred if the corresponding token receives attention:

$$  
v_j=x_jW_V  
$$

The similarity between query $q_i$ and key $k_j$ is measured by their dot product:

$$  
s_{ij}=q_i^\top k_j  
$$

For all token pairs:

$$  
S=QK^\top  
$$

The scores are scaled by $\sqrt{d_k}$:

$$  
\frac{QK^\top}{\sqrt{d_k}}  
$$

Without scaling, the variance of the dot products tends to grow with the key dimension $d_k$. Large scores can push softmax toward highly saturated outputs with very small gradients.

The normalized attention matrix is:
 **$$ 

A=

\operatorname{softmax}  
\left(  
\frac{QK^\top}{\sqrt{d_k}}  
\right)  
$$
Finally:

$$  
Z=AV  
$$

Therefore, scaled dot-product attention is:

$$ 

\boxed{
\operatorname{Attention}(Q,K,V)=

\operatorname{softmax}  
\left(  
\frac{QK^\top}{\sqrt{d_k}}  
\right)V  
}  
$$

This operation has two stages:

1. Use $QK^\top$ to determine **where to retrieve information from**.
2. Use $AV$ to determine **what information is retrieved**.

---

# **8. Multi-Head Attention**

The 2017 self-attention model already allowed multiple attention summaries with localized attention. The Transformer extends this idea by giving each head its own query, key, and value transformations.

For head $m$:

$$  
Q_m=XW_m^Q  
$$

$$  
K_m=XW_m^K  
$$

$$  
V_m=XW_m^V  
$$

Then:

$$

\operatorname{head}_m=

\operatorname{Attention}(Q_m,K_m,V_m)  
$$

Because each head has independently learned matrices, different heads can learn different relation spaces.

The heads are concatenated:

$$

H_{\text{cat}}=

\operatorname{Concat}  
\left(  
\operatorname{head}_1,\ldots,\operatorname{head}_h  
\right)  
$$

and projected back into the model dimension:

$$

\operatorname{MHA}(X)=

H_{\text{cat}}W^O  
$$

Thus:

$$\boxed{\operatorname{MHA}(X)=

\operatorname{Concat}  
\left(  
\operatorname{head}_1,\ldots,\operatorname{head}_h  
\right)W^O  
}  
$$

Each head operates in a lower-dimensional learned subspace.

The heads do not receive completely different tokens. They receive different learned projections of the same token representations.

Multi-head attention therefore allows the model to construct several learned graphs over the same sequence:

$$  
A^{(1)},A^{(2)},\ldots,A^{(h)}  
$$

Each head may learn a different pattern of interaction.

---

# **9. What Was Lost When Recurrence Was Removed?**

Recurrence provided more than information transport.

It also implicitly encoded:

1. Sequence order.
2. Direction of time.
3. Causal visibility.

When recurrence was removed, the Transformer needed explicit replacements for these properties.

| **Function previously provided by recurrence** | **Transformer replacement**                  |
| ---------------------------------------------- | -------------------------------------------- |
| Transport information across positions         | Self-attention                               |
| Represent token order                          | Positional encoding                          |
| Prevent access to future tokens                | Causal masking                               |
| Transform each contextual state                | Position-wise FFN                            |
| Stabilize deep computation                     | Residual connections and layer normalization |
|                                                |                                              |
|                                                |                                              |

The full Transformer architecture is easier to understand as the collection of components required after recurrence was removed.

---

# **10. Positional Encoding: Restoring Order**

Pure self-attention is permutation equivariant.

Without positional information, rearranging the input tokens rearranges the outputs in the same way, but attention itself cannot distinguish:

$$  
(x_1,x_2,x_3)  
$$

from:

$$  
(x_3,x_1,x_2)  
$$

based only on token content.

Therefore, the Transformer adds a positional representation:


$$ 
X^{(0)}=

E_{\text{token}}  
+  
E_{\text{position}}  
$$

In the original Transformer, sinusoidal positional encodings were used:

 **$$ 
PE_{(pos,2i)}=

\sin  
\left(  
\frac{pos}{10000^{2i/d_{\text{model}}}}  
\right)  
$$

$$ 

PE_{(pos,2i+1)}=

\cos  
\left(  
\frac{pos}{10000^{2i/d_{\text{model}}}}  
\right)  
$$

Each sequence position is mapped into a vector of sine and cosine values at different frequencies.

The token representation therefore contains both:

- what the token is,
- where the token occurs.

Self-attention models relationships based on content, while positional encoding restores the sequence’s order structure.

---

# **11. Masked Attention: Restoring Causality**

## **The Problem**

During autoregressive generation, token $x_i$ should only depend on previous or current tokens:

$$  
p(x_i\mid x_1,\ldots,x_{i-1})  
$$

It must not inspect future targets:

$$  
x_{i+1},x_{i+2},\ldots  
$$

An RNN naturally satisfies this because:

$$  
h_i=f(h_{i-1},x_i)  
$$

The forward hidden state has no path from the future.

Self-attention, however, initially creates a complete graph. Without intervention, every position can attend to every other position.

## **The Causal Mask**

A mask $M$ is added to the attention scores:


$$

\operatorname{MaskedAttention}(Q,K,V)=

\operatorname{softmax}  
\left(  
\frac{QK^\top}{\sqrt{d_k}}+M  
\right)V  
$$

For causal attention:

$$ 

M_{ij}=

\begin{cases}  
0, & j\le i\  
-\infty, & j>i  
\end{cases}  
$$

The score matrix becomes lower triangular after softmax:

$$  
A=  
\begin{bmatrix}  
a_{11} & 0 & 0 & \cdots & 0\  
a_{21} & a_{22} & 0 & \cdots & 0\  
a_{31} & a_{32} & a_{33} & \cdots & 0\  
\vdots & \vdots & \vdots & \ddots & \vdots\  
a_{n1} & a_{n2} & a_{n3} & \cdots & a_{nn}  
\end{bmatrix}  
$$

Position $i$ may attend to:

$$  
1,2,\ldots,i  
$$

but not to:

$$  
i+1,\ldots,n  
$$

## **Structural Interpretation**

An RNN preserves causality by restricting the direction of transport:

$$  
x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_n  
$$

A masked Transformer preserves causality by restricting visibility:

$$  
x_i  
\rightarrow  
{x_1,\ldots,x_i}  
$$

Therefore:

- **RNN:** causality through sequential execution.
- **Transformer:** causality through graph pruning.

The Transformer preserves direct long-range connections to the past while deleting all edges pointing into the future.

---

# **12. Position-Wise Feed-Forward Network**

Attention moves and combines information between tokens, akin to node updates in graph learning.

After this communication step, every token is processed independently by a feed-forward network:



 **$$

\operatorname{FFN}(x)=

W_2\sigma(W_1x+b_1)+b_2  
$$

The same FFN parameters are applied at every sequence position.

For each token $i$:

$$** 

**z_i’**

\operatorname{FFN}(z_i)  
$$

There is no communication between tokens inside the FFN. Communication has already occurred in the attention layer.

A useful interpretation is:

- **Attention:** horizontal processing across tokens.
- **FFN:** vertical processing within each token representation.

Attention determines which information should be collected.

The FFN transforms the collected representation into more useful features.

---

# **13. Residual Connections and Layer Normalization**

Removing recurrence does not remove the difficulty of training deep networks.

Transformer blocks are therefore equipped with residual connections and normalization.

For a sublayer $F$:

$$  
y=x+F(x)  
$$

The residual connection allows the layer to learn a modification of the existing representation rather than replacing it completely.

It also creates a direct gradient path:

# 

# # **$$** 

**\frac{\partial y}{\partial x}**

I+\frac{\partial F(x)}{\partial x}  
$$

[[Layer normalization]] stabilizes the scale of each token representation:

**$$\operatorname{LayerNorm}(x)=

\gamma  
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}  
+\beta  
$$

The original Transformer used the general pattern:

$$  
\operatorname{LayerNorm}(x+F(x))  
$$

which is now often called **post-normalization**.

Many modern Transformers instead use **pre-normalization**:

$$  
x+F(\operatorname{LayerNorm}(x))  
$$

The important structural idea is the same:

- Residual connections preserve an information highway.
- Layer normalization stabilizes the activations.
- Their combination allows many Transformer blocks to be stacked.

---

# **14. The Complete Transformer Block**

![[Pasted image 20260710131500.png]]

A basic Transformer block contains two main sublayers:

1. Multi-head self-attention.
2. Position-wise feed-forward network.

Using the original post-normalization structure:

$$
H’=

\operatorname{LayerNorm}  
\left(  
H+\operatorname{MHA}(H)  
\right)  
$$

Then:
$$H’’=

\operatorname{LayerNorm}  
\left(  
H’+\operatorname{FFN}(H’)  
\right)  
$$

The full flow is:

$$  
H  
\rightarrow  
\text{Multi-Head Attention}  \rightarrow  
\text{Add and Norm}  
\rightarrow  
\text{FFN}  
\rightarrow  
\text{Add and Norm} 
$$

Stacking these blocks produces increasingly contextualized representations:

$$  
H^{(0)}  
\rightarrow  
H^{(1)}  
\rightarrow  
\cdots  
\rightarrow  
H^{(L)}  
$$

In each layer:

- Attention changes how tokens communicate.
- The FFN changes how each token represents the collected information.
- Residual connections preserve earlier information.
- Normalization stabilizes the computation.

---

# **15. Encoder and Decoder Attention**

The original Transformer contains both an encoder and a decoder.

## **Encoder Self-Attention**

The encoder may attend bidirectionally across the entire input sequence:

$$ 

A_{\text{encoder}}=

\operatorname{softmax}  
\left(  
\frac{QK^\top}{\sqrt{d_k}}  
\right)  
$$

Every input token can interact with every other non-padding input token.

This is similar to the bidirectional context previously constructed by a BiLSTM, but it is constructed through direct attention connections.

## **Decoder Masked Self-Attention**

The decoder must generate autoregressively, so it uses causal masking:

$$
A_{\text{decoder}}=

\operatorname{softmax}  
\left(  
\frac{QK^\top}{\sqrt{d_k}}+M  
\right)  
$$

Every output position can see only itself and previous output positions.

## **Encoder–Decoder Cross-Attention**

The decoder also needs information from the encoded input.

In cross-attention:

$$  
Q=H_{\text{decoder}}W_Q  
$$

$$  
K=H_{\text{encoder}}W_K  
$$

$$  
V=H_{\text{encoder}}W_V  
$$

Thus:

$$
\operatorname{CrossAttention}=

\operatorname{softmax}  
\left(  
\frac{  
Q_{\text{decoder}}  
K_{\text{encoder}}^\top  
}{  
\sqrt{d_k}  
}  
\right)  
V_{\text{encoder}}  
$$

The decoder asks a question using its query and retrieves relevant information from the encoder’s keys and values.

The original decoder block therefore contains:

1. Masked decoder self-attention.
2. Encoder–decoder cross-attention.
3. Position-wise FFN.

Decoder-only models such as GPT remove the separate encoder and cross-attention stages, retaining masked self-attention.

---

# **16. The Historical Progression**

The development can be summarized in four stages.

## **Stage 1: Recurrent Representation**

The RNN or LSTM constructs contextual hidden states sequentially:

$$  
X  
\rightarrow  
H  
$$

## **Stage 2: Attention over Recurrent States**

Attention selects and combines the states produced by the RNN:

$$  
X  
\rightarrow  
H  
\rightarrow  
AH  
$$

## **Stage 3: Self-Attention Reveals Global Interaction**

Self-attention shows that every position can be related directly to every other position:

$$  
h_i  
\longleftrightarrow  
h_j  
$$

The attention matrix becomes a learned relation structure over the sequence.

## **Stage 4: The Transformer Removes Recurrence**

The Transformer derives the attention inputs directly from the token representations:

$$  
Q=XW_Q,\qquad  
K=XW_K,\qquad  
V=XW_V  
$$

and constructs contextual representations using:

# 

# # **$$** 

**H’**

\operatorname{softmax}  
\left(  
\frac{QK^\top}{\sqrt{d_k}}  
\right)V  
$$

The progression is therefore:

$$  
\boxed{  
\text{RNN}  
\rightarrow  
\text{RNN + Attention}  
\rightarrow  
\text{RNN + Self-Attention}  
\rightarrow  
\text{Self-Attention without RNN}  
}  
$$

The major conceptual leap was not merely adding attention.

It was recognizing that attention could perform the contextual interaction that recurrence had previously been responsible for.

---

# **Final Perspective**

The Transformer replaced a fixed sequential transport system with a learned interaction graph.

The recurrent model says:

Move information through the sequence one state at a time.

The Transformer says:

Determine directly which tokens need information from which other tokens.

This creates the transition:

$$  
\text{Sequential state transport}  
\longrightarrow  
\text{Parallel relational inference}  
$$

Self-attention provides the direct interactions, but removing recurrence creates additional responsibilities:

- Positional encoding restores order.
- Causal masking restores temporal visibility constraints.
- Multi-head attention learns several relation structures.
- The FFN transforms each contextual representation.
- Residual connections and layer normalization allow the architecture to scale deeply.

That combination—not attention alone—is what produces the Transformer.