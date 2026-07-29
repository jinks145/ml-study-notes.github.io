
## Relative Positional Encodings

Recall that [[Attention#Self-Attention|self-attention can be interpreted as a probabilistic graph]].

In vanilla self-attention,

$$
P(\operatorname{attend}_j \mid q_i)
=
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt{d_k}}
\right).
$$

The probability depends only on the semantic compatibility between the query $q_i$ and key $k_j$. However, we would also like attention to account for the **relative position** between tokens $x\_i$ and $x_j$.

To achieve this, we introduce a learned relative positional embedding

$$
a_{ij}=f(i-j)\in\mathbb{R}^{d},
$$

where $a_{ij}$ encodes the relative displacement from token $i$ to token $j$.

The attention distribution is therefore extended from

$$
P(\operatorname{attend}_j\mid q_i)
$$

to

$$
P(\operatorname{attend}_j\mid q_i,a_{ij}).
$$

Conceptually, we write

$$
P(\operatorname{attend}_j\mid q_i,a_{ij})
=
\operatorname{softmax}(q_i^\top k_j,a_{ij}),
$$

although the semantic similarity and positional information must first be combined into a single scalar compatibility score.

The unnormalized compatibility score is

$$
e_{ij}
=
\frac{
q_i^\top k_j
+
q_i^\top a_{ij}^{K}
}{
\sqrt{d_k}
}.
$$

Applying softmax across all candidate key positions $j$,

$$
\alpha_{ij}
=
\operatorname{softmax}_j(e_{ij})
=
\frac{
\exp(e_{ij})
}{
\sum_{\ell}\exp(e_{i\ell})
}.
$$

Substituting the relative-position-aware compatibility score,

$$
\alpha_{ij}
=
\frac{
\exp\left(
\frac{
q_i^\top k_j
+
q_i^\top a_{ij}^{K}
}{
\sqrt{d_k}
}
\right)
}{
\sum_{\ell}
\exp\left(
\frac{
q_i^\top k_{\ell}
+
q_i^\top a_{i\ell}^{K}
}{
\sqrt{d_k}
}
\right)
}.
$$

Therefore,

$$
P(\operatorname{attend}_j\mid q_i,a_{ij})
=
\alpha_{ij}.
$$

giving the normalized attention weights


The relative positional embedding may also be added to the value vectors, producing

$$
z_i
=
\sum_j
\alpha_{ij}
\left(
v_j+a_{ij}^{V}
\right),
$$

where

$$
q_i=x_iW_Q,\qquad
k_j=x_jW_K,\qquad
v_j=x_jW_V.
$$

Substituting the query and key projections into the attention score yields

$$
e_{ij}
=
\frac{
(x_iW_Q)(x_jW_K)^\top
+
(x_iW_Q)(a_{ij}^{K})^\top
}{
\sqrt{d_k}
}.
$$

Thus, the compatibility score naturally decomposes into two terms:

$$
e_{ij}
=
\underbrace{q_i^\top k_j}_{\text{content similarity}}
+
\underbrace{q_i^\top a_{ij}^{K}}_{\text{relative positional bias}}.
$$

The first term measures semantic compatibility between tokens, while the second biases the attention according to their relative displacement.
### RoPE
Now, let's replace $a$ with the distance/similarity measure. Recall that each attention score is computed from the inner product

$$

q_i^\top k_j.

$$

Because this quantity is a scalar,

$$

q_i^\top k_j

=

k_j^\top q_i.

$$ When the query and key vectors are normalized, this inner product becomes cosine similarity. In standard scaled dot-product attention, however, the factor $1/\sqrt{d_k}$ is primarily used to control the magnitude of the logits rather than to compute correlation.
With additive relative positional encoding, the attention logit is



$$

e_{ij}=

\frac{  
q_i^\top k_j  
+  
q_i^\top a_{ij}^{K}  
}{  
\sqrt{d_k}  
}.  
$$

RoPE replaces the additive relative-position vector $a_{ij}^{K}$ with a position-dependent rotation of the query and key vectors.

Let

$$

\tilde q_i = R_iq_i,

\qquad

\tilde k_j = R_jk_j,

$$

where $R\_i$ is a rotation matrix determined by the position of token $i$end:math:text$.

The attention logit now becomes

$$

e_{ij}^{\mathrm{RoPE}}

=

\frac{

\tilde q_i^\top\tilde k_j

}{

\sqrt{d_k}

}.

$$

Substituting the rotated vectors,

$$

e_{ij}^{\mathrm{RoPE}}

=

\frac{

(R_iq_i)^\top(R_jk_j)

}{

\sqrt{d_k}

}.

$$

Since

$$

(R_iq_i)^\top

=

q_i^\top R_i^\top,

$$

we obtain

$$

e_{ij}^{\mathrm{RoPE}}

=

\frac{

q_i^\top R_i^\top R_jk_j

}{

\sqrt{d_k}

}.

$$

Rotation matrices satisfy

$$

R_i^\top R_j

=

R_{j-i},

$$

therefore,

$$

e_{ij}^{\mathrm{RoPE}}

=

\frac{

q_i^\top R_{j-i}k_j

}{

\sqrt{d_k}

}.

$$

Applying the softmax gives

$$

P(\operatorname{attend}_j\mid q_i,j-i)

=

\alpha_{ij}

=

\operatorname{softmax}_j

\left(

\frac{

q_i^\top R_{j-i}k_j

}{

\sqrt{d_k}

}

\right),

$$

or equivalently,

$$

\alpha_{ij}

=

\frac{

\exp\left(

\dfrac{

q_i^\top R_{j-i}k_j

}{

\sqrt{d_k}

}

\right)

}{

\sum_{\ell}

\exp\left(

\dfrac{

q_i^\top R_{\ell-i}k_{\ell}

}{

\sqrt{d_k}

}

\right)

}.

$$

Hence, RoPE replaces the additive relative-position score

$$

q_i^\top k_j

+

q_i^\top a_{ij}^{K}

$$

with the rotation-aware compatibility score

$$

q_i^\top R_{j-i}k_j.

$$

Rather than learning a separate positional embedding, the relative displacement is encoded directly into the geometry of the query-key inner product.

