## [[Attention#Self-Attention|Self-Attention]]

For a sequence of length $n$,

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right),
\qquad
O=AV.
$$

During autoregressive decoding, recomputing the entire matrix for every newly generated token would repeatedly recompute the same previous attention scores.

---

## KV Cache

Instead, cache every previously computed key and value:

$$
K_{\text{cache}}
=
[k_1,k_2,\ldots,k_t],
$$

$$
V_{\text{cache}}
=
[v_1,v_2,\ldots,v_t].
$$

When generating token $t$, compute only the new query

$$
q_t,
$$

and perform attention against the cached keys

$$
A_{t,:}
=
\operatorname{softmax}
\left(
\frac{
q_tK_{1:t}^{\top}
}{
\sqrt{d_k}
}
\right).
$$

The resulting attention output is

$$
o_t=A_{t,:}V_{1:t}.
$$

Thus,

```
q₁ → cache [k₁,v₁]
q₂ → cache [k₁,k₂ | v₁,v₂]
q₃ → cache [k₁,k₂,k₃ | v₁,v₂,v₃]
⋮
```

Only one new row of the attention matrix is computed each decoding step.

> **KV cache = growing key/value matrices.**  
> **Autoregressive decoding = compute one new attention row per generated token.**



---