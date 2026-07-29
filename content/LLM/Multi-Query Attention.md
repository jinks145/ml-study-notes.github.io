## Limitations of Multi-head Attention

[[Transformers#**8. Multi-Head Attention**|Multi-head attention]] requires splitting the $Q$, $K$, and $V$ tensors into $h$ heads.

However, during autoregressive inference, each newly generated token must repeatedly load the cached keys and values for every head. On modern GPUs, this becomes **memory-bandwidth bound** rather than compute-bound, especially with .

With

- Memory: $O(bn^2d + nd^2)$
- Computation: $O(bnd^2)$

the arithmetic intensity is approximately

$$
O\!\left(\frac1k+\frac1{bn}\right),
$$

meaning memory movement dominates the computation.

Hence, **Multi-Query Attention (MQA)** was proposed.

---

## Multi-Query Attention

Instead of splitting $Q$, $K$, and $V$,

- Split only the queries into $h$ heads.
- Share a single $K$.
- Share a single $V$.

In other words,

$$
(Q_1,\ldots,Q_h,\;K,\;V).
$$

The complexity becomes

$$
O(bnd+bn^2k+nd^2),
$$

with arithmetic intensity

$$
O\!\left(
\frac1d+\frac{n}{dh}+\frac1b
\right).
$$

Thus, the KV cache is much smaller, requiring significantly fewer memory reads while maintaining nearly the same computation.

---

## Grouped Query Attention (GQA)

Sharing one key and value across every head slightly reduces model quality.

To balance memory efficiency and accuracy, we partition the heads into groups.

Instead of

- one KV pair per head (MHA), or
- one KV pair for all heads (MQA),

we use

- one KV pair per **group** of heads.

This recovers much of the modeling capacity of Multi-Head Attention while retaining most of the memory savings of Multi-Query Attention.

## So, the big idea:
MHA is compute-cheap but KV-cache expensive. MQA trades a little expressiveness for a much smaller KV cache. GQA is the compromise that essentially all modern LLMs (Llama 2/3, Gemma, Qwen, Mistral) settled on. That one sentence is probably enough to recall the whole idea six months from now.