#llm-improvements
## Swish Function
A family of mathematical function where it's an interpolation of linear function and ReLU function.
![[Pasted image 20260715105433.png]]
$$\text{swish}_{\beta}(x)=\frac{x}{1+e^{-\beta{x}}}$$
## GLU with Swish Function
Since SwiGLU is a variant of the Gated Linear Unit (GLU) that replaces the standard Sigmoid gate with the **Swish** function, it is defined as:

$$\text{SwiGLU}(x, y) = \text{swish}_{\beta}(x) \otimes y$$

Where:
- $\text{swish}_{\beta}(x) = \frac{x}{1 + e^{-\beta x}}$ (as defined in your note).
- $\otimes$ represents element-wise multiplication.

In many modern LLM architectures, the input is often split into two halves ($x$ and $y$) before being passed through this function to allow for dynamic gating of information.