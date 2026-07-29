[[Layer normalization]] helps stabilize the gradient during training but can be computationally expensive because it requires calculating both the mean and the variance of the activations.

We modify this by removing the mean centering step, focusing only on the root mean square of 

the inputs:
$$ RMS(a) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}{a_i^2}
}$$


$$ \text{RMSNorm}(a) = \frac{a}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} a_i^2 + \epsilon}} $$

### Key Improvements:
- **Computational Efficiency:** By removing the "expensive" mean centering step, RMSNorm reduces the number of operations required during the forward and backward passes.
- **Gradient Stability:** Like Layer Normalization, it still provides the necessary normalization to stabilize gradients across different layers.
- **Performance:** In many modern architectures (such as Llama), RMSNorm is preferred over standard LayerNorm because it achieves similar or better performance with lower overhead.

*(Note: I added $\epsilon$ to the denominator, which is the standard small constant used in practice to prevent division by zero.)*
so the expensive mean centering is not needed.
