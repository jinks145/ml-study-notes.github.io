

#### **ReLU (Rectified Linear Unit)**
  - **Description:** Outputs the input directly if it is positive; otherwise, it outputs zero.
  - **Formula:** $f(x) = \max(0, x)$
  - **Key Characteristics:** Highly efficient for hidden layers and helps prevent vanishing gradients.

#### **Sigmoid**
  - **Description:** Maps any real-valued number into a range between 0 and 1.
  - **Formula:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
  - **Key Characteristics:** Primarily used for binary classification output layers; prone to vanishing gradients in deep networks.

### Gated Linear Unit (GLU)

The Gated Linear Unit is an activation function that uses a "gating" mechanism to control the flow of information. It is particularly effective in complex architectures like Transformers and LSTMs because it allows the network to dynamically decide which information to pass through.

- **Description:** It takes two inputs (or splits one input into two halves) and multiplies them together, where one part acts as a "gate" to filter the other.
- **Formula:** $GLU(x, y) = x \otimes \sigma(y)$
  *(Where $\otimes$ is element-wise multiplication and $\sigma$ is the sigmoid function)*
- **Key Characteristics:**
  - **Dynamic Filtering:** The gate ($\sigma(y)$) determines how much of the signal ($x$) should be allowed to pass.
  - **Performance:** It helps in capturing complex dependencies and has been shown to outperform standard ReLU in certain deep learning tasks.
  - **Variants:** A common variant is the **[[SwiGLU]]**, which replaces the sigmoid gate with the Swish activation function, widely used in modern LLMs (like Llama).