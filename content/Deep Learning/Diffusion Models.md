## **Key Idea**

A diffusion model can be understood as extending the latent-variable and variational-inference ideas introduced in [[Variational Autoencoder|VAE]].

In a VAE, we use a learned encoder
$$q_\phi(z\mid x)$$
to map the observed data ($x$) into a latent distribution, and a learned decoder
$$p_\theta(x\mid z)$$
to reconstruct or generate data from that latent representation.

Diffusion models modify this structure. Instead of learning a single encoder that immediately maps the data into a latent variable, diffusion defines a **fixed sequence of stochastic transformations** that gradually adds noise:

$$x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_T$$

The final state ($x_T$) becomes approximately Gaussian noise. The model then learns the reverse sequence:

$$x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0$$

Thus, the transition from VAE to diffusion can be summarized as:

```text
VAE:
x ── learned encoder ──► z
z ── learned decoder ──► x

Diffusion:
x₀ ── fixed noising chain ──► x₁ ──► ··· ──► xT
xT ── learned denoising chain ──► ··· ──► x₁ ──► x₀
```

The VAE uses a single compressed latent variable ($z$), while diffusion introduces an entire sequence of latent variables:
$$x_1, x_2, \ldots, x_T$$

Therefore, diffusion does not remove the neural network. Rather, it replaces the VAE’s learned encoder with a fixed Gaussian noising process and replaces the single decoder step with a sequence of learned denoising transitions. Both models are still latent-variable generative models, and both can be derived using the Evidence Lower Bound.

---

## **From VAE Latent Variables to a Diffusion Chain**

For a VAE, the latent-variable model is
$$p_\theta(x,z) = p(z)p_\theta(x\mid z)$$

The approximate posterior is modeled by the encoder:
$$q_\phi(z\mid x)$$

The log-likelihood is decomposed as
$$\log p_\theta(x) = \mathcal{L}_{\mathrm{ELBO}} + D_{\mathrm{KL}}(q_\phi(z\mid x) \parallel p_\theta(z\mid x))$$

In a diffusion model, we replace the single latent variable ($z$) with a sequence of latent variables:
$$x_{1:T} = (x_1, x_2, \ldots, x_T)$$

The forward distribution becomes
$$q(x_{1:T}\mid x_0) = \prod_{t=1}^{T} q(x_t\mid x_{t-1})$$

The learned generative model becomes
$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)$$

Therefore, the VAE formulation
$$x \leftrightarrow z$$
is extended into the Markov chain
$$x_0 \leftrightarrow x_1 \leftrightarrow \cdots \leftrightarrow x_T$$

The mathematical machinery remains similar, but the latent representation is now distributed across many time steps rather than concentrated in a single variable.