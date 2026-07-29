
## Big Picture

```text
Data X
   │
Encoder
   ▼
Latent Space Representation
   │
Decoder
   ▼
Output Y
```

In ML, we often want to transform data such as images and text into an abstract representation form that allows us to apply inference and modeling. We call this this representation space the latent space. The most basic form is
 [[Principal Component Analysis (PCA)]], where we apply a linear transformation.

⇒ Use the latent space to understand the data.

But PCA and other forms of unsupervised learning are not flexible, as they only compress and reconstruct.


---


## Autoencoder
Deep learning says:

compress → encode → reconstruct → decode.

```text
Input X
    │
Encoder
f(x)
    │
    ▼
Latent Variable

z = f(x)

    │
Decoder
g(z)
    │
    ▼
Output
```

Notice that the decoder's output is simply trying to reconstruct the original input.

Hence we call it an **Autoencoder**.

## Variational Autoencoder

But it's the same as other unsupervised solutions as it is still doing the same job as PCA.
Compressing to latent space.

But they both have a flaw:

- There is no guarantee that the original encoding function is invertible.
- Information is lost as well.

So, it is better to treat this as mapping $X$ to the same latent representation $z$.

```text
x₁ ─┐
    ├──► z
x₂ ─┘
```

But we don't know what the latent space $z$ looks like.

This is where a **latent variable model** comes in.

We can assume that, as $begin:math:text$n \\rightarrow \\infty$end:math:text$,

$$
z \sim \mathcal N(0,I),
$$

Since we do not know the latent space \(z\), we make a modeling assumption.

A reasonable choice is

$$
z \sim \mathcal N(0,I),
$$

since the CLT tells us that many independent sources of variation tend toward a Gaussian distribution. While this does not prove the latent variables are Gaussian, it makes for a simple and mathematically convenient prior.

Then,

```text
          Encoder
        q(z|x)

x ----------------> latent distribution
                        │
                  sample z
                        │
                        ▼
                  Decoder
                  p(x|z)
                        │
                        ▼
                 reconstructed x'
```

The encoder no longer produces a single latent vector.

Instead, it predicts a **distribution** over the latent space, from which we sample a latent variable before reconstructing the original input.

---
## VAE as a Latent Variable Model

Being a latent variable model, we can apply the same trick used in the [[Expectation-Maximization (EM) Algorithm]] as in GMMs.

So, we are trying to maximize the probability of reconstructing the original sample $x$ from a latent variable $z$ aka, trying to maximize the reconstruction probability from the decoder.
Thus, the two notations naturally pop up,

$$
p(x)
\qquad\text{and}\qquad
p(x|z),
$$

since we are trying to infer the latent variable given the $x$. So, we are asking
> *Given a latent hypothesis \(z\), how plausible is it that the observed data \(x\) is true?*

as well. Hence the [[Probability as Logic|bayesian viewpoint]] is needed.
This naturally leads to the joint probability

$$

p(x,z)

=

p(x|z)p(z),

$$

But this time, unlike GMMs, there are an uncountable number of latent variables $z$ and is continuous.

Therefore,

$$
p(x)
=
\int p(x|z,w)p(z)\,dz.
$$

From [[Probability Theory]], we know that

$$
p(z|x)
=
\frac{p(x|z)p(z)}{p(x)},
$$

by Bayes' rule.

In other words, inference is simply updating our belief about the latent variable after observing the data. This is the same probabilistic logic we used throughout latent variable models and the EM algorithm.

Unfortunately,

$$
p(x)
=
\int p(x|z,w)p(z)\,dz
$$

is a multivariate integral which makes it tough to evaluate, making the exact posterior

$$
p(z|x)
$$

difficult to compute.

Instead, we approximate the posterior using

$$
q(z|x),
$$

and derive the **Evidence Lower Bound (ELBO)** using [[Variational Inference]],

$$
\mathrm{ELBO}
=
\mathbb E_{q(z|x)}
\left[
\log p(x|z)
\right]
-
D_{KL}
\!\left(
q(z|x)
\,\|\,p(z)
\right).
$$

The first term encourages the decoder to accurately reconstruct the observed sample, while the second term regularizes the latent space by encouraging the inferred latent distribution to stay close to our prior.

In other words, we are minimizing the **statistical distance** between the inferred latent distribution and our prior while simultaneously reconstructing the observed sample.
And that distance is called the [[KL divergence]].
## Amortized Inference
Also, since we apply it on neural networks, we can apply $W$ to create a single distribution with respect to the network parameters rather than solving for each latent variable individually.

So, all we have to do is optimize

$$

\theta=(w,\phi),

$$

instead of optimizing every latent variable at the same time.

We call this **amortized inference**.

## Changing ELBO for VAE

The ELBO becomes

$$
\mathcal L_n(w,\phi)
=
\int
q_\phi(z_n|x_n)
\log
\frac{p_w(x_n|z_n)p(z_n)}
{q_\phi(z_n|x_n)}
\,dz_n,
\qquad
\theta=(w,\phi).
$$

We can separate it into

$$
\mathcal L_n(w,\phi)
=
\underbrace{
\mathbb E_{q_\phi(z|x)}
\left[
\log p_w(x|z)
\right]
}_{\text{Reconstruction}}
-
\underbrace{
D_{KL}
\left(
q_\phi(z|x)
\,\|\,p(z)
\right)
}_{\text{Analytically computable}}.
$$

Unlike GMMs, we still need to compute the expectation in the first term.

Since evaluating the integral directly is difficult, we instead apply a simple Monte Carlo estimator.

$$
\mathcal L_n(w,\theta)
\approx
\frac1L
\sum_{l=1}^{L}
\log p(x_n|z_n^{(l)},w),
\qquad
z_n^{(l)}
\sim
q_\phi(z|x_n).
$$

Now, being a Monte Carlo estimator, computing the expectation is easy enough.

---
### Reparametarization

The problem is that the latent distribution

$$
q_\phi(z|x)
$$

depends on the encoder parameters $\phi$.

Therefore, we cannot differentiate through the sampling operation directly.

Since we are simply taking random samples,

$$
z
\sim
q_\phi(z|x),
$$

the sampling operation itself is **not differentiable**.

Instead, we rewrite the sampling process as

$$
\boxed{
z
=
\mu(x,\phi)
+
\sigma(x,\phi)\epsilon,
\qquad
\epsilon
\sim
\mathcal N(0,I).
}
$$

Now all of the randomness lives inside

$$
\epsilon,
$$

which is independent of the network parameters.

Therefore,

$$
z
$$

becomes a differentiable function of

$$
\mu
\quad\text{and}\quad
\sigma,
$$

allowing gradients to flow through the encoder.

Finally, assuming a Gaussian decoder,

$$
-\log p(x|z)
\propto
\|x-\hat x\|^2,
$$

so the reconstruction objective naturally reduces to the familiar **MSE loss**.

In other words, we have transformed the stochastic sampling process into a deterministic computation graph with randomness injected only through

$$
\epsilon.
$$

---

## Training via Backpropagation

Now, we can simply take the gradients with respect to

$$
\phi
\quad\text{and}\quad
\theta,
$$

and the optimizer will simultaneously update both the encoder and decoder during training.

We have effectively combined both sides of the EM algorithm into a single end-to-end optimization procedure.

```text
                ε ~ N(0,I)
                     │
                 Sample ε
                     │
                     ▼
Input
  │
Encoder qϕ(z|x)
  │
μ(x), σ(x)
  │
z = μ + σε
  │
Decoder pθ(x|z)
  │
Reconstructed x'
```

As we found earlier, the objective is

$$
\mathcal L(\theta,\phi)
=
\mathbb E_{q_\phi(z|x)}
\left[
\log p_\theta(x|z)
\right]
-
D_{KL}
\!\left(
q_\phi(z|x)
\,\|\,p(z)
\right).
$$

where

- the reconstruction term

$$
\mathbb E_{q_\phi(z|x)}
\left[
\log p_\theta(x|z)
\right]
$$

encourages the decoder to accurately reconstruct the observed sample, and

- the KL divergence

$$
D_{KL}
\!\left(
q_\phi(z|x)
\,\|\,p(z)
\right)
$$

minimizes the **statistical distance** between the inferred latent distribution and our prior.

Assuming a Gaussian decoder,

$$
-\log p_\theta(x|z)
\propto
\|x-\hat x\|^2,
$$

so the reconstruction objective naturally reduces to the familiar **MSE loss**.

Therefore, after applying the reparameterization trick, we can simply take gradients with respect to

$$
\theta
\quad\text{and}\quad
\phi,
$$

and optimize the entire network end-to-end using backpropagation.

Unlike the EM algorithm, which alternates between the E-step and M-step, the encoder and decoder are optimized simultaneously in a single end-to-end optimization procedure.

---
### Closed-form KL Divergence

Since both the encoder distribution

$$
q_\phi(z|x)
=
\mathcal N(\mu(x),\Sigma(x))
$$

and the prior

$$
p(z)
=
\mathcal N(0,I)
$$

are multivariate Gaussian distributions, the KL divergence can be computed analytically.

The general KL divergence between two multivariate Gaussians is

$$
D_{KL}(p_1\|p_2)
=
\frac12
\left[
\log\frac{|\Sigma_2|}{|\Sigma_1|}
-
n
+
\operatorname{tr}(\Sigma_2^{-1}\Sigma_1)
+
(\mu_2-\mu_1)^T
\Sigma_2^{-1}
(\mu_2-\mu_1)
\right].
$$

For VAEs,

$$
p_1=q(z|x),
\qquad
p_2=p(z),
$$

where

$$
\mu_1=\mu,
\qquad
\Sigma_1=\Sigma,
\qquad
\mu_2=\mathbf0,
\qquad
\Sigma_2=I.
$$

Substituting these into the general formula gives

$$
D_{KL}
\!\left(
q(z|x)
\,\|\,p(z)
\right)
=
\frac12
\sum_i
\left(
\mu_i^2
+
\sigma_i^2
-
\log\sigma_i^2
-
1
\right).
$$

Therefore, unlike the reconstruction term, the KL divergence has a **closed-form expression** and does not require Monte Carlo estimation.

---