## Definition

For a random variable $X$, the variance is defined as

$$
\operatorname{Var}(X)
=
\mathbb{E}\!\left[(X-\mathbb{E}[X])^2\right].
$$
## Actually...

We previously interpreted the [[Expectation]] as the **posterior tendency** of a random variable.

Suppose we are now interested not in the tendency itself, but in how far the random variable tends to deviate from that tendency. Following the same philosophy, we simply choose a different transformation

$$
g(X)=\left(X-\mathbb{E}[X]\right)^2.
$$

This transformation measures the squared distance from the tendency (or centroid). Taking its expectation defines the [[Variance]], which quantifies the average spread of the distribution around its tendency.

## Easy Calculation

$$
\begin{aligned}
\operatorname{Var}(X)
&=
\mathbb{E}\!\left[(X-\mathbb{E}[X])^2\right] \\
&=
\mathbb{E}\!\left[
X^2
-
2X\mathbb{E}[X]
+
(\mathbb{E}[X])^2
\right] \\
&=
\mathbb{E}[X^2]
-
2\mathbb{E}[X]\mathbb{E}[X]
+
(\mathbb{E}[X])^2 \\
&=
\boxed{
\mathbb{E}[X^2]
-
(\mathbb{E}[X])^2
}.
\end{aligned}
$$
## Geometry

Variance measures the squared distance from the centroid of the distribution.

In one dimension,

$$
\operatorname{Var}(X)
=
\mathbb{E}\!\left[\|X-\mu\|^2\right].
$$

More generally, for vector-valued random variables,

$$
\mathbb{E}\!\left[\|X-\mu\|^2\right]
$$

measures the average squared Euclidean distance from the centroid, while the [[Covariance]] matrix records how this variation is distributed across different directions.