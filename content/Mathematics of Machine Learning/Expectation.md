## Definition
$$\sum_{i=1}^{n}{x_ip(x_i)}$$ 
## Why $x_ip(x_i)$?
In statistics textbooks, usually, it is explained as weighted averages. However, this interpretation is not natural given the idea that expectation measures tendencies.
## From Inference
We can use the idea of [[Probability as Logic]] to make that apparent.
Suppose we wish to Infer/estimate a random variable $X$ given some information (or evidence) $E$. The key idea is that a random variable can be decomposed into mutually exclusive binary propositions.

  

$$

X=\sum_i x_i\mathbf{1}_{\{X=x_i\}},

$$

  

where

  

$$

\mathbf{1}_{\{X=x_i\}}=

\begin{cases}

1, & X=x_i,\\

0, & X\neq x_i.

\end{cases}

$$

  

Each indicator simply answers the yes-or-no question:

  

> "Is the value of $X$ equal to $x_i$?"

  

If we knew the true world, exactly one indicator would be $1$ and the rest would be $0$. Before observing the outcome, however, each indicator is assigned a posterior plausibility

  

$$

P(X=x_i\mid E).

$$

  

Since

  

$$

\mathbb{E}\!\left[\mathbf{1}_{\{X=x_i\}}\mid E\right]

=

P(X=x_i\mid E),

$$

  

taking expectations gives

  

$$

\begin{aligned}

\mathbb{E}[X\mid E]

&=

\mathbb{E}\!\left[

\sum_i x_i\mathbf{1}_{\{X=x_i\}}

\;\middle|\;E

\right] \\

&=

\sum_i x_i

\mathbb{E}\!\left[

\mathbf{1}_{\{X=x_i\}}

\;\middle|\;E

\right] \\

&=

\sum_i x_iP(X=x_i\mid E).

\end{aligned}

$$

  

Thus, expectation can be interpreted as the **posterior tendency** of a random variable. We first decompose the variable into binary propositions (possible worlds), assign each proposition its posterior plausibility, and then reconstruct the variable by summing over all possibilities. If $X$ itself is binary, this immediately reduces to

  

$$

\mathbb{E}[X\mid E]

=

P(X=1\mid E),

$$

  

which explains why probability is simply a special case of expectation.

## Geometry / Linear Algebra

For a discrete random variable, define

$$

X=

\begin{bmatrix}

x_1\\

x_2\\

\vdots\\

x_n

\end{bmatrix},

\qquad

p=

\begin{bmatrix}

P(X=x_1)\\

P(X=x_2)\\

\vdots\\

P(X=x_n)

\end{bmatrix}.

$$

Then

$$

\mathbb{E}[X]

=

p^\top X.

$$

This reveals two useful geometric interpretations:

1. The expectation is the projection of the possible values onto the probability vector and act as the center mass.

2. The expectation is a positive linear functional on the space of random variables.

Consequently,

$$

\mathbb{E}[aX+bY]

=

a\mathbb{E}[X]

+

b\mathbb{E}[Y],

$$

which explains why expectation behaves like a linear operator throughout probability and machine learning.

## Expectation as tendency
The notion that expectation represents the **posterior tendency** of a random variable has a powerful consequence. By choosing different transformations of the random variable before taking the expectation, we obtain different summaries of the distribution.

That is, instead of considering

$$  
\mathbb{E}[X],  
$$  
  
we may consider  
  
$$  
\mathbb{E}[g(X)],  
$$
for some transformation $g$.

For example,

- Choosing

$$
g(X)=\left(X-\mathbb{E}[X]\right)^2
$$

measures the squared distance from the tendency (or centroid). Taking its expectation defines the [[Variance]], which quantifies the average deviation of the distribution from its tendency.

- Choosing

  $$

  g(X)=\log P(X)

  $$

  measures the average log-plausibility of outcomes, leading to [[Entropy]], [[Cross Entropy]], and the expected log-likelihood used throughout statistical inference and machine learning.

