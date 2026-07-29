## Idea

Humans have difficulty reasoning about high-dimensional objects, and datasets are no exception. Statistics provides a way to transform data into simpler representations that preserve the information relevant to a question of interest. In this sense, statistics is the study of extracting useful information from data.

## Definition

Let

$$  
X=(X_1,\ldots,X_n)  
$$

be a random sample. A **statistic** is a function

$$  
h:\mathcal{X}^n\rightarrow\mathbb{R}^m,  
$$

that depends only on the observed sample and not on any unknown parameters.

Common examples include

$$  
\bar X,\qquad S^2,\qquad \max_i X_i.  
$$
Note that these examples boil down to a scalar value, a human-digestible information.
So, a statistic is simply a transformation that maps the data into a representation better suited for inference.

## Why do we use them?

Probability models describe our uncertainty about the world, while data consists of observations (propositions) about that world. Statistics transform these observations into quantities that make inference easier.

That is,

$$  
X  
\;\xrightarrow{\,h\,}\;  
T,  
$$

where $T=h(X)$ is a representation that preserves the information relevant to the inference problem.  
## Basic Representations

Many statistical methods can be viewed as compositions of simple statistics. Two of the most fundamental are:

1. [[Expectation]] summarizes the posterior tendency of a random variable by decomposing it into binary propositions and combining their posterior plausibilities.

2. [[Variance]] measures the average squared deviation from the expectation, quantifying the uncertainty or spread around that tendency.