## Recurrent Neural Networks as Dynamical Systems

### Core Idea

A Recurrent Neural Network (RNN) can be viewed as a discrete-time dynamical system where the neural network acts as the learned transition operator.

Reference:
[[Deep Learning.pdf#page=392&selection=422,0,537,1|Deep Learning, page 392]]

The hidden state evolves according to

$$
h_t = F(h_{t-1}, x_t)
$$

where

- $x_t$ = input at time $t$
- $h_t$ = hidden state (memory)
- $F$ = learned transition operator

The hidden state acts as a compressed summary of the past.
This view opens up the possibilty The operator viewpoint also suggests that sequence information can be propagated in more than one temporal direction. This observation motivates the construction of [[Bi-Directional RNN]]s, which learn separate transition operators for the forward and reversed sequences, allowing each time step to access both past and future context.

---

## Function Composition View

The transition operator can itself be composed of multiple functions.

For example,

$$
z_t = g(x_t)
$$

$$
h_t = F(h_{t-1}, z_t)
$$

where

- $g$ = feature extractor / encoder
- $F$ = recurrent state transition

The full model can be written as

$$
h_t = F(h_{t-1}, g(x_t))
$$

with output

$$
y_t = q(h_t)
$$

where $q$ is the readout operator.

This resembles a learned state-space model:

$$
\text{State Update: } h_t = F(h_{t-1}, x_t)
$$

$$
\text{Observation: } y_t = q(h_t)
$$

---

## Unfolding Through Time

Because the same transition operator is reused at every time step,

$$
h_t \rightarrow h_{t+1} \rightarrow \cdots \rightarrow h_{t+k}
$$

the recurrent graph can be unfolded into a chain of identical copies.

![[Pasted image 20260619174336.png]]

Important:

- The model itself may process arbitrarily long sequences.
- Unfolding only determines how much of the computational graph is exposed during training.
- Full BPTT unfolds the entire sequence.
- Truncated BPTT unfolds only a window of length $k$.

Thus, $k$ determines how far gradients are allowed to travel backward through time.

---

## Backpropagation Through Time (BPTT)

Since the hidden states are recursively connected,

$$
h_t \leftarrow h_{t+1} \leftarrow \cdots \leftarrow h_{t+k}
$$

gradients must propagate through

1. the neural network parameters,
2. the hidden-state chain across time.

This process is called Backpropagation Through Time (BPTT).

Conceptually, gradients travel:

$$
L
\rightarrow y_t
\rightarrow h_t
\rightarrow h_{t-1}
\rightarrow \cdots
\rightarrow h_{t-k}
$$

The shared parameters receive gradient contributions from every time step.

---

## Probabilistic Interpretation

For sequence prediction,

$$
p(y_{1:T}\mid x_{1:T})
=
\prod_{t=1}^{T}
p(y_t \mid y_{<t}, x_{\le t})
$$

The training objective is Maximum Likelihood Estimation (MLE):

$$
\max_\theta
\sum_{t=1}^{T}
\log p(y_t \mid y_{<t}, x_{\le t})
$$

or equivalently minimizing negative log-likelihood / cross-entropy.

If we look at this objective, we can see that this is kinda like HMM where it uses a markov graph, but we are trying to maximize the likelyhood of

$$
p(y_{t+1}\mid y_t, x_t, x_{t+1})
$$

which yields

$$
p(y_t, y_{t+1}\mid x_t, x_{t+1})
=
p(y_{t+1}\mid y_t, x_t, x_{t+1})
\,p(y_t\mid x_t)
$$

and therefore

$$
\log p(y_t, y_{t+1}\mid x_t, x_{t+1})
=
\log p(y_{t+1}\mid y_t, x_t, x_{t+1})
+
\log p(y_t\mid x_t)
$$
 Note: this probabilistic interpretation is useful esp 
---

## Teacher Forcing

During autoregressive training, the model predicts

$$
y_t
=
f(y_{t-1}, x_t, h_{t-1})
$$

A problem arises because prediction errors accumulate over time.

Teacher forcing addresses this by feeding the true previous output:

$$
y_t^{\text{true}}
$$

instead of the model prediction

$$
\hat y_t
$$

during training.

The objective becomes

$$
\max_\theta
\sum_t
\log p(
y_t
\mid
y_{<t}^{\text{true}},
x_{\le t}
)
$$

Teacher forcing:

- stabilizes training,
- prevents error accumulation,
- improves optimization.

However, it **does not eliminate BPTT**.

Gradients must still propagate through the hidden-state dynamics.
Teacher forcing only changes the inputs used during training.
