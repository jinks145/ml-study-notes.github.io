#ml-math 
## Motivation  
### Need to Translate  
Images and signals can be viewed as functions defined over a domain (space or time), and therefore inhabit a function space with its own geometry. 

![[Pasted image 20260623135103.png]]

But images and signals need processing if we want to understand our data. So the problem often becomes a matter of finding a transformation operator $\phi$ that maps into a [[Representation Learning##Representation Space|representation space]] where we can use it for a downstream task. A natural choice is a linear transformation

$$

  

Z = WX,

  

$$
where $X$ is a vectorized representation of the signal or image and $W$ represents a linear operator $\phi$ because a matrix multiplication is computationally easy to implement and optimize. Such operators can project the data into a representation space better suited for downstream tasks. Problem solved? no. Images and signals shift, so we need to account for that.  
### Translation Symmetry
#### Shifting

Images and signals often undergo translations. For images, intensity patterns may move across space, while for signals they may be delayed in time. Although their coordinates change, the underlying pattern remains the same.

For example, consider the function

$$
f(t)=e^{i\pi\omega t}.
$$

A shift by $\tau$ is represented by the operator

$$
(T_\tau f)(t)=f(t-\tau).
$$

The shifted function has the same shape and structure as the original function, but appears at a different location in the domain.

The operator $T_\tau$ is called a translation (or shift) operator. By varying $\tau$, it generates a family of functions

$$
\{T_\tau f\}_{\tau \in \mathbb R},
$$

which all represent the same pattern appearing at different locations.
### Shift Operator Matrix

We defined the shift operator in function space as

$$
(T_\tau f)(t)=f(t-\tau).
$$

To implement this, we discretize the function over \(K\) points:

$$
x =
\begin{bmatrix}
f(t_0)\\
f(t_1)\\
\vdots\\
f(t_{K-1})
\end{bmatrix}.
$$

A one-step circular shift can be represented by a matrix \(S\):

$$
Sx =
\begin{bmatrix}
x_{K-1}\\
x_0\\
x_1\\
\vdots\\
x_{K-2}
\end{bmatrix}.
$$

So \(S\) is the discrete shift operator. Applying it repeatedly gives shifted copies:

$$
x,\quad Sx,\quad S^2x,\quad \dots,\quad S^{K-1}x.
$$

If we stack these shifted copies as rows, we obtain a circulant matrix:

$$
C(x)=
\begin{bmatrix}
x_0 & x_1 & x_2 & \cdots & x_{K-1}\\
x_{K-1} & x_0 & x_1 & \cdots & x_{K-2}\\
x_{K-2} & x_{K-1} & x_0 & \cdots & x_{K-3}\\
\vdots & \vdots & \vdots & \ddots & \vdots
\end{bmatrix}.
$$

Thus, the shift operator is implemented by a shift matrix, while the circulant matrix is obtained by collecting all shifted copies of a discretized function.

## So, Why Convolution?

Now we can put the pieces together. We want a transformation operator $\phi$ that maps an image or signal into a representation space for downstream tasks. A general linear map

$$
Z = WX
$$

can do this, but it treats every location independently and does not naturally account for shifts.

For images and signals, the same pattern may appear at different locations. Therefore, instead of learning a completely new weight vector for every location, we reuse the same weight/filter across shifted positions.

If $S_i$ is a shift operator, then convolution computes

$$
z_i = \langle X, S_i W\rangle,
$$

or equivalently,

$$
z_i = \langle S_i X, W\rangle.
$$

So convolution is a structured linear transformation: it applies the same local detector across shifts of the input. This gives the model translation equivariance: if the input shifts, the feature map shifts with it.


### Implications

1. The circular shift matrix \(S\) is diagonalizable by the discrete Fourier transform:

$$
S = F^{-1}\Lambda F.
$$

This means that Fourier modes are eigenvectors of the shift operator. In other words, complex exponentials are the natural basis functions for understanding translations.

This is why convolution and Fourier analysis are deeply connected: convolution is built from shifts, and shifts become simple multiplication in the Fourier basis.
2. The weight function \(W\) can be treated as a learnable parameter.

In convolution, \(W\) is not a full arbitrary matrix. Instead, it is a small shared pattern, called a kernel or filter, that is applied across shifted locations:

$$
z_i = \langle X, S_i W\rangle.
$$

Thus, a convolutional layer learns the kernel \(W\), while the shift structure is fixed by the geometry of the domain. This is the basic learnable component of a [[Convolutional Neural Network]].