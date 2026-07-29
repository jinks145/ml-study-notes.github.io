## Statements as Degrees of Plausibility

Let $f(Q\mid P)$ denote the plausibility assigned to proposition $Q$ given background information $P$.

1. If
   $$
   P \Rightarrow Q,
   $$
   then
   $$
   f(Q\mid P)=1.
   $$

2. If
   $$
   P \Rightarrow \bar Q,
   $$
   then
   $$
   f(Q\mid P)=0.
   $$

3. In general, if $P$ does not determine whether $Q$ is true,
   $$
   0<f(Q\mid P)<1.
   $$

4. Since $Q$ and $\bar Q$ are mutually exclusive and exhaustive,
   $$
   Q+\bar Q=\text{True},
   $$
   so
   $$
   f(Q\mid P)+f(\bar Q\mid P)=1.
   $$

5. Likewise,
   $$
   Q\bar Q=\text{False},
   $$
   and therefore
   $$
   f(Q\bar Q\mid P)=0.
   $$


## Product Rule

1. More supporting prior information should not make a conclusion less plausible.

So the plausibility-combination rule should be monotone:

$$
F_1(x,y)>0,\qquad F_2(x,y)>0.
$$

Here, $F_1$ and $F_2$ denote partial derivatives with respect to the first and second arguments.

The plausibility scale is bounded:

$$
0 \leq x \leq 1.
$$

The key consistency condition is associativity:

$$
F(F(x,y),z)=F(x,F(y,z)).
$$

Solving this functional equation gives, after re-scaling,

$$
F(x,y)=xy.
$$

Therefore,

$$
p(AB\mid C)=p(A\mid BC)p(B\mid C).
$$

## Sum Rule

Define the complement function:

$$
p(\bar A\mid B)=S(p(A\mid B)).
$$

Using double negation,

$$
\bar{\bar A}=A,
$$

we get

$$
S(S(x))=x.
$$

Using the fact that $A$ and $\bar A$ are exhaustive,

$$
A+\bar A = 1,
$$

and consistency with the product rule, Jaynes derives

$$
S(x)=1-x.
$$

Thus,

$$
p(\bar A\mid B)=1-p(A\mid B).
$$

Special cases:

$$
S(0)=1,\qquad S(1)=0.
$$

These are boundary cases, not logistic functions.