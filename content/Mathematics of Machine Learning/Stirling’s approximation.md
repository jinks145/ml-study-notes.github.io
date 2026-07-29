#ml-math
Recall that

  

$$

  

\log n!

  

=

  

\log(1\cdot 2 \cdots n)

  

=

  

\sum_{j=1}^{n}\log j$$.
For large $n$, we approximate the sum using an integral:
$$\sum_{j=1}^{n}\log j \approx \int_1^n \log x\,dx$$
Since
$$\int \log x\,dx=x\log x-x,$$
we obtain
$$\int_1^n \log x\,dx =
(n\log n-n)-(1\log 1-1)$$.
Since $\log 1 = 0$,
$$\int_1^n \log x\,dx =n\log n-n+1$$.

The difference between the sum and the integral grows only logarithmically, so
$$
  

\boxed{

  

\log n!

  

=

  

n\log n-n+O(\log n).

  

}$$
Ignoring lower-order terms yields the familiar approximation
$$\boxed{

  

\log n!

  

\approx

  

n\log n-n.

  

}$$
  