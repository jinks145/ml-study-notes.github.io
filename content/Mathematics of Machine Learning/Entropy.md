#ml-statistics
Suppose we are playing poker.

Our opponent has a hidden hand, and we want to identify it. Imagine that after each game, a friend secretly observes the opponent’s hand and sends us a message describing what it was. If the message is too long, communication is inefficient. If it is too short, different hands become indistinguishable.

So the question is:

**What is the minimum average message length needed to communicate the hidden hand?**

Let

$$S=\{s_1,\dots,s_m\}$$

be the set of possible hidden states.

Let the communication alphabet be C, with

|$C|=b$.

A message of length $\ell$ can distinguish

$b^\ell$

possibilities. For Instance,
A 2-bit message has two slots:

  

$\_ \quad \_$

  

Each slot has 2 possible values:
$0 \text{ or } 1$.

So the possible messages are:

$00,\;01,\;10,\;11$.

That gives

  

$2\times 2 = 2^2 = 4$

  

possible messages.

Therefore, to distinguish among M states,

$b^\ell \ge M$.

Taking logs,

$\ell \ge \log_b M$, we can find our length of the message.

So state counting says:

$\ell_{\min}=\log_b M$.

Now suppose we observe N games. State $s_k$ appears $n_k$ times, with

$\sum_{k=1}^m n_k=N$.

The empirical probability is

$p_k=\frac{n_k}{N}$.

Now ask:

$\text{How many different sequences of }N\text{ games have these same counts?}$

That number is

$\frac{N!}{n_1!n_2!\cdots n_m!}$.

Since one message must identify one possible sequence, the total message length needed is approximately

$$\log_b \left( \frac{N!}{n_1!n_2!\cdots n_m!} \right)$$.

So the average message length per game is

$$\frac1N \log_b \left( \frac{N!}{n_1!n_2!\cdots n_m!} \right)$$.

Now apply [[Stirling’s approximation]]. Stirling is naturally written with natural logs:

$$\ln n!\approx n\ln n-n + O(\log n)$$.

First compute in natural logs:

$$\ln \left( \frac{N!}{\prod_k n_k!} \right) = \ln N!-\sum_k \ln n_k!$$.

Using Stirling,

$$= (N\ln N- N + O(\log N)) - \sum_k(n_k\ln n_k-n_k + O(\log n_k))$$.

Since

$$\sum_k n_k=N$$,

the linear terms cancel:

$$-N+\sum_k n_k=0$$.

So

$$\ln \left( \frac{N!}{\prod_k n_k!} \right) = N\ln N-\sum_k n_k\ln n_k + \sum_k{O(\log n_k)} + O(\log N)$$
$$\ln \left( \frac{N!}{\prod_k n_k!} \right) = N\ln N-\sum_k n_k\ln n_k + O(\log N)$$
.

Substitute

$$n_k=Np_k$$.

Then

$N\ln N-\sum_k Np_k\ln(Np_k) + O(\log N)$.

Expand:

$N\ln N - N\sum_k p_k(\ln N+\ln p_k) + + O(\log N)$.

Since

$\sum_k p_k=1$,

we get

$N\ln N-N\ln N-N\sum_k p_k\ln p_k + O(\log N)$.

Thus,

$\ln \left( \frac{N!}{\prod_k n_k!} \right) = -N\sum_k p_k\ln p_k + O(\log N)$.

Now convert from natural logs to base b:

$\log_b x=\frac{\ln x}{\ln b}$.

Therefore,

$\log_b \left( \frac{N!}{\prod_k n_k!} \right) = \frac{1}{\ln b} \ln \left( \frac{N!}{\prod_k n_k!} \right)$.

So

$$\log_b \left( \frac{N!}{\prod_k n_k!} \right) \approx \frac{1}{\ln b} \left( -N\sum_k p_k\ln p_k \right) + O(\log n)$$.

Since

$\log_b p_k=\frac{\ln p_k}{\ln b}$,

we get
$$
\log_b \left( \frac{N!}{\prod_k n_k!} \right)= \left( -N\sum_k p_k\log_b p_k \right) + O(\log n)
$$

.
Divide by N:

$$
\frac1N \log_b \left( \frac{N!}{\prod_k n_k!} \right)= \left( -\sum_k p_k\log_b p_k \right) + O(\frac{\log n}{N})
$$.
As $N \to \infty$,  $\lim_{N \to \infty}O(\frac{\log n}{N}) =0$.
Thus, we have reached the definition of 

$$H_b(X) = -\sum_k p_k\log_b p_k$$.

So Shannon entropy is the average message length per observation needed to identify one typical sequence of hidden states.

Equivalently,

$$\#\text{typical sequences} \approx b^{NH_b(X)}$$.

So identifying one such sequence requires about

$$NH_b(X)
$$
symbols total, or

$$H_b(X)
$$
symbols per game.

For poker:

- high entropy means many opponent hands remain plausible;
- lower entropy means betting behavior has narrowed the possibilities;
- entropy zero means the hidden hand is known exactly.
$$
P(X=s^\*)=1 \quad\Rightarrow\quad H_b(X)=0
$$
So the final interpretation is:

$$\boxed{ \text{Entropy is the minimum average message length needed to identify the hidden state.} }$$

Base 2 gives bits.  
Base e gives nats.  
Base b gives symbols from an alphabet of size b.


This core idea—that identifying a hidden state requires a certain average message length—allows us to *quantify* information. Information becomes the logarithm of the number of distinguishable possibilities, normalized by the number of observations. In this way, information can be measured as a length: the average number of symbols needed to identify what actually happened.
Since we defined our entropy using logarithmic growth, the base only serves to scale and $\log_b x=\frac{\ln x}{\ln b}$. So, changing bases just rescales entropy by a constant. Therefore, we can  abstract away from alphabets into nats usually(choice can be arbitrary :smile:).