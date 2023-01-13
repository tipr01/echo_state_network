# Implementation of an echo state network to predict the solution of the lorenz system
## Echo state network
-   Data: use *ẋ*(*t*) = *f*(*x*(*t*)), with
    *x*(0) = *x̂* ∈ ℝ<sup>*n*<sub>*x*</sub></sup> to generate data pairs
    (*x*(*δ*<sub>*i*</sub>,*x̂*<sub>*i*</sub>),*x̂*<sub>*i*</sub>)<sub>*i* = 1, …, *d*</sub>

-   Split the set {1, …, *d*} into {1, …, *w*}, *w* \< *d*, the so
    called washout, and {*w* + 1, …, *d*}

-   Reservoir
    *r* = (*r*<sub>1</sub>,…,*r*<sub>*N*</sub>) ∈ ℝ<sup>*N*</sup> where
    *N* ∈ ℕ denotes the reservoir size

-   Let *γ* ∈ ℝ, called leakage or leaking rate and *f*, *g* be two
    activation functions. Furthermore let
    *r*<sup>0</sup> ∈ ℝ<sup>*N*</sup> be a random choosed initial value
    of the reservoir state, *W*<sub>r</sub> ∈ ℝ<sup>*N* × *N*</sup> a
    random adjacency matrix with spectral radius *ρ*(*W*) and
    *W*<sub>in</sub> ∈ ℝ<sup>*N* × *n*<sub>*x*</sub></sup> a random
    sparse matrix. Then the reservoir dynamics is given by
    $$*r*(*k*+1) = (1−*γ*)*r*(*k*) + *γ* *f*(*W*<sub>in</sub>*x̂*(*k*)+*W*<sub>r</sub>*r*(*k*)),  *r*(0) = *r*<sup>0</sup>,$$
    where *k* = 1, …, *d* is discrete time. Here
    *r*(*k*) ∈ ℝ<sup>*N*</sup> denotes the reservoir state vector.

-   The task is to learn a matrix
    *W*<sub>out</sub> ∈ ℝ<sup>*n*<sub>*x*</sub> × *N*</sup> such that
    *x̂*<sup>+</sup>(*k*) = *g*(*W*<sub>out</sub>*r*(*k*))
    matches *x̂*(*k*) as well as possible for all *k* ∈ {*w*, …, *d*}.
    Assume that *g* is the identity function. Then
    $$W\_{\mathrm{out}}=\mathop{\mathrm{arg\\,min}}\_{W^{\mathrm{out}} \\, \in \\, \mathbb{R}^{n_x \times N}} \sum\_{k = w}^{d} \|\| \hat{x}(k) -  W^{\mathrm{out}} r(k)\|\|\_2^2.$$
    Typically *W*<sub>out</sub> can be computed using so called Tikhonov
    regularization:
    *W*<sub>out</sub> = *X**R*<sup>*T*</sup>(*R**R*<sup>*T*</sup>+*β**I*)<sup>−1</sup>,
    where *β* is a regularization term, *I* ∈ ℝ<sup>*N* × *N*</sup> the
    identity matrix, *X* ∈ ℝ<sup>*n*<sub>*x*</sub> × *d* − *w*</sup>
    denotes the matrix
    *X* = \[*x̂*(*w*)\| … \|*x̂*(*d*)\]
    and
    *R* = \[*r*(*w*)\| … \|*r*(*d*)\] ∈ ℝ<sup>*N* × *d* − *w*</sup>.

-   We consider the reservoir dynamics
    $$*r̂*(*i*+1) = (1−*γ*)*r̂*(*i*) + *γ* *f*(*W*<sub>in</sub>*W*<sub>out</sub>*r̂*(*i*)+*W*<sub>r</sub>*r̂*(*i*)),  *r̂*(0) = *r*(*d*).$$
    The prediction is given by
    *x̂*<sup>+</sup>(*i*) = *g*(*W*<sub>out</sub>*r̂*(*i*)),
    i.e., the output *x̂*<sup>+</sup>(*i*) should match the actual
    solution *x*(*i*) as well as possible for *i* \> *d*.




## Data generation
We consider the system of three ordinary differential equations

$$ \displaystyle \begin{array}{rcl}
\dot{X} &=& a(Y-X) \\ 
\dot{Y} &=& X(b-Z)-Y \\ 
\dot{Z} &=& XY-cZ 
\end{array} $$

with $a = 10, b = 30$ and $c=2$. In order to avoid influences of the possibly random initial value on the reservoir state, we will compute 
```python
washout = 500
```
steps without training the output matrix.



Solving this system by 
```python
solve_ivp(spc.lorenz, (0, tmax), x0, method='RK45', t_eval=t, args=(a, b, c))
```






## Statistical analysis
Comparison of actual solution $x_i$ with the prediction $y_i$ for $i \in \{ d, \ldots, \text{prediction} \} $:
Let
$$\mu = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}$$
and 
$$\nu = \frac{1}{n} \sum_{j=1}^n \delta_{y_j}$$.
Now we integrate the testfunction $f$ with respect to the measure $\mu - \nu$ to get the ...
