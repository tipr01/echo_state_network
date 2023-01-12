$Implementation of an echo state network to predict the solution of the lorenz system$ 
Comparison of actual solution $x_i$ with the prediction $y_i$ for $i \in \{d, \ldots, prediction\}$:
Let
$$\mu = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}$$
and 
$$\nu = \frac{1}{n} \sum_{j=1}^n \delta_{y_j}$$.
Now we integrate the testfunction $f$ with respect to the measure $\mu - \nu$ to get the ...
