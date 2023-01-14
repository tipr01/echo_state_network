# Implementation of an echo state network to predict the solution of the lorenz system

We consider the system of three ordinary differential equations

$$ \displaystyle \begin{array}{rcl}
\dot{X} &=& a(Y-X) \\ 
\dot{Y} &=& X(b-Z)-Y \\ 
\dot{Z} &=& XY-cZ 
\end{array} $$

with $a = 10, b = 30$ and $c=2$. 
