# IPG
interpolated policy gradient for my RL course final project

slides : https://hackmd.io/@kBcUQuNKSQKpa1_ZlUZhCw/HkmeljRUn

## Usage
``python main.py [args]``

### arguments
- ``--algo``: The algorithm used.
- ``--env``: The environment used to train on.

#### Q-prop Algorithm

##### Control Variate
- Covariance of random variables $X$ and $Y$ : $Cov(X, Y) = E[(X-\mu_X)(Y-\mu_Y)]$
- Correlation coefficient of $X$, $Y$ : $\rho_{XY} = \frac{\sum (X-\mu_X)(Y-\mu_Y)}{\sqrt{\sum(X-\mu_X)^2\sum(Y-\mu_Y)^2}}$

The combination of $X$ and $Y$ with minimal variance is $X^* = X + c(Y - \mu_Y)$, where $c = -\frac{Cov(X, Y)}{Var(Y)}$ and the variance is $Var(X^*) = Var(X) + c^2Var(Y) + 2cCov(X, Y)$

<img src="https://latex.codecogs.com/svg.image?{\color{White}&space;\begin{align*}Var(X^*)&space;&&space;=&space;&space;Var(X)&space;&plus;&space;c^2Var(Y)&space;&plus;&space;2cCov(X,&space;Y)&space;\\&space;&&space;=&space;Var(X)&space;&plus;&space;(-\frac{Cov(X,&space;Y)}{Var(Y)})^2Var(Y)&space;&plus;&space;2(-\frac{Cov(X,&space;Y)}{Var(Y)})Cov(X,&space;Y)&space;\\&space;&&space;=&space;Var(X)&space;&plus;&space;\frac{Cov(X,&space;Y)^2}{Var(Y)}&space;-&space;2\frac{Cov(X,&space;Y)^2}{Var(Y)}&space;\\&space;&&space;=&space;Var(X)&space;-&space;\frac{Cov(X,&space;Y)^2}{Var(Y)}&space;\\&space;&space;&&space;=&space;[1&space;-&space;Var(X)&space;&plus;&space;\frac{Cov(X,&space;Y)^2}{Var(X)Var(Y)}]Var(X)&space;\\&space;&&space;=&space;(1&space;-&space;\rho_{XY}^2)Var(X)\end{align*}}&space;" title="https://latex.codecogs.com/svg.image?{\color{White} \begin{align*}Var(X^*) & = Var(X) + c^2Var(Y) + 2cCov(X, Y) \\ & = Var(X) + (-\frac{Cov(X, Y)}{Var(Y)})^2Var(Y) + 2(-\frac{Cov(X, Y)}{Var(Y)})Cov(X, Y) \\ & = Var(X) + \frac{Cov(X, Y)^2}{Var(Y)} - 2\frac{Cov(X, Y)^2}{Var(Y)} \\ & = Var(X) - \frac{Cov(X, Y)^2}{Var(Y)} \\ & = [1 - Var(X) + \frac{Cov(X, Y)^2}{Var(X)Var(Y)}]Var(X) \\ & = (1 - \rho_{XY}^2)Var(X)\end{align*}} " />


Use Taylor Expansion of the off policy critic as a control variate to reduce high variance.
<img width="1061" alt="image" src="https://github.com/mmi366127/IPG/assets/77866896/5ce5f1da-7b6b-4bec-a3f7-8f4cc171ab6b">



Code:
https://github.com/brain-research/mirage-rl-qprop/blob/9717d88ccc37062c15e19c7f9f1e6fb4dcf7a371/rllab/sampler/base.py#L103

https://github.com/HarveyYan/RL-Robotic-Control
