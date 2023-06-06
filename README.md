# IPG
interpolated policy gradient for my RL course final project


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

![image](https://github.com/mmi366127/IPG/assets/77866896/b274d939-1eae-46a0-8861-47ec8a63227e)

<img src="https://latex.codecogs.com/svg.image?&space;\begin{align*}Var(X^*)&space;&&space;=&space;&space;Var(X)&space;&plus;&space;c^2Var(Y)&space;&plus;&space;2cCov(X,&space;Y)&space;\\&space;&&space;=&space;Var(X)&space;&plus;&space;(-\frac{Cov(X,&space;Y)}{Var(Y)})^2Var(Y)&space;&plus;&space;2(-\frac{Cov(X,&space;Y)}{Var()Y})Cov(X,&space;Y)&space;\\&space;&&space;=&space;Var(X)&space;&plus;&space;\frac{Cov(X,&space;Y)^2}{Var(Y)}&space;-&space;2\frac{Cov(X,&space;Y)^2}{Var(Y)}&space;\\&space;&&space;=&space;Var(X)&space;-&space;\frac{Cov(X,&space;Y)^2}{Var(Y)}&space;\\&space;&space;&&space;=&space;[1&space;-&space;Var(X)&space;&plus;&space;\frac{Cov(X,&space;Y)^2}{Var(X)Var(Y)}]Var(X)&space;\\&space;&&space;=&space;(1&space;-&space;\rho_{XY})Var(X)\end{align*}" title="https://latex.codecogs.com/svg.image? \begin{align*}Var(X^*) & = Var(X) + c^2Var(Y) + 2cCov(X, Y) \\ & = Var(X) + (-\frac{Cov(X, Y)}{Var(Y)})^2Var(Y) + 2(-\frac{Cov(X, Y)}{Var()Y})Cov(X, Y) \\ & = Var(X) + \frac{Cov(X, Y)^2}{Var(Y)} - 2\frac{Cov(X, Y)^2}{Var(Y)} \\ & = Var(X) - \frac{Cov(X, Y)^2}{Var(Y)} \\ & = [1 - Var(X) + \frac{Cov(X, Y)^2}{Var(X)Var(Y)}]Var(X) \\ & = (1 - \rho_{XY})Var(X)\end{align*}" />

Use Taylor Expansion of the off policy critic as a control variate to reduce high variance.
<img width="1138" alt="image" src="https://github.com/mmi366127/IPG/assets/77866896/410c7ae7-49cb-4977-9d29-81153cc204a5">


Code:
https://github.com/brain-research/mirage-rl-qprop/blob/9717d88ccc37062c15e19c7f9f1e6fb4dcf7a371/rllab/sampler/base.py#L103
