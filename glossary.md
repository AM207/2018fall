#### bernoulli distribution
A random variable $X \sim \text{Bernoulli}(p)$ has a bernoulli distribution if it takes on a value of 1 with probability $p$, and a value of 0 with probability $1-p$. This can be thought of as a coin flip where heads is 1 with probability $p$. 

#### bias
When using an estimator $\hat{\theta}$ to determine the true value of a parameter $\theta$, the bias of that estimator is the difference its expected value and the true value of the parameter
$$\text{Bias}_\theta [\hat{\theta}] = E[\hat{\theta}]- \theta$$

#### bias-variance tradeoff
Suppose we want to model some process of the form $y = f(x) + \epsilon$ for datapoints $x_1, \dots, x_n$ and $\epsilon$ is white noise with mean 0 and variance $\sigma^2$. We want to create a function $\hat{f}(x)$ to estimate the true function $f(x)$. One metric that we could use to determine how well our model fits would be to choose $\hat{f}(x)$ that minimizes our mean squared error: $(y - \hat{f}(x))^2$. 

The bias-variance tradeoff comes from decomposing the expected mean square error:
$$E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$
where 
$$\text{Bias}[\hat{f}(x)] = E[\hat{f}(x) - f(x)]$$
and 
$$\text{Var}[\hat{f}(x)] = E[\hat{f}(x)^2] - E[\hat{f}(x)]^2$$

This result shows us that there are two components of the mean squared error that we want to minimize:

1. bias is any error resulting from poor assumptions, however an increase in bias may prevent overfitting
2. variance is error resulting from small changes in the training set, reducing variance reduces overfotting

#### binomial distribution
A random variable $X \sim \text{Bin}(n, p)$ has a binomial distribution if $X = Y_1 + \dots + Y_n$ and each $Y_i$ is an independent bernoulli random variable that takes on a value of 1 with probability $p$, and a value of 0 with probability $1-p$. 

This can be thought of as flipping a coin with heads occuring with probability $p$, $n$ times, and then counting the number of heads.

#### bootstrap
Boostrapping is the process of estimating properties of an estimator by using random sampling with replacement. This process can be used to approximate a distribution.

#### box-muller
The box-muller transform was developed as a more computationally efficient alternative to the inverse transform sampling method. It stems from the fact that one can generate two independent random normal variables ($Z_0, Z_1$) from two independent uniform random variables ($U_1, U_2$)
$$Z_0  = R \cos(\theta) = \sqrt{-2\ln U_1} \cos(2\pi U_2)$$
$$Z_1  = R \sin(\theta) = \sqrt{-2\ln U_1} \sin(2\pi U_2)$$
This is derived using two-dimensional Cartesian coordinates.

#### calculus
The mathematical study of continuous change. The two components are differential calculus and integral calculus. 

#### cdf
The Cumuluative Distribution function $F_X(x)$ of a real value random variable $X$ is the probability that $X$ will take on a value less than or equal to $x$, i.e.
$$F_X(x) = P(X \leq x)$$
This function must have the property: $F(-\infty) = 0$ and $F(\infty) = 1$ and must be non-decreasing and right-continuous. 

#### central limit theorem
The Central Limit Theorem (CLT) tells us that for any set of independent random variables $X_1, \dots, X_n$, as $n \rightarrow \infty$, the mean of the $n$ random variables approaches a distribution that is normal.
$$\sqrt{n}\left(\frac{1}{n} \sum\limits_{i=1}^nX_i - \mu\right) \rightarrow^d N(0, \sigma^2)$$
where Var[$X_i] = \sigma^2$

#### combinatoric optimization
There are often times when we wish to find an optimal combination over a finite set of object, without exhaustively checking every possible combination. This general set of optimization problems are labled combinatoric optimiaztion. The travelling salesman problem and minimum spanning tree problem are examples of problems that can be solved by using combinatoric optimization.

#### conditional distribution
Given two jointly distribution random variables $X$ and $Y$, the conditional probability distribution of $Y$ given $X$ is the probability distribution of $Y$ when $X$ is known to be a particular value.
$$p_Y(y | X = x) = P(Y = y | X = x) = \frac{P(X=x \cap Y = y)}{P(X=x)}$$ 
Conditional distributions are key to Bayesian statistics:
$$f_Y(y | X=x) = \frac{f_{X,Y}(x,y)}{f_X(x)}= \frac{f_{X|Y}(x|y)f(y)}{f_X(x)}$$

#### convex
In Euclidean space, a convex set is a region such that, for every pair of points within the region, every point on the straight line segment that joins the pair of points is also within the region. For example
- a solid cube is a convex set but anything hollow, such as a crescent shape, is not convex. 
- $y =x^2$ is a convex function while $y = x^3$ is not


#### cross-entropy
The cross-entropy between two probability distributions $p$ and $q$ over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set, if a coding scheme is used that is optimized for an "unnatural" probability distribution $q$, rather than the "true" distribution $p$. 
$$H(p,q) = E_p[-\log q] = H(p) + D_{KL}(p || q)$$
where $H(p)$ is the entropy and $D_{KL}(p || q)$ is the KL divergence of $q$ from $p$. 

Cross-entropy is a way to measure the difference between two functions in terms of their KL divergence and entropy. This measurement can be used as a cost function for various optimization problems.

#### frequentist statistics
Frequentist statistics refers to the field of statistics that emphasizes the frequency or proportion of the data. The goal of frequentist statistics is to learn some metric about the data, and from this hypothesis testing and confidence intervals have come about. Frequentist statistics is often compared to bayesian statistics. 

Frequentist statistics can be understood well through an example. Let us imagine we hear some beeping sound in our home and we want to determine where the sound is coming from. The frequentist approach would be to take all of the data we have and come up with the areas of the house where the data says the sound is most likely coming from. The bayesian approach would be to first set a prior over the locations where we believe the sound may be coming from, and then using the data to adjust our belief of where the sound may be coming from. 

#### poisson distribution
The poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occuring in a fixed interval of time and/or space, if the event occurs with a known average rate and independently of the time since the last event. 

For example, imagine you randomly receive letters and on average you receive 4 letters per day. If you receive a letter at 9:01am, this gives you no information about whether another piece of mail will come at 9:02am, 9:03am, ... Thus if $X$ is a random variable that represents the number of pieces of mail you received today, then $X \sim \text{Pois}(4)$

#### entropy
entropy $H$ of a discrete random variable $X$ with positive values $\{x_1, \dots, x_n\}$ and probability mass function $P(X)$ is:
$$H(X) = E[I(X)] = E[-\ln P(X)] = -\sum\limits_{i=1}^nP(x_i)\log_b P(X_i)$$
where $I$ is the information content of $X$

Entropy is an attempt to measure the expected value of the information passed in each message or random variable.



#### deterministic error
Deterministic error is equivalent to the bias of approximating a function $f$ by using a function $g$. Typically, we never know the true underlying function $f$, and so the deterministic error refers to the fact that you may have a hard time distinguishing this error from error resulting from measurement and noise.

#### deviance
Deviance is a quality-of-fit statistic for a model that is often used for statistical hypothesis testing. It is a generalization of the idea of using the sum of squares of residuals in ordinary least squares to cases where model-fitting is achieved by maximum likelihood.

The deviance for a model $M_0$ on a dataset $y$ is defined as
$$D(y) = -2(\log(p(y|\hat{\theta}_0))-\log(p(y|\hat{\theta}_s)))$$
where $\hat{\theta}_0$ denotes the fitted values of the parametesr in the model $M_0$ while $\hat{\theta}_s$ denotes the fitted parameters for the saturated model. Here the saturated model is a model with a parameter for every observation so that the data are fitted exactly. 

#### distributions
Distributions are objects that generalize the classical notion of functions in mathematical analysis. Distributions make it possible to differentiate functions whose derivatives do not exist in the classical sense. 

#### expectations
The expected value of a random variable is the long-run average value of repetions of the experiment it represents. 

If $X$ is a discrete random variable, we can determine the expected value, $E[X]$:
$$E[X] = \sum\limits_{i=1}^\infty x_ip_i$$
when $x_i$ are the possible discrete values that $X$ can take on and $p_i=  P(X = x_i)$

If $X$ is a continuous random variable, we can determine the expected value, $E[X]$:
$$E[X] = \int\limits_{-\infty}^\infty xf(x)dx$$
where $f(x)$ is the pdf of $X$.

#### global minimum
The global minimum of the function $f(\theta)$ is 
$$\min_{\theta'} f(\theta')$$

For many optimization problems, we attempt to minimize some objective function $f$ over parameter space $\theta$. We hope to find the global minimum but often times we find ourselves stuck in local minima.

#### inference
Statistical inference uses mathematics to draw conclusions in the presence of uncertainty. 

#### jensen's inequality
Jensen's inequality relates the value of a convex function of an integral to the integral of the convex function.

If $X$ is a random variable and $g$ is a convex function, then
$$g(E[X]) \leq E[g(X)]$$

#### kl-divergence
The Kullback-Leibler divergence is a measure of the non-symmetric difference between two probability distributions $P$ and $Q$. In other words, it is an attempt to measure the difference between two distributions by taking the expected logarithmic difference between two probabilities $P$ and $Q$.
$$D_{KL}(P || Q) = \sum\limits_i P(i) \log \frac{P(i)}{Q(i)}$$
and for continuous random variables
$$D_{KL}(P || Q) = \int\limits_{-\infty}^\infty p(x) \log \frac{p(x)}{q(x)}dx$$

#### exponential distribution
The exponential distribution is the probability distribution that describes the time between events in a Poisson process. For example, if $Y$ is the number of pieces of mail I receive today and $Y\sim \text{Pois}(\lambda)$, then if we let $X$ be the time between each letter receieved, then $X \sim \text{Exp}(\lambda)$

#### probability
Probability is the measure of the likelihood than an event will occur. Probability is quantified as a number between 0 and 1 where 0 is impossibility and 1 is certainty.

#### lotus
The Law of the unconscious statistician is a theorem used to calculate the expected value of a function $g(X)$ of a random variable $X$ when one knows the probability distribution of $X$ but one does not explicitly know the distribution of $g(X)$.

For discrete random variables:
$$E[g(X)] = \sum\limits_x g(x)f_X(x)$$

For continuous random variables:
$$E[g(X)] = \int\limits_{-\infty}^\infty g(x)f_X(x)dx$$ 

#### empirical distribution

#### empirical risk minimization

#### energy

#### gradient descent

#### hoeffding's inequality

#### hypothesis space

#### importance sampling

#### integration

#### inverse transform

#### lasso

#### law of large numbers

#### likelihood

#### likelihood-ratio

#### linear regression

#### log-likelihood

#### logistic regression

#### marginal distribution

#### markov chain

#### maxent

#### maximum likelihood

#### mcmc

#### metropolis-hastings

#### minibatch sgd

#### monte-carlo

#### normal distribution

#### optimization

#### out-of-sample error

#### parametric model

#### pdf

#### pmf

#### posterior

#### probabilistic distribution

#### proposal

#### regularization

#### rejection sampling

#### rejection sampling on steroids

#### ridge

#### sampling

#### sampling distribution

#### sampling distribution of variance

#### sgd

#### simulated annealing

#### standard error

#### staistical mechanics

#### stochastic noise

#### stratification

#### test error

#### testing set

#### theano

#### training error

#### training set

#### travelling salesman problem

#### uniform distribution

#### validation error

#### variance

#### variance reduction

#### boltzmann distribution

#### box loop

#### complexity parameter

#### cross-validation

#### Bayesian Regression

#### Bayesian Statistics

#### Differentiation

#### Discrete Random Variable

#### Fundamental Rules of Probability

#### Gaussian Processes

#### Generalized Linear Models (GLMs)

#### Generative Adversarial Net (GAN)

#### Generative Models

#### Hidden Markov Chain (HMC)

#### Hierarchical Model

#### Independence

#### Independent Identically Distributed (IID)

#### Intersection

#### Latent Variable

#### Neural Network (NN)

#### Non-Parametric

#### Population

#### Posterior Predictive

#### Random Variable

#### Resampling

#### Union


