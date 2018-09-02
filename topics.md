---
title: Topic Index
shorttitle: Topics
layout: default
---

These are not listed in the order of when they will be covered, or even the depth in which they will be covered (this is one course, after all). But these are all topics we will be touching on. Some will be covered in class, some in homework, some in lab, and some you will be expected to read on your own.

Expect this list to either shrink, or for some topics to be replaced, as the semester goes on!

### introduction

- why this course
- problems you can solve
- the [Box loop](wiki/boxloop.html)
- representing models graphically

### Probability

- the basics of [Probability](wiki/probability.html)
- [probability mass functions, densities and cumulative functions](wiki/distributions.html)
- distributions [example](wiki/distrib-example.html)
- [expectations and integration](wiki/Expectations.html)

### Distributions

- Gaussian Distribution
- Bernoulli Distribution
- Binomial Distribution
- Poisson Distribution
- Exponential Distribution

### Basic Stats and Monte Carlo

- [law of large numbers](wiki/Expectations.html)
- pdf's vs sampling
- monte-carlo for [integrals](wiki/montecarlointegrals.html)
- [sampling and central limit theorem](wiki/SamplingCLT.html)

### Frequentist Statistics

- [Frequentist principles](wiki/frequentist.html) with sampling distributions and bootstrap
- [frequentist example](wiki/frequentist-example.html)
- [Sampling distributions and bootstrap](wiki/frequentist.html)
- p-values and confidence intervals

### sampling methods

- sampling vs simulation
- the [inverse method](wiki/inversetransform.html) from the cdf
- [rejection sampling](wiki/rejectionsampling.html)
- [importance sampling](wiki/importancesampling.html)
- SIR
- 2D and marginals from a sampling perspective

### Maximum Likelihood and Risk

- [maximum likelihood and log-likelihood](wiki/MLE.html)
- density estimation vs supervised learning
- covariates and linear regression: decision risk
- logistic regression

### Machine Learning a model

- approximation (ERM) vs Statistics
- bias and variance
- cross-validation
- regularization
- classification via decision risk

### Optimization

- basic optimization
- gradient free methods
- gradient based methods
- stochastic gradient descent(SGD)
- convexity and Jensen's inequality
- theano and automatic differentiation
- SGD using Theano for logistic regression

### Information Theory and Statistical mechanics

- entropy and cross-entropy
- KL divergence and deviance
- model comparison with likelihood ratios and AIC
- maximum entropy distributions: binomial and normal
- the exponential family of distributions
- statistical mechanics: stationarity and the ensembles
- the boltzmann distribution


### Combinatoric optimization and markov chains

- combinatoric optimization methods
- markov chains
- simulated annealing
- the simulated annealing markov chain
- the traveling salesman problem

### Hidden variables and learning

- hidden variables
- mixture models and unsupervised learning
- generative vs discriminative models
- missing data and Data Augmentation
- the expectation maximization algorithm
- EM algorithm, statistical version
- Applications of EM

### Basic Bayesian Stats

- the meaning of bayes theorem
- MLE of a binomial and beta-binomial bayesian updating
- the formal structure of bayesian inference and the globe throw example
- posteriors, marginal posteriors and posterior predictives
- frequentist equivalences to bayesian stats
- priors and their choice

### Even more bayes

- MAP, plugin predictive, and point estimates
- posterior predictive intervals
- shrinkage and regularization
- empirical bayes and the (ever more) bayesian hierarchy
- hierarchical models and regularization: using empirical bayes
- combining multiple experiments: bayesian meta-analysis

### Machine Learning and Decision Making from a bayesian perspective

- point estimates from decision theory: decision risk
- the bayesian structure of machine learning through posterior predictives
- generative models revisited and LDA
- hyper-parameters in a bayesian setup.
- are we playing with parameters or with models?
- multistage decision analysis

### MCMC

- when is MCMC needed? (why not always use importance sampling)
- details of the markov chain and the proposal distribution
- how to write a Metropolis-Hastings (MH) sampler
- MCMC convergence tuning and diagnostics: burnin, thinning, and autocorrelation
- the structure of pymc
- gibbs sampling, a simpler version of MCMC
- different kinds of gibbs
- the relationship of gibbs to Data Augmentation and EM
- Hierarchical model full bayesian: alternating MH and gibbs for different posteriors (rats)
- Missing data from a sampling perspective

### Convergence and Model checking

- Convergence problems with MCMC and gibbs: correlations and efficiency
- Gelman Rubin and Gewecke tests
- External Validation of models using holdout sets
- Posterior predictive checking, posterior replications
- Posterior predictive p-values
- Interesting ideas to fix convergence issues

### More sampling

- Slice sampler
- Mechanics and Statistical Mechanics for HMC
- Hamiltonian Monte Carlo
- NUTS and other improvements on HMC
- HMC convergence vs others

### From density models to regression

- regression as bayesian updating
- normal prior as ridge regression
- exponential family and glms with a link function
- a bayesian glm example
- exposure and zero-inflation in glms
- overdispersion in glms
- hierarchical GLMs: radon example

### Model comparision and selection

- out of sample performance
- evidence
- bayes ratios
- cross validation (LOO) for model selection
- BIC/WAIC/DIC etc measures: KL and deviance out of sample.
- model averaging and ensembles

### Variational Algorithms

- normal approximation
- marginal posterior modes with EM
- variational inference
- expectation propagation
- ADVI

### Non-IID temporal models

- time series and dealing with conditional dependence on previous times
- Hidden Markov Models (HMM)
- viterbi and other algorithms
- stochastic processes
- Kalman filters
- Sequential Monte Carlo
- Particle Filters

### Covariance and Gaussian Processes

- glms with a covariance in intercepts and slopes
- spatial autocorrelation in glms
- gaussian processes
- gaussian processes for regression
- the capacity of models
- bayesian non-parametrics

### Long Running models in this course

- Rat Tumors
- Kidney Cancer
- Oceanic tools
- Radon in houses
- Chimpanzees
- Drinking Monks
