---
title: Formal Tests for Convergence
shorttitle: gewecke
noline: 1
layout: wiki
keywords: ['bayesian', 'MCMC', 'mcmc engineering', 'formal tests']
summary: "The Gewecke test takes two non-overlapping samples and compares their means, trying to see how similar they are. The Gelman-Rubin test compares the winthin-chain to between-chain variance."
---

So far, we have looked for stationarity visually. We have inspected traces to see if they are white-noisy. We have made histograms over subsets of the chain, and traceplots of multiple chains with different starting points. Let us now look at some formal tests.

## Gewecke Test: Testing for difference in Means

Let $H_0$ be the null hypothesis that there is no difference in the means of the distributions from each of the stationary chains. That is

$$H_0 : \mu_{\theta_1}  - \mu_{\theta_2} = 0 \implies \mu_{\theta_1 - \theta_2} = 0$$

In other words, the distribution of the difference in chain samples is a 0-mean distribution.

The standard deviation of this distribution is:

$$\sigma_{\theta_1 - \theta_2} = \sqrt{\frac{var(\theta_1)}{n_1} + \frac{var(\theta_2)}{n_2} }$$

Now, assume the usual rejection of $H_0$ if the p-value is below the 5% limit: that is, if $H_0$ is correct, there is only a 5% chance of the absolute value of the mean difference being larger than 2 standard deviations (1.96 to be precise). That is:

$$\vert \mu_{\theta_1}  - \mu_{\theta_2}  \vert < 2 \sigma_{\theta_1 - \theta_2} $$



## Gelman-Rubin Test

This test also uses multiple chains. It compares between-chain and within-chain variance. If these are very different we havent converged yet.

Lets assume that we have m chains, each of length n. The sample variance of the $j$th chain is:

$$ s_j^2 =  \frac{1}{n-1} \sum_i (\theta_{ij} - \mu_{\theta_j})^2$$

Let $w$ be the mean of the within-chain variances. Then:

$$w = \frac{1}{m} \sum_j s_j^2$$

Note that we expect the winthin changes to be all equal asymptotocally as $n \to \infty$ as we have then reached stationarity.

Let $\mu$ be the mean of the chain means:

$$\mu = \frac{1}{m} \sum_j \mu_{\theta_j}$$

The between chain variance can then be written as:

$$B = \frac{n}{m-1}\sum_j (\mu_{\theta_j} - \mu)^2$$

This is the variance of the chain means multiplied by the number of samples in each chain.

We use the weighted average of these two to estimate the variance of the stationary distribution:

$$\hat{Var}(\theta) = (1 - \frac{1}{n})w + \frac{1}{n} B$$

Since the starting points of our chains are likely not from the stationary distribution, this overestimates our variance, but is unbiased under stationarity. There $n \to \infty$ and only the first term sruvives and gives us $w$.

Lets define the ratio of the estimated distribution variance to the asymptotic one.:

$$\hat{R} = \sqrt{\frac{\hat{Var}(\theta)}{w}}$$

Stationarity would imply that this value is 1. The departure from stationarity, the overestimation then shoes up in a ratio larger than 1.



## Autocorrelation and Mixing: Effective Sample size

As we have seen, autocorrelation and stationarity are related but not identical concepts. A large autocorrelation may happen due to strong correlations in parameters (which can be measured), or due to smaller step sizes which are not letting us explore a distribution well. In other words, our mixing is poor. Unidentifiability also typically causes large correlations (and autocorrelations) as two parameters may carry the same information.

The general observation that can be made is that problems in sampling often cause strong autocorrelation, but autocorrelation by itself does not mean that our sampling is wrong. But it is something we should always investigate, and only use our samples if we are convinced that the autocorrelation is benign.

A good measure to have that depends on autocorrelation is effective sample size(ESS). With autocorrelation, the 'iid'ness of our draws decreases. The ESS quantifies how much. The exact derivation involves us going into time series theory, which we do not have the time to do here. Instead we shall just write the result our:

$$n_{eff} = \frac{mn}{1 + 2 \sum_{\Delta t}\rho_{\Delta t}}$$

