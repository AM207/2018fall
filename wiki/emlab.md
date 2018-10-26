---
title: EM Lab
shorttitle: emlab
notebook: emlab.ipynb
noline: 1
summary: ""
keywords: ['maximum likelihood', 'mixture model', 'full-data likelihood', 'x-likelihood', 'latent variables', 'log-likelihood', 'training set', 'normal distribution', 'z-posterior']
data: ['ncog.txt']
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}




```python
%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import pymc3 as pm
```


$$\newcommand{\isum}{\sum_{i}}$$
$$\newcommand{\zsum}{\sum_{k=1}^{K}}$$
$$\newcommand{\zsumi}{\sum_{\{z_i\}}}$$

## The EM Algorithm

### E-step

These observations set up the EM algorithm for us. If we choose, in the **E-step**, at some (possibly initial) value of the parameters $\theta_{old}$,

$$q(z) = p(z  \vert  x, \theta_{old}),$$ 

we then set the Kullback Liebler divergence to 0, and thus $\mathcal{L}(q, \theta)$ to the log-likelihood at $\theta_{old}$,  and maximizing the lower bound. 

Using this missing data posterior, conditioned on observed data, and $\theta_{old}$, we compute the expectation of the missing data with respect to the posterior and use it later.

![](images/klsplitestep.png)

### Now the **M-step**. 

Since after the E-step, the lower bound touches the log-likelihood, any maximization of this ELBO from its current value with respect to $\theta$ will also “push up” on the likelihood itself. Thus M step guaranteedly modifies the parameters $\theta$ to increase (or keep same) the likelihood of the observed data.

Thus we hold now the distribution $q(z)$ fixed at the hidden variable posterior calculated at $\theta_{old}$, and maximize $\mathcal{L}(q, \theta)$ with respect to $\theta$ to obtain new parameter values $\theta_{new}$. This is a regular maximization.

The distribution $q$, calculated as it is at $\theta_{old}$ will not in general equal the new posterior distribution $p(z \vert x,\theta_{new})$, and hence there will be a nonzero KL divergence. Thus the increase in the log-likelihood will be greater than the increase in the lower bound $\mathcal{L}$, as illustrated below.

The M in “M-step” and “EM” stands for “maximization”.

![](images/klsplitmstep.png)

### The process

Note that since $\mathcal{L}$ is maximized with respect to $\theta$, one can equivalently maximize the expectation of the full-data log likelihood $\mathrm{E_q[\ell( x,z  \vert  \theta)]}$ in the M-step since the difference is purely a function of $q$. Furthermore, if the joint distribution $p(x, z \vert  \theta)$ is a member of the exponential family, the log-likelihood will have a particularly simple form and will lead to a much simpler maximization than that of the incomple-data log-likelihood $p(x \vert \theta)$.

We now set $\theta_{old} = \theta_{new}$ and repeat the process. This **EM algorithm** is presented and  illustrated below:

![](images/emupdate.png)

1. We start with the log-likelihood $p(x  \vert  \theta)$(red curve) and the initial guess $\theta_{old}$ of the parameter values
2. Until convergence (the $\theta$ values dont change too much):
    1. E-step: Evaluate the hidden variable posterior $q(z, \theta_{old}) = p(z  \vert  x, \theta_{old})$ which gives rise to a lower bound function of $\theta$: $\mathcal{L}(q(z, \theta_{old}), \theta)$(blue curve) whose value equals the value of $p(x  \vert  \theta)$ at $\theta_{old}$.
    2. M-step: maximize the lower bound function with respect to $\theta$ to get $\theta_{new}$.
    3. Set $\theta_{old} = \theta_{new}$
    
One iteration more is illustrated above, where the subsequent E-step constructs a new lower-bound function that is tangential to the log-likelihood at $\theta_{new}$, and whose value at $\theta_{new}$ is higher than the lower bound at $\theta_{old}$ from the previous step.

Thus

$$\ell(\theta_{t+1}) \ge \mathcal{L}(q(z,\theta_t), \theta_{t+1}) \ge \mathcal{L}(q(z,\theta_t), \theta_{t}) = \ell(\theta_t)$$

The first equality follows since $\mathcal{L}$ is a lower bound on $\ell$, the second from the M-step's maximization of $\mathcal{L}$, and the last from the vanishing of the KL-divergence after the E-step. As a consequence, you **must** observe monotonic increase of the observed-data log likelihood $\ell$ across iterations. **This is a  powerful debugging tool for your code**.

## The Gaussian Mixture model using EM

We dont know how to solve for the MLE of the unsupervised problem. The EM algorithm comes to the rescue. As described above here is the algorithm:


* Repeat until convergence 
*  E-step: For each $i,j$ calculate 

$$ w_{i,j} = q_i(z_i=j)=p(z_i=j \vert  x_i, \lambda, \mu, \Sigma) $$
     
* M-step: We need to maximize, with respect to our parameters the
  
$$
\begin{eqnarray}
 \mathcal{L} &=& \sum_i \sum_{z_i} q_i(z_i) \log \frac{p(x_i,z_i  \vert \lambda, \mu, \Sigma)}{q_i(z_i)} \nonumber \\
 \mathcal{L} &=& \sum_i \sum_{j=i}^{k}  q_i(z_i=j) \log \frac{p(x_i \vert z_i=j , \mu, \Sigma) p(z_i=j \vert \lambda)}{q_i(z_i=j)} \\
 \mathcal{L} & =&  \sum_{i=1}^{m} \sum_{j=i}^{k} w_{i,j}  \log \left[   \frac{ \frac{1}{ (2\pi)^{n/2} \vert \Sigma_j \vert ^{1/2}} \exp \left(    -\frac{1}{2}(x_i-\mu_j)^T \Sigma_j^{-1} (x_i-\mu_j) \right)  \, \lambda_j   }{w_{i,j}}\right]
\end{eqnarray}
$$

Taking the derivatives yields the following updating formulas:

$$
\begin{eqnarray}
 \lambda_j &=& \frac{1}{m} \sum_{i=1}^m w_{i,j} \nonumber \\ 
 \mu_j&=& \frac{ \sum_{i=1}^m  w_{i,j} \, x_i}{ \sum_{i=1}^m  w_{i,j}} \nonumber \\ 
 \Sigma_j &=& \frac{ \sum_{i=1}^m  w_{i,j} \, (x_i-\mu_j)(x_i-\mu_j)^T}{ \sum_{i=1}^m  w_{i,j}}
\end{eqnarray}
$$

To calculate the E-step we basically calculating the posterior of the  $z$'s given the $x$'s and the
current estimate of our parameters. We can use Bayes rule 

$$ w_{i,j}= p(z_i=j \vert  x_i, \lambda, \mu, \Sigma) = \frac{p( x_i \vert  z_i=j,  \mu, \Sigma)\, p(z_i=j \vert \lambda)}{\sum_{l=1}^k p(x_i  \vert  z_i=l,  \mu, \Sigma) \, p(z_i=l \vert \lambda)} $$

Where $p(x_i  \vert  z_i =j,  \mu, \Sigma)$ is the density of the Gaussian with mean $\mu_j$ and covariance 
$\Sigma_j$ at $x_i$ and $p(z_i=j \vert  \lambda)$ is simply $\lambda_j$. 
If we to compare these formulas in the M-step with the ones we found in GDA we can see
that are very similar except that instead of using $\delta$ functions we use the $w$'s. Thus the EM algorithm corresponds here to a weighted maximum likelihood and the weights are interpreted as the 'probability' of coming from that Gaussian instead of the deterministic 
$\delta$ functions. Thus we have achived a **soft clustering** (as opposed to k-means in the unsupervised case and classification in the supervised case).



```python
#In 1-D
# True parameter values
mu_true = [2, 5]
sigma_true = [0.6, 0.6]
lambda_true = .4
n = 1000

# Simulate from each distribution according to mixing proportion psi
z = np.random.binomial(1, lambda_true, n)
x = np.array([np.random.normal(mu_true[i], sigma_true[i]) for i in z])

plt.hist(x, bins=20);
```



![png](emlab_files/emlab_8_0.png)




```python
#The 1000 z's we used in our simulation but which we shall promptly forget
z
```





    array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
           0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
           0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
           0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
           1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
           0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
           1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
           1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0,
           1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1,
           1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
           0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0,
           1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0,
           0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
           0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
           1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
           1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1,
           1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
           1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0,
           0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
           0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
           1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
           1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
           1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
           1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1,
           0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
           1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
           0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])





```python
#from Bios366 lecture notes
from scipy.stats.distributions import norm

def Estep(x, mu, sigma, lam):
    a = lam * norm.pdf(x, mu[0], sigma[0])
    b = (1. - lam) * norm.pdf(x, mu[1], sigma[1])
    return b / (a + b)


```




```python
def Mstep(x, w):
    lam = np.mean(1.-w) 
    
    mu = [np.sum((1-w) * x)/np.sum(1-w), np.sum(w * x)/np.sum(w)]
    
    sigma = [np.sqrt(np.sum((1-w) * (x - mu[0])**2)/np.sum(1-w)), 
             np.sqrt(np.sum(w * (x - mu[1])**2)/np.sum(w))]
    
    return mu, sigma, lam
```




```python
print(lambda_true, mu_true, sigma_true)
# Initialize values
mu = np.random.normal(4, 10, size=2)
sigma = np.random.uniform(0, 5, size=2)
lam = np.random.random()
print("Initials, mu:", mu)
print("Initials, sigma:", sigma)
print("Initials, lam:", lam)

# Stopping criterion
crit = 1e-15

# Convergence flag
converged = False

# Loop until converged
iterations=1


while not converged:
    # E-step
    if np.isnan(mu[0]) or np.isnan(mu[1]) or np.isnan(sigma[0]) or np.isnan(sigma[1]):
        print("Singularity!")
        break
        
    w = Estep(x, mu, sigma, lam)

    # M-step
    mu_new, sigma_new, lam_new = Mstep(x, w)
    
    # Check convergence
    converged = ((np.abs(lam_new - lam) < crit) 
                 & np.all(np.abs((np.array(mu_new) - np.array(mu)) < crit))
                 & np.all(np.abs((np.array(sigma_new) - np.array(sigma)) < crit)))
    mu, sigma, lam = mu_new, sigma_new, lam_new
    iterations +=1           

print("Iterations", iterations)
print('A: N({0:.4f}, {1:.4f})\nB: N({2:.4f}, {3:.4f})\nlam: {4:.4f}'.format(
                        mu_new[0], sigma_new[0], mu_new[1], sigma_new[1], lam_new))
```


    0.4 [2, 5] [0.6, 0.6]
    Initials, mu: [ 2.77851696  4.82539293]
    Initials, sigma: [ 1.92145614  2.00233266]
    Initials, lam: 0.21517291484839296
    Iterations 51
    A: N(2.0413, 0.5991)
    B: N(5.0397, 0.6518)
    lam: 0.6144


The output of the estep is indeed our classification probability, if we want to use our unsupervized model to make a classification.



```python
zpred = 1*(Estep(x, mu, sigma, lam) >= 0.5)
```




```python
zpred
```





    array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
           0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
           0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
           0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
           1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
           0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
           1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
           1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0,
           1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1,
           1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
           0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0,
           1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0,
           0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
           0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
           1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
           1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
           1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
           1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0,
           0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
           0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
           1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
           1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
           1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
           1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1,
           0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
           1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
           0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])





```python
np.mean(z==zpred)
```





    0.99399999999999999



## Survival Analysis using EM

The critical problem in survival analysis is that data tends to be right-censored. Imaginw a medical study following a cohort which has got cancer treatment. The study ends 5 years from start. Upto then, some patients have died. We record the time they died. But some were alive past the 5 years, and we stop recording, so we dont know, how long further from 5 years they will continue to live.

There are some key concepts and definitions we need.

Define T as a random lifetime taken from the population.

The **survival function** $S(t)$ of a population is defined as:

$$S(t) = P(T > t), $$

the probability of surviving past time t. The function $S$ is 1 minus the CDF of T, or $$S(t) = 1 - CDF_T(t)$$


The **Hazard Function** is defined as the probability that the death event occurs at time t, conditional on the fact that the death event has not occurred until time t. 

In other words, we care about:

$$\lambda(t) =  \lim_{\delta t \rightarrow 0 } \; \frac{P( t \le T \le t + \delta t \mid T > t)}{\delta t}$$

Using Bayes Theorem we can see that:

$$\lambda(t) = \frac{-S'(t)}{S(t)}$$

Solving this differential equation needs:

$$S(t) = \exp\left( -\int_0^t \lambda(z) \mathrm{d}z \right)$$

For a constant Hazard Function, this is just an exponential model:

$$f_T(t) = \lambda(t)\exp\left( -\int_0^t \lambda(z) \mathrm{d}z \right)$$

(Formulae are taken from the docs for Python's lifelines package)

If you think we are being too morbid here, rest assured that survival analysis finds applications in many places, survival of politicians in office, survival os machinery and circuits. For example, a Weibull distribution which is concave is used to model all of early failures, random failures, and age related wear and tear, while the exponential distribution is used when u dont know how old the parts are (as then any memory based distribution is out). 

In general, you might even assume a non-parametric hazard and use techniques such as Kaplan-Meier estimation and Cox's proportional hazards (for regressions on subject covariates).

But let us focus on the exponential distribution.

As we said before, when our study ends, some patients are still alive. So we have incomplete data $z_m$ as the unseen lifetime for those censored, and the data we observe:

$$x_m = (a_m, d_m)$$

where $a_m$ is the age reached, and $d_m =1$ if the data was not censored and $d_m = 0$ if it was.

We assume that the age of death is exponentially distributed with constant hazard $\lambda$, so that

$$\ell(\{x_m\}, \{z_m\} \mid \lambda) = log \left( \prod_{m=1}^{n} \frac{1}{\lambda} e^{-\lambda x_m} \right)$$

Splitting the data up into censored and non-censored data, and expanding the log-likelihood we get:

$$\ell(\{x_m\}, \{z_m\} \mid \lambda) = -n log \lambda - \frac{1}{\lambda} \sum_{m=1}^{n} x_m = -n log \lambda - \frac{1}{\lambda} \sum_{m=1}^{n}  \left( a_m d_m + (1 - d_m) z_m \right)$$

Now the $Q$ function is the z-posterior expectation of the full-data likelihood, so that:

$$Q(\lambda, \lambda_{old}) = E_z \left[\ell(\{x_m\}, \{z_m\} \mid \lambda) \mid \lambda_{old} \right]$$

Thus we only need the expectations of the $z$ with respect to their posterior. Since we assume exponential distribution, we are assuming that our data is "memoryless". That is, given that we have survived till censoring, or expected survival time beyond that is simply $\lambda_{old}$, giving a total expected age equal to censoring-time plus $\lambda_{old}$.

Thus:

$$Q(\lambda, \lambda_{old}) = -n log \lambda - \frac{1}{\lambda}  \sum_{m=1}^{n} \left( a_m d_m + (1 - d_m) (c_m + \lambda_{old}) \right)$$

That does the e-step. The m-step simply differentiates this with respect to lambda to find the x-data MLE.

$$\lambda_{MLE} = \frac{1}{n} \left[\sum_{m=1}^{n} \left( a_m d_m + (1 - d_m) (c_m + \lambda_{old}) \right) \right]$$



```python
def iteration(lamo, data):
    n = data.shape[0]
    summer = np.sum(data.a*data.d + (1-data.d)*(data.c + lamo))
    return summer/n
def run_em(data, lam_init, n_iter):
    lamo = lam_init
    for i in range(n_iter):
        lam_mle = iteration(lamo, data)
        lamo = lam_mle
        print("iteration i", i, lamo)
```


Hers is the data description:

```
Head and neck cancer from the Northeren California Oncology Group (NCOG)
Data matrix gives detailed time information, with t (months till death/censoring) being used in our analysis.
The column d is the death(1)/censoring(0) indicator, and arm the treatment arm (A Chemotherapy, B Chemotherapy + Radiation)
```



```python
sdata = pd.read_csv("data/ncog.txt", sep=" ")
sdata.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>t</th>
      <th>d</th>
      <th>arm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>6</td>
      <td>78</td>
      <td>248</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>12</td>
      <td>78</td>
      <td>160</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>7</td>
      <td>78</td>
      <td>319</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>11</td>
      <td>78</td>
      <td>277</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>12</td>
      <td>78</td>
      <td>440</td>
      <td>1</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>





```python
sdata[sdata.d==0]
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>t</th>
      <th>d</th>
      <th>arm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>7</td>
      <td>78</td>
      <td>319</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>21</th>
      <td>6</td>
      <td>11</td>
      <td>80</td>
      <td>1226</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>24</th>
      <td>22</td>
      <td>1</td>
      <td>81</td>
      <td>74</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>27</th>
      <td>12</td>
      <td>6</td>
      <td>81</td>
      <td>1116</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>28</th>
      <td>6</td>
      <td>8</td>
      <td>81</td>
      <td>1412</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>30</th>
      <td>15</td>
      <td>7</td>
      <td>81</td>
      <td>1349</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>31</th>
      <td>23</td>
      <td>10</td>
      <td>81</td>
      <td>185</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>46</th>
      <td>23</td>
      <td>11</td>
      <td>83</td>
      <td>523</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>50</th>
      <td>6</td>
      <td>7</td>
      <td>84</td>
      <td>279</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>54</th>
      <td>4</td>
      <td>12</td>
      <td>78</td>
      <td>2146</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>55</th>
      <td>5</td>
      <td>3</td>
      <td>79</td>
      <td>2297</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>56</th>
      <td>12</td>
      <td>3</td>
      <td>79</td>
      <td>528</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>59</th>
      <td>16</td>
      <td>4</td>
      <td>79</td>
      <td>169</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>63</th>
      <td>13</td>
      <td>9</td>
      <td>79</td>
      <td>2023</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>67</th>
      <td>27</td>
      <td>3</td>
      <td>80</td>
      <td>1771</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>68</th>
      <td>14</td>
      <td>2</td>
      <td>80</td>
      <td>1897</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>74</th>
      <td>19</td>
      <td>12</td>
      <td>80</td>
      <td>1642</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>79</th>
      <td>18</td>
      <td>9</td>
      <td>81</td>
      <td>1245</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>81</th>
      <td>29</td>
      <td>10</td>
      <td>81</td>
      <td>1331</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>82</th>
      <td>5</td>
      <td>3</td>
      <td>82</td>
      <td>1092</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>89</th>
      <td>20</td>
      <td>5</td>
      <td>83</td>
      <td>759</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>92</th>
      <td>6</td>
      <td>10</td>
      <td>83</td>
      <td>613</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>93</th>
      <td>23</td>
      <td>11</td>
      <td>83</td>
      <td>547</td>
      <td>0</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
</div>





```python
sdata['a'] = sdata['c'] = sdata.t
```




```python
sdata.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>t</th>
      <th>d</th>
      <th>arm</th>
      <th>a</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>6</td>
      <td>78</td>
      <td>248</td>
      <td>1</td>
      <td>A</td>
      <td>248</td>
      <td>248</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>12</td>
      <td>78</td>
      <td>160</td>
      <td>1</td>
      <td>A</td>
      <td>160</td>
      <td>160</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>7</td>
      <td>78</td>
      <td>319</td>
      <td>0</td>
      <td>A</td>
      <td>319</td>
      <td>319</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>11</td>
      <td>78</td>
      <td>277</td>
      <td>1</td>
      <td>A</td>
      <td>277</td>
      <td>277</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>12</td>
      <td>78</td>
      <td>440</td>
      <td>1</td>
      <td>A</td>
      <td>440</td>
      <td>440</td>
    </tr>
  </tbody>
</table>
</div>





```python
A_avg = np.mean(sdata[sdata.arm=='A'].t)
B_avg = np.mean(sdata[sdata.arm=='B'].t)
A_avg, B_avg
```





    (357.84313725490193, 639.2)





```python
run_em(sdata[sdata.arm=='A'], A_avg, 10)
```


    iteration i 0 420.991926182
    iteration i 1 432.135830111
    iteration i 2 434.102401392
    iteration i 3 434.449443383
    iteration i 4 434.510686087
    iteration i 5 434.521493623
    iteration i 6 434.523400835
    iteration i 7 434.523737402
    iteration i 8 434.523796796
    iteration i 9 434.523807278




```python
run_em(sdata[sdata.arm=='B'], B_avg, 15)
```


    iteration i 0 838.062222222
    iteration i 1 899.930469136
    iteration i 2 919.178368176
    iteration i 3 925.166603432
    iteration i 4 927.029609957
    iteration i 5 927.609211987
    iteration i 6 927.789532618
    iteration i 7 927.84563237
    iteration i 8 927.863085626
    iteration i 9 927.868515528
    iteration i 10 927.870204831
    iteration i 11 927.870730392
    iteration i 12 927.8708939
    iteration i 13 927.870944769
    iteration i 14 927.870960595


## Other resons why is EM important.

We have motivated the EM algorithm using mixture models and missing data, but that is not its only place of use. 

Since MLE's can overfit, we often prefer to use MAP estimation. EM is a perfectly reasonable method for MAP estimation in mixture models; you just need to multiply in the prior.

Basically the EM algorithm has a similar setup to the data augmentation problem and can be used in any problem which has a similar structure. Suppose for example you have two parameters $\phi$ and $\gamma$ in a posterior estimation, with daya $y$. Say that we'd like to estimate the posterior $p(\phi  \vert  y)$. It may be relatively hard to estimate this, but suppose we can  work with $p(\phi  \vert  \gamma, y)$ and $p(\gamma  \vert  \phi, y)$. Then you can use the structure of the EM algorithm to estimate the marginal posterior of any one parameter. Start with:

$$log p(\phi  \vert  y) = log p(\gamma, \phi  \vert  y) - log p(\gamma  \vert  \phi, y)$$

Notice the similarity of this to the above expressions with $\phi$ as $x$, $y$ as $\theta$, and $\gamma$ as $z$. Thus the same derivations apply toany problem with this structure.

This structure can also be used in type-2 likelihood or emprical bayes estimation.
