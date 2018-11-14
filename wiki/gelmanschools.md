---
title: Gelman Schools and Hierarchical Pathology
shorttitle: gelmanschools
notebook: gelmanschools.ipynb
noline: 1
summary: ""
keywords: ['normal-normal model', 'hierarchical normal-normal model', 'hierarchical', 'divergences', 'non-centered hierarchical model', 'sampling distribution']
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


From Gelman:

>a simple hierarchical model based on the normal distribu- tion, in which observed data are normally distributed with a different mean for each ‘group’ or ‘experiment,’ with known observation variance, and a normal population distribution for the group means. This model is sometimes termed the one-way normal random-effects model with known data variance and is widely applicable, being an important special case of the hierarchical normal linear model,...

## Statement of the model

The particular example we will deal with is called the 8-schools example, and is described thus:

>A study was performed for the Educational Testing Service to analyze the effects of special coaching programs on test scores. Separate randomized experiments were performed to estimate the effects of coaching programs for the SAT-V (Scholastic Aptitude Test- Verbal) in each of eight high schools. The outcome variable in each study was the score on a special administration of the SAT-V, a standardized multiple choice test administered by the Educational Testing Service and used to help colleges make admissions decisions; the scores can vary between 200 and 800, with mean about 500 and standard deviation about 100. The SAT examinations are designed to be resistant to short-term efforts directed specifically toward improving performance on the test; instead they are designed to reflect knowledge acquired and abilities developed over many years of education. Nevertheless, each of the eight schools in this study considered its short-term coaching program to be successful at increasing SAT scores. Also, there was no prior reason to believe that any of the eight programs was more effective than any other or that some were more similar in effect to each other than to any other.

>the estimated coaching effects are $\bar{y}_j$, and their sampling variances, $\sigma_j^2$... The estimates $\bar{y}_j$ are obtained by independent experiments and have approximately normal sampling distributions with sampling variances that are known, for all practical purposes, because the sample sizes in all of the eight experiments were relatively large, over thirty students in each school 

![](images/8schools.png)

Notice the bar on the y's and the mention of standard errors (rather than standard deviations) in the third column in the table above. Why is this?

The answer is that these are means taken over many (> 30) students in each of the schools. The general structure of this model can be written thus:

>Consider $J$ independent experiments, with experiment $j$ estimating the parameter $\theta_j$ from $n_j$ independent normally distributed data points, $y_{ij}$, each with known error variance $\sigma^2$; that is,

$$y_{ij} \vert \theta_j \sim N(\theta_j, \sigma^2), \, i = 1,...,n_j; j = 1,...,J.$$

We label the sample mean of each group $j$ as

$$\bar{y_j} = \frac{1}{n_j} \sum_{i=1}^{n_j} y_{ij}$$

with sampling variance:

$$\sigma_j^2 = \sigma^2/n_j$$

  
>We can then write the likelihood for each $\theta_j$ using the sufficient statistics, $\bar{y}_j$:

$$\bar{y_j} \vert \theta_j \sim N(\theta_j,\sigma_j^2).$$

This is

>a notation that will prove useful later because of the flexibility in allowing a separate variance $\sigma_j^2$ for the mean of each group $j$. ...all expressions will be implicitly conditional on the known values $\sigma_j^2$.... Although rarely strictly true, the assumption of known variances at the sampling level of the model is often an adequate approximation.

>The treatment of the model provided ... is also appropriate for situations in which the variances differ for reasons other than the number of data points in the experiment. In fact, the likelihood  can appear in much more general contexts than that stated here. For example, if the group sizes $n_j$ are large enough, then the means $\bar{y_j}$ are approximately normally distributed, given $\theta_j$, even when the data $y_{ij}$ are not. 

## Setting up the hierarchical model

We'll set up the modelled in what is called a "Centered" parametrization which tells us how $\theta_i$ is modelled: it is written to be directly dependent as a normal distribution from the hyper-parameters. 



```python
J = 8
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])
sigma = np.array([15, 10, 16, 11,  9, 11, 10, 18])
```


We set up our priors in a Hierarchical model to use this centered parametrization. We can say: the $\theta_j$ is drawn from a Normal hyper-prior distribution with parameters $\mu$ and $\tau$. Once we get a $\theta_j$ then can draw the means from it given the data $\sigma_j$ and one such draw corresponds to our data.

$$
\mu \sim \mathcal{N}(0, 5)\\
\tau \sim \text{Half-Cauchy}(0, 5)\\
\theta_{j} \sim \mathcal{N}(\mu, \tau)\\
\bar{y_{j}} \sim \mathcal{N}(\theta_{j}, \sigma_{j})
$$

where $j \in \{1, \ldots, 8 \}$ and the
$\{ y_{j}, \sigma_{j} \}$ are given as data



```python
with pm.Model() as schools1:

    mu = pm.Normal('mu', 0, sd=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sd=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)
```




```python
with schools1:
    trace1 = pm.sample(5000, init=None, njobs=2, tune=500)
```


    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta, tau, mu]
    Sampling 2 chains: 100%|██████████| 11000/11000 [00:21<00:00, 510.72draws/s]
    There were 80 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 139 divergences after tuning. Increase `target_accept` or reparameterize.
    The number of effective samples is smaller than 10% for some parameters.




```python
pm.summary(trace1)
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
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>4.207586</td>
      <td>3.387222</td>
      <td>0.128527</td>
      <td>-2.364228</td>
      <td>10.815423</td>
      <td>443.438897</td>
      <td>1.003844</td>
    </tr>
    <tr>
      <th>theta__0</th>
      <td>6.212825</td>
      <td>5.831719</td>
      <td>0.159452</td>
      <td>-4.128980</td>
      <td>18.410220</td>
      <td>977.723211</td>
      <td>1.000012</td>
    </tr>
    <tr>
      <th>theta__1</th>
      <td>4.745148</td>
      <td>4.785493</td>
      <td>0.135035</td>
      <td>-3.945520</td>
      <td>14.722419</td>
      <td>869.961535</td>
      <td>1.001362</td>
    </tr>
    <tr>
      <th>theta__2</th>
      <td>3.688946</td>
      <td>5.384133</td>
      <td>0.135945</td>
      <td>-6.881554</td>
      <td>14.602095</td>
      <td>1194.801444</td>
      <td>1.002863</td>
    </tr>
    <tr>
      <th>theta__3</th>
      <td>4.599909</td>
      <td>5.106523</td>
      <td>0.142366</td>
      <td>-5.636394</td>
      <td>14.969515</td>
      <td>866.237232</td>
      <td>1.001245</td>
    </tr>
    <tr>
      <th>theta__4</th>
      <td>3.345345</td>
      <td>4.826723</td>
      <td>0.134081</td>
      <td>-6.022840</td>
      <td>12.653606</td>
      <td>1026.652251</td>
      <td>1.003787</td>
    </tr>
    <tr>
      <th>theta__5</th>
      <td>3.861150</td>
      <td>4.892589</td>
      <td>0.126074</td>
      <td>-6.373854</td>
      <td>13.138098</td>
      <td>1186.571117</td>
      <td>1.003271</td>
    </tr>
    <tr>
      <th>theta__6</th>
      <td>6.239858</td>
      <td>5.293900</td>
      <td>0.156773</td>
      <td>-3.199409</td>
      <td>17.364483</td>
      <td>803.695943</td>
      <td>1.000267</td>
    </tr>
    <tr>
      <th>theta__7</th>
      <td>4.747230</td>
      <td>5.385314</td>
      <td>0.132700</td>
      <td>-5.646697</td>
      <td>15.551957</td>
      <td>1463.209855</td>
      <td>1.000123</td>
    </tr>
    <tr>
      <th>tau</th>
      <td>3.884377</td>
      <td>3.099556</td>
      <td>0.118285</td>
      <td>0.424971</td>
      <td>10.005174</td>
      <td>572.684459</td>
      <td>1.003743</td>
    </tr>
  </tbody>
</table>
</div>



The Gelman-Rubin statistic seems fine, but notice how small the effective-n's are? Something is not quite right. Lets see traceplots.



```python
pm.traceplot(trace1);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3449: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](gelmanschools_files/gelmanschools_14_1.png)


Its hard to pick the thetas out but $\tau$ looks not so white-noisy. Lets zoom in:



```python
pm.traceplot(trace1, varnames=['tau_log__'])
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3449: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x119e794a8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x119e9f978>]], dtype=object)




![png](gelmanschools_files/gelmanschools_16_2.png)


There seems to be some stickiness at lower values in the trace. Zooming in even more helps us see this better:



```python
plt.plot(trace1['tau_log__'], alpha=0.6)
plt.axvline(5000, color="r")
#plt.plot(short_trace['tau_log_'][5000:], alpha=0.6);
```





    <matplotlib.lines.Line2D at 0x119f09978>




![png](gelmanschools_files/gelmanschools_18_1.png)


We plot the cumulative mean of $log(\tau)$ as time goes on. This definitely shows some problems. Its biased above the value you would expect from many many samples.



```python
# plot the estimate for the mean of log(τ) cumulating mean
logtau = trace1['tau_log__']
mlogtau = [np.mean(logtau[:i]) for i in np.arange(1, len(logtau))]
plt.figure(figsize=(15, 4))
plt.axhline(0.7657852, lw=2.5, color='gray')
plt.plot(mlogtau, lw=2.5)
plt.ylim(0, 2)
plt.xlabel('Iteration')
plt.ylabel('MCMC mean of log(tau)')
plt.title('MCMC estimation of cumsum log(tau)')
```





    Text(0.5, 1.0, 'MCMC estimation of cumsum log(tau)')




![png](gelmanschools_files/gelmanschools_20_1.png)


## The problem with curvature

In-fact the "sticky's" in the traceplot are trying to drag the $\tau$ trace down to the true value, eventually. If we wait for the heat-death of the universe long this will happen, given MCMC's guarantees. But clearly we want to get there with a finite number of samples.

We can diagnose whats going on by looking for divergent traces in pymc3. In newer versions, these are obtained by a special boolean component of the trace.



```python
divergent = trace1['diverging']
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size/len(trace1)
print('Percentage of Divergent %.5f' % divperc)
```


    Number of Divergent 219
    Percentage of Divergent 0.04380


What does divergent mean? These are situations in which our symplectic integrator has gone Kaput, as illustrated in this diagram below from Betancourt's review:

![](images/sympldiv.png)

When a MCMC sampler (this problem is worse in non HMC models) encounters a region of high curvature it gets stuck, and goes off elsewhere, after a while. In the HMC scenario, the symplectic integrator diverges. What happens is that in encounters a region of high curvature which our timestep size $\epsilon$ is not able to resolve. Things become numerically unstable and the integrator veers off towards infinite energies, clearly not conserving energy any more.

![](images/sympldiv2.png)

Where is this curvature coming from? Things become a bit easier to understand if we plot the joint distribution of one of the thetas against a hyper-parameter. And we see that there is a triangular funnel structure with the hyperparameter spanning orders of magnitude in value. Using pymc we can plot where the divergences occur, and while they can occur anywhere they seem to be clustered in the neck of the distribution.



```python
theta_trace = trace1['theta']
theta0 = theta_trace[:, 0]
plt.figure(figsize=(10, 6))
plt.scatter(theta0[divergent == 0], logtau[divergent == 0], color='r', s=10, alpha=0.05)
plt.scatter(theta0[divergent == 1], logtau[divergent == 1], color='g', s=10, alpha=0.9)
plt.axis([-20, 50, -6, 4])
plt.ylabel('log(tau)')
plt.xlabel('theta[0]')
plt.title('scatter plot between log(tau) and theta[0]')
plt.show()
```



![png](gelmanschools_files/gelmanschools_25_0.png)


Two things have now happened...we have an increasing inability to integrate in the neck of this funnel, and we have lost confidence that our sampler is now actually characterizing this funnel well.

`pymc3` warning system also captures this, and the information can be drawn from there as well



```python
trace1.report.ok
```





    False





```python
trace1.report._chain_warnings[0][0]
```





    SamplerWarning(kind=<WarningType.DIVERGENCE: 1>, message='Energy change in leapfrog step is too large: 1458.41782758.', level='debug', step=2, exec_info=None, extra={'theta': array([ 4.87711111,  2.25999236,  2.34774179,  2.1574986 ,  3.79069814,
            4.34545496,  3.76649258,  3.32757565]), 'tau_log__': array(-0.3984309958799402), 'mu': array(4.635052778361571)})



## Funnels in hierarchical models

As is discussed in Betancourt and Girolami (2015) from where the following diagrams are taken, a funnel structure is common in Hierarchical models, and reflects strong correlations between down-tree parameters such as thetas, and uptree parameters such as $\phi = \mu, \tau$ here.

![](images/girolam1.png)

The funnel between $v = log(\tau)$ and $\theta_i$ in the hierarchical Normal-Normal model looks like this:

![](images/girolam2.png)

The problem is that a sampler must sample both the light and dark regions as both have enough probability mass.

### Divergences are good things

This is because they can help us diagnose problems in our samplers. Chances are that in a region with divergences the sampler is having problems exploring.

There is a second reason besides curvature and symplectic integration which affects the efficiency of HMC. This is the range of transitions a HMC sampler can make. For a euclidean mass-matrix sampler, the transitions are of the order of the range in kinetic energy, which itself is chi-squared distributed (since p is Gaussian and a sum of gaussians is a chi-squared). Thus, in expectation, the variation in $K$ is of oder $d/2$ where $d$ is the dimension of the target. Since hierarchical structures correlate variables between levels, they also induce large changes in energy density, which our transitions dont explore well.

## Non-centered Model

So what is to be done? We could change the kinetic energy using methods such as Riemannian Monte-Carlo HMC, but thats beyond our scope. But, just as in the case with the regression example earlier, a re-parametrization comes to our rescue. We want to reduce the levels in the hierarchy, as shown here:

![](images/girolam3.png)

We change our model to:

$$
\mu \sim \mathcal{N}(0, 5)\\
\tau \sim \text{Half-Cauchy}(0, 5)\\
\nu_{j} \sim \mathcal{N}(0, 1)\\
\theta_{j} = \mu + \tau\nu_j \\
\bar{y_{j}} \sim \mathcal{N}(\theta_{j}, \sigma_{j})
$$

Notice how we have factored the dependency of $\theta$ on $\phi = \mu, \tau$ into a deterministic
transformation between the layers, leaving the
actively sampled variables uncorrelated. 

This does two things for us: it reduces steepness and curvature, making for better stepping. It also reduces the strong change in densities, and makes sampling from the transition distribution easier.



```python
with pm.Model() as schools2:
    mu = pm.Normal('mu', mu=0, sd=5)
    tau = pm.HalfCauchy('tau', beta=5)
    nu = pm.Normal('nu', mu=0, sd=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * nu)
    obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)
```




```python
with schools2:
    trace2 = pm.sample(5000, init=None, njobs=2, tune=500)
```


    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [nu, tau, mu]
    Sampling 2 chains: 100%|██████████| 11000/11000 [00:09<00:00, 1102.28draws/s]
    There was 1 divergence after tuning. Increase `target_accept` or reparameterize.
    There were 3 divergences after tuning. Increase `target_accept` or reparameterize.




```python
pm.summary(trace2)
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
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>4.347176</td>
      <td>3.286204</td>
      <td>0.033158</td>
      <td>-2.076250</td>
      <td>10.809128</td>
      <td>9280.625148</td>
      <td>0.999958</td>
    </tr>
    <tr>
      <th>nu__0</th>
      <td>0.333415</td>
      <td>0.998524</td>
      <td>0.008853</td>
      <td>-1.542916</td>
      <td>2.345821</td>
      <td>10770.381759</td>
      <td>0.999930</td>
    </tr>
    <tr>
      <th>nu__1</th>
      <td>0.095830</td>
      <td>0.925314</td>
      <td>0.008496</td>
      <td>-1.748745</td>
      <td>1.913819</td>
      <td>12623.663126</td>
      <td>0.999962</td>
    </tr>
    <tr>
      <th>nu__2</th>
      <td>-0.090668</td>
      <td>0.973473</td>
      <td>0.010201</td>
      <td>-2.003096</td>
      <td>1.811096</td>
      <td>11384.279142</td>
      <td>0.999909</td>
    </tr>
    <tr>
      <th>nu__3</th>
      <td>0.071129</td>
      <td>0.940188</td>
      <td>0.007578</td>
      <td>-1.732506</td>
      <td>1.882510</td>
      <td>12046.790749</td>
      <td>0.999905</td>
    </tr>
    <tr>
      <th>nu__4</th>
      <td>-0.153937</td>
      <td>0.951805</td>
      <td>0.008784</td>
      <td>-2.014699</td>
      <td>1.727808</td>
      <td>12065.548667</td>
      <td>0.999984</td>
    </tr>
    <tr>
      <th>nu__5</th>
      <td>-0.070166</td>
      <td>0.944777</td>
      <td>0.009325</td>
      <td>-1.890132</td>
      <td>1.813137</td>
      <td>10954.542775</td>
      <td>0.999954</td>
    </tr>
    <tr>
      <th>nu__6</th>
      <td>0.358106</td>
      <td>0.981190</td>
      <td>0.011142</td>
      <td>-1.515111</td>
      <td>2.341151</td>
      <td>9123.178041</td>
      <td>1.000136</td>
    </tr>
    <tr>
      <th>nu__7</th>
      <td>0.065285</td>
      <td>0.970545</td>
      <td>0.008104</td>
      <td>-1.823876</td>
      <td>1.990456</td>
      <td>13758.237973</td>
      <td>0.999901</td>
    </tr>
    <tr>
      <th>tau</th>
      <td>3.552034</td>
      <td>3.185068</td>
      <td>0.036809</td>
      <td>0.002162</td>
      <td>9.845447</td>
      <td>7236.920715</td>
      <td>0.999933</td>
    </tr>
    <tr>
      <th>theta__0</th>
      <td>6.188267</td>
      <td>5.563705</td>
      <td>0.054012</td>
      <td>-3.875958</td>
      <td>18.548927</td>
      <td>9263.543222</td>
      <td>1.000048</td>
    </tr>
    <tr>
      <th>theta__1</th>
      <td>4.863323</td>
      <td>4.650979</td>
      <td>0.043388</td>
      <td>-4.070784</td>
      <td>14.487559</td>
      <td>10923.364832</td>
      <td>0.999942</td>
    </tr>
    <tr>
      <th>theta__2</th>
      <td>3.873658</td>
      <td>5.244453</td>
      <td>0.060124</td>
      <td>-6.738831</td>
      <td>14.406468</td>
      <td>8794.224031</td>
      <td>1.000048</td>
    </tr>
    <tr>
      <th>theta__3</th>
      <td>4.721012</td>
      <td>4.754331</td>
      <td>0.043573</td>
      <td>-4.633450</td>
      <td>14.515977</td>
      <td>10975.024446</td>
      <td>1.000119</td>
    </tr>
    <tr>
      <th>theta__4</th>
      <td>3.580124</td>
      <td>4.644715</td>
      <td>0.046571</td>
      <td>-6.319482</td>
      <td>12.434017</td>
      <td>10236.673808</td>
      <td>0.999948</td>
    </tr>
    <tr>
      <th>theta__5</th>
      <td>4.015354</td>
      <td>4.769508</td>
      <td>0.047606</td>
      <td>-6.101058</td>
      <td>13.014267</td>
      <td>10498.887075</td>
      <td>0.999901</td>
    </tr>
    <tr>
      <th>theta__6</th>
      <td>6.215937</td>
      <td>5.086602</td>
      <td>0.052506</td>
      <td>-3.160603</td>
      <td>17.322192</td>
      <td>9862.673399</td>
      <td>1.000478</td>
    </tr>
    <tr>
      <th>theta__7</th>
      <td>4.755671</td>
      <td>5.289659</td>
      <td>0.053902</td>
      <td>-6.135223</td>
      <td>15.392630</td>
      <td>10221.686590</td>
      <td>1.000014</td>
    </tr>
  </tbody>
</table>
</div>





```python
pm.traceplot(trace2);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3449: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](gelmanschools_files/gelmanschools_34_1.png)




```python
pm.traceplot(trace2, varnames=['tau_log__'])
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3449: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x11b0064e0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11afb0400>]], dtype=object)




![png](gelmanschools_files/gelmanschools_35_2.png)


Ok, so this seems to look better!



```python
plt.plot(trace2['tau_log__'], alpha=0.6)
plt.axvline(5000, color="r")
```





    <matplotlib.lines.Line2D at 0x11af00dd8>




![png](gelmanschools_files/gelmanschools_37_1.png)


And the effective number of iterations hs improved as well:



```python
pm.diagnostics.gelman_rubin(trace2), pm.diagnostics.effective_n(trace2)
```





    ({'mu': 0.99995806128958298,
      'nu': array([ 0.99993023,  0.99996187,  0.99990919,  0.99990459,  0.99998409,
              0.99995408,  1.00013553,  0.99990075]),
      'tau': 0.99993259236404675,
      'theta': array([ 1.00004779,  0.99994164,  1.00004808,  1.00011865,  0.99994786,
              0.9999015 ,  1.00047802,  1.00001387])},
     {'mu': 9280.6251483004617,
      'nu': array([ 10770.38175948,  12623.66312606,  11384.27914234,  12046.79074917,
              12065.54866653,  10954.54277482,   9123.17804143,  13758.23797319]),
      'tau': 7236.9207149620033,
      'theta': array([  9263.54322197,  10923.36483198,   8794.22403058,  10975.02444636,
              10236.67380755,  10498.88707531,   9862.67339945,  10221.6865905 ])})



And we reach the true value better as the number of samples increases, decreasing our bias



```python
# plot the estimate for the mean of log(τ) cumulating mean
logtau = trace2['tau_log__']
mlogtau = [np.mean(logtau[:i]) for i in np.arange(1, len(logtau))]
plt.figure(figsize=(15, 4))
plt.axhline(0.7657852, lw=2.5, color='gray')
plt.plot(mlogtau, lw=2.5)
plt.ylim(0, 2)
plt.xlabel('Iteration')
plt.ylabel('MCMC mean of log(tau)')
plt.title('MCMC estimation of cumsum log(tau)')
```





    Text(0.5, 1.0, 'MCMC estimation of cumsum log(tau)')




![png](gelmanschools_files/gelmanschools_41_1.png)


How about our divergences? They have decreased too.



```python
divergent = trace2['diverging']
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size/len(trace2)
print('Percentage of Divergent %.5f' % divperc)
```


    Number of Divergent 4
    Percentage of Divergent 0.00080




```python
theta_trace = trace2['theta']
theta0 = theta_trace[:, 0]
plt.figure(figsize=(10, 6))
plt.scatter(theta0[divergent == 0], logtau[divergent == 0], color='r', s=10, alpha=0.05)
plt.scatter(theta0[divergent == 1], logtau[divergent == 1], color='g', s=20, alpha=0.9)
plt.axis([-20, 50, -6, 4])
plt.ylabel('log(tau)')
plt.xlabel('theta[0]')
plt.title('scatter plot between log(tau) and theta[0]')
plt.show()
```



![png](gelmanschools_files/gelmanschools_44_0.png)


Look how much longer the funnel actually is. And we have explored this much better.



```python
theta01 = trace1['theta'][:, 0]
logtau1 = trace1['tau_log__']

theta02 = trace2['theta'][:, 0]
logtau2 = trace2['tau_log__']


plt.figure(figsize=(10, 6))
plt.scatter(theta01, logtau1, alpha=.05, color="b", label="original", s=10)
plt.scatter(theta02, logtau2, alpha=.05, color="r", label='reparametrized', s=10)
plt.axis([-20, 50, -6, 4])
plt.ylabel('log(tau)')
plt.xlabel('theta[0]')
plt.title('scatter plot between log(tau) and theta[0]')
plt.legend()

```





    <matplotlib.legend.Legend at 0x116002860>




![png](gelmanschools_files/gelmanschools_46_1.png)


It may not be possible in all models to achieve this sort of decoupling. In that case, Riemannian HMC, where we generalize the mass matrix to depend upon position, explicitly tackling high-curvature, can help.
