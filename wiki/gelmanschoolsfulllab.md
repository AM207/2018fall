---
title: Gelman Schools, end to end
shorttitle: gelmanschoolsfulllab
notebook: gelmanschoolsfulllab.ipynb
noline: 1
summary: ""
keywords: ['energy', 'hamiltonian monte carlo', 'nuts', 'leapfrog', 'canonical distribution', 'microcanonical distribution', 'transition distribution', 'marginal energy distribution', 'data augmentation', 'classical mechanics', 'detailed balance', 'statistical mechanics', 'divergences', 'step-size', 'non-centered hierarchical model', 'hierarchical']
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




```python
import arviz as az
```


## Centered parametrization

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
    trace1 = pm.sample(10000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta, tau, mu]
    Sampling 2 chains: 100%|██████████| 21000/21000 [00:38<00:00, 543.35draws/s]
    There were 192 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 152 divergences after tuning. Increase `target_accept` or reparameterize.
    The number of effective samples is smaller than 10% for some parameters.




```python
#TODO: ARVIZ, E-BFMI
```




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
      <td>4.421838</td>
      <td>3.335100</td>
      <td>0.076147</td>
      <td>-2.521747</td>
      <td>10.889159</td>
      <td>1666.484372</td>
      <td>0.999995</td>
    </tr>
    <tr>
      <th>theta__0</th>
      <td>6.607782</td>
      <td>5.966534</td>
      <td>0.098909</td>
      <td>-4.777644</td>
      <td>18.768973</td>
      <td>3882.800476</td>
      <td>0.999965</td>
    </tr>
    <tr>
      <th>theta__1</th>
      <td>5.116251</td>
      <td>4.933157</td>
      <td>0.080553</td>
      <td>-4.688147</td>
      <td>15.092325</td>
      <td>3583.157313</td>
      <td>1.000047</td>
    </tr>
    <tr>
      <th>theta__2</th>
      <td>3.829182</td>
      <td>5.527924</td>
      <td>0.095912</td>
      <td>-7.645811</td>
      <td>14.453706</td>
      <td>3228.973893</td>
      <td>0.999984</td>
    </tr>
    <tr>
      <th>theta__3</th>
      <td>4.825545</td>
      <td>4.929238</td>
      <td>0.078297</td>
      <td>-4.987601</td>
      <td>14.733869</td>
      <td>4188.067465</td>
      <td>0.999956</td>
    </tr>
    <tr>
      <th>theta__4</th>
      <td>3.478191</td>
      <td>4.851977</td>
      <td>0.092242</td>
      <td>-6.182416</td>
      <td>13.371734</td>
      <td>2511.345157</td>
      <td>1.000152</td>
    </tr>
    <tr>
      <th>theta__5</th>
      <td>4.023393</td>
      <td>4.996055</td>
      <td>0.082124</td>
      <td>-6.402197</td>
      <td>13.759797</td>
      <td>3842.954606</td>
      <td>1.000052</td>
    </tr>
    <tr>
      <th>theta__6</th>
      <td>6.660372</td>
      <td>5.290873</td>
      <td>0.092386</td>
      <td>-2.831968</td>
      <td>18.095978</td>
      <td>3666.135589</td>
      <td>0.999959</td>
    </tr>
    <tr>
      <th>theta__7</th>
      <td>4.880685</td>
      <td>5.525661</td>
      <td>0.079835</td>
      <td>-5.816082</td>
      <td>16.624295</td>
      <td>5084.452672</td>
      <td>0.999955</td>
    </tr>
    <tr>
      <th>tau</th>
      <td>4.138336</td>
      <td>3.127551</td>
      <td>0.085624</td>
      <td>0.639250</td>
      <td>10.236035</td>
      <td>1256.355294</td>
      <td>0.999959</td>
    </tr>
  </tbody>
</table>
</div>





```python
pm.trace_to_dataframe(trace1).corr()
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
      <th>mu</th>
      <th>theta__0</th>
      <th>theta__1</th>
      <th>theta__2</th>
      <th>theta__3</th>
      <th>theta__4</th>
      <th>theta__5</th>
      <th>theta__6</th>
      <th>theta__7</th>
      <th>tau</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>1.000000</td>
      <td>0.446534</td>
      <td>0.542310</td>
      <td>0.563762</td>
      <td>0.547569</td>
      <td>0.580491</td>
      <td>0.565741</td>
      <td>0.469521</td>
      <td>0.544208</td>
      <td>-0.116377</td>
    </tr>
    <tr>
      <th>theta__0</th>
      <td>0.446534</td>
      <td>1.000000</td>
      <td>0.336050</td>
      <td>0.192429</td>
      <td>0.309346</td>
      <td>0.211380</td>
      <td>0.225359</td>
      <td>0.427040</td>
      <td>0.273259</td>
      <td>0.401492</td>
    </tr>
    <tr>
      <th>theta__1</th>
      <td>0.542310</td>
      <td>0.336050</td>
      <td>1.000000</td>
      <td>0.299167</td>
      <td>0.308855</td>
      <td>0.289636</td>
      <td>0.289145</td>
      <td>0.346080</td>
      <td>0.321100</td>
      <td>0.109375</td>
    </tr>
    <tr>
      <th>theta__2</th>
      <td>0.563762</td>
      <td>0.192429</td>
      <td>0.299167</td>
      <td>1.000000</td>
      <td>0.300424</td>
      <td>0.354484</td>
      <td>0.336760</td>
      <td>0.209144</td>
      <td>0.314495</td>
      <td>-0.192125</td>
    </tr>
    <tr>
      <th>theta__3</th>
      <td>0.547569</td>
      <td>0.309346</td>
      <td>0.308855</td>
      <td>0.300424</td>
      <td>1.000000</td>
      <td>0.309420</td>
      <td>0.330811</td>
      <td>0.308911</td>
      <td>0.316515</td>
      <td>0.028588</td>
    </tr>
    <tr>
      <th>theta__4</th>
      <td>0.580491</td>
      <td>0.211380</td>
      <td>0.289636</td>
      <td>0.354484</td>
      <td>0.309420</td>
      <td>1.000000</td>
      <td>0.351088</td>
      <td>0.200968</td>
      <td>0.312110</td>
      <td>-0.235397</td>
    </tr>
    <tr>
      <th>theta__5</th>
      <td>0.565741</td>
      <td>0.225359</td>
      <td>0.289145</td>
      <td>0.336760</td>
      <td>0.330811</td>
      <td>0.351088</td>
      <td>1.000000</td>
      <td>0.234143</td>
      <td>0.292970</td>
      <td>-0.143010</td>
    </tr>
    <tr>
      <th>theta__6</th>
      <td>0.469521</td>
      <td>0.427040</td>
      <td>0.346080</td>
      <td>0.209144</td>
      <td>0.308911</td>
      <td>0.200968</td>
      <td>0.234143</td>
      <td>1.000000</td>
      <td>0.278426</td>
      <td>0.390982</td>
    </tr>
    <tr>
      <th>theta__7</th>
      <td>0.544208</td>
      <td>0.273259</td>
      <td>0.321100</td>
      <td>0.314495</td>
      <td>0.316515</td>
      <td>0.312110</td>
      <td>0.292970</td>
      <td>0.278426</td>
      <td>1.000000</td>
      <td>0.022809</td>
    </tr>
    <tr>
      <th>tau</th>
      <td>-0.116377</td>
      <td>0.401492</td>
      <td>0.109375</td>
      <td>-0.192125</td>
      <td>0.028588</td>
      <td>-0.235397</td>
      <td>-0.143010</td>
      <td>0.390982</td>
      <td>0.022809</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The Gelman-Rubin statistic seems fine, but notice how small the effective-n's are? Something is not quite right. Lets see traceplots. Also notice the strong correlations between some $\theta$s and $\tau$ and some $\theta$s and $\mu$.



```python
pm.traceplot(trace1);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3449: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_13_1.png)


Its hard to pick the thetas out but $\tau$ looks not so white-noisy. Lets zoom in:



```python
trace1.varnames
```





    ['mu', 'tau_log__', 'theta', 'tau']





```python
pm.traceplot(trace1, varnames=['tau_log__'])
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3449: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x12138a6a0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1213accc0>]], dtype=object)




![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_16_2.png)


There seems to be some stickiness at lower values in the trace. Zooming in even more helps us see this better:



```python
plt.plot(trace1['tau_log__'], alpha=0.6)
plt.axvline(10000, color="r")
#plt.plot(short_trace['tau_log_'][5000:], alpha=0.6);
```





    <matplotlib.lines.Line2D at 0x12138a5f8>




![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_18_1.png)


### Tracking divergences



```python
divergent = trace1['diverging']
divergent
```





    array([False, False, False, ..., False, False, False], dtype=bool)





```python
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size/len(trace1)
print('Percentage of Divergent %.5f' % divperc)
```


    Number of Divergent 344
    Percentage of Divergent 0.03440




```python
def biasplot(trace):
    logtau = trace['tau_log__']
    mlogtau = [np.mean(logtau[:i]) for i in np.arange(1, len(logtau))]
    plt.figure(figsize=(8, 2))
    plt.axhline(0.7657852, lw=2.5, color='gray')
    plt.plot(mlogtau, lw=2.5)
    plt.ylim(0, 2)
    plt.xlabel('Iteration')
    plt.ylabel('MCMC mean of log(tau)')
    plt.title('MCMC estimation of cumsum log(tau)')
```




```python
biasplot(trace1)
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_23_0.png)




```python
def funnelplot(trace):
    logtau = trace['tau_log__']
    divergent = trace['diverging']
    theta_trace = trace['theta']
    theta0 = theta_trace[:, 0]
    plt.figure(figsize=(5, 3))
    plt.scatter(theta0[divergent == 0], logtau[divergent == 0], s=10, color='r', alpha=0.1)
    plt.scatter(theta0[divergent == 1], logtau[divergent == 1], s=10, color='g')
    plt.axis([-20, 50, -6, 4])
    plt.ylabel('log(tau)')
    plt.xlabel('theta[0]')
    plt.title('scatter plot between log(tau) and theta[0]')
    plt.show()
```




```python
funnelplot(trace1)
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_25_0.png)


You can also get an idea of your acceptance rate. 65% is decent for NUTS.



```python
np.mean(trace1['mean_tree_accept'])
```





    0.74708717366214583





```python
azdata1 = az.from_pymc3(
    trace=trace1)
```




```python
az.bfmi(azdata1.sample_stats.energy)
```





    array([ 0.27481475,  0.28938651])





```python
az.plot_autocorr(azdata1);
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_30_0.png)


### Where are the divergences coming from?

Divergences can be a sign of the symplectic integration going off to infinity, or a false positive. False positives occur because instead of waiting for infinity, some heuristics are used. This is typically true of divergences not deep in the funnel, where the curvature of the target distribution is high.





## The effect of step-size

Looking at the docs for the `NUTS` sampler at https://pymc-devs.github.io/pymc3/api/inference.html#module-pymc3.step_methods.hmc.nuts , we see that we can co-erce a smaller step-size $\epsilon$, and thus an ok symplectic integration from our sampler by increasing the target acceptance rate.

If we do this, then we have geometric ergodicity (we go everywhere!) between the Hamiltonian transitions (ie in the leapfrogs) and the target distribution. This should result in the divergence rate decreasing.

But if for some reason we do not have geometric ergodicity, then divergences will persist. This can happen deep in the funnel, where even drastic decreases in the step size are not able to explore the highly curved geometry.




```python
with schools1:
    step = pm.NUTS(target_accept=.85)
    trace1_85 = pm.sample(10000, step=step)
with schools1:
    step = pm.NUTS(target_accept=.90)
    trace1_90 = pm.sample(10000, step=step)
with schools1:
    step = pm.NUTS(target_accept=.95)
    trace1_95 = pm.sample(10000, step=step)
with schools1:
    step = pm.NUTS(target_accept=.99)
    trace1_99 = pm.sample(10000, step=step)
```


    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta, tau, mu]
    Sampling 2 chains: 100%|██████████| 21000/21000 [00:32<00:00, 640.18draws/s] 
    There were 228 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 602 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.593123095327, but should be close to 0.85. Try to increase the number of tuning steps.
    The estimated number of effective samples is smaller than 200 for some parameters.
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta, tau, mu]
    Sampling 2 chains: 100%|██████████| 21000/21000 [00:35<00:00, 598.26draws/s]
    There were 838 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.70706900873, but should be close to 0.9. Try to increase the number of tuning steps.
    There were 284 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.658746391353, but should be close to 0.9. Try to increase the number of tuning steps.
    The number of effective samples is smaller than 10% for some parameters.
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta, tau, mu]
    Sampling 2 chains: 100%|██████████| 21000/21000 [01:22<00:00, 253.67draws/s]
    There were 206 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.872542834364, but should be close to 0.95. Try to increase the number of tuning steps.
    There were 190 divergences after tuning. Increase `target_accept` or reparameterize.
    The number of effective samples is smaller than 10% for some parameters.
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta, tau, mu]
    Sampling 2 chains: 100%|██████████| 21000/21000 [03:31<00:00, 99.48draws/s] 
    There were 24 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 89 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.9516802872, but should be close to 0.99. Try to increase the number of tuning steps.
    The number of effective samples is smaller than 10% for some parameters.




```python
for t in [trace1_85, trace1_90, trace1_95, trace1_99]:
    print("Acceptance", np.mean(t['mean_tree_accept']), "Step Size", np.mean(t['step_size']), "Divergence", np.sum(t['diverging']))
```


    Acceptance 0.692500227779 Step Size 0.247497861579 Divergence 830
    Acceptance 0.682924981865 Step Size 0.172543729336 Divergence 1123
    Acceptance 0.893198124991 Step Size 0.101784236938 Divergence 396
    Acceptance 0.964135746997 Step Size 0.0653667162649 Divergence 113




```python
for t in [trace1_85, trace1_90, trace1_95, trace1_99]:
    biasplot(t)
    funnelplot(t)
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_35_0.png)



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_35_1.png)



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_35_2.png)



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_35_3.png)



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_35_4.png)



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_35_5.png)



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_35_6.png)



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_35_7.png)




```python
plt.plot(trace1_99['tau_log__'])
```





    [<matplotlib.lines.Line2D at 0x121b72278>]




![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_36_1.png)




```python
df99 = pm.trace_to_dataframe(trace1_99)
df99.head()
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
      <th>mu</th>
      <th>theta__0</th>
      <th>theta__1</th>
      <th>theta__2</th>
      <th>theta__3</th>
      <th>theta__4</th>
      <th>theta__5</th>
      <th>theta__6</th>
      <th>theta__7</th>
      <th>tau</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.693776</td>
      <td>14.477088</td>
      <td>10.353145</td>
      <td>18.408850</td>
      <td>0.004432</td>
      <td>8.185481</td>
      <td>13.967940</td>
      <td>8.531192</td>
      <td>9.507294</td>
      <td>6.308449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.407051</td>
      <td>0.709888</td>
      <td>16.086012</td>
      <td>2.926935</td>
      <td>10.425556</td>
      <td>7.994480</td>
      <td>3.554593</td>
      <td>18.974226</td>
      <td>13.514982</td>
      <td>8.005093</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.084759</td>
      <td>8.556587</td>
      <td>1.263262</td>
      <td>9.449619</td>
      <td>4.694946</td>
      <td>5.025554</td>
      <td>6.031463</td>
      <td>1.203137</td>
      <td>2.035244</td>
      <td>2.936942</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.586736</td>
      <td>8.860204</td>
      <td>6.887298</td>
      <td>5.738129</td>
      <td>5.422712</td>
      <td>4.859962</td>
      <td>5.804170</td>
      <td>7.349872</td>
      <td>6.061337</td>
      <td>2.313194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.967213</td>
      <td>7.087187</td>
      <td>7.038952</td>
      <td>5.864002</td>
      <td>7.126083</td>
      <td>4.261743</td>
      <td>6.588078</td>
      <td>9.380298</td>
      <td>4.836616</td>
      <td>1.379508</td>
    </tr>
  </tbody>
</table>
</div>





```python
df99.corr()
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
      <th>mu</th>
      <th>theta__0</th>
      <th>theta__1</th>
      <th>theta__2</th>
      <th>theta__3</th>
      <th>theta__4</th>
      <th>theta__5</th>
      <th>theta__6</th>
      <th>theta__7</th>
      <th>tau</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>1.000000</td>
      <td>0.499408</td>
      <td>0.578907</td>
      <td>0.598298</td>
      <td>0.584340</td>
      <td>0.586283</td>
      <td>0.599065</td>
      <td>0.503385</td>
      <td>0.573302</td>
      <td>-0.080196</td>
    </tr>
    <tr>
      <th>theta__0</th>
      <td>0.499408</td>
      <td>1.000000</td>
      <td>0.373228</td>
      <td>0.239095</td>
      <td>0.334952</td>
      <td>0.214745</td>
      <td>0.272826</td>
      <td>0.461365</td>
      <td>0.341573</td>
      <td>0.398513</td>
    </tr>
    <tr>
      <th>theta__1</th>
      <td>0.578907</td>
      <td>0.373228</td>
      <td>1.000000</td>
      <td>0.336129</td>
      <td>0.374738</td>
      <td>0.342857</td>
      <td>0.356699</td>
      <td>0.374444</td>
      <td>0.371255</td>
      <td>0.101561</td>
    </tr>
    <tr>
      <th>theta__2</th>
      <td>0.598298</td>
      <td>0.239095</td>
      <td>0.336129</td>
      <td>1.000000</td>
      <td>0.342234</td>
      <td>0.379636</td>
      <td>0.375234</td>
      <td>0.258921</td>
      <td>0.329257</td>
      <td>-0.165129</td>
    </tr>
    <tr>
      <th>theta__3</th>
      <td>0.584340</td>
      <td>0.334952</td>
      <td>0.374738</td>
      <td>0.342234</td>
      <td>1.000000</td>
      <td>0.348511</td>
      <td>0.376360</td>
      <td>0.354740</td>
      <td>0.365683</td>
      <td>0.054737</td>
    </tr>
    <tr>
      <th>theta__4</th>
      <td>0.586283</td>
      <td>0.214745</td>
      <td>0.342857</td>
      <td>0.379636</td>
      <td>0.348511</td>
      <td>1.000000</td>
      <td>0.370492</td>
      <td>0.222877</td>
      <td>0.330311</td>
      <td>-0.226055</td>
    </tr>
    <tr>
      <th>theta__5</th>
      <td>0.599065</td>
      <td>0.272826</td>
      <td>0.356699</td>
      <td>0.375234</td>
      <td>0.376360</td>
      <td>0.370492</td>
      <td>1.000000</td>
      <td>0.286706</td>
      <td>0.349603</td>
      <td>-0.122573</td>
    </tr>
    <tr>
      <th>theta__6</th>
      <td>0.503385</td>
      <td>0.461365</td>
      <td>0.374444</td>
      <td>0.258921</td>
      <td>0.354740</td>
      <td>0.222877</td>
      <td>0.286706</td>
      <td>1.000000</td>
      <td>0.351618</td>
      <td>0.405648</td>
    </tr>
    <tr>
      <th>theta__7</th>
      <td>0.573302</td>
      <td>0.341573</td>
      <td>0.371255</td>
      <td>0.329257</td>
      <td>0.365683</td>
      <td>0.330311</td>
      <td>0.349603</td>
      <td>0.351618</td>
      <td>1.000000</td>
      <td>0.093862</td>
    </tr>
    <tr>
      <th>tau</th>
      <td>-0.080196</td>
      <td>0.398513</td>
      <td>0.101561</td>
      <td>-0.165129</td>
      <td>0.054737</td>
      <td>-0.226055</td>
      <td>-0.122573</td>
      <td>0.405648</td>
      <td>0.093862</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>





```python
from itertools import cycle, islice
from pandas.plotting import parallel_coordinates as pc
def pcplot(t):
    dvs = sum(t['diverging']==True)
    dft = pm.trace_to_dataframe(t)
    dftsi = dft.reset_index()
    plt.figure(figsize=(16, 10))
    orderedticks = ['index', 'mu', 'tau']+["theta__{}".format(i) for i in range(8)]
    div_colors = list(islice(cycle(['g', 'g', 'g', 'g', 'g']), None, int(dvs)))
    pc(dftsi[t['diverging']==True][orderedticks], 'index', color=div_colors, alpha=0.2);
    undiv_colors = list(islice(cycle(['k', 'k', 'k', 'k', 'k']), None, 200))
    pc(dftsi[t['diverging']==False].sample(200)[orderedticks], 'index', color=undiv_colors, alpha=0.04);
    plt.gca().legend_.remove()
pcplot(trace1_99)
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_39_0.png)




```python
az.plot_parallel(az.from_pymc3(trace=trace1_99))
```





    <matplotlib.axes._subplots.AxesSubplot at 0x123be30b8>




![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_40_1.png)


The divergences decrease, but dont totally go away, showing that we have lost some geometric ergodicity. And as we get to a very small step size we explore the funnel much better, but we are now taking our sampler more into a MH like random walk regime, and our sampler looks very strongly autocorrelated.

We know the fix, it is to move to a

## Non-centered paramerization

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
    trace2 = pm.sample(10000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [nu, tau, mu]
    Sampling 2 chains: 100%|██████████| 21000/21000 [00:29<00:00, 723.95draws/s]
    There were 12 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 13 divergences after tuning. Increase `target_accept` or reparameterize.


And we reach the true value better as the number of samples increases, decreasing our bias



```python
biasplot(trace2)
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_46_0.png)


How about our divergences? They have decreased too.



```python
divergent = trace2['diverging']
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size/len(trace2)
print('Percentage of Divergent %.5f' % divperc)
```


    Number of Divergent 25
    Percentage of Divergent 0.00250




```python
funnelplot(trace2)
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_49_0.png)




```python
pcplot(trace2)
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_50_0.png)


The divergences are infrequent and do not seem to concentrate anywhere, indicative of false positives. Lowering the step size should make them go away.

### A smaller step size



```python
with schools2:
    step = pm.NUTS(target_accept=.95)
    trace2_95 = pm.sample(10000, step=step, init="jitter+adapt_diag")
```


    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [nu, tau, mu]
    Sampling 2 chains: 100%|██████████| 21000/21000 [00:58<00:00, 357.60draws/s]




```python
biasplot(trace2_95)
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_54_0.png)




```python
funnelplot(trace2_95)
```



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_55_0.png)


Indeed at a smaller step-size our false-positive divergences go away, and the lower curvature in our parametrization ensures geometric ergodicity deep in our funnel



```python
prior = pm.sample_prior_predictive(model=schools2)
posterior_predictive = pm.sample_ppc(trace2_95, 500, schools2)
```


    100%|██████████| 500/500 [00:00<00:00, 1152.89it/s]




```python
azdatagood = az.from_pymc3(
    trace = trace2_95,
    prior=prior,
    posterior_predictive = posterior_predictive )
```




```python
az.bfmi(azdatagood.sample_stats.energy)
```





    array([ 0.96607181,  0.94140452])



## Path length L

If we choose too small a $L$ we are returning our HMC sampler to a random walk. How long must a leapfrog run explore a level set of the Hamiltonian (ie of the canonical distribution $p(p,q)$ beofre we force an accept-reject step and a momentum resample?

Clearly if we go too long we'll be coming back to the neighborhood of areas we might have reached in smaller trajectories. NUTS is one approach to adaptively fix this by not letting trajectories turn on themselves.

In the regular HMC sampler, for slightly complex problems, $L=100$ maybe a good place to start. For a fixed step-size $\epsilon$, we can now check the level of autocorrelation. If it is too much, we want a larger $L$.

Now, the problem with a fixed $L$ is that one $L$ does not work everywhere in a distribution. To see this, note that tails correspond to much higher energies. Thus the level-set surfaces are larger, and a fixed length $L$ trajectory only explores a small portion of this set before a momentum resampling takes us off. This is why a dynamic method like NUTS is a better choice.

## Tuning HMC(NUTS)

This requires preliminary runs. In `pymc3` some samples are dedicated to this, and an adaptive tuning is carried out according to algorithm 6 in the original NUTS paper: http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf .

But we have seem how to play with step-size within the context of the NUTS sampler, something which we might need to do for tetchy models. Clearly too large an $\epsilon$ can lead to loss of symplecticity. And too small a step-size will get us to a random walk for finite sized $L$. Clearly the adaptive approach in NUTS is a good idea.

In pymc3's regular HMC sampler a knob `step_rand` is available which allows us a distribution to sample a step-size from. In the case that we are not adaptively tuning as in NUTS, allowing for this randomness allows for occasionally small values of $\epsilon$ even for large meaned distributions. Note that this should be done at the star of leapfrog, not in-between.

Another place where this is useful is where the exact solution to the Hamiltonian equations (gaussian distribution, harmonic oscillator hamiltonian resulting) has periodicity. If $L\epsilon$ is chosen to be $2\pi$ then our sampler will lack ergodicity. In such a case choose $\epsilon$ from a distribution (or for that matter, $L$).

Finally there are multiple ways to tweak the mass matrix. One might use a variational posterior to obtain a approximate covariance matrix for the target distribution $p(q)$. Or one could use the tuning samples for this purpose. But choosing the mass matrix as the inverse of the covariance matrix of the target is highly recommended, as it will maximally decorrelate parameters of the target distribution.

The covariance matrix also establishes a scale for each parameter. This scale can be used to tweak step size as well. Intuitively the variances are measures of curvature along a particular dimension, and choosing a stepsize in each parameter which accomodates this difference is likely to help symplecticity. I do not believe this optimization is available within pymc3. This does not mean you are out of luck: you could simply redefine the parameters in a scaled form.

If you are combining HMC with other samplers, such as MH for discrete parameters in a gibbs based conditional scheme, then you might prefer smaller $L$ parameters to allow for the other parameters to be updated faster.


## Efficiency of momentum resampling

When we talked about the most Hamiltonian-trajectory momentum resampling, we talked about its efficiency. The point there was that you want the marginal energy distribution to match the transition distribution induced by momentum resampling.

`pymc3` gives us some handy stats to calculate this:



```python
def resample_plot(t):
    sns.distplot(t['energy']-t['energy'].mean(), label="P(E)")
    sns.distplot(np.diff(t['energy']), label = "p(E | q)")
    plt.legend();
    plt.xlabel("E - <E>")
    
```


So let us see this for our original trace



```python
resample_plot(trace1);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6499: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_65_1.png)


Awful. The momentum resamples here will do a very slow job of traversing this distribution. This is indicative of the second issue we were having with this centered model (the first was a large step size for the curvature causing loss of symplectivity): the momentum resampling simply cannot provide enough energy to traverse the large energy changes that occur in this hierarchical model.

Note the caveat with such a plot obtained from our chains: it only tells us about the energies it explored: not the energies it ought to be exploring, as can be seen in the plot with `trace1_99` below. Still, a great diagnostic.



```python
resample_plot(trace1_99)
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6499: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_68_1.png)


The match is much better for the non-centered version of our model.



```python
resample_plot(trace2)
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6499: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_70_1.png)




```python
resample_plot(trace2_95)
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6499: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_71_1.png)




```python
az.plot_energy(azdatagood)
```





    <matplotlib.axes._subplots.AxesSubplot at 0x132673438>




![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_72_1.png)




```python
az.plot_forest(azdatagood, var_names=['theta'], kind='ridgeplot', combined=False, ridgeplot_overlap=3)
```





    (<Figure size 720x288 with 3 Axes>,
     array([<matplotlib.axes._subplots.AxesSubplot object at 0x13377add8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x133a29908>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x133a51dd8>], dtype=object))




![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_73_1.png)




```python
az.plot_ppc(azdatagood, alpha=0.1)
```





    array([<matplotlib.axes._subplots.AxesSubplot object at 0x134176898>], dtype=object)




![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_74_1.png)




```python
az.plot_ppc(azdatagood, kind='cumulative', alpha=0.1)
```





    array([<matplotlib.axes._subplots.AxesSubplot object at 0x12265bef0>], dtype=object)




![png](gelmanschoolsfulllab_files/gelmanschoolsfulllab_75_1.png)

