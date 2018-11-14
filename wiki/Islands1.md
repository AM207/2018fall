---
title: Poisson Regression -  tools on islands, part 1
shorttitle: Islands1
notebook: Islands1.ipynb
noline: 1
summary: ""
keywords: ['poisson distribution', 'poisson regression', 'glm', 'centering', 'counterfactual plot', 'regression', 'interaction-term', 'oceanic tools', 'parameter correlation']
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


From Mcelreath:

>The island societies of Oceania provide a natural experiment in technological evolution. Different historical island populations possessed tool kits of different size. These kits include fish hooks, axes, boats, hand plows, and many other types of tools. A number of theories predict that larger populations will both develop and sustain more complex tool kits. So the natural variation in population size induced by natural variation in island size in Oceania provides a natural experiment to test these ideas. It's also suggested that contact rates among populations effectively increase population size, as it's relevant to technological evolution. So variation in contact rates among Oceanic societies is also relevant. (McElreath 313)

![](images/islands.png)

## Setting up the model and data

Some points to take into account:

- sample size is not  umber of rows, after all this is a count model
- the data is small, so we will need regularizing to avoid overfitting
- outcome will be `total_tools` which we will model as proportional to `log(population)` as theory says it depends on order of magnitude
- number of tools incereases with `contact` rate
- we will, over multiple attempts, be testing the idea that the impact of population on tool counts is increased by high `contact`. This is an example of an **interaction**. Specifically this is a **positive** interaction between `log(population)` and `contact`.



```python
df=pd.read_csv("data/islands.csv", sep=';')
df
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
      <th>culture</th>
      <th>population</th>
      <th>contact</th>
      <th>total_tools</th>
      <th>mean_TU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malekula</td>
      <td>1100</td>
      <td>low</td>
      <td>13</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tikopia</td>
      <td>1500</td>
      <td>low</td>
      <td>22</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Santa Cruz</td>
      <td>3600</td>
      <td>low</td>
      <td>24</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Yap</td>
      <td>4791</td>
      <td>high</td>
      <td>43</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lau Fiji</td>
      <td>7400</td>
      <td>high</td>
      <td>33</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Trobriand</td>
      <td>8000</td>
      <td>high</td>
      <td>19</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chuuk</td>
      <td>9200</td>
      <td>high</td>
      <td>40</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Manus</td>
      <td>13000</td>
      <td>low</td>
      <td>28</td>
      <td>6.6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tonga</td>
      <td>17500</td>
      <td>high</td>
      <td>55</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hawaii</td>
      <td>275000</td>
      <td>low</td>
      <td>71</td>
      <td>6.6</td>
    </tr>
  </tbody>
</table>
</div>





```python
df['logpop']=np.log(df.population)
df['clevel']=(df.contact=='high')*1
df
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
      <th>culture</th>
      <th>population</th>
      <th>contact</th>
      <th>total_tools</th>
      <th>mean_TU</th>
      <th>logpop</th>
      <th>clevel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malekula</td>
      <td>1100</td>
      <td>low</td>
      <td>13</td>
      <td>3.2</td>
      <td>7.003065</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tikopia</td>
      <td>1500</td>
      <td>low</td>
      <td>22</td>
      <td>4.7</td>
      <td>7.313220</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Santa Cruz</td>
      <td>3600</td>
      <td>low</td>
      <td>24</td>
      <td>4.0</td>
      <td>8.188689</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Yap</td>
      <td>4791</td>
      <td>high</td>
      <td>43</td>
      <td>5.0</td>
      <td>8.474494</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lau Fiji</td>
      <td>7400</td>
      <td>high</td>
      <td>33</td>
      <td>5.0</td>
      <td>8.909235</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Trobriand</td>
      <td>8000</td>
      <td>high</td>
      <td>19</td>
      <td>4.0</td>
      <td>8.987197</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chuuk</td>
      <td>9200</td>
      <td>high</td>
      <td>40</td>
      <td>3.8</td>
      <td>9.126959</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Manus</td>
      <td>13000</td>
      <td>low</td>
      <td>28</td>
      <td>6.6</td>
      <td>9.472705</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tonga</td>
      <td>17500</td>
      <td>high</td>
      <td>55</td>
      <td>5.4</td>
      <td>9.769956</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hawaii</td>
      <td>275000</td>
      <td>low</td>
      <td>71</td>
      <td>6.6</td>
      <td>12.524526</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Lets write down the model we plan to fit.

### M1

$$
\begin{eqnarray}
T_i & \sim & Poisson(\lambda_i)\\
log(\lambda_i) & = & \alpha + \beta_P log(P_i) + \beta_C C_i + \beta_{PC} C_i log(P_i)\\
\alpha & \sim & N(0,100)\\
\beta_P & \sim & N(0,1)\\
\beta_C & \sim & N(0,1)\\
\beta_{PC} & \sim & N(0,1)
\end{eqnarray}
$$

The $\beta$s have strongly regularizing priors on them, because the sample is small, while the $\alpha$ prior is essentially a flat prior.

## Implementation in pymc



```python
import theano.tensor as t
with pm.Model() as m1:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    betapc = pm.Normal("betapc", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop + betac*df.clevel + betapc*df.clevel*df.logpop
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    
```




```python
with m1:
    trace=pm.sample(10000, njobs=2)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betapc, betac, betap]
    Sampling 2 chains: 100%|██████████| 21000/21000 [01:17<00:00, 269.30draws/s]
    The acceptance probability does not match the target. It is 0.88659413662, but should be close to 0.8. Try to increase the number of tuning steps.




```python
pm.summary(trace)
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
      <th>betap</th>
      <td>0.264372</td>
      <td>0.035345</td>
      <td>0.000443</td>
      <td>0.194924</td>
      <td>0.333177</td>
      <td>6225.835834</td>
      <td>1.000744</td>
    </tr>
    <tr>
      <th>betac</th>
      <td>-0.083529</td>
      <td>0.838570</td>
      <td>0.009645</td>
      <td>-1.700037</td>
      <td>1.597633</td>
      <td>6954.452047</td>
      <td>1.000067</td>
    </tr>
    <tr>
      <th>betapc</th>
      <td>0.042019</td>
      <td>0.091893</td>
      <td>0.001062</td>
      <td>-0.147786</td>
      <td>0.214488</td>
      <td>6987.273344</td>
      <td>1.000076</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>0.934902</td>
      <td>0.366691</td>
      <td>0.004492</td>
      <td>0.208745</td>
      <td>1.645146</td>
      <td>6253.602039</td>
      <td>1.000709</td>
    </tr>
  </tbody>
</table>
</div>





```python
pm.traceplot(trace);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3449: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](Islands1_files/Islands1_13_1.png)




```python
pm.autocorrplot(trace);
```



![png](Islands1_files/Islands1_14_0.png)


Our traces an autocorrelations look pretty good. `pymc3` does quick work on the model

### Posteriors



```python
pm.plot_posterior(trace);
```



![png](Islands1_files/Islands1_16_0.png)


Looking at the posteriors reveals something interesting. The posterior for $\beta_p$ is, as expected from theory, showing a positive effect. The posterior is fairly tightly constrained. The posteriors for $\beta_c$ and $\beta_{pc}$ both overlap 0 substantially, and seem comparatively poorly constrained.

At this point you might be willing to say that there is no substantial effect of contact rate, directly or through the interaction.

You would be wrong.

### Posterior check with counterfactual predictions.

Lets get $\lambda$ traces for high-contact and low contact



```python
lamlow = lambda logpop: trace['alpha']+trace['betap']*logpop
lamhigh = lambda logpop: trace['alpha']+(trace['betap'] + trace['betapc'])*logpop + trace['betac'] 
```


Now let us see what happens at an intermediate log(pop) of 8:



```python
sns.distplot(lamhigh(8) - lamlow(8));
plt.axvline(0);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6499: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](Islands1_files/Islands1_20_1.png)


We can see evidence of a fairly strong positive effect of contact in this "counterfactual posterior", with most of the weight above 0.

So what happened?

### Posterior scatter plots

We make posterior scatter plots and this give us the answer.



```python
def postscat(trace, thevars):
    d={}
    for v in thevars:
        d[v] = trace.get_values(v)
    df = pd.DataFrame.from_dict(d)
    return sns.pairplot(df)
```




```python
postscat(trace,trace.varnames)
```





    <seaborn.axisgrid.PairGrid at 0x116913160>




![png](Islands1_files/Islands1_23_1.png)


Look at the very strong negative correlations between $\alpha$ and $\beta_p$, and the very strong ones between $\beta_c$ and $\beta_{pc}$. The latter is the cause for the 0-overlaps. When $\beta_c$ is high, $\beta_{pc}$ must be low, and vice-versa. As a result, its not enough to observe just the marginal uncertainty of each parameter; you must look at the joint uncertainty of the correlated variables.

You would have seen that this might be a problem if you looked at $n_{eff}$:



```python
pm.effective_n(trace)
```





    {'alpha': 6253.6020387847238,
     'betac': 6954.4520465741789,
     'betap': 6225.8358335534804,
     'betapc': 6987.2733444159694}



## Fixing by centering

As usual, centering the log-population fixes things:



```python
df.logpop_c = df.logpop - df.logpop.mean()
```


      """Entry point for launching an IPython kernel.




```python
with pm.Model() as m1c:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    betapc = pm.Normal("betapc", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop_c + betac*df.clevel + betapc*df.clevel*df.logpop_c
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
```




```python
with m1c:
    trace1c = pm.sample(10000, njobs=2)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betapc, betac, betap]
    Sampling 2 chains: 100%|██████████| 21000/21000 [00:16<00:00, 1284.28draws/s]




```python
pm.effective_n(trace1c)
```





    {'alpha': 13059.09068128995,
     'betac': 13486.057010456128,
     'betap': 14787.07665103561,
     'betapc': 14659.70637220611}





```python
pm.summary(trace1c)
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
      <th>betap</th>
      <td>0.263160</td>
      <td>0.035343</td>
      <td>0.000270</td>
      <td>0.191687</td>
      <td>0.330985</td>
      <td>14787.076651</td>
      <td>0.999950</td>
    </tr>
    <tr>
      <th>betac</th>
      <td>0.285020</td>
      <td>0.116811</td>
      <td>0.000988</td>
      <td>0.051495</td>
      <td>0.510018</td>
      <td>13486.057010</td>
      <td>0.999965</td>
    </tr>
    <tr>
      <th>betapc</th>
      <td>0.064929</td>
      <td>0.168961</td>
      <td>0.001466</td>
      <td>-0.270487</td>
      <td>0.394762</td>
      <td>14659.706372</td>
      <td>1.000043</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>3.311022</td>
      <td>0.089226</td>
      <td>0.000771</td>
      <td>3.140614</td>
      <td>3.491820</td>
      <td>13059.090681</td>
      <td>0.999956</td>
    </tr>
  </tbody>
</table>
</div>





```python
postscat(trace1c,trace1c.varnames)
```





    <seaborn.axisgrid.PairGrid at 0x111642160>




![png](Islands1_files/Islands1_32_1.png)




```python
pm.plot_posterior(trace1c);
```



![png](Islands1_files/Islands1_33_0.png)


How do we decide whether the interaction is significant or not? We'll use model comparison to achieve this!
