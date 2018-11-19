---
title: Poisson Regression - tools on islands part 2
shorttitle: Islands2
notebook: Islands2.ipynb
noline: 1
summary: ""
keywords: ['poisson distribution', 'poisson regression', 'glm', 'centering', 'counterfactual plot', 'regression', 'interaction-term', 'model-comparison', 'varying intercept', 'waic', 'hierarchical', 'overdispersion', 'oceanic tools']
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


We go back to our island tools data set to illustrate

- model comparison using WAIC
- model averaging using WAIC
- fighting overdispersion by making a hierarchical regression model.



```python
df=pd.read_csv("data/islands.csv", sep=';')
df
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
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
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
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





```python
def postscat(trace, thevars):
    d={}
    for v in thevars:
        d[v] = trace.get_values(v)
    df = pd.DataFrame.from_dict(d)
    g = sns.pairplot(df, diag_kind="kde", plot_kws={'s':10})
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
    return g
```


## Centered Model

As usual, centering the log-population fixes things:



```python
df.logpop_c = df.logpop - df.logpop.mean()
```




```python
from theano import tensor as t
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
    trace1c = pm.sample(5000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betapc, betac, betap]
    100%|██████████| 6000/6000 [00:08<00:00, 677.13it/s]




```python
pm.traceplot(trace1c);
```



![png](Islands2_files/Islands2_10_0.png)




```python
pm.autocorrplot(trace1c);
```



![png](Islands2_files/Islands2_11_0.png)




```python
pm.effective_n(trace1c)
```





    {'alpha': 6874.0, 'betac': 5729.0, 'betap': 8246.0, 'betapc': 6812.0}





```python
postscat(trace1c,trace1c.varnames);
```



![png](Islands2_files/Islands2_13_0.png)




```python
pm.plot_posterior(trace1c);
```



![png](Islands2_files/Islands2_14_0.png)


## Model comparison for interaction significance

This is an example of feature selection, where we want to decide whether we should keep the interaction term or not, that is, whether the interaction is significant or not? We'll use model comparison to achieve this!

We can see some summary stats from this model:



```python
dfsum=pm.summary(trace1c)
dfsum
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
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
      <td>0.262633</td>
      <td>0.036001</td>
      <td>0.000406</td>
      <td>0.193631</td>
      <td>0.335489</td>
      <td>8246.0</td>
      <td>1.000064</td>
    </tr>
    <tr>
      <th>betac</th>
      <td>0.283962</td>
      <td>0.116508</td>
      <td>0.001489</td>
      <td>0.054386</td>
      <td>0.507466</td>
      <td>5729.0</td>
      <td>0.999912</td>
    </tr>
    <tr>
      <th>betapc</th>
      <td>0.069461</td>
      <td>0.171149</td>
      <td>0.002120</td>
      <td>-0.255683</td>
      <td>0.412522</td>
      <td>6812.0</td>
      <td>0.999901</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>3.311851</td>
      <td>0.089442</td>
      <td>0.001194</td>
      <td>3.132732</td>
      <td>3.484661</td>
      <td>6874.0</td>
      <td>1.000020</td>
    </tr>
  </tbody>
</table>
</div>





```python
pm.dic(trace1c, m1c)
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/ipykernel_launcher.py:1: DeprecationWarning: dic has been deprecated. Use `waic` or `loo` instead.
      """Entry point for launching an IPython kernel.





    95.700834543217113





```python
pm.waic(trace1c, m1c)
```


            log predictive densities exceeds 0.4. This could be indication of
            WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
            
      """)





    WAIC_r(WAIC=84.092688492420706, WAIC_se=12.194906568168383, p_WAIC=7.0510128813693722)



### Sampling from multiple different centered models

**(A)** Our complete model

**(B)** A model with no interaction



```python
with pm.Model() as m2c_nopc:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop_c + betac*df.clevel
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2c_nopc = pm.sample(5000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betac, betap]
    100%|██████████| 6000/6000 [00:07<00:00, 795.67it/s]


**(C)** A model with no contact term



```python
with pm.Model() as m2c_onlyp:
    betap = pm.Normal("betap", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop_c
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2c_onlyp = pm.sample(5000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betap]
    100%|██████████| 6000/6000 [00:04<00:00, 1270.55it/s]


**(D)** A model with only the contact term



```python
with pm.Model() as m2c_onlyc:
    betac = pm.Normal("betac", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha +  betac*df.clevel
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2c_onlyc = pm.sample(5000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betac]
    100%|██████████| 6000/6000 [00:06<00:00, 968.85it/s]


**(E)** A model with only the intercept.



```python
with pm.Model() as m2c_onlyic:
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2c_onlyic = pm.sample(5000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha]
    100%|██████████| 6000/6000 [00:03<00:00, 1904.33it/s]


We create a dictionary from these models and their traces, so that we can track the names as well



```python
modeldict=dict(m1c=(m1c, trace1c), m2c_nopc = (m2c_nopc, trace2c_nopc),
              m2c_onlyp=(m2c_onlyp, trace2c_onlyp),
              m2c_onlyc=(m2c_onlyc, trace2c_onlyc),
              m2c_onlyic=(m2c_onlyic, trace2c_onlyic))
```




```python
names, models, traces=zip(*[(a, b, c) for a, (b, c) in modeldict.items()])
```


## Comparing the models using WAIC

Finally we use `pm.compare` to create a dataframe of comparisions, and do some pandas stuff to label the rows.



```python
comparedf = pm.compare(traces, models, method="pseudo-BMA")
comparedf.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>79.06</td>
      <td>4.24</td>
      <td>0</td>
      <td>0.87</td>
      <td>11.06</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>84.09</td>
      <td>7.05</td>
      <td>5.04</td>
      <td>0.07</td>
      <td>12.19</td>
      <td>3.77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84.43</td>
      <td>3.75</td>
      <td>5.37</td>
      <td>0.06</td>
      <td>8.94</td>
      <td>7.93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>141.65</td>
      <td>8.38</td>
      <td>62.6</td>
      <td>0</td>
      <td>31.7</td>
      <td>32.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>150.44</td>
      <td>16.94</td>
      <td>71.38</td>
      <td>0</td>
      <td>44.67</td>
      <td>44.44</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





```python
temp=comparedf.sort_index()
temp['name']=names
comparedf = temp.sort_values('WAIC').set_index('name')
comparedf
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m2c_nopc</th>
      <td>79.06</td>
      <td>4.24</td>
      <td>0</td>
      <td>0.87</td>
      <td>11.06</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m1c</th>
      <td>84.09</td>
      <td>7.05</td>
      <td>5.04</td>
      <td>0.07</td>
      <td>12.19</td>
      <td>3.77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyp</th>
      <td>84.43</td>
      <td>3.75</td>
      <td>5.37</td>
      <td>0.06</td>
      <td>8.94</td>
      <td>7.93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyic</th>
      <td>141.65</td>
      <td>8.38</td>
      <td>62.6</td>
      <td>0</td>
      <td>31.7</td>
      <td>32.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyc</th>
      <td>150.44</td>
      <td>16.94</td>
      <td>71.38</td>
      <td>0</td>
      <td>44.67</td>
      <td>44.44</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



From McElreath, here is how to read this table:

>(1)	WAIC is obviously WAIC for each model. Smaller WAIC indicates better estimated out-of-sample deviance.

>(2)	pWAIC is the estimated effective number of parameters. This provides a clue as to how flexible each model is in fitting the sample.

>(3)	dWAIC is the difference between each WAIC and the lowest WAIC. Since only relative deviance matters, this column shows the differences in relative fashion.

>(4)	weight is the AKAIKE WEIGHT for each model. These values are transformed information criterion values. I'll explain them below.

>(5)	SE is the standard error of the WAIC estimate. WAIC is an estimate, and provided the sample size N is large enough, its uncertainty will be well approximated by its standard error. So this SE value isn't necessarily very precise, but it does provide a check against overconfidence in differences between WAIC values.

>(6)	dSE is the standard error of the difference in WAIC between each model and the top-ranked model. So it is missing for the top model. 

>The weight for a model i in a set of m models is given by:

$$w_i = \frac{exp(-\frac{1}{2}dWAIC_i)}{\sum_j exp(-\frac{1}{2}dWAIC_j)}$$

>The Akaike weight formula might look rather odd, but really all it is doing is putting WAIC on a probability scale, so it just undoes the multiplication by −2 and then exponentiates to reverse the log transformation. Then it standardizes by dividing by the total. So each weight will be a number from 0 to 1, and the weights together always sum to 1. Now larger values are better.

>But what do these weights mean? 

>Akaike's interpretation:

>A model's weight is an estimate of the probability that the model will make the best predictions on new data, conditional on the set of models considered...the Akaike weights are analogous to posterior probabilities of models, conditional on expected future data.

>So you can heuristically read each weight as an estimated probability that each model will perform best on future data. In simulation at least, interpreting weights in this way turns out to be appropriate. (McElreath 199-200)

We can make visual comparison plots in the style of McElreath's book. We can see that all the weight is in the no-interaction, full, and only log(population) models.



```python
pm.compareplot(comparedf)
```





    <matplotlib.axes._subplots.AxesSubplot at 0x1141f93c8>




![png](Islands2_files/Islands2_35_1.png)


### Comparing for non-centered models

We can redo the coparison for non-centered models



```python
with pm.Model() as m1:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    betapc = pm.Normal("betapc", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop + betac*df.clevel + betapc*df.clevel*df.logpop
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace1 = pm.sample(10000, tune=2000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betapc, betac, betap]
    100%|██████████| 12000/12000 [01:23<00:00, 143.29it/s]




```python
with pm.Model() as m2_onlyp:
    betap = pm.Normal("betap", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2_onlyp = pm.sample(10000, tune=2000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betap]
    100%|██████████| 12000/12000 [00:26<00:00, 445.51it/s]
    The acceptance probability does not match the target. It is 0.893725363713, but should be close to 0.8. Try to increase the number of tuning steps.
    The number of effective samples is smaller than 25% for some parameters.




```python
with pm.Model() as m2_nopc:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop + betac*df.clevel
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2_nopc = pm.sample(10000, tune=2000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betac, betap]
    100%|██████████| 12000/12000 [00:47<00:00, 251.99it/s]
    The number of effective samples is smaller than 25% for some parameters.




```python
modeldict2=dict(m1=(m1, trace1), m2_nopc = (m2_nopc, trace2_nopc),
              m2_onlyp=(m2_onlyp, trace2_onlyp),
              m2_onlyc=(m2c_onlyc, trace2c_onlyc),
              m2_onlyic=(m2c_onlyic, trace2c_onlyic))
```




```python
names2, models2, traces2=zip(*[(a, b, c) for a, (b, c) in modeldict2.items()])
```




```python
comparedf2 = pm.compare(traces2, models2, method="pseudo-BMA")
temp=comparedf2.sort_index()
temp['name']=names2
comparedf2 = temp.sort_values('WAIC').set_index('name')
comparedf2
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m2_nopc</th>
      <td>78.95</td>
      <td>4.17</td>
      <td>0</td>
      <td>0.61</td>
      <td>11.12</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m1</th>
      <td>80.05</td>
      <td>4.83</td>
      <td>1.1</td>
      <td>0.35</td>
      <td>11.24</td>
      <td>1.25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2_onlyp</th>
      <td>84.39</td>
      <td>3.73</td>
      <td>5.44</td>
      <td>0.04</td>
      <td>8.94</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2_onlyic</th>
      <td>141.65</td>
      <td>8.38</td>
      <td>62.7</td>
      <td>0</td>
      <td>31.7</td>
      <td>32.89</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2_onlyc</th>
      <td>150.44</td>
      <td>16.94</td>
      <td>71.48</td>
      <td>0</td>
      <td>44.67</td>
      <td>44.48</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



What we find now is that the full-model has much more weight.



```python
pm.compareplot(comparedf2)
```





    <matplotlib.axes._subplots.AxesSubplot at 0x116788f28>




![png](Islands2_files/Islands2_44_1.png)


In either the centered or non-centered case, our top model excludes the interaction, but the second top model includes it. In the centered case, the non-interacting model has most of the weight, while in the non-centered model, the weights were more equally shared.

In a situation where the interaction model has so much weight, we can say its probably overfit. So in a sense, centering even helps us with our overfitting issues by clearly preferring the sans-interaction model, as it removes correlation and thus spurious weight being borrowed.

## Computing the (counterfactual) posterior predictive for checking

We now write some code to compute the posterior predictive at artbitrary points without having to use theano shared vaiables and sample_ppc, in two different counterfactual situations of low contact and high contact. Since some of our models omit certain terms, we use traces with 0s in them to construct a general function to do this.



```python
def trace_or_zero(trace, name):
    if name in trace.varnames:
        return trace[name]
    else:
        return np.zeros(2*len(trace))
```




```python
# why the 2 above? see output below
len(trace1c), len(trace1c['alpha'])
```





    (5000, 10000)





```python
from scipy.stats import poisson
def compute_pp(lpgrid, trace, contact=0):
    alphatrace = trace_or_zero(trace, 'alpha')
    betaptrace = trace_or_zero(trace, 'betap')
    betactrace = trace_or_zero(trace, 'betac')
    betapctrace = trace_or_zero(trace, 'betapc')
    tl=2*len(trace)
    gl=lpgrid.shape[0]
    lam = np.empty((gl, tl))
    lpgrid_c = lpgrid - lpgrid.mean()
    for i, v in enumerate(lpgrid):
        temp = alphatrace + betaptrace*lpgrid_c[i] + betactrace*contact + betapctrace*contact*lpgrid_c[i]
        lam[i,:] = poisson.rvs(np.exp(temp))
    return lam
```


We compute the posterior predictive in the counterfactual cases: remember what we are doing here is turning on and off a feature.



```python
lpgrid = np.linspace(6,13,30)
pplow = compute_pp(lpgrid, trace1c)
pphigh = compute_pp(lpgrid, trace1c, contact=1)
```


We compute the medians and the hpds, and plot these against the data



```python
pplowmed = np.median(pplow, axis=1)
pplowhpd = pm.stats.hpd(pplow.T)
pphighmed = np.median(pphigh, axis=1)
pphighhpd = pm.stats.hpd(pphigh.T)
```




```python
with sns.plotting_context('poster'):
    plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'.', color="g")
    plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'.', color="r")
    plt.plot(lpgrid, pphighmed, color="g", label="c=1")
    plt.fill_between(lpgrid, pphighhpd[:,0], pphighhpd[:,1], color="g", alpha=0.2)
    plt.plot(lpgrid, pplowmed, color="r", label="c=0")
    plt.fill_between(lpgrid, pplowhpd[:,0], pplowhpd[:,1], color="r", alpha=0.2)
```



![png](Islands2_files/Islands2_54_0.png)


This is for the full centered model. The high contact predictive and data is in green. We undertake this exercise as a prelude to ensembling the models with high Akaike weights

## Ensembling

Ensembles are a good way to combine models where one model may be good at something and the other at something else. Ensembles also help with overfitting if the variance cancels out between the ensemble members: they would all probably overfit in slightly different ways. Lets write a function to do our ensembling for us.



```python
def ensemble(grid, modeldict, comparedf, modelnames, contact=0):
    accum_pp=0
    accum_weight=0
    for m in modelnames:
        weight = comparedf.loc[m]['weight']
        pp = compute_pp(grid, modeldict[m][1], contact)
        print(m, weight, np.median(pp))
        accum_pp += pp*weight
        accum_weight +=weight
    return accum_pp/accum_weight
        
```




```python
ens_pp_low = ensemble(lpgrid, modeldict, comparedf, ['m1c', 'm2c_nopc', 'm2c_onlyp'])
```


    m1c 0.07 28.0
    m2c_nopc 0.87 28.0
    m2c_onlyp 0.06 33.0




```python
ens_pp_high = ensemble(lpgrid, modeldict, comparedf, ['m1c', 'm2c_nopc', 'm2c_onlyp'], contact=1)
```


    m1c 0.07 37.0
    m2c_nopc 0.87 37.0
    m2c_onlyp 0.06 32.0




```python
with sns.plotting_context('poster'):
    pplowmed = np.median(ens_pp_low, axis=1)
    pplowhpd = pm.stats.hpd(ens_pp_low.T)
    pphighmed = np.median(ens_pp_high, axis=1)
    pphighhpd = pm.stats.hpd(ens_pp_high.T)
    plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
    plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
    plt.plot(lpgrid, pphighmed, color="g")
    plt.fill_between(lpgrid, pphighhpd[:,0], pphighhpd[:,1], color="g", alpha=0.2)
    plt.plot(lpgrid, pplowmed, color="r")
    plt.fill_between(lpgrid, pplowhpd[:,0], pplowhpd[:,1], color="r", alpha=0.2)

```



![png](Islands2_files/Islands2_60_0.png)


The ensemble gives sensible limits and even regularizes down the green band at high population by giving more weight to the no-interaction model.

## Hierarchical Modelling

**Overdispersion** is a problem one finds in most poisson models where the variance of the data is larger than the mean, which is the constraint the poisson distribution imposes.

To simplify things, let us consider here, only the model with log(population). Since there is no contact variable, there are no counterfactual plots and we can view the posterior predictive.



```python
ppsamps = compute_pp(lpgrid, trace2c_onlyp)
ppmed = np.median(ppsamps, axis=1)
pphpd = pm.stats.hpd(ppsamps.T)
plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
plt.plot(lpgrid, ppmed, color="b")
plt.fill_between(lpgrid, pphpd[:,0], pphpd[:,1], color="b", alpha=0.1)
#plt.ylim([0, 300])
```





    <matplotlib.collections.PolyCollection at 0x113f09b70>




![png](Islands2_files/Islands2_63_1.png)


By taking the ratio of the posterior-predictive variance to the posterior-predictive mean, we see that the model is overdispersed.



```python
ppvar=np.var(ppsamps, axis=1)
ppmean=np.mean(ppsamps, axis=1)
```




```python
ppvar/ppmean
```





    array([ 1.30421519,  1.26489919,  1.2693647 ,  1.20461164,  1.25536688,
            1.19957498,  1.1793642 ,  1.17456651,  1.14728935,  1.15605154,
            1.09427345,  1.12326509,  1.13490696,  1.09674559,  1.12324651,
            1.10038695,  1.11064864,  1.11575808,  1.11499395,  1.14792792,
            1.15350445,  1.18526221,  1.22732124,  1.29480132,  1.30994366,
            1.41243637,  1.48317552,  1.58211591,  1.67981142,  1.79674707])



Overdispersion can be fixed by considering a mixture model. We shall see this next week. But hierarchical modelling is also a great way to do this.

### Varying Intercepts hierarchical model

What we are basically doing is splitting the intercept into a value constant across the societies and a residual which is society dependent. It is this residual that we will assume is drawn from a gaussian with 0 mean and `sigmasoc` ($\sigma_{society}$) standard deviation. Since there is a varying intercept for **every** observation, $\sigma_{society}$ lands up as an estimate of overdispersion amongst societies.



```python
with pm.Model() as m3c:
    betap = pm.Normal("betap", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    sigmasoc = pm.HalfCauchy("sigmasoc", 1)
    alphasoc = pm.Normal("alphasoc", 0, sigmasoc, shape=df.shape[0])
    loglam = alpha + alphasoc + betap*df.logpop_c 
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
```




```python
with m3c:
    trace3 = pm.sample(5000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alphasoc, sigmasoc_log__, alpha, betap]
    100%|██████████| 6000/6000 [00:15<00:00, 386.45it/s]
    The number of effective samples is smaller than 25% for some parameters.


Notice that we are fitting 13 parameters to 10 points. Ordinarily this would scream overfitting, but thefocus of our parameters is at different levels, and in the hierarchial set up, 10 of these parameters are really pooled together from one sigma. So the effective number of parameters is something lower.



```python
pm.traceplot(trace3)
```





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x115b51198>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x115e669b0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x114fe6b38>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x116842588>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x115eeeb00>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1162fdcf8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1188bf5f8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x119c85400>]], dtype=object)




![png](Islands2_files/Islands2_71_1.png)




```python
np.mean(trace3['diverging'])
```





    0.0





```python
pm.summary(trace3)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
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
      <td>0.259400</td>
      <td>0.079289</td>
      <td>0.001079</td>
      <td>0.103152</td>
      <td>0.421723</td>
      <td>4488.0</td>
      <td>0.999985</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>3.448287</td>
      <td>0.120423</td>
      <td>0.001770</td>
      <td>3.200975</td>
      <td>3.677543</td>
      <td>3919.0</td>
      <td>0.999915</td>
    </tr>
    <tr>
      <th>alphasoc__0</th>
      <td>-0.210082</td>
      <td>0.243427</td>
      <td>0.003397</td>
      <td>-0.728636</td>
      <td>0.234597</td>
      <td>5045.0</td>
      <td>0.999912</td>
    </tr>
    <tr>
      <th>alphasoc__1</th>
      <td>0.035751</td>
      <td>0.218266</td>
      <td>0.002822</td>
      <td>-0.380003</td>
      <td>0.503895</td>
      <td>6020.0</td>
      <td>0.999900</td>
    </tr>
    <tr>
      <th>alphasoc__2</th>
      <td>-0.050208</td>
      <td>0.192201</td>
      <td>0.002430</td>
      <td>-0.435754</td>
      <td>0.335806</td>
      <td>6532.0</td>
      <td>0.999932</td>
    </tr>
    <tr>
      <th>alphasoc__3</th>
      <td>0.322955</td>
      <td>0.185538</td>
      <td>0.002634</td>
      <td>-0.032226</td>
      <td>0.692806</td>
      <td>4217.0</td>
      <td>0.999951</td>
    </tr>
    <tr>
      <th>alphasoc__4</th>
      <td>0.038710</td>
      <td>0.176232</td>
      <td>0.002034</td>
      <td>-0.315874</td>
      <td>0.383337</td>
      <td>6676.0</td>
      <td>1.000008</td>
    </tr>
    <tr>
      <th>alphasoc__5</th>
      <td>-0.320876</td>
      <td>0.206979</td>
      <td>0.002859</td>
      <td>-0.745401</td>
      <td>0.055176</td>
      <td>4959.0</td>
      <td>1.000079</td>
    </tr>
    <tr>
      <th>alphasoc__6</th>
      <td>0.140740</td>
      <td>0.172141</td>
      <td>0.002258</td>
      <td>-0.185500</td>
      <td>0.497224</td>
      <td>5333.0</td>
      <td>0.999988</td>
    </tr>
    <tr>
      <th>alphasoc__7</th>
      <td>-0.172323</td>
      <td>0.185173</td>
      <td>0.002393</td>
      <td>-0.548562</td>
      <td>0.189961</td>
      <td>6245.0</td>
      <td>1.000587</td>
    </tr>
    <tr>
      <th>alphasoc__8</th>
      <td>0.272111</td>
      <td>0.174620</td>
      <td>0.002321</td>
      <td>-0.055450</td>
      <td>0.624651</td>
      <td>4475.0</td>
      <td>1.000024</td>
    </tr>
    <tr>
      <th>alphasoc__9</th>
      <td>-0.094881</td>
      <td>0.282000</td>
      <td>0.003884</td>
      <td>-0.695863</td>
      <td>0.450101</td>
      <td>4451.0</td>
      <td>1.000008</td>
    </tr>
    <tr>
      <th>sigmasoc</th>
      <td>0.309760</td>
      <td>0.124590</td>
      <td>0.002286</td>
      <td>0.103297</td>
      <td>0.569363</td>
      <td>2491.0</td>
      <td>0.999920</td>
    </tr>
  </tbody>
</table>
</div>



We can ask the WAIC how many effective parameters it has, and it tells us roughly 5. Thus you really care about the number of hyper-parameters you have, and not so much about the lower level parameters.



```python
pm.waic(trace3, m3c)
```


            log predictive densities exceeds 0.4. This could be indication of
            WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
            
      """)





    WAIC_r(WAIC=70.176914155345884, WAIC_se=2.5510130691737838, p_WAIC=5.0619370952485738)



We now write code where now we use sampling from the normal corresponding to $\sigma_{society}$ to simulate our societies. Again, we dont use theano's shareds, opting simply to generate samples for the residual intercepts for multiple societies. How many? As many as the traces. You might have thought you only need to generate as many as there are grid points, ie 30, but at the end the posterior predictive must marginalize over the traces at all these points, and thus marginalizing over the full trace at each point suffices!



```python
def compute_pp2(lpgrid, trace, contact=0):
    alphatrace = trace_or_zero(trace, 'alpha')
    betaptrace = trace_or_zero(trace, 'betap')
    sigmasoctrace = trace_or_zero(trace, 'sigmasoc')
    tl=2*len(trace)
    gl=lpgrid.shape[0]
    lam = np.empty((gl, tl))
    lpgrid_c = lpgrid - lpgrid.mean()
    #simulate. 5000 alphasocs gen here
    alphasoctrace=np.random.normal(0, sigmasoctrace)
    for i, v in enumerate(lpgrid):
        temp = alphatrace + betaptrace*lpgrid_c[i] + alphasoctrace
        lam[i,:] = poisson.rvs(np.exp(temp))
    return lam
```




```python
ppsamps = compute_pp2(lpgrid, trace3)

```




```python
ppmed = np.median(ppsamps, axis=1)
pphpd = pm.stats.hpd(ppsamps.T)
plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
plt.plot(lpgrid, ppmed, color="b")
plt.fill_between(lpgrid, pphpd[:,0], pphpd[:,1], color="b", alpha=0.1)
```





    <matplotlib.collections.PolyCollection at 0x118d9d358>




![png](Islands2_files/Islands2_79_1.png)


The envelope of predictions is much wider here, but overlaps all the points! This is because of the varying intercepts, and it reflects the fact that there is much more variation in the data than is expected from a pure poisson model.

## Cross Validation and stacking BMA in pymc3



```python
comparedf = pm.compare(traces, models, method="pseudo-BMA", ic='LOO')
comparedf.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LOO</th>
      <th>pLOO</th>
      <th>dLOO</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>79.84</td>
      <td>4.63</td>
      <td>0</td>
      <td>0.93</td>
      <td>11.17</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>85.26</td>
      <td>4.17</td>
      <td>5.43</td>
      <td>0.06</td>
      <td>8.96</td>
      <td>8.01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>89.92</td>
      <td>9.96</td>
      <td>10.08</td>
      <td>0.01</td>
      <td>13.53</td>
      <td>6.11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>141.77</td>
      <td>8.44</td>
      <td>61.93</td>
      <td>0</td>
      <td>31.74</td>
      <td>32.63</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151.24</td>
      <td>17.34</td>
      <td>71.4</td>
      <td>0</td>
      <td>44.91</td>
      <td>44.37</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





```python
temp=comparedf.sort_index()
temp['name']=names
comparedf = temp.sort_values('LOO').set_index('name')
comparedf
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LOO</th>
      <th>pLOO</th>
      <th>dLOO</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m2c_nopc</th>
      <td>79.84</td>
      <td>4.63</td>
      <td>0</td>
      <td>0.93</td>
      <td>11.17</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyp</th>
      <td>85.26</td>
      <td>4.17</td>
      <td>5.43</td>
      <td>0.06</td>
      <td>8.96</td>
      <td>8.01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m1c</th>
      <td>89.92</td>
      <td>9.96</td>
      <td>10.08</td>
      <td>0.01</td>
      <td>13.53</td>
      <td>6.11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyic</th>
      <td>141.77</td>
      <td>8.44</td>
      <td>61.93</td>
      <td>0</td>
      <td>31.74</td>
      <td>32.63</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyc</th>
      <td>151.24</td>
      <td>17.34</td>
      <td>71.4</td>
      <td>0</td>
      <td>44.91</td>
      <td>44.37</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





```python
pm.compareplot(comparedf)
```





    <matplotlib.axes._subplots.AxesSubplot at 0x11738c6d8>




![png](Islands2_files/Islands2_84_1.png)




```python
comparedf_s = pm.compare(traces, models, method="stacking", ic='WAIC')
comparedf_s.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>79.06</td>
      <td>4.24</td>
      <td>0</td>
      <td>0.76</td>
      <td>11.06</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>84.09</td>
      <td>7.05</td>
      <td>5.04</td>
      <td>0</td>
      <td>12.19</td>
      <td>3.77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84.43</td>
      <td>3.75</td>
      <td>5.37</td>
      <td>0.24</td>
      <td>8.94</td>
      <td>7.93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>141.65</td>
      <td>8.38</td>
      <td>62.6</td>
      <td>0</td>
      <td>31.7</td>
      <td>32.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>150.44</td>
      <td>16.94</td>
      <td>71.38</td>
      <td>0</td>
      <td>44.67</td>
      <td>44.44</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





```python
temp=comparedf_s.sort_index()
temp['name']=names
comparedf_s = temp.sort_values('WAIC').set_index('name')
comparedf_s
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m2c_nopc</th>
      <td>79.06</td>
      <td>4.24</td>
      <td>0</td>
      <td>0.76</td>
      <td>11.06</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m1c</th>
      <td>84.09</td>
      <td>7.05</td>
      <td>5.04</td>
      <td>0</td>
      <td>12.19</td>
      <td>3.77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyp</th>
      <td>84.43</td>
      <td>3.75</td>
      <td>5.37</td>
      <td>0.24</td>
      <td>8.94</td>
      <td>7.93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyic</th>
      <td>141.65</td>
      <td>8.38</td>
      <td>62.6</td>
      <td>0</td>
      <td>31.7</td>
      <td>32.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyc</th>
      <td>150.44</td>
      <td>16.94</td>
      <td>71.38</td>
      <td>0</td>
      <td>44.67</td>
      <td>44.44</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


