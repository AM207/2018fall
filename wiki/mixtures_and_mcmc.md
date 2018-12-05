---
title: Mixtures and MCMC
shorttitle: mixtures_and_mcmc
notebook: mixtures_and_mcmc.ipynb
noline: 1
summary: ""
keywords: ['supervised learning', 'semi-supervised learning', 'unsupervised learning', 'mixture model', 'gaussian mixture model', 'pymc3', 'label-switching', 'identifiability', 'normal distribution', 'pymc3 potentials']
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}


We now do a study of learning mixture models with MCMC. We have already done this in the case of the Zero-Inflated Poisson Model, and will stick to Gaussian Mixture models for now.



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
import theano.tensor as tt
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


## Mixture of 2 Gaussians, the old faithful data

We start by considering waiting times from the Old-Faithful Geyser at Yellowstone National Park.



```python
ofdata=pd.read_csv("data/oldfaithful.csv")
ofdata.head()
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
      <th>eruptions</th>
      <th>waiting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.600</td>
      <td>79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.800</td>
      <td>54</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.333</td>
      <td>74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.283</td>
      <td>62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.533</td>
      <td>85</td>
    </tr>
  </tbody>
</table>
</div>





```python
sns.distplot(ofdata.waiting);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_4_1.png)


Visually, there seem to be two components to the waiting time, so let us model this using a mixture of two gaussians. Remember that this is a unsupervized model, and all we are doing is modelling $p(x)$ , with the assumption that there are two clusters and a hidden variable $z$ that indexes them.

Notice that these gaussians seem well separated. The separation of gaussians impacts how your sampler will perform.



```python
with pm.Model() as ofmodel:
    p1 = pm.Uniform('p', 0, 1)
    p2 = 1 - p1
    p = tt.stack([p1, p2])
    assignment = pm.Categorical("assignment", p, 
                                shape=ofdata.shape[0])
    sds = pm.Uniform("sds", 0, 40, shape=2)
    centers = pm.Normal("centers", 
                        mu=np.array([50, 80]), 
                        sd=np.array([20, 20]), 
                        shape=2)
    
    # and to combine it with the observations:
    observations = pm.Normal("obs", mu=centers[assignment], sd=sds[assignment], observed=ofdata.waiting)
```




```python
with ofmodel:
    #step1 = pm.Metropolis(vars=[p, sds, centers])
    #step2 = pm.CategoricalGibbsMetropolis(vars=[assignment])
    #oftrace_full = pm.sample(10000, step=[step1, step2])
    oftrace_full = pm.sample(10000)
```


    Multiprocess sampling (2 chains in 2 jobs)
    CompoundStep
    >NUTS: [centers, sds, p]
    >BinaryGibbsMetropolis: [assignment]
    Sampling 2 chains: 100%|██████████| 21000/21000 [04:43<00:00, 73.96draws/s]




```python
pm.model_to_graphviz(ofmodel)
```





![svg](mixtures_and_mcmc_files/mixtures_and_mcmc_8_0.svg)





```python
oftrace = oftrace_full[2000::5]
pm.traceplot(oftrace);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3604: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_9_1.png)




```python
pm.summary(oftrace)
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
      <th>assignment__0</th>
      <td>0.999687</td>
      <td>0.017675</td>
      <td>0.000311</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>assignment__1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__2</th>
      <td>0.993125</td>
      <td>0.082630</td>
      <td>0.001504</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3248.753445</td>
      <td>0.999745</td>
    </tr>
    <tr>
      <th>assignment__3</th>
      <td>0.050937</td>
      <td>0.219870</td>
      <td>0.004231</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3173.388070</td>
      <td>1.000029</td>
    </tr>
    <tr>
      <th>assignment__4</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__5</th>
      <td>0.000313</td>
      <td>0.017675</td>
      <td>0.000311</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>assignment__6</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__7</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__8</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__9</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__10</th>
      <td>0.000625</td>
      <td>0.024992</td>
      <td>0.000437</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3208.032629</td>
      <td>0.999687</td>
    </tr>
    <tr>
      <th>assignment__11</th>
      <td>0.999687</td>
      <td>0.017675</td>
      <td>0.000311</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>assignment__12</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__13</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__14</th>
      <td>0.999687</td>
      <td>0.017675</td>
      <td>0.000311</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>assignment__15</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__16</th>
      <td>0.046875</td>
      <td>0.211371</td>
      <td>0.003711</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3205.553814</td>
      <td>0.999696</td>
    </tr>
    <tr>
      <th>assignment__17</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__18</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__19</th>
      <td>0.999375</td>
      <td>0.024992</td>
      <td>0.000437</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3208.032629</td>
      <td>0.999687</td>
    </tr>
    <tr>
      <th>assignment__20</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__21</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__22</th>
      <td>0.999062</td>
      <td>0.030604</td>
      <td>0.000533</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3209.379383</td>
      <td>0.999792</td>
    </tr>
    <tr>
      <th>assignment__23</th>
      <td>0.817500</td>
      <td>0.386256</td>
      <td>0.006151</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3120.023664</td>
      <td>0.999729</td>
    </tr>
    <tr>
      <th>assignment__24</th>
      <td>0.993125</td>
      <td>0.082630</td>
      <td>0.001629</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3235.543269</td>
      <td>1.001749</td>
    </tr>
    <tr>
      <th>assignment__25</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__26</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__27</th>
      <td>0.997188</td>
      <td>0.052958</td>
      <td>0.001177</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3222.014229</td>
      <td>0.999722</td>
    </tr>
    <tr>
      <th>assignment__28</th>
      <td>0.998437</td>
      <td>0.039498</td>
      <td>0.000681</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3213.694327</td>
      <td>0.999750</td>
    </tr>
    <tr>
      <th>assignment__29</th>
      <td>0.998750</td>
      <td>0.035333</td>
      <td>0.000612</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3212.072926</td>
      <td>0.999687</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>assignment__247</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__248</th>
      <td>0.576562</td>
      <td>0.494103</td>
      <td>0.008319</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2947.001300</td>
      <td>1.000571</td>
    </tr>
    <tr>
      <th>assignment__249</th>
      <td>0.995938</td>
      <td>0.063608</td>
      <td>0.001140</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2794.058393</td>
      <td>0.999712</td>
    </tr>
    <tr>
      <th>assignment__250</th>
      <td>0.000625</td>
      <td>0.024992</td>
      <td>0.000437</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1.000313</td>
    </tr>
    <tr>
      <th>assignment__251</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__252</th>
      <td>0.983750</td>
      <td>0.126436</td>
      <td>0.002048</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3312.296435</td>
      <td>0.999907</td>
    </tr>
    <tr>
      <th>assignment__253</th>
      <td>0.983125</td>
      <td>0.128803</td>
      <td>0.001897</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3192.025887</td>
      <td>0.999899</td>
    </tr>
    <tr>
      <th>assignment__254</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__255</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__256</th>
      <td>0.949375</td>
      <td>0.219231</td>
      <td>0.004141</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2517.301618</td>
      <td>1.000086</td>
    </tr>
    <tr>
      <th>assignment__257</th>
      <td>0.999687</td>
      <td>0.017675</td>
      <td>0.000311</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>assignment__258</th>
      <td>0.000625</td>
      <td>0.024992</td>
      <td>0.000437</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1.000313</td>
    </tr>
    <tr>
      <th>assignment__259</th>
      <td>0.999687</td>
      <td>0.017675</td>
      <td>0.000311</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>assignment__260</th>
      <td>0.999687</td>
      <td>0.017675</td>
      <td>0.000311</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>assignment__261</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__262</th>
      <td>0.003438</td>
      <td>0.058529</td>
      <td>0.000978</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3224.662207</td>
      <td>0.999944</td>
    </tr>
    <tr>
      <th>assignment__263</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__264</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__265</th>
      <td>0.013750</td>
      <td>0.116451</td>
      <td>0.002088</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3296.006530</td>
      <td>0.999716</td>
    </tr>
    <tr>
      <th>assignment__266</th>
      <td>0.995938</td>
      <td>0.063608</td>
      <td>0.001140</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2823.618545</td>
      <td>1.000870</td>
    </tr>
    <tr>
      <th>assignment__267</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__268</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__269</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__270</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>assignment__271</th>
      <td>0.993125</td>
      <td>0.082630</td>
      <td>0.001295</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3247.614839</td>
      <td>0.999916</td>
    </tr>
    <tr>
      <th>centers__0</th>
      <td>54.627527</td>
      <td>0.733327</td>
      <td>0.011941</td>
      <td>53.263593</td>
      <td>56.069912</td>
      <td>2857.383302</td>
      <td>1.000063</td>
    </tr>
    <tr>
      <th>centers__1</th>
      <td>80.087834</td>
      <td>0.523151</td>
      <td>0.008877</td>
      <td>79.104464</td>
      <td>81.134251</td>
      <td>3013.404106</td>
      <td>0.999852</td>
    </tr>
    <tr>
      <th>p</th>
      <td>0.361858</td>
      <td>0.030777</td>
      <td>0.000567</td>
      <td>0.302382</td>
      <td>0.419931</td>
      <td>2753.669629</td>
      <td>1.000464</td>
    </tr>
    <tr>
      <th>sds__0</th>
      <td>6.018165</td>
      <td>0.585472</td>
      <td>0.010211</td>
      <td>4.961674</td>
      <td>7.231991</td>
      <td>2942.626870</td>
      <td>0.999759</td>
    </tr>
    <tr>
      <th>sds__1</th>
      <td>5.947338</td>
      <td>0.416616</td>
      <td>0.008702</td>
      <td>5.151874</td>
      <td>6.773165</td>
      <td>2566.297885</td>
      <td>0.999898</td>
    </tr>
  </tbody>
</table>
<p>277 rows × 7 columns</p>
</div>





```python
pm.autocorrplot(oftrace, varnames=['p', 'centers', 'sds']);
```



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_11_0.png)




```python
oftrace['centers'].mean(axis=0)
```





    array([54.62752749, 80.0878344 ])



We can visualize the two clusters, suitably scales by the category-belonging probability by taking the posterior means. Note that this misses any smearing that might go into making the posterior predictive



```python
from scipy.stats import norm
x = np.linspace(20, 120, 500)
# for pretty colors later in the book.
colors = ["#348ABD", "#A60628"] if oftrace['centers'][-1, 0] > oftrace['centers'][-1, 1] \
    else ["#A60628", "#348ABD"]

posterior_center_means = oftrace['centers'].mean(axis=0)
posterior_std_means = oftrace['sds'].mean(axis=0)
posterior_p_mean = oftrace["p"].mean()

plt.hist(ofdata.waiting, bins=20, histtype="step", normed=True, color="k",
     lw=2, label="histogram of data")
y = posterior_p_mean * norm.pdf(x, loc=posterior_center_means[0],
                                scale=posterior_std_means[0])
plt.plot(x, y, label="Cluster 0 (using posterior-mean parameters)", lw=3)
plt.fill_between(x, y, color=colors[1], alpha=0.3)

y = (1 - posterior_p_mean) * norm.pdf(x, loc=posterior_center_means[1],
                                      scale=posterior_std_means[1])
plt.plot(x, y, label="Cluster 1 (using posterior-mean parameters)", lw=3)
plt.fill_between(x, y, color=colors[0], alpha=0.3)

plt.legend(loc="upper left")
plt.title("Visualizing Clusters using posterior-mean parameters");
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_14_1.png)


## A tetchy 3 Gaussian Model

Let us set up our data. Our analysis here follows that of https://colindcarroll.com/2018/07/20/why-im-excited-about-pymc3-v3.5.0/ , and we have chosen 3 gaussians reasonably close to each other to show the problems that arise!



```python
mu_true = np.array([-2, 0, 2])
sigma_true = np.array([1, 1, 1])
lambda_true = np.array([1/3, 1/3, 1/3])
n = 100
from scipy.stats import multinomial
# Simulate from each distribution according to mixing proportion psi
z = multinomial.rvs(1, lambda_true, size=n)
data=np.array([np.random.normal(mu_true[i.astype('bool')][0], sigma_true[i.astype('bool')][0]) for i in z])
sns.distplot(data, bins=50);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    //anaconda/envs/py3l/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_16_1.png)




```python
np.savetxt("data/3gv2.dat", data)
```




```python
with pm.Model() as mof:
    #p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=3)
    p=[1/3, 1/3, 1/3]

    # cluster centers
    means = pm.Normal('means', mu=0, sd=10, shape=3)


    #sds = pm.HalfCauchy('sds', 5, shape=3)
    sds = np.array([1., 1., 1.])
    
    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=data.shape[0])

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sd=1., #sds[category],
                       observed=data)

```




```python
with mof:
    tripletrace_full = pm.sample(10000)
```


    Multiprocess sampling (2 chains in 2 jobs)
    CompoundStep
    >NUTS: [means]
    >CategoricalGibbsMetropolis: [category]
    Sampling 2 chains: 100%|██████████| 21000/21000 [01:39<00:00, 210.02draws/s]
    The estimated number of effective samples is smaller than 200 for some parameters.




```python
trace_mof=tripletrace_full[3000::7]
#pm.traceplot(trace_mof, varnames=["means", "p", "sds"]);
pm.traceplot(trace_mof, varnames=["means"], combined=True);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3604: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_20_1.png)




```python
pm.autocorrplot(trace_mof, varnames=['means']);
```



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_21_0.png)


## Problems with clusters and sampling

Some of the traces seem ok, but the autocorrelation is quite bad. And there is label-switching .This is because there are major problems with using MCMC for clustering.

AND THIS IS WITHOUT MODELING $p$ OR $\sigma$. It gets much worse otherwise! (it would be better if the gaussians were quite widely separated out).

These are firstly, the lack of parameter identifiability (the so called label-switching problem) and secondly, the multimodality of the posteriors.

We have seen non-identifiability before. Switching labels on the means and z's, for example, does not change the likelihoods. The problem with this is that cluster parameters cannot be compared across chains: what might be a cluster parameter in one chain could well belong to the other cluster in the second chain. Even within a single chain, indices might swap leading to a telltale back and forth in the traces for long chains or data not cleanly separated.

Also, the (joint) posteriors can be highly multimodal. One form of multimodality is the non-identifiability, though even without identifiability issues the posteriors are highly multimodal.

To quote the Stan manual:
>Bayesian inference fails in cases of high multimodality because there is no way to visit all of the modes in the posterior in appropriate proportions and thus no way to evaluate integrals involved in posterior predictive inference.
In light of these two problems, the advice often given in fitting clustering models is to try many different initializations and select the sample with the highest overall probability. It is also popular to use optimization-based point estimators such as expectation maximization or variational Bayes, which can be much more efficient than sampling-based approaches.

### Some mitigation via ordering in pymc3

But this is not a panacea. Sampling is still very hard.




```python
import theano.tensor as tt
import pymc3.distributions.transforms as tr


with pm.Model() as mof2:
    
    p = [1/3, 1/3, 1/3]

    # cluster centers
    means = pm.Normal('means', mu=0, sd=10, shape=3,
                  transform=tr.ordered,
                  testval=np.array([-1, 0, 1]))


                                         
    # measurement error
    #sds = pm.Uniform('sds', lower=0, upper=20, shape=3)

    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=data.shape[0])

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sd=1., #sds[category],
                       observed=data)

```




```python
with mof2:
    tripletrace_full2 = pm.sample(10000)
```


    Multiprocess sampling (2 chains in 2 jobs)
    CompoundStep
    >NUTS: [means]
    >CategoricalGibbsMetropolis: [category]
    Sampling 2 chains: 100%|██████████| 21000/21000 [03:42<00:00, 94.44draws/s] 
    The acceptance probability does not match the target. It is 0.9255675538672136, but should be close to 0.8. Try to increase the number of tuning steps.
    The number of effective samples is smaller than 10% for some parameters.




```python
trace_mof2 = tripletrace_full2[3000::5]
pm.traceplot(trace_mof2, varnames=["means"], combined=True);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3604: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_26_1.png)




```python
pm.autocorrplot(trace_mof2, varnames=["means"]);
```



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_27_0.png)


## Full sampling is horrible, even with potentials

Now lets put Dirichlet based (and this is a strongly centering Dirichlet) prior on the probabilities



```python
from scipy.stats import dirichlet
ds = dirichlet(alpha=[10,10,10]).rvs(1000)
```




```python
"""
Visualize points on the 3-simplex (eg, the parameters of a
3-dimensional multinomial distributions) as a scatter plot 
contained within a 2D triangle.
David Andrzejewski (david.andrzej@gmail.com)
"""
import numpy as NP
import matplotlib.pyplot as P
import matplotlib.ticker as MT
import matplotlib.lines as L
import matplotlib.cm as CM
import matplotlib.colors as C
import matplotlib.patches as PA

def plotSimplex(points, fig=None, 
                vertexlabels=['1','2','3'],
                **kwargs):
    """
    Plot Nx3 points array on the 3-simplex 
    (with optionally labeled vertices) 
    
    kwargs will be passed along directly to matplotlib.pyplot.scatter    
    Returns Figure, caller must .show()
    """
    if(fig == None):        
        fig = P.figure()
    # Draw the triangle
    l1 = L.Line2D([0, 0.5, 1.0, 0], # xcoords
                  [0, NP.sqrt(3) / 2, 0, 0], # ycoords
                  color='k')
    fig.gca().add_line(l1)
    fig.gca().xaxis.set_major_locator(MT.NullLocator())
    fig.gca().yaxis.set_major_locator(MT.NullLocator())
    # Draw vertex labels
    fig.gca().text(-0.05, -0.05, vertexlabels[0])
    fig.gca().text(1.05, -0.05, vertexlabels[1])
    fig.gca().text(0.5, NP.sqrt(3) / 2 + 0.05, vertexlabels[2])
    # Project and draw the actual points
    projected = projectSimplex(points)
    P.scatter(projected[:,0], projected[:,1], **kwargs)              
    # Leave some buffer around the triangle for vertex labels
    fig.gca().set_xlim(-0.2, 1.2)
    fig.gca().set_ylim(-0.2, 1.2)

    return fig    

def projectSimplex(points):
    """ 
    Project probabilities on the 3-simplex to a 2D triangle
    
    N points are given as N x 3 array
    """
    # Convert points one at a time
    tripts = NP.zeros((points.shape[0],2))
    for idx in range(points.shape[0]):
        # Init to triangle centroid
        x = 1.0 / 2
        y = 1.0 / (2 * NP.sqrt(3))
        # Vector 1 - bisect out of lower left vertex 
        p1 = points[idx, 0]
        x = x - (1.0 / NP.sqrt(3)) * p1 * NP.cos(NP.pi / 6)
        y = y - (1.0 / NP.sqrt(3)) * p1 * NP.sin(NP.pi / 6)
        # Vector 2 - bisect out of lower right vertex  
        p2 = points[idx, 1]  
        x = x + (1.0 / NP.sqrt(3)) * p2 * NP.cos(NP.pi / 6)
        y = y - (1.0 / NP.sqrt(3)) * p2 * NP.sin(NP.pi / 6)        
        # Vector 3 - bisect out of top vertex
        p3 = points[idx, 2]
        y = y + (1.0 / NP.sqrt(3) * p3)
      
        tripts[idx,:] = (x,y)

    return tripts


```




```python
plotSimplex(ds, s=20);
```



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_31_0.png)


The idea behind a `Potential` is something that is not part of the likelihood, but enforces a constraint by setting the probability to 0 if the constraint is violated. We use it here to give each cluster some membership and to order the means to remove the non-identifiability problem. See below for how its used.

The sampler below has a lot of problems. 



```python
with pm.Model() as mofb:
    p = pm.Dirichlet('p', a=np.array([10., 10., 10.]), shape=3)
    # ensure all clusters have some points
    p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(p) < .1, -np.inf, 0))
    # cluster centers
    means = pm.Normal('means', mu=0, sd=10, shape=3, transform=tr.ordered,
                  testval=np.array([-1, 0, 1]))

    category = pm.Categorical('category',
                              p=p,
                              shape=data.shape[0])

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sd=1., #sds[category],
                       observed=data)


```




```python
with mofb:
    tripletrace_fullb = pm.sample(10000, nuts_kwargs=dict(target_accept=0.95))
```


    Multiprocess sampling (2 chains in 2 jobs)
    CompoundStep
    >NUTS: [means, p]
    >CategoricalGibbsMetropolis: [category]
    Sampling 2 chains: 100%|██████████| 21000/21000 [06:13<00:00, 56.23draws/s]
    There were 10 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 7 divergences after tuning. Increase `target_accept` or reparameterize.
    The number of effective samples is smaller than 10% for some parameters.




```python
trace_mofb = tripletrace_fullb[3000::5]
pm.traceplot(trace_mofb, varnames=["means", "p"], combined=True);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3604: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](mixtures_and_mcmc_files/mixtures_and_mcmc_35_1.png)




```python
pm.summary(trace_mofb)
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
      <th>category__0</th>
      <td>1.406071</td>
      <td>0.575480</td>
      <td>0.014954</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1362.583316</td>
      <td>1.000115</td>
    </tr>
    <tr>
      <th>category__1</th>
      <td>0.228571</td>
      <td>0.421610</td>
      <td>0.010938</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1306.422985</td>
      <td>1.000472</td>
    </tr>
    <tr>
      <th>category__2</th>
      <td>0.395357</td>
      <td>0.496897</td>
      <td>0.011826</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1674.615993</td>
      <td>1.000276</td>
    </tr>
    <tr>
      <th>category__3</th>
      <td>0.108929</td>
      <td>0.311550</td>
      <td>0.008695</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1307.190266</td>
      <td>1.001074</td>
    </tr>
    <tr>
      <th>category__4</th>
      <td>0.824643</td>
      <td>0.552559</td>
      <td>0.011552</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2328.351050</td>
      <td>0.999864</td>
    </tr>
    <tr>
      <th>category__5</th>
      <td>0.280714</td>
      <td>0.452516</td>
      <td>0.011192</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1662.033357</td>
      <td>1.000363</td>
    </tr>
    <tr>
      <th>category__6</th>
      <td>1.948929</td>
      <td>0.221760</td>
      <td>0.004981</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1702.198664</td>
      <td>1.002138</td>
    </tr>
    <tr>
      <th>category__7</th>
      <td>1.712143</td>
      <td>0.473583</td>
      <td>0.012564</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1256.089935</td>
      <td>0.999754</td>
    </tr>
    <tr>
      <th>category__8</th>
      <td>0.900357</td>
      <td>0.569837</td>
      <td>0.011499</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2245.021084</td>
      <td>0.999675</td>
    </tr>
    <tr>
      <th>category__9</th>
      <td>1.739286</td>
      <td>0.454218</td>
      <td>0.012043</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1421.562668</td>
      <td>1.000357</td>
    </tr>
    <tr>
      <th>category__10</th>
      <td>0.432500</td>
      <td>0.506121</td>
      <td>0.013448</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1749.523067</td>
      <td>0.999755</td>
    </tr>
    <tr>
      <th>category__11</th>
      <td>1.758214</td>
      <td>0.445337</td>
      <td>0.011746</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1291.113963</td>
      <td>1.000946</td>
    </tr>
    <tr>
      <th>category__12</th>
      <td>1.916071</td>
      <td>0.282386</td>
      <td>0.006534</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1765.304573</td>
      <td>1.000003</td>
    </tr>
    <tr>
      <th>category__13</th>
      <td>1.866429</td>
      <td>0.342284</td>
      <td>0.009551</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1477.039629</td>
      <td>0.999995</td>
    </tr>
    <tr>
      <th>category__14</th>
      <td>0.225000</td>
      <td>0.419289</td>
      <td>0.010990</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1472.910821</td>
      <td>1.001048</td>
    </tr>
    <tr>
      <th>category__15</th>
      <td>0.459286</td>
      <td>0.513864</td>
      <td>0.011694</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1619.209679</td>
      <td>1.000851</td>
    </tr>
    <tr>
      <th>category__16</th>
      <td>0.649643</td>
      <td>0.528238</td>
      <td>0.011004</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2333.805817</td>
      <td>0.999647</td>
    </tr>
    <tr>
      <th>category__17</th>
      <td>0.440000</td>
      <td>0.511273</td>
      <td>0.012340</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1803.682268</td>
      <td>1.000142</td>
    </tr>
    <tr>
      <th>category__18</th>
      <td>1.989286</td>
      <td>0.102954</td>
      <td>0.002396</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2055.892030</td>
      <td>0.999691</td>
    </tr>
    <tr>
      <th>category__19</th>
      <td>1.957143</td>
      <td>0.202535</td>
      <td>0.004574</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1996.770513</td>
      <td>1.001435</td>
    </tr>
    <tr>
      <th>category__20</th>
      <td>0.445714</td>
      <td>0.510514</td>
      <td>0.013029</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1827.687813</td>
      <td>0.999651</td>
    </tr>
    <tr>
      <th>category__21</th>
      <td>1.989643</td>
      <td>0.101242</td>
      <td>0.002039</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2505.025728</td>
      <td>0.999954</td>
    </tr>
    <tr>
      <th>category__22</th>
      <td>1.930000</td>
      <td>0.255147</td>
      <td>0.006387</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1414.514214</td>
      <td>1.000427</td>
    </tr>
    <tr>
      <th>category__23</th>
      <td>1.332143</td>
      <td>0.587704</td>
      <td>0.014147</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1797.023673</td>
      <td>1.001351</td>
    </tr>
    <tr>
      <th>category__24</th>
      <td>1.733571</td>
      <td>0.457963</td>
      <td>0.012479</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1184.444494</td>
      <td>1.000431</td>
    </tr>
    <tr>
      <th>category__25</th>
      <td>1.515714</td>
      <td>0.556555</td>
      <td>0.015364</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1331.826974</td>
      <td>0.999807</td>
    </tr>
    <tr>
      <th>category__26</th>
      <td>0.425000</td>
      <td>0.500089</td>
      <td>0.012622</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1455.116136</td>
      <td>0.999694</td>
    </tr>
    <tr>
      <th>category__27</th>
      <td>1.979643</td>
      <td>0.141219</td>
      <td>0.002913</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2378.339776</td>
      <td>1.003031</td>
    </tr>
    <tr>
      <th>category__28</th>
      <td>1.722143</td>
      <td>0.468214</td>
      <td>0.012763</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1469.880006</td>
      <td>1.000669</td>
    </tr>
    <tr>
      <th>category__29</th>
      <td>0.873214</td>
      <td>0.554202</td>
      <td>0.011719</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2572.986079</td>
      <td>1.000211</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>category__76</th>
      <td>0.473929</td>
      <td>0.516892</td>
      <td>0.012308</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1568.551705</td>
      <td>1.000044</td>
    </tr>
    <tr>
      <th>category__77</th>
      <td>1.940714</td>
      <td>0.237666</td>
      <td>0.005468</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2057.523309</td>
      <td>1.000546</td>
    </tr>
    <tr>
      <th>category__78</th>
      <td>1.552857</td>
      <td>0.545821</td>
      <td>0.014423</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1278.184527</td>
      <td>1.001185</td>
    </tr>
    <tr>
      <th>category__79</th>
      <td>1.197143</td>
      <td>0.603317</td>
      <td>0.012732</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2419.356924</td>
      <td>0.999812</td>
    </tr>
    <tr>
      <th>category__80</th>
      <td>0.119643</td>
      <td>0.325642</td>
      <td>0.008502</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1457.209338</td>
      <td>0.999644</td>
    </tr>
    <tr>
      <th>category__81</th>
      <td>0.671429</td>
      <td>0.535762</td>
      <td>0.011043</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2416.678137</td>
      <td>0.999991</td>
    </tr>
    <tr>
      <th>category__82</th>
      <td>1.471786</td>
      <td>0.560539</td>
      <td>0.014711</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1408.654254</td>
      <td>0.999643</td>
    </tr>
    <tr>
      <th>category__83</th>
      <td>0.687857</td>
      <td>0.540894</td>
      <td>0.011061</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2184.065312</td>
      <td>1.000089</td>
    </tr>
    <tr>
      <th>category__84</th>
      <td>1.832500</td>
      <td>0.379117</td>
      <td>0.009391</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1442.318862</td>
      <td>0.999651</td>
    </tr>
    <tr>
      <th>category__85</th>
      <td>0.238929</td>
      <td>0.428101</td>
      <td>0.010072</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1580.222999</td>
      <td>0.999649</td>
    </tr>
    <tr>
      <th>category__86</th>
      <td>1.001786</td>
      <td>0.574764</td>
      <td>0.011643</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2431.781980</td>
      <td>0.999884</td>
    </tr>
    <tr>
      <th>category__87</th>
      <td>0.714286</td>
      <td>0.546884</td>
      <td>0.011661</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2298.065679</td>
      <td>0.999726</td>
    </tr>
    <tr>
      <th>category__88</th>
      <td>1.627143</td>
      <td>0.519869</td>
      <td>0.014501</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1153.871436</td>
      <td>0.999673</td>
    </tr>
    <tr>
      <th>category__89</th>
      <td>1.979286</td>
      <td>0.142426</td>
      <td>0.002728</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2325.559161</td>
      <td>0.999668</td>
    </tr>
    <tr>
      <th>category__90</th>
      <td>1.987143</td>
      <td>0.112658</td>
      <td>0.002501</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1955.167689</td>
      <td>1.000005</td>
    </tr>
    <tr>
      <th>category__91</th>
      <td>1.411429</td>
      <td>0.573845</td>
      <td>0.012387</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2121.534066</td>
      <td>0.999742</td>
    </tr>
    <tr>
      <th>category__92</th>
      <td>0.287857</td>
      <td>0.454339</td>
      <td>0.011087</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1656.472820</td>
      <td>1.000951</td>
    </tr>
    <tr>
      <th>category__93</th>
      <td>0.120714</td>
      <td>0.325795</td>
      <td>0.009032</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1414.023539</td>
      <td>0.999648</td>
    </tr>
    <tr>
      <th>category__94</th>
      <td>0.817857</td>
      <td>0.560327</td>
      <td>0.011872</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1951.110342</td>
      <td>0.999649</td>
    </tr>
    <tr>
      <th>category__95</th>
      <td>1.003571</td>
      <td>0.596407</td>
      <td>0.012872</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2002.102565</td>
      <td>0.999679</td>
    </tr>
    <tr>
      <th>category__96</th>
      <td>0.076786</td>
      <td>0.266251</td>
      <td>0.007683</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1365.135665</td>
      <td>0.999645</td>
    </tr>
    <tr>
      <th>category__97</th>
      <td>1.140000</td>
      <td>0.584661</td>
      <td>0.010782</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2331.457873</td>
      <td>0.999697</td>
    </tr>
    <tr>
      <th>category__98</th>
      <td>1.993571</td>
      <td>0.079920</td>
      <td>0.001462</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2601.475216</td>
      <td>1.000921</td>
    </tr>
    <tr>
      <th>category__99</th>
      <td>0.199643</td>
      <td>0.400624</td>
      <td>0.010216</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1420.458431</td>
      <td>1.000063</td>
    </tr>
    <tr>
      <th>p__0</th>
      <td>0.360226</td>
      <td>0.081588</td>
      <td>0.003317</td>
      <td>0.204306</td>
      <td>0.519089</td>
      <td>519.487415</td>
      <td>1.002060</td>
    </tr>
    <tr>
      <th>p__1</th>
      <td>0.325225</td>
      <td>0.071761</td>
      <td>0.001919</td>
      <td>0.172866</td>
      <td>0.459570</td>
      <td>1294.128430</td>
      <td>0.999978</td>
    </tr>
    <tr>
      <th>p__2</th>
      <td>0.314550</td>
      <td>0.065550</td>
      <td>0.002548</td>
      <td>0.189302</td>
      <td>0.444908</td>
      <td>587.062612</td>
      <td>1.001335</td>
    </tr>
    <tr>
      <th>means__0</th>
      <td>-1.886959</td>
      <td>0.263363</td>
      <td>0.007660</td>
      <td>-2.388613</td>
      <td>-1.365366</td>
      <td>1295.486095</td>
      <td>1.002200</td>
    </tr>
    <tr>
      <th>means__1</th>
      <td>-0.407361</td>
      <td>0.565056</td>
      <td>0.029250</td>
      <td>-1.451171</td>
      <td>0.683029</td>
      <td>286.606094</td>
      <td>1.002899</td>
    </tr>
    <tr>
      <th>means__2</th>
      <td>1.948775</td>
      <td>0.303411</td>
      <td>0.009837</td>
      <td>1.418676</td>
      <td>2.599857</td>
      <td>791.034984</td>
      <td>1.001906</td>
    </tr>
  </tbody>
</table>
<p>106 rows × 7 columns</p>
</div>



### Making Problems go away

A lot will go away when identifiability improves through separated gaussians. But that changes the data. If we want any further improvement on this data, we are going to have to stop sampling so many discrete categoricals. And for that we will need a marginalization trick.
