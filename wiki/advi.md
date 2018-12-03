---
title: ADVI
shorttitle: advi
notebook: advi.ipynb
noline: 1
summary: ""
keywords: ['variational inference', 'elbo', 'kl-divergence', 'normal distribution', 'mean-field approximation', 'advi', 'optimization', 'sgd', 'minibatch sgd']
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


    //anaconda/envs/py3l/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


## From CAVI to Stochastic CAVI to ADVI

One of the challenges of any posterior inference problem is the ability to scale. While VI is faster than the traditional MCMC, the CAVI algorithm described above fundamentally doesn't scale as it needs to run through the **entire dataset** each iteration. An alternative that is sometimes recommended is the Stochastic CAVI that uses gradient-based optimization. Using this approach, the algorithm only requires a subsample of the data set to iteratively optimize local and global parameters of the model. 

Stochastic CAVI is specifically used for conditionally conjugate models, but the ideas from it are applicable outside: the use of gradient (for gradient ascent) and the use of SGD style techniques: minibatch or fully stochastic.

Finally, we have seen how to implement SGD in Theano, and how pymc3 uses automatic differentiation under the hood to provide gradients for its NUTS sampler. This idea is used to replace CAVI with an automatically-calculated gradient-ascent algorithm, with stochastic updates that allow us to scale by not requiring the use of the complete dataset at each iteration.

## ADVI in pymc3: approximating a gaussian



```python
data = np.random.randn(100)
```




```python
with pm.Model() as model: 
    mu = pm.Normal('mu', mu=0, sd=1, testval=0)
    sd = pm.HalfNormal('sd', sd=1)
    n = pm.Normal('n', mu=mu, sd=sd, observed=data)
```




```python
advifit = pm.ADVI( model=model)
```




```python
advifit.fit(n=50000)
```


    Average Loss = 144.82: 100%|██████████| 50000/50000 [00:35<00:00, 1419.68it/s]
    Finished [100%]: Average Loss = 144.82





    <pymc3.variational.approximations.MeanField at 0x10d2737f0>





```python
elbo = -advifit.hist
```




```python
plt.plot(elbo[::10]);
```



![png](advi_files/advi_9_0.png)




```python
advifit.approx.shared_params, type(advifit.approx.shared_params)
```





    ({'mu': mu, 'rho': rho}, dict)





```python
advifit.approx.mean.eval(), advifit.approx.std.eval()
```





    (array([ 0.13169556, -0.0110224 ]), array([0.10678595, 0.07751521]))





```python
m = advifit.approx.mean.eval()[0]
s = advifit.approx.std.eval()[1]
m,s
```





    (0.13169556333305704, 0.07751520792262655)





```python
sig = np.exp(advifit.approx.mean.eval()[1])
sig
```





    0.9890381286804355





```python
trace = advifit.approx.sample(10000)
```




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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>0.131541</td>
      <td>0.106210</td>
      <td>0.000990</td>
      <td>-0.073015</td>
      <td>0.338539</td>
    </tr>
    <tr>
      <th>sd</th>
      <td>0.990258</td>
      <td>0.077982</td>
      <td>0.000825</td>
      <td>0.834474</td>
      <td>1.141754</td>
    </tr>
  </tbody>
</table>
</div>





```python
pm.traceplot(trace)
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3604: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x12a98dbe0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x12a90d080>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x12a92e6a0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x12a956cc0>]],
          dtype=object)




![png](advi_files/advi_16_2.png)




```python
trace['mu'].mean(), trace['sd'].mean()
```





    (0.13154130334352263, 0.9902577471791679)





```python
pred = pm.sample_ppc(trace, 10000, model=model)
```


    100%|██████████| 10000/10000 [00:06<00:00, 1563.56it/s]




```python
pred['n'].shape
```





    (5000, 100)





```python
with model:
    trace_nuts = pm.sample(10000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sd, mu]
    Sampling 2 chains: 100%|██████████| 22000/22000 [00:10<00:00, 2132.93draws/s]




```python
pm.summary(trace_nuts)
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
      <td>0.134987</td>
      <td>0.098675</td>
      <td>0.000987</td>
      <td>-0.059616</td>
      <td>0.326348</td>
      <td>10983.688378</td>
      <td>0.999913</td>
    </tr>
    <tr>
      <th>sd</th>
      <td>0.993436</td>
      <td>0.070409</td>
      <td>0.000609</td>
      <td>0.858776</td>
      <td>1.133757</td>
      <td>11117.739069</td>
      <td>0.999915</td>
    </tr>
  </tbody>
</table>
</div>



### Comparing the mu parameter



```python
sns.kdeplot(trace_nuts['mu'], label='NUTS')
sns.kdeplot(trace['mu'], label='ADVI')
plt.legend();
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](advi_files/advi_23_1.png)




```python
sns.kdeplot(trace_nuts['sd'], label='NUTS')
sns.kdeplot(trace['sd'], label='ADVI')
plt.legend();
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](advi_files/advi_24_1.png)


### Comparing the data to the posterior-predictive



```python
pred['n'][:,0].shape
```





    (10000,)





```python
sns.distplot(data)
sns.kdeplot(pred['n'][:,0])
sns.kdeplot(pred['n'][:,1])
sns.kdeplot(pred['n'][:,50])
sns.kdeplot(pred['n'][:,99])
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")





    <matplotlib.axes._subplots.AxesSubplot at 0x12debe0f0>




![png](advi_files/advi_27_2.png)


## ADVI: what does it do?

Remember that in Variational inference, we decompose an aprroximate posterior in the mean-field approximation into a product of per-latent-variable posteriors. The approximate posterior is chosen from a pre-specified family of distributions to "variationally" minimize the KL-divergence (equivalently to maximize the ELBO) between itself and the true posterior.

$$ ELBO(q) = E_q[(log(p(z,x))] - E_q[log(q(z))] $$ 


This means that the ELBO must be painstakingly calculated and optimized with custom CAVI updates for each new model, and an approximating family chosen. If you choose to use a gradient based optimizer then you must supply gradients.

From the ADVI paper:

>ADVI solves this problem automatically. The user specifies the model, expressed as a program, and ADVI automatically generates a corresponding variational algorithm. The idea is to first automatically transform the inference problem into a common space and then to solve the variational optimization. Solving the problem in this common space solves variational inference for all models in a large class. 

Here is what ADVI does for us:

(1) The model undergoes transformations such that the latent parameters are transformed to representations where the 'new" parameters are unconstrained on the real-line. Specifically the joint $p(x, \theta)$ transforms to $p(x, \eta)$ where $\eta$ is unconstrained. We then define the approximating density $q$ and the posterior in terms of these transformed variable and minimize the KL-divergence between the transformed densities. This is done for *ALL* latent variables so that all of them are now defined on the same space. As a result we can use the same variational family for ALL parameters, and indeed for ALL models, as every parameter for every model is now defined on all of R. It should be clear from this that Discrete parameters must be marginalized out.

![](images/TransformtoR.png)

Optimizing the KL-divergence implicitly assumes that the support of the approximating density lies within the support of the posterior. These transformations make sure that this is the case

(2) Ok, so now we must maximize our suitably transformed ELBO (the log full-data posterior will gain an additional term which is the determinant of the log of the Jacobian). Remember in variational inference that we are optimizing an expectation value with respect to the transformed approximate posterior. This posterior contains our transformed latent parameters so the gradient of this expectation is not simply defined.

What are we to do?

(3) We first choose as our family of approximating densities mean-field normal distributions. We'll tranform the always positive $\sigma$ params by simply taking their logs. 

The choice of Gaussians may make you think that we are doing a laplace (taylor series) approximation around the posterior mode, which is another method for approximate inference. This is not what we are doing here.

We still havent solved the problem of taking the gradient. Basically what we want to do is to push the gradient inside the expectation. For this, the distribution we use to calculate the expectation must be free of parameters we might compute the gradient with respect to.

So we indulge ourselves another transformation, which takes the approximate 1-D gaussian $q$ and standardizes it. The determinant of the jacobian of this transform is 1. This is the REPARAMETERIZATION TRICK and variants are available for other approximating families.

As a result of this, we can now compute the integral as a monte-carlo estimate over a standard Gaussian--superfast, and we can move the gradient inside the expectation (integral) to boot. This means that our job now becomes the calculation of the gradient of the full-data joint-distribution.

(4) We can replace full $x$ data by just one point (SGD) or mini-batch (some-$x$) and thus use noisy gradients to optimize the variational distribution. An
adaptively tuned step-size is used to provide good convergence.

## Demonstrating ADVI in pymc3

We wish to sample a 2D Posterior which looks something like below. Here the x and y axes are parameters.



```python
cov=np.array([[1,0.8],[0.8,1]])
data = np.random.multivariate_normal([0,0], cov, size=1000)
sns.kdeplot(data, alpha=0.4);
plt.scatter(data[:,0], data[:,1], s=10, alpha=0.2)
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    <matplotlib.collections.PathCollection at 0x12dede0b8>




![png](advi_files/advi_30_2.png)




```python
np.std(data[:,0]),np.std(data[:,1])
```





    (1.0213287306989942, 1.0081473951333644)



Ok, so we just set up a simple sampler with no observed data



```python
import theano.tensor as tt
cov=np.array([[0,0.8],[0.8,0]], dtype=np.float64)
with pm.Model() as mdensity:
    density = pm.MvNormal('density', mu=[0,0], cov=tt.fill_diagonal(cov,1), shape=2)

```


We try and retrieve the posterior by sampling



```python
with mdensity:
    mdtrace=pm.sample(10000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [density]
    Sampling 2 chains: 100%|██████████| 21000/21000 [00:21<00:00, 954.68draws/s] 
    The number of effective samples is smaller than 25% for some parameters.




```python
pm.traceplot(mdtrace);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3604: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](advi_files/advi_36_1.png)


We do a pretty good job:



```python
plt.scatter(mdtrace['density'][:,0], mdtrace['density'][:,1], s=5, alpha=0.1)
```





    <matplotlib.collections.PathCollection at 0x12e3b3438>




![png](advi_files/advi_38_1.png)


But when we sample using ADVI, the mean-field approximation means that we lose our correlation:



```python
mdvar = pm.ADVI(model=mdensity)
mdvar.fit(n=40000)
```


    Average Loss = 0.51667: 100%|██████████| 40000/40000 [00:41<00:00, 959.09it/s]
    Finished [100%]: Average Loss = 0.51766





    <pymc3.variational.approximations.MeanField at 0x12db6d8d0>





```python
plt.plot(-mdvar.hist[::10])
```





    [<matplotlib.lines.Line2D at 0x12f276208>]




![png](advi_files/advi_41_1.png)




```python
samps=mdvar.approx.sample(5000)
```




```python
plt.scatter(samps['density'][:,0], samps['density'][:,1], s=5, alpha=0.3)
```





    <matplotlib.collections.PathCollection at 0x12f60b6a0>




![png](advi_files/advi_43_1.png)


A full rank fit also models the covariance parameters, and thus restores our correlation at the cost of more variational parameters to fit...



```python
mdvar_fr = pm.FullRankADVI(model=mdensity)
mdvar_fr.fit(n=40000)
```


      0%|          | 0/40000 [00:00<?, ?it/s]//anaconda/envs/py3l/lib/python3.6/site-packages/theano/tensor/subtensor.py:2320: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      out[0][inputs[2:]] = inputs[1]
    //anaconda/envs/py3l/lib/python3.6/site-packages/theano/tensor/basic.py:6592: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    //anaconda/envs/py3l/lib/python3.6/site-packages/theano/tensor/subtensor.py:2190: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      rval = inputs[0].__getitem__(inputs[1:])
    Average Loss = 0.015506: 100%|██████████| 40000/40000 [00:59<00:00, 666.68it/s]   
    Finished [100%]: Average Loss = 0.01521





    <pymc3.variational.approximations.FullRank at 0x12f6047f0>





```python
plt.plot(-mdvar_fr.hist[::10])
```





    [<matplotlib.lines.Line2D at 0x12f306b38>]




![png](advi_files/advi_46_1.png)




```python
samps2=mdvar_fr.approx.sample(5000)
plt.scatter(samps2['density'][:,0], samps2['density'][:,1], s=5, alpha=0.3)
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/theano/tensor/subtensor.py:2320: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      out[0][inputs[2:]] = inputs[1]





    <matplotlib.collections.PathCollection at 0x12d59e8d0>




![png](advi_files/advi_47_2.png)

