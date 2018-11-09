---
title: Tumors inference with PyMC3.
shorttitle: tumor_pymc3
notebook: tumor_pymc3.ipynb
noline: 1
summary: ""
keywords: ['beta-binomial', 'hierarchical', 'pymc3', 'posterior predictive']
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
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
import pandas as pd

import pymc3 as pm
from pymc3 import Model, Normal, HalfNormal
import time
```


# Tumors in rats

Let us try to do full Bayesian inference with PyMC3 for the rat tumor example that we have solved using Gibbs sampling in a previous lab. Remember that the goal is to estimate $\theta$, the probability of developing a tumor in a population of female rats that have not received treatement. Data of a particular experiment shows that 4 out of 14 rats develop the tumor. But we also have historical literature data for other 70 experiments, which give estimates $\theta_i$. 

For convenience, we adopt a prior for $\theta_i$ from the conjugate *Beta* family: $\theta_i \sim Beta(\alpha, \beta)$. If we had an expert telling us the values for $\alpha$ and $\beta$, we could just calculate the posterior using the conjugate rules. But we do not usually have that. We have, instead, the historical data, that we can use to perform inference for $\theta_i$, the tumor probability in the current experiment

It is natural the model the number $y_i$ of tumors for *each* experiment performed on a total of $n_i$ rats as a binomial:

$$
p(y_i \vert \theta_i; n_i) =  Binom(n_i, y_i, \theta_i)
$$

We can now write a joint posterior distribution for the $\theta$s, $\alpha$ and $\beta$, assuming partial pooling (i.e., hierarchical Bayes), where the $\theta_i$ is assumed to be different for each experiment, but all drawn from the same *Beta* distribution with parameteres $\alpha$ and $\beta$:

$$p( \theta_i, \alpha, \beta  \vert  y_i, n_i) \propto p(\alpha, \beta) \, p(\theta_i  \vert  \alpha, \beta) \, p(y_i  \vert  \theta_i, n_i,\alpha, \beta)$$
or for the whole dataset:
$$ p( \Theta, \alpha, \beta  \vert  Y, \{n_i\}) \propto p(\alpha, \beta) \prod_{i=1}^{70} Beta(\theta_i, \alpha, \beta) \prod_{i=1}^{70} Binom(n_i, y_i, \theta_i)$$

So we only need to figure out the prior for the hyperparameters: $p(\alpha,\beta)$. We have shown that it is convenient to use uniform priors on the alernative variables $\mu$ (the mean of the beta distribution) and $\nu$:

$$\mu = \frac{\alpha}{\alpha+\beta}$$
$$\nu = (\alpha+\beta)^{-1/2}$$

which yiels a prior for $\alpha$ and $\beta$ of the form:

$$p(\alpha,\beta) \sim (\alpha+\beta)^{-5/2}$$

Let us firs load the data:



```python
tumordata="""0 20 
0 20 
0 20 
0 20 
0 20 
0 20 
0 20 
0 19 
0 19 
0 19 
0 19 
0 18 
0 18 
0 17 
1 20 
1 20 
1 20 
1 20 
1 19 
1 19 
1 18 
1 18 
3 27 
2 25 
2 24 
2 23 
2 20 
2 20 
2 20 
2 20 
2 20 
2 20 
1 10 
5 49 
2 19 
5 46 
2 17 
7 49 
7 47 
3 20 
3 20 
2 13 
9 48 
10 50 
4 20 
4 20 
4 20 
4 20 
4 20 
4 20 
4 20 
10 48 
4 19 
4 19 
4 19 
5 22 
11 46 
12 49 
5 20 
5 20 
6 23 
5 19 
6 22 
6 20 
6 20 
6 20 
16 52 
15 46 
15 47 
9 24 
"""
```


And now let us create two arrays, one for the observed tumors and one for the total number of rats in each of the 70 experiments.



```python
tumortuples=[e.strip().split() for e in tumordata.split("\n")]
tumory=np.array([np.int(e[0].strip()) for e in tumortuples if len(e) > 0])
tumorn=np.array([np.int(e[1].strip()) for e in tumortuples if len(e) > 0])
print(tumory, tumorn)
print(np.shape(tumory))
```


    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  3  2  2
      2  2  2  2  2  2  2  1  5  2  5  2  7  7  3  3  2  9 10  4  4  4  4  4  4
      4 10  4  4  4  5 11 12  5  5  6  5  6  6  6  6 16 15 15  9] [20 20 20 20 20 20 20 19 19 19 19 18 18 17 20 20 20 20 19 19 18 18 27 25 24
     23 20 20 20 20 20 20 10 49 19 46 17 49 47 20 20 13 48 50 20 20 20 20 20 20
     20 48 19 19 19 22 46 49 20 20 23 19 22 20 20 20 52 46 47 24]
    (70,)


Just to have some intuition let us get the naive probabilities ($y_i/n_i$) of developing a tumor for each of the 70 experiments, and let's print the mean and the standard deviation:



```python
tumor_rat = [float(e[0])/float(e[1]) for e in zip(tumory, tumorn)]
#print (tumor_rat)
tmean = np.mean(tumor_rat)
tvar = np.var(tumor_rat)
print(tmean, tvar)
```


    0.13600653889 0.0105576406236


We now write the model in PyMC3. PyMC3 will by default use the NUTS algorithm, unless told otherwise. Once the step method has been chosen, PyMC3 will also optimize for the parameters of the method (step sizes, proposal distributions, scaling, starting values for the parameters, etc), but we can also manually set those. Here we use both NUTS and Metropolis to perform the sampling. First let us load the relevant probability distributions

### Setting up the PyMC3 model

Now let us set up the model. Note the simplification with respect to the Gibbs sampler we have used earlier. Because PyMC3 takes care of refining the parameters of the selected step method, or uses gradient-based methods for the sampling, it does not require us to specify the conditionals distributions for $\alpha$ and $\beta$. We only need to specify the priors for $\mu$ and $\nu$, and then write expressions for $\alpha$ and $\beta$ as a function of $\mu$ and $\nu$. Note that we use the ```pm.Deterministic``` function to define $\alpha$ and $\beta$ and give them proper names. Without a name, these variables will not be included in the trace.



```python
# pymc3
from pymc3 import Uniform, Normal, HalfNormal, HalfCauchy, Binomial, Beta, sample, Model # Import relevant distributions

N = tumorn.shape[0]

with Model() as tumor_model:

    # Uniform priors on the mean and variance of the Beta distributions
    mu = Uniform("mu",0.00001,1.)
    nu = Uniform("nu",0.00001,1.)
    #nu = HalfCauchy("nu", beta = 1.)

    # Calculate hyperparameters alpha and beta as a function of mu and nu
    alpha = pm.Deterministic('alpha', mu/(nu*nu))
    beta = pm.Deterministic('beta', (1.-mu)/(nu*nu))
    
    # Priors for each theta
    thetas = Beta('theta', alpha, beta, shape=N)
    
    # Data likelihood
    obs_deaths = Binomial('obs_deaths', n=tumorn, p=thetas, observed=tumory)
```




```python
from pymc3 import find_MAP

with tumor_model:
    # instantiate sampler
    step = pm.Metropolis()
    
    # draw 2000 posterior samples
    tumor_trace = pm.sample(200000, step=step)
```


    Multiprocess sampling (2 chains in 2 jobs)
    CompoundStep
    >Metropolis: [theta]
    >Metropolis: [nu]
    >Metropolis: [mu]
    Sampling 2 chains: 100%|██████████| 401000/401000 [04:26<00:00, 1505.99draws/s]
    The gelman-rubin statistic is larger than 1.2 for some parameters.
    The estimated number of effective samples is smaller than 200 for some parameters.




```python
mtrace = tumor_trace[100000::25]
pm.summary(mtrace)
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
      <td>5.000669e-01</td>
      <td>6.287878e-05</td>
      <td>0.000006</td>
      <td>4.999661e-01</td>
      <td>5.001860e-01</td>
      <td>3.234569</td>
      <td>1.360175</td>
    </tr>
    <tr>
      <th>nu</th>
      <td>3.611176e-04</td>
      <td>8.753924e-05</td>
      <td>0.000008</td>
      <td>2.014466e-04</td>
      <td>5.066631e-04</td>
      <td>1.942442</td>
      <td>1.652039</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>4.727586e+06</td>
      <td>2.801670e+06</td>
      <td>261826.826631</td>
      <td>1.566256e+06</td>
      <td>1.042954e+07</td>
      <td>2.390785</td>
      <td>1.526307</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>4.726268e+06</td>
      <td>2.800994e+06</td>
      <td>261764.660399</td>
      <td>1.566473e+06</td>
      <td>1.042653e+07</td>
      <td>2.392376</td>
      <td>1.525924</td>
    </tr>
    <tr>
      <th>theta__0</th>
      <td>5.000391e-01</td>
      <td>1.800322e-04</td>
      <td>0.000018</td>
      <td>4.997121e-01</td>
      <td>5.003853e-01</td>
      <td>9.831093</td>
      <td>0.999910</td>
    </tr>
    <tr>
      <th>theta__1</th>
      <td>5.001020e-01</td>
      <td>1.329097e-04</td>
      <td>0.000013</td>
      <td>4.998569e-01</td>
      <td>5.003674e-01</td>
      <td>26.436879</td>
      <td>1.099340</td>
    </tr>
    <tr>
      <th>theta__2</th>
      <td>5.001533e-01</td>
      <td>1.659936e-04</td>
      <td>0.000016</td>
      <td>4.998351e-01</td>
      <td>5.004404e-01</td>
      <td>12.091202</td>
      <td>1.007664</td>
    </tr>
    <tr>
      <th>theta__3</th>
      <td>4.999507e-01</td>
      <td>2.331874e-04</td>
      <td>0.000023</td>
      <td>4.994840e-01</td>
      <td>5.003726e-01</td>
      <td>3.010059</td>
      <td>1.402556</td>
    </tr>
    <tr>
      <th>theta__4</th>
      <td>5.002441e-01</td>
      <td>1.948635e-04</td>
      <td>0.000019</td>
      <td>4.999769e-01</td>
      <td>5.006585e-01</td>
      <td>7.904370</td>
      <td>1.194238</td>
    </tr>
    <tr>
      <th>theta__5</th>
      <td>5.000091e-01</td>
      <td>2.041813e-04</td>
      <td>0.000020</td>
      <td>4.996871e-01</td>
      <td>5.003607e-01</td>
      <td>3.004450</td>
      <td>1.384933</td>
    </tr>
    <tr>
      <th>theta__6</th>
      <td>4.999200e-01</td>
      <td>1.604717e-04</td>
      <td>0.000015</td>
      <td>4.996485e-01</td>
      <td>5.002131e-01</td>
      <td>2.712364</td>
      <td>1.401897</td>
    </tr>
    <tr>
      <th>theta__7</th>
      <td>5.001111e-01</td>
      <td>2.240313e-04</td>
      <td>0.000022</td>
      <td>4.995906e-01</td>
      <td>5.004315e-01</td>
      <td>5.876380</td>
      <td>1.179545</td>
    </tr>
    <tr>
      <th>theta__8</th>
      <td>5.001553e-01</td>
      <td>1.604234e-04</td>
      <td>0.000015</td>
      <td>4.999116e-01</td>
      <td>5.005172e-01</td>
      <td>11.259525</td>
      <td>1.049591</td>
    </tr>
    <tr>
      <th>theta__9</th>
      <td>5.000615e-01</td>
      <td>1.719356e-04</td>
      <td>0.000017</td>
      <td>4.996643e-01</td>
      <td>5.003404e-01</td>
      <td>10.994698</td>
      <td>1.065421</td>
    </tr>
    <tr>
      <th>theta__10</th>
      <td>5.000502e-01</td>
      <td>1.883012e-04</td>
      <td>0.000018</td>
      <td>4.997173e-01</td>
      <td>5.004508e-01</td>
      <td>19.463268</td>
      <td>1.001215</td>
    </tr>
    <tr>
      <th>theta__11</th>
      <td>5.000578e-01</td>
      <td>1.575548e-04</td>
      <td>0.000015</td>
      <td>4.997069e-01</td>
      <td>5.003717e-01</td>
      <td>11.026666</td>
      <td>1.058203</td>
    </tr>
    <tr>
      <th>theta__12</th>
      <td>5.000756e-01</td>
      <td>1.698583e-04</td>
      <td>0.000016</td>
      <td>4.997752e-01</td>
      <td>5.003683e-01</td>
      <td>3.960245</td>
      <td>1.319004</td>
    </tr>
    <tr>
      <th>theta__13</th>
      <td>5.000070e-01</td>
      <td>1.786940e-04</td>
      <td>0.000017</td>
      <td>4.996324e-01</td>
      <td>5.003333e-01</td>
      <td>7.675879</td>
      <td>1.120249</td>
    </tr>
    <tr>
      <th>theta__14</th>
      <td>5.001392e-01</td>
      <td>2.366688e-04</td>
      <td>0.000023</td>
      <td>4.997431e-01</td>
      <td>5.005740e-01</td>
      <td>6.399215</td>
      <td>1.087414</td>
    </tr>
    <tr>
      <th>theta__15</th>
      <td>4.999389e-01</td>
      <td>1.429021e-04</td>
      <td>0.000014</td>
      <td>4.996461e-01</td>
      <td>5.002203e-01</td>
      <td>10.493800</td>
      <td>1.191283</td>
    </tr>
    <tr>
      <th>theta__16</th>
      <td>5.000264e-01</td>
      <td>1.527704e-04</td>
      <td>0.000015</td>
      <td>4.997020e-01</td>
      <td>5.002922e-01</td>
      <td>8.520209</td>
      <td>1.152993</td>
    </tr>
    <tr>
      <th>theta__17</th>
      <td>4.999154e-01</td>
      <td>1.854030e-04</td>
      <td>0.000018</td>
      <td>4.995322e-01</td>
      <td>5.002151e-01</td>
      <td>3.399206</td>
      <td>1.321350</td>
    </tr>
    <tr>
      <th>theta__18</th>
      <td>5.000417e-01</td>
      <td>1.937594e-04</td>
      <td>0.000019</td>
      <td>4.996465e-01</td>
      <td>5.004034e-01</td>
      <td>10.537999</td>
      <td>1.006096</td>
    </tr>
    <tr>
      <th>theta__19</th>
      <td>5.001232e-01</td>
      <td>1.455825e-04</td>
      <td>0.000014</td>
      <td>4.998634e-01</td>
      <td>5.004082e-01</td>
      <td>7.147684</td>
      <td>1.126504</td>
    </tr>
    <tr>
      <th>theta__20</th>
      <td>5.000995e-01</td>
      <td>1.986575e-04</td>
      <td>0.000019</td>
      <td>4.996207e-01</td>
      <td>5.004306e-01</td>
      <td>8.625162</td>
      <td>1.128328</td>
    </tr>
    <tr>
      <th>theta__21</th>
      <td>5.001336e-01</td>
      <td>1.439817e-04</td>
      <td>0.000014</td>
      <td>4.998322e-01</td>
      <td>5.003975e-01</td>
      <td>17.871090</td>
      <td>1.044630</td>
    </tr>
    <tr>
      <th>theta__22</th>
      <td>5.000656e-01</td>
      <td>1.165241e-04</td>
      <td>0.000011</td>
      <td>4.998488e-01</td>
      <td>5.002999e-01</td>
      <td>29.477362</td>
      <td>1.066967</td>
    </tr>
    <tr>
      <th>theta__23</th>
      <td>5.000234e-01</td>
      <td>1.151251e-04</td>
      <td>0.000011</td>
      <td>4.998260e-01</td>
      <td>5.002795e-01</td>
      <td>29.281858</td>
      <td>1.000100</td>
    </tr>
    <tr>
      <th>theta__24</th>
      <td>5.001275e-01</td>
      <td>1.492349e-04</td>
      <td>0.000014</td>
      <td>4.998598e-01</td>
      <td>5.004422e-01</td>
      <td>18.561020</td>
      <td>1.000380</td>
    </tr>
    <tr>
      <th>theta__25</th>
      <td>5.000606e-01</td>
      <td>2.124652e-04</td>
      <td>0.000021</td>
      <td>4.997016e-01</td>
      <td>5.005374e-01</td>
      <td>6.958177</td>
      <td>1.007100</td>
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
      <th>theta__40</th>
      <td>5.001775e-01</td>
      <td>1.871297e-04</td>
      <td>0.000018</td>
      <td>4.998663e-01</td>
      <td>5.005601e-01</td>
      <td>15.097984</td>
      <td>1.000517</td>
    </tr>
    <tr>
      <th>theta__41</th>
      <td>5.000726e-01</td>
      <td>2.399461e-04</td>
      <td>0.000024</td>
      <td>4.994396e-01</td>
      <td>5.004034e-01</td>
      <td>5.370408</td>
      <td>1.161731</td>
    </tr>
    <tr>
      <th>theta__42</th>
      <td>5.000981e-01</td>
      <td>1.682222e-04</td>
      <td>0.000016</td>
      <td>4.997105e-01</td>
      <td>5.003691e-01</td>
      <td>9.326048</td>
      <td>1.066904</td>
    </tr>
    <tr>
      <th>theta__43</th>
      <td>4.999549e-01</td>
      <td>2.211898e-04</td>
      <td>0.000022</td>
      <td>4.995577e-01</td>
      <td>5.003209e-01</td>
      <td>2.878013</td>
      <td>1.395574</td>
    </tr>
    <tr>
      <th>theta__44</th>
      <td>5.000656e-01</td>
      <td>1.583165e-04</td>
      <td>0.000015</td>
      <td>4.998096e-01</td>
      <td>5.003818e-01</td>
      <td>16.496075</td>
      <td>1.002571</td>
    </tr>
    <tr>
      <th>theta__45</th>
      <td>4.999942e-01</td>
      <td>1.608783e-04</td>
      <td>0.000016</td>
      <td>4.996740e-01</td>
      <td>5.002416e-01</td>
      <td>13.352827</td>
      <td>1.106972</td>
    </tr>
    <tr>
      <th>theta__46</th>
      <td>5.001250e-01</td>
      <td>1.663445e-04</td>
      <td>0.000016</td>
      <td>4.998137e-01</td>
      <td>5.005003e-01</td>
      <td>19.084825</td>
      <td>1.051638</td>
    </tr>
    <tr>
      <th>theta__47</th>
      <td>5.000473e-01</td>
      <td>1.914011e-04</td>
      <td>0.000019</td>
      <td>4.996274e-01</td>
      <td>5.003780e-01</td>
      <td>9.577494</td>
      <td>1.078678</td>
    </tr>
    <tr>
      <th>theta__48</th>
      <td>5.000222e-01</td>
      <td>1.519040e-04</td>
      <td>0.000015</td>
      <td>4.996622e-01</td>
      <td>5.003131e-01</td>
      <td>17.045429</td>
      <td>1.075165</td>
    </tr>
    <tr>
      <th>theta__49</th>
      <td>4.999924e-01</td>
      <td>1.781156e-04</td>
      <td>0.000017</td>
      <td>4.996150e-01</td>
      <td>5.003196e-01</td>
      <td>9.367578</td>
      <td>1.013653</td>
    </tr>
    <tr>
      <th>theta__50</th>
      <td>5.001496e-01</td>
      <td>1.675654e-04</td>
      <td>0.000016</td>
      <td>4.998011e-01</td>
      <td>5.004607e-01</td>
      <td>19.557957</td>
      <td>1.006750</td>
    </tr>
    <tr>
      <th>theta__51</th>
      <td>5.001616e-01</td>
      <td>2.013671e-04</td>
      <td>0.000020</td>
      <td>4.998633e-01</td>
      <td>5.006041e-01</td>
      <td>5.466876</td>
      <td>1.193948</td>
    </tr>
    <tr>
      <th>theta__52</th>
      <td>5.001461e-01</td>
      <td>1.623611e-04</td>
      <td>0.000016</td>
      <td>4.998244e-01</td>
      <td>5.005076e-01</td>
      <td>9.804783</td>
      <td>1.007959</td>
    </tr>
    <tr>
      <th>theta__53</th>
      <td>5.000805e-01</td>
      <td>1.783615e-04</td>
      <td>0.000017</td>
      <td>4.997702e-01</td>
      <td>5.004180e-01</td>
      <td>8.014413</td>
      <td>1.182770</td>
    </tr>
    <tr>
      <th>theta__54</th>
      <td>5.000096e-01</td>
      <td>1.369743e-04</td>
      <td>0.000013</td>
      <td>4.997913e-01</td>
      <td>5.002760e-01</td>
      <td>9.935164</td>
      <td>1.032204</td>
    </tr>
    <tr>
      <th>theta__55</th>
      <td>5.001198e-01</td>
      <td>1.293484e-04</td>
      <td>0.000012</td>
      <td>4.998772e-01</td>
      <td>5.003745e-01</td>
      <td>22.275240</td>
      <td>1.002406</td>
    </tr>
    <tr>
      <th>theta__56</th>
      <td>5.000050e-01</td>
      <td>1.478372e-04</td>
      <td>0.000014</td>
      <td>4.996954e-01</td>
      <td>5.002809e-01</td>
      <td>19.140730</td>
      <td>1.019614</td>
    </tr>
    <tr>
      <th>theta__57</th>
      <td>5.000829e-01</td>
      <td>1.775987e-04</td>
      <td>0.000017</td>
      <td>4.997650e-01</td>
      <td>5.004582e-01</td>
      <td>5.644519</td>
      <td>1.155416</td>
    </tr>
    <tr>
      <th>theta__58</th>
      <td>5.000725e-01</td>
      <td>1.508687e-04</td>
      <td>0.000014</td>
      <td>4.997883e-01</td>
      <td>5.003515e-01</td>
      <td>7.784004</td>
      <td>1.199499</td>
    </tr>
    <tr>
      <th>theta__59</th>
      <td>5.000540e-01</td>
      <td>1.414569e-04</td>
      <td>0.000013</td>
      <td>4.997392e-01</td>
      <td>5.002932e-01</td>
      <td>5.673277</td>
      <td>1.243000</td>
    </tr>
    <tr>
      <th>theta__60</th>
      <td>4.999827e-01</td>
      <td>1.757847e-04</td>
      <td>0.000017</td>
      <td>4.995909e-01</td>
      <td>5.002680e-01</td>
      <td>3.259067</td>
      <td>1.349769</td>
    </tr>
    <tr>
      <th>theta__61</th>
      <td>5.000376e-01</td>
      <td>1.778730e-04</td>
      <td>0.000017</td>
      <td>4.996996e-01</td>
      <td>5.003708e-01</td>
      <td>13.820166</td>
      <td>1.012429</td>
    </tr>
    <tr>
      <th>theta__62</th>
      <td>5.000823e-01</td>
      <td>1.784750e-04</td>
      <td>0.000017</td>
      <td>4.997655e-01</td>
      <td>5.004359e-01</td>
      <td>11.812727</td>
      <td>1.032088</td>
    </tr>
    <tr>
      <th>theta__63</th>
      <td>4.999897e-01</td>
      <td>2.581503e-04</td>
      <td>0.000025</td>
      <td>4.994230e-01</td>
      <td>5.004484e-01</td>
      <td>2.906859</td>
      <td>1.420999</td>
    </tr>
    <tr>
      <th>theta__64</th>
      <td>5.000612e-01</td>
      <td>1.383971e-04</td>
      <td>0.000013</td>
      <td>4.998412e-01</td>
      <td>5.003580e-01</td>
      <td>15.026356</td>
      <td>1.002756</td>
    </tr>
    <tr>
      <th>theta__65</th>
      <td>5.000979e-01</td>
      <td>2.324506e-04</td>
      <td>0.000023</td>
      <td>4.997207e-01</td>
      <td>5.006178e-01</td>
      <td>2.778490</td>
      <td>1.432332</td>
    </tr>
    <tr>
      <th>theta__66</th>
      <td>5.001369e-01</td>
      <td>1.670882e-04</td>
      <td>0.000016</td>
      <td>4.998011e-01</td>
      <td>5.004797e-01</td>
      <td>8.248015</td>
      <td>1.008637</td>
    </tr>
    <tr>
      <th>theta__67</th>
      <td>5.000645e-01</td>
      <td>2.151833e-04</td>
      <td>0.000021</td>
      <td>4.996766e-01</td>
      <td>5.005471e-01</td>
      <td>3.908657</td>
      <td>1.318731</td>
    </tr>
    <tr>
      <th>theta__68</th>
      <td>5.000867e-01</td>
      <td>1.333601e-04</td>
      <td>0.000013</td>
      <td>4.998251e-01</td>
      <td>5.003215e-01</td>
      <td>10.326565</td>
      <td>1.044398</td>
    </tr>
    <tr>
      <th>theta__69</th>
      <td>5.001163e-01</td>
      <td>1.660164e-04</td>
      <td>0.000016</td>
      <td>4.997761e-01</td>
      <td>5.004282e-01</td>
      <td>6.906468</td>
      <td>1.179012</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 7 columns</p>
</div>





```python
from pymc3 import traceplot

traceplot(mtrace, varnames=['alpha','beta','theta'])
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3449: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x1086012e8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x184a15d68>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x184a3f518>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1887e3ac8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x12fe6db38>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x184a1e940>]], dtype=object)




![png](tumor_pymc3_files/tumor_pymc3_13_2.png)


I have used the first half of the original samples for burnin. Note that we need many iterations, and a significant amount of thinning in order to make it converge and have uncorrelated samples. We plot the $\alpha$ and $\beta$ marginals and create a 2D histogram or KDE plot (sns.kdeplot in seaborn) of the marginal posterior density in the space $x = \alpha/\beta$, $y = log(\alpha + \beta)$. Further down we also look at the autocorrelation plots for $\alpha$, $\beta$, and $\theta_1$.

#### Autocorrelation



```python
pm.autocorrplot(mtrace,varnames=['alpha','beta'])
```





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x11355b3c8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x17f2d8da0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x180214080>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x181a0c9b0>]], dtype=object)




![png](tumor_pymc3_files/tumor_pymc3_16_1.png)


#### NUTS

Let's try with NUTS now:



```python
with tumor_model:
    tumor_trace = pm.sample(40000)#, step, start=mu)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta, nu, mu]
    Sampling 2 chains: 100%|██████████| 81000/81000 [03:33<00:00, 379.93draws/s]
    The number of effective samples is smaller than 25% for some parameters.


Discussion on warmup and adaptation: https://andrewgelman.com/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/



```python
tt = tumor_trace[5000::]
pm.summary(tt)
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
      <td>0.142671</td>
      <td>0.013482</td>
      <td>0.000049</td>
      <td>0.116257</td>
      <td>0.169103</td>
      <td>59109.621269</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>nu</th>
      <td>0.256804</td>
      <td>0.043600</td>
      <td>0.000395</td>
      <td>0.173587</td>
      <td>0.343456</td>
      <td>14076.577169</td>
      <td>1.000003</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>2.356234</td>
      <td>0.885711</td>
      <td>0.009062</td>
      <td>1.009642</td>
      <td>4.090366</td>
      <td>11920.240119</td>
      <td>1.000041</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>14.232511</td>
      <td>5.335247</td>
      <td>0.053187</td>
      <td>5.901158</td>
      <td>24.734979</td>
      <td>12760.797477</td>
      <td>1.000032</td>
    </tr>
    <tr>
      <th>theta__0</th>
      <td>0.062545</td>
      <td>0.041190</td>
      <td>0.000155</td>
      <td>0.001426</td>
      <td>0.142216</td>
      <td>57114.307338</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__1</th>
      <td>0.062545</td>
      <td>0.041059</td>
      <td>0.000168</td>
      <td>0.000175</td>
      <td>0.140762</td>
      <td>61814.151395</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__2</th>
      <td>0.062449</td>
      <td>0.041228</td>
      <td>0.000180</td>
      <td>0.000782</td>
      <td>0.141716</td>
      <td>58898.117106</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__3</th>
      <td>0.062585</td>
      <td>0.041198</td>
      <td>0.000179</td>
      <td>0.000580</td>
      <td>0.141435</td>
      <td>58406.712770</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__4</th>
      <td>0.062153</td>
      <td>0.040964</td>
      <td>0.000176</td>
      <td>0.000585</td>
      <td>0.141045</td>
      <td>60698.101006</td>
      <td>1.000010</td>
    </tr>
    <tr>
      <th>theta__5</th>
      <td>0.062497</td>
      <td>0.041606</td>
      <td>0.000164</td>
      <td>0.000292</td>
      <td>0.142410</td>
      <td>63046.941193</td>
      <td>0.999993</td>
    </tr>
    <tr>
      <th>theta__6</th>
      <td>0.062770</td>
      <td>0.041512</td>
      <td>0.000187</td>
      <td>0.000105</td>
      <td>0.141784</td>
      <td>63023.817321</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__7</th>
      <td>0.064208</td>
      <td>0.042452</td>
      <td>0.000168</td>
      <td>0.000571</td>
      <td>0.145774</td>
      <td>61077.079804</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__8</th>
      <td>0.064273</td>
      <td>0.042562</td>
      <td>0.000181</td>
      <td>0.000700</td>
      <td>0.146872</td>
      <td>57544.041581</td>
      <td>0.999992</td>
    </tr>
    <tr>
      <th>theta__9</th>
      <td>0.064479</td>
      <td>0.042579</td>
      <td>0.000186</td>
      <td>0.000359</td>
      <td>0.146103</td>
      <td>64880.809084</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__10</th>
      <td>0.064248</td>
      <td>0.042197</td>
      <td>0.000180</td>
      <td>0.000564</td>
      <td>0.145700</td>
      <td>62979.973136</td>
      <td>1.000004</td>
    </tr>
    <tr>
      <th>theta__11</th>
      <td>0.066199</td>
      <td>0.043631</td>
      <td>0.000201</td>
      <td>0.000633</td>
      <td>0.149525</td>
      <td>60156.065174</td>
      <td>0.999999</td>
    </tr>
    <tr>
      <th>theta__12</th>
      <td>0.066337</td>
      <td>0.043481</td>
      <td>0.000172</td>
      <td>0.001097</td>
      <td>0.150348</td>
      <td>64173.087755</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__13</th>
      <td>0.068089</td>
      <td>0.044675</td>
      <td>0.000193</td>
      <td>0.000411</td>
      <td>0.153202</td>
      <td>60800.136182</td>
      <td>1.000011</td>
    </tr>
    <tr>
      <th>theta__14</th>
      <td>0.090601</td>
      <td>0.047484</td>
      <td>0.000176</td>
      <td>0.010498</td>
      <td>0.181864</td>
      <td>86662.352407</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__15</th>
      <td>0.090951</td>
      <td>0.048220</td>
      <td>0.000165</td>
      <td>0.009726</td>
      <td>0.184524</td>
      <td>83324.203128</td>
      <td>1.000006</td>
    </tr>
    <tr>
      <th>theta__16</th>
      <td>0.090458</td>
      <td>0.047725</td>
      <td>0.000177</td>
      <td>0.010900</td>
      <td>0.183688</td>
      <td>79788.348745</td>
      <td>1.000009</td>
    </tr>
    <tr>
      <th>theta__17</th>
      <td>0.090624</td>
      <td>0.048168</td>
      <td>0.000196</td>
      <td>0.010582</td>
      <td>0.184257</td>
      <td>83635.915830</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__18</th>
      <td>0.093077</td>
      <td>0.048915</td>
      <td>0.000165</td>
      <td>0.011609</td>
      <td>0.188544</td>
      <td>87567.638908</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__19</th>
      <td>0.093323</td>
      <td>0.049052</td>
      <td>0.000158</td>
      <td>0.009923</td>
      <td>0.187825</td>
      <td>85158.958841</td>
      <td>0.999988</td>
    </tr>
    <tr>
      <th>theta__20</th>
      <td>0.095923</td>
      <td>0.050479</td>
      <td>0.000169</td>
      <td>0.012296</td>
      <td>0.195612</td>
      <td>90532.353113</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__21</th>
      <td>0.095991</td>
      <td>0.050579</td>
      <td>0.000175</td>
      <td>0.011063</td>
      <td>0.194170</td>
      <td>81859.084380</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__22</th>
      <td>0.122798</td>
      <td>0.049803</td>
      <td>0.000157</td>
      <td>0.034500</td>
      <td>0.220019</td>
      <td>95776.052091</td>
      <td>0.999995</td>
    </tr>
    <tr>
      <th>theta__23</th>
      <td>0.104218</td>
      <td>0.047498</td>
      <td>0.000164</td>
      <td>0.023454</td>
      <td>0.199208</td>
      <td>84808.659289</td>
      <td>1.000001</td>
    </tr>
    <tr>
      <th>theta__24</th>
      <td>0.106730</td>
      <td>0.048697</td>
      <td>0.000185</td>
      <td>0.023302</td>
      <td>0.203106</td>
      <td>94290.434437</td>
      <td>0.999991</td>
    </tr>
    <tr>
      <th>theta__25</th>
      <td>0.109694</td>
      <td>0.050011</td>
      <td>0.000162</td>
      <td>0.022267</td>
      <td>0.207214</td>
      <td>102235.578657</td>
      <td>0.999986</td>
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
      <th>theta__40</th>
      <td>0.146865</td>
      <td>0.058558</td>
      <td>0.000183</td>
      <td>0.041777</td>
      <td>0.261984</td>
      <td>103652.372571</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__41</th>
      <td>0.147525</td>
      <td>0.065778</td>
      <td>0.000207</td>
      <td>0.030102</td>
      <td>0.274830</td>
      <td>105007.444810</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__42</th>
      <td>0.176222</td>
      <td>0.047421</td>
      <td>0.000175</td>
      <td>0.086746</td>
      <td>0.269802</td>
      <td>79145.084384</td>
      <td>1.000013</td>
    </tr>
    <tr>
      <th>theta__43</th>
      <td>0.185808</td>
      <td>0.047745</td>
      <td>0.000169</td>
      <td>0.099794</td>
      <td>0.283515</td>
      <td>82452.543457</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__44</th>
      <td>0.174455</td>
      <td>0.063212</td>
      <td>0.000161</td>
      <td>0.060966</td>
      <td>0.300280</td>
      <td>101300.501555</td>
      <td>0.999997</td>
    </tr>
    <tr>
      <th>theta__45</th>
      <td>0.174581</td>
      <td>0.063210</td>
      <td>0.000230</td>
      <td>0.062059</td>
      <td>0.301569</td>
      <td>95290.715514</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__46</th>
      <td>0.174436</td>
      <td>0.063561</td>
      <td>0.000197</td>
      <td>0.058815</td>
      <td>0.298835</td>
      <td>90052.314424</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__47</th>
      <td>0.174391</td>
      <td>0.063421</td>
      <td>0.000225</td>
      <td>0.059492</td>
      <td>0.299640</td>
      <td>90037.774671</td>
      <td>1.000008</td>
    </tr>
    <tr>
      <th>theta__48</th>
      <td>0.174598</td>
      <td>0.063581</td>
      <td>0.000213</td>
      <td>0.063073</td>
      <td>0.304042</td>
      <td>101955.788989</td>
      <td>1.000021</td>
    </tr>
    <tr>
      <th>theta__49</th>
      <td>0.174643</td>
      <td>0.063113</td>
      <td>0.000226</td>
      <td>0.059909</td>
      <td>0.298779</td>
      <td>91349.607255</td>
      <td>1.000003</td>
    </tr>
    <tr>
      <th>theta__50</th>
      <td>0.174376</td>
      <td>0.063243</td>
      <td>0.000188</td>
      <td>0.061179</td>
      <td>0.300758</td>
      <td>88346.091753</td>
      <td>1.000005</td>
    </tr>
    <tr>
      <th>theta__51</th>
      <td>0.192097</td>
      <td>0.049554</td>
      <td>0.000171</td>
      <td>0.100407</td>
      <td>0.291313</td>
      <td>96696.415050</td>
      <td>0.999998</td>
    </tr>
    <tr>
      <th>theta__52</th>
      <td>0.179781</td>
      <td>0.065145</td>
      <td>0.000211</td>
      <td>0.062991</td>
      <td>0.309483</td>
      <td>85222.955448</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__53</th>
      <td>0.179466</td>
      <td>0.064806</td>
      <td>0.000202</td>
      <td>0.064987</td>
      <td>0.310693</td>
      <td>92152.363771</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__54</th>
      <td>0.179614</td>
      <td>0.064603</td>
      <td>0.000202</td>
      <td>0.062636</td>
      <td>0.307698</td>
      <td>93720.581239</td>
      <td>1.000077</td>
    </tr>
    <tr>
      <th>theta__55</th>
      <td>0.191451</td>
      <td>0.063981</td>
      <td>0.000211</td>
      <td>0.073988</td>
      <td>0.317621</td>
      <td>93847.064895</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__56</th>
      <td>0.213938</td>
      <td>0.052120</td>
      <td>0.000174</td>
      <td>0.114844</td>
      <td>0.316490</td>
      <td>92842.549308</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__57</th>
      <td>0.219551</td>
      <td>0.051596</td>
      <td>0.000199</td>
      <td>0.121534</td>
      <td>0.320302</td>
      <td>71468.589126</td>
      <td>1.000073</td>
    </tr>
    <tr>
      <th>theta__58</th>
      <td>0.202631</td>
      <td>0.067303</td>
      <td>0.000233</td>
      <td>0.078683</td>
      <td>0.335506</td>
      <td>79052.827788</td>
      <td>0.999995</td>
    </tr>
    <tr>
      <th>theta__59</th>
      <td>0.202387</td>
      <td>0.067428</td>
      <td>0.000223</td>
      <td>0.077808</td>
      <td>0.335611</td>
      <td>78964.861693</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__60</th>
      <td>0.212595</td>
      <td>0.066060</td>
      <td>0.000234</td>
      <td>0.092429</td>
      <td>0.344337</td>
      <td>86759.061230</td>
      <td>0.999992</td>
    </tr>
    <tr>
      <th>theta__61</th>
      <td>0.208208</td>
      <td>0.069223</td>
      <td>0.000268</td>
      <td>0.082436</td>
      <td>0.345336</td>
      <td>79764.736883</td>
      <td>1.000003</td>
    </tr>
    <tr>
      <th>theta__62</th>
      <td>0.218662</td>
      <td>0.067560</td>
      <td>0.000229</td>
      <td>0.094363</td>
      <td>0.352974</td>
      <td>77136.531129</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__63</th>
      <td>0.230253</td>
      <td>0.071304</td>
      <td>0.000256</td>
      <td>0.098650</td>
      <td>0.372085</td>
      <td>80093.322450</td>
      <td>0.999991</td>
    </tr>
    <tr>
      <th>theta__64</th>
      <td>0.230473</td>
      <td>0.070972</td>
      <td>0.000244</td>
      <td>0.100047</td>
      <td>0.370713</td>
      <td>81977.147169</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__65</th>
      <td>0.230547</td>
      <td>0.071464</td>
      <td>0.000285</td>
      <td>0.100718</td>
      <td>0.374298</td>
      <td>74543.109055</td>
      <td>1.000055</td>
    </tr>
    <tr>
      <th>theta__66</th>
      <td>0.268448</td>
      <td>0.054589</td>
      <td>0.000191</td>
      <td>0.164719</td>
      <td>0.375830</td>
      <td>77466.938358</td>
      <td>0.999995</td>
    </tr>
    <tr>
      <th>theta__67</th>
      <td>0.278563</td>
      <td>0.057947</td>
      <td>0.000238</td>
      <td>0.168726</td>
      <td>0.392856</td>
      <td>67498.481305</td>
      <td>0.999992</td>
    </tr>
    <tr>
      <th>theta__68</th>
      <td>0.274266</td>
      <td>0.057061</td>
      <td>0.000196</td>
      <td>0.167632</td>
      <td>0.388138</td>
      <td>75987.954939</td>
      <td>0.999991</td>
    </tr>
    <tr>
      <th>theta__69</th>
      <td>0.282718</td>
      <td>0.073329</td>
      <td>0.000329</td>
      <td>0.148560</td>
      <td>0.432725</td>
      <td>60596.486567</td>
      <td>0.999994</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 7 columns</p>
</div>





```python
pm.autocorrplot(tt, varnames=['alpha','beta'])
```





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x18c574d30>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x18c56d7b8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1965c1b38>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1965b6eb8>]], dtype=object)




![png](tumor_pymc3_files/tumor_pymc3_21_1.png)




```python
from pymc3 import traceplot

#traceplot(bioassay_trace[500:], varnames=['alpha'])
traceplot(tt, varnames=['alpha','beta','theta']);
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_base.py:3449: MatplotlibDeprecationWarning: 
    The `ymin` argument was deprecated in Matplotlib 3.0 and will be removed in 3.2. Use `bottom` instead.
      alternative='`bottom`', obj_type='argument')



![png](tumor_pymc3_files/tumor_pymc3_22_1.png)




```python
fig = plt.subplots(1, 2, figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(tt['alpha'], tt['beta'],'.', alpha=0.1)
sns.kdeplot(tt['alpha'], tt['beta'])
plt.xlabel(r"$\alpha$",size=20)
plt.ylabel(r"$\beta$",size=20)
plt.subplot(1,2,2)
plt.plot(np.log(tt['alpha']/tt['beta']), np.log(tt['alpha']+tt['beta']),'.', alpha=0.1)
sns.kdeplot(np.log(tt['alpha']/tt['beta']), np.log(tt['alpha']+tt['beta']))
plt.xlabel(r"$\log(\alpha/\beta)$",size=20)
plt.ylabel(r"$\log(\alpha+\beta)$",size=20)
```





    Text(0, 0.5, '$\\log(\\alpha+\\beta)$')




![png](tumor_pymc3_files/tumor_pymc3_23_1.png)


Note the advantage of using gradients for sampling (stay tuned for Hamiltonian Monte Carlo). We need way less samples to converge to a similar result as with Metropolis, and autocorrelation plots look beter. Let us move to checking convergence for the NUTS sampler, using the Geweke diagnostic. It is important to check that both $\alpha$ and $\beta$ has converged.



```python
tt.varnames
```





    ['mu_interval__',
     'nu_interval__',
     'theta_logodds__',
     'mu',
     'nu',
     'alpha',
     'beta',
     'theta']





```python
pm.plot_posterior(tt, varnames=['alpha', 'beta'])
```





    array([<matplotlib.axes._subplots.AxesSubplot object at 0x17fce90f0>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x1086839b0>], dtype=object)




![png](tumor_pymc3_files/tumor_pymc3_26_1.png)




```python
plt.figure(figsize=(8, 20))
pm.forestplot(tt, );
```



![png](tumor_pymc3_files/tumor_pymc3_27_0.png)




```python
from pymc3 import geweke
z1 = geweke(tt, intervals=15)[0]
z2 = geweke(tt, intervals=15)[1]
```




```python
tt.get_values('alpha', chains=0).shape
```





    (35000,)





```python
fig = plt.subplots(1, 2, figsize=(12, 4))
plt.subplot(1,2,1)
plt.scatter(*z1['alpha'].T)
plt.axhline(-1, 0, 1, linestyle='dotted')
plt.axhline(1, 0, 1, linestyle='dotted')


plt.subplot(1,2,2)
plt.scatter(*z1['beta'].T)
plt.axhline(-1, 0, 1, linestyle='dotted')
plt.axhline(1, 0, 1, linestyle='dotted')

```





    <matplotlib.lines.Line2D at 0x18db456a0>




![png](tumor_pymc3_files/tumor_pymc3_30_1.png)




```python
from pymc3 import sample_ppc
with tumor_model:
    tumor_sim = sample_ppc(tt, samples=500)
```


    100%|██████████| 500/500 [00:03<00:00, 138.18it/s]




```python
tumor_sim['obs_deaths'].T[59].shape
```





    (500,)



Let's plot a few of the posterior predictives and the observed data:



```python
fig = plt.subplots(1, 4, figsize=(12, 5))
plt.subplot(1,4,1)
plt.hist(tumor_sim['obs_deaths'].T[59])
plt.plot(tumory[59]+0.5, 1, 'ro')
plt.subplot(1,4,2)
plt.hist(tumor_sim['obs_deaths'].T[49])
plt.plot(tumory[49]+0.5, 1, 'ro')
plt.subplot(1,4,3)
plt.hist(tumor_sim['obs_deaths'].T[39])
plt.plot(tumory[39]+0.5, 1, 'ro')
plt.subplot(1,4,4)
plt.hist(tumor_sim['obs_deaths'].T[29])
plt.plot(tumory[29]+0.5, 1, 'ro')
```





    [<matplotlib.lines.Line2D at 0x19942e198>]




![png](tumor_pymc3_files/tumor_pymc3_34_1.png)


A more meaningful plot is the observed tumor rates on the x-axis against posterior medians for each of the 70 $\theta$'s on the y axis, along with error bars obtained from finding the 2.5 and 97.5 percentiles. With ```df_summary``` we can get the summary with the means and the percentiles directly into a pandas dataframe:



```python
from pymc3 import summary

df_sum = summary(tt, varnames=['theta'])
df_sum.head()
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
      <th>theta__0</th>
      <td>0.062545</td>
      <td>0.041190</td>
      <td>0.000155</td>
      <td>0.001426</td>
      <td>0.142216</td>
      <td>57114.307338</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__1</th>
      <td>0.062545</td>
      <td>0.041059</td>
      <td>0.000168</td>
      <td>0.000175</td>
      <td>0.140762</td>
      <td>61814.151395</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__2</th>
      <td>0.062449</td>
      <td>0.041228</td>
      <td>0.000180</td>
      <td>0.000782</td>
      <td>0.141716</td>
      <td>58898.117106</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>theta__3</th>
      <td>0.062585</td>
      <td>0.041198</td>
      <td>0.000179</td>
      <td>0.000580</td>
      <td>0.141435</td>
      <td>58406.712770</td>
      <td>0.999987</td>
    </tr>
    <tr>
      <th>theta__4</th>
      <td>0.062153</td>
      <td>0.040964</td>
      <td>0.000176</td>
      <td>0.000585</td>
      <td>0.141045</td>
      <td>60698.101006</td>
      <td>1.000010</td>
    </tr>
  </tbody>
</table>
</div>





```python
medianthetas = df_sum['mean'].values
lowerthetas = df_sum['hpd_2.5'].values
upperthetas = df_sum['hpd_97.5'].values

elowertheta = medianthetas - lowerthetas
euppertheta = upperthetas - medianthetas
```


Our naive, non-Bayesian estimate of the probabilities would have been just the ratio of rats with tumor to total number of observed rats in each experiment:



```python
ratios=tumory.astype(float)/tumorn
```


Now let us compare those naive estimates to our posterior estimates:



```python
plt.errorbar(ratios, 
             medianthetas, yerr=[lowerthetas,upperthetas], fmt='o', alpha=0.5)
plt.plot([0,0.5],[0,0.5],'k-')
plt.xlabel("observed rates",size=15)
plt.ylabel("posterior median of rate parameters",size=15)

plt.xlim(-0.1,0.5)
```





    (-0.1, 0.5)




![png](tumor_pymc3_files/tumor_pymc3_41_1.png)


Also see this problem in the pymc3 examples: https://docs.pymc.io/notebooks/GLM-hierarchical-binominal-model.html
