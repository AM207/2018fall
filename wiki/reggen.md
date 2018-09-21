---
title: Generating And Fitting Regression Data
shorttitle: reggen
notebook: reggen.ipynb
noline: 1
summary: ""
keywords: ['multi-variate normal', 'linear regression', 'correlation']
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}




```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
```


## A trick to generate data

We generate data from a gaussian with standard deviation 1 and means given by:

$$\mu_i = 0.15 x_{1,i} - 0.4 x_{2,i}, y \sim N(\mu, 1).$$

This is a **2 parameter** model.

We use an interesting trick to generate this data, directly using the regression coefficients as correlations with the response variable.

Lets start in 2D



```python
rho=[0.15, -0.4] # correlation with y
n_dim = 1 + len(rho)
Rho = np.eye(n_dim)
for i,r in enumerate(rho):
    Rho[0, i+1] = r
Rho
```





    array([[ 1.  ,  0.15, -0.4 ],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.  ,  0.  ,  1.  ]])





```python
index_lower = np.tril_indices(n_dim, -1)
Rho[index_lower] = Rho.T[index_lower]
Rho
```





    array([[ 1.  ,  0.15, -0.4 ],
           [ 0.15,  1.  ,  0.  ],
           [-0.4 ,  0.  ,  1.  ]])





```python
mean = n_dim * [0.]
samples = np.random.multivariate_normal(mean, Rho, size=100)
samples
```





    array([[ 0.84829468,  1.86357579, -0.39473678],
           [-0.2201655 ,  0.10651643,  0.88117214],
           [-0.4694467 ,  1.01881081,  0.61019291],
           [ 0.97238626,  0.97415043, -0.69144442],
           [-0.18560854,  0.10045763, -0.76464303],
           [-1.70293835, -1.31029031,  1.06756914],
           [ 0.67821329,  1.9591632 , -1.06810272],
           [-1.25007846,  0.60104172,  0.41335409],
           [ 1.38362573, -0.20020628, -1.27849534],
           [ 1.42074458, -0.06381162,  1.25155729],
           [-0.71426815,  0.67477458,  0.38833928],
           [-0.58061311,  0.33207564,  0.02361399],
           [-0.59322056, -1.36695479,  0.63820656],
           [ 0.14823059,  1.17110299, -0.34436122],
           [ 0.63092864,  0.71524888, -0.68165195],
           [ 0.29954188,  1.28327759, -0.06481651],
           [-0.38609811, -0.70228668,  1.47769783],
           [-0.2830174 ,  1.36178387, -1.41310613],
           [-0.16544907, -0.19139881, -0.39996352],
           [-0.57846011,  1.5933223 ,  0.89445915],
           [-0.71243852, -0.62190687,  1.33086044],
           [-0.59305509, -0.41859279, -0.9301691 ],
           [-0.31862974,  0.0919313 , -0.43707989],
           [-0.37998334,  0.58489863, -0.98396642],
           [ 0.9863531 ,  0.72120315,  0.64776917],
           [ 0.89592025,  0.09647819, -1.22667958],
           [-1.05236921,  0.67291205,  1.35502779],
           [ 0.14038859, -0.29442144,  0.10617583],
           [ 1.95234632,  0.64369668, -1.26011343],
           [ 0.79761038, -0.85069265, -0.76658547],
           [ 0.39113478,  0.87653103,  0.42298565],
           [ 0.96649324,  1.3705299 , -1.18880219],
           [ 0.01805473, -0.60373274,  0.22047191],
           [ 2.21388287,  0.9892857 ,  0.13090843],
           [-0.44651357, -0.63320017,  0.75019402],
           [ 0.4969435 ,  1.33691532,  0.71902038],
           [-1.19189085, -0.76064319, -0.31596461],
           [-1.03003591, -0.3191257 ,  0.56842621],
           [ 0.30716251,  0.10647805,  0.16286304],
           [-1.35027751,  0.65173137,  1.63197819],
           [-0.78170283,  0.21362455,  1.37974478],
           [ 1.12264292, -0.07592556, -0.91714744],
           [ 0.75498325, -0.53966961, -0.16471459],
           [-1.35877505, -0.70913814, -1.41090652],
           [ 0.10671348,  1.19590082,  0.67322524],
           [-0.68418793, -2.25179149,  0.28370276],
           [ 1.4911018 ,  2.07738105, -1.39300423],
           [ 0.43697093,  0.06133477, -1.44926685],
           [ 0.17102509, -0.84678608, -0.73387107],
           [-0.64063258,  0.49115781,  0.23454946],
           [-1.01267405, -0.03990269,  0.9765875 ],
           [ 0.65403794,  0.58651412, -0.31162163],
           [ 0.30157724,  0.91919912, -1.55372535],
           [-0.22127222,  0.31922825,  1.49988373],
           [-1.52887519, -0.38711862,  0.71018256],
           [-0.1670647 ,  0.42258057,  0.23205335],
           [-1.20161439,  1.70682925,  0.79890039],
           [-2.33599968, -0.32576889,  0.89234008],
           [ 0.80218063, -0.61221003, -0.16369216],
           [ 0.61798034, -1.02821704,  0.26791116],
           [ 0.05504228,  0.72733467,  0.38308278],
           [-0.27684371, -0.63385337, -0.8580844 ],
           [-0.6376338 ,  0.40402547, -0.30787444],
           [ 1.40448067, -0.2304404 , -0.16642549],
           [ 2.11813954,  1.40501024, -0.96754945],
           [ 0.01459605, -1.30137268, -0.77768442],
           [-1.25435394,  0.60752406,  0.65659762],
           [ 1.08397832,  0.76594246, -1.75191981],
           [ 1.42700019,  1.26809649, -0.46503917],
           [-0.3495552 , -2.76888682, -0.37371275],
           [ 0.12181154, -0.66414857,  0.08826891],
           [-0.41436876,  0.81574386,  0.32965788],
           [-1.65278622, -0.25389301,  0.84035176],
           [-0.63203123,  0.61358441,  0.73414382],
           [-0.91358216,  0.55890725,  0.30369196],
           [ 0.04674343,  0.76753364,  1.38347599],
           [ 0.14664759, -0.37460711, -1.04643953],
           [ 0.6759038 , -1.04578362,  0.35434335],
           [-0.23100848,  0.20752056, -1.35946668],
           [ 0.88636279,  0.14186637, -2.1955578 ],
           [ 0.47192492,  0.30398009, -0.14475692],
           [-0.92941733,  0.81270497,  0.64266262],
           [-1.70115043,  0.05558197,  1.15819739],
           [-1.90246527, -0.52489892,  2.94850424],
           [ 0.11614657, -2.01446352, -0.61270621],
           [-0.45321877,  0.03671184,  0.6536504 ],
           [-0.40468291, -0.85662248, -0.35441041],
           [ 1.31835095,  0.62017641, -2.3056487 ],
           [-1.03290025,  0.16448292, -0.14982922],
           [ 0.61372111, -0.08778994, -0.44367042],
           [ 1.54586397,  1.53606156, -0.30466412],
           [ 0.57250264,  0.65593102,  0.28969349],
           [ 0.19877124,  0.17627301, -1.21124608],
           [-0.84964953, -1.35959095,  1.0272965 ],
           [ 1.162175  ,  0.19057154, -0.19975374],
           [ 0.62203791,  0.2594764 , -0.73826795],
           [ 0.52176301, -0.7400816 , -2.66810076],
           [ 2.38195794,  0.96428855, -0.34786335],
           [ 0.00445255, -2.5835561 , -0.17519434],
           [ 1.12178194, -0.254197  , -0.54977673]])





```python
plt.scatter(samples[:,1], samples[:,0]) #marginal
```





    <matplotlib.collections.PathCollection at 0x114e440b8>




![png](reggen_files/reggen_7_1.png)




```python
plt.scatter(samples[:,2], samples[:,0]) #marginal
```





    <matplotlib.collections.PathCollection at 0x114f84e80>




![png](reggen_files/reggen_8_1.png)




```python
plt.scatter(samples[:,1], samples[:,2]) #marginal
```





    <matplotlib.collections.PathCollection at 0x1150d8710>




![png](reggen_files/reggen_9_1.png)




```python
def calculate_corr_matrix(k, rho):
    n_dim = 1 + len(rho)
    if n_dim < k:
        n_dim = k
    Rho = np.eye(n_dim)
    for i,r in enumerate(rho):
        Rho[0, i+1] = r
    index_lower = np.tril_indices(n_dim, -1)
    Rho[index_lower] = Rho.T[index_lower]
    return Rho, n_dim
```




```python
calculate_corr_matrix(2, [0.15, -0.4])
```





    (array([[ 1.  ,  0.15, -0.4 ],
            [ 0.15,  1.  ,  0.  ],
            [-0.4 ,  0.  ,  1.  ]]), 3)





```python
calculate_corr_matrix(3, [0.15, -0.4])
```





    (array([[ 1.  ,  0.15, -0.4 ],
            [ 0.15,  1.  ,  0.  ],
            [-0.4 ,  0.  ,  1.  ]]), 3)





```python
calculate_corr_matrix(4, [0.15, -0.4])
```





    (array([[ 1.  ,  0.15, -0.4 ,  0.  ],
            [ 0.15,  1.  ,  0.  ,  0.  ],
            [-0.4 ,  0.  ,  1.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  1.  ]]), 4)





```python
calculate_corr_matrix(5, [0.15, -0.4])
```





    (array([[ 1.  ,  0.15, -0.4 ,  0.  ,  0.  ],
            [ 0.15,  1.  ,  0.  ,  0.  ,  0.  ],
            [-0.4 ,  0.  ,  1.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  1.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  1.  ]]), 5)





```python
def generate_data(N, k, rho=[0.15, -0.4]):
    Rho, n_dim = calculate_corr_matrix(k, rho)
    mean = n_dim * [0.]
    Xtrain = np.random.multivariate_normal(mean, Rho, size=N)
    Xtest = np.random.multivariate_normal(mean, Rho, size=N)
    ytrain = Xtrain[:,0].copy()
    Xtrain[:,0]=1.
    ytest = Xtest[:,0].copy()
    Xtest[:,0]=1.
    #print(Xtrain)
    #print(Xtrain.shape, Xtrain[:,:k].shape)
    return Xtrain[:,:k], ytrain, Xtest[:,:k], ytest
```


We want to generate data for 5 different cases, a one parameter (intercept) fit, a two parameter (intercept and $x_1$), three parameters (add a $x_2), and four and five parameters. Here is what the data looks like for 2 and 3 parameters:



```python
generate_data(5,2)
```





    (array([[ 1.        ,  0.78804338],
            [ 1.        , -0.17559308],
            [ 1.        ,  2.61106682],
            [ 1.        ,  0.16187352],
            [ 1.        ,  0.94676314]]),
     array([ 0.66633307,  0.09760313,  1.05121673, -0.99508296, -1.2087932 ]),
     array([[ 1.        ,  0.55519384],
            [ 1.        ,  0.07493834],
            [ 1.        , -1.7226416 ],
            [ 1.        ,  0.27709465],
            [ 1.        ,  0.78917846]]),
     array([ 0.69627859,  0.97390882,  1.71695311, -1.7152763 , -0.04044233]))





```python
generate_data(5,3)
```





    (array([[ 1.        , -1.36143601,  0.03148762],
            [ 1.        ,  1.12974765, -0.49766819],
            [ 1.        , -0.53260886, -0.57664704],
            [ 1.        ,  0.20249861, -0.19959547],
            [ 1.        ,  0.90036475,  0.50690876]]),
     array([-1.48977234, -0.59190736, -0.40522914, -0.91222955, -1.70553949]),
     array([[ 1.        , -1.64668283, -1.80285072],
            [ 1.        , -0.10180807,  0.91731483],
            [ 1.        , -0.13906829, -1.55263645],
            [ 1.        , -2.11138676,  0.22651539],
            [ 1.        , -0.8860697 , -1.35889235]]),
     array([ 1.31449753,  0.69134707,  1.78590081,  0.10361084,  1.95151795]))



And for four and 5 parameters



```python
generate_data(5,4)
```





    (array([[ 1.        ,  0.15021561,  1.69863925, -1.2804319 ],
            [ 1.        ,  0.08304854,  1.14454472,  0.33821796],
            [ 1.        ,  0.68227058,  0.6163912 , -1.5910914 ],
            [ 1.        , -1.10381808, -0.88655867, -0.08370383],
            [ 1.        ,  0.19388402,  0.59994127, -0.62826231]]),
     array([-0.73357551, -1.66825412, -0.23778661, -1.05959313,  1.20576895]),
     array([[ 1.        ,  1.97511727, -0.50345436, -0.01065413],
            [ 1.        ,  0.95802976,  0.80566122,  1.78646255],
            [ 1.        ,  0.54619592, -1.01789944,  1.88597392],
            [ 1.        ,  0.64961521, -1.22012602,  1.12468855],
            [ 1.        ,  0.44773808,  1.19129354,  0.03779606]]),
     array([ 1.27321518, -0.64355957,  0.93028705,  1.90772648, -0.42072479]))





```python
generate_data(5,5)
```





    (array([[ 1.        ,  0.27817718,  0.59341622, -0.33465159, -2.58984203],
            [ 1.        , -1.15578301,  0.11206437,  0.56905775,  0.05500154],
            [ 1.        , -0.69974996,  1.37556309,  1.55261713, -0.61371482],
            [ 1.        ,  0.8837714 ,  0.04687932, -0.59032411,  0.27833392],
            [ 1.        , -0.49619511,  0.81998447,  0.9305664 ,  1.34997944]]),
     array([-2.80059115,  0.21019808, -0.9332254 , -0.2448186 ,  0.35210238]),
     array([[ 1.        , -0.47399572, -3.13636432, -0.47067828, -1.03602178],
            [ 1.        ,  0.11283849, -0.28687251, -0.29365302, -0.09671533],
            [ 1.        , -0.0811073 ,  0.72964077, -0.992109  ,  0.22584973],
            [ 1.        ,  0.69415671, -0.14770833, -1.42307707,  0.71640962],
            [ 1.        , -1.20924232, -1.80826492,  0.55658883, -2.83933828]]),
     array([ 0.77906954,  0.70536224, -2.08137173, -1.59671007,  0.30048861]))





```python
from scipy.stats import norm
import statsmodels.api as sm
```


## Analysis, n=20

Here is the main loop of our analysis. We take the 5 models we talked about. For each model we generate 10000 samples of the data, split into an equal sized (N=20 each) training and testing set. We fit the regression on the training set, and calculate the **deviance** on the training set. 

What is **deviance**. We will come to the concept of deviance soon, but for now, its just 

$$ Deviance  =  -2 \times \ell  =  -2 \times log({\cal L}) ,$$

computed over the data set in question.

Thus the Deviance is just a loss function.

Notice how we have simply used the `logpdf` from `scipy.stats`. You can easily do this for other distributions.

We then use the fit to calculate the $\mu$ on the test set, and calculate the deviance there. We then find the average and the standard deviation across the 10000 simulations.

Why do we do 10000 simulations? These are our **multiple samples from some hypothetical population**.



```python
reps=10000
results_20 = {}
for k in range(1,6):
    trdevs=np.zeros(reps)
    tedevs=np.zeros(reps)
    for r in range(reps):
        Xtr, ytr, Xte, yte = generate_data(20, k)
        ols = sm.OLS(ytr, Xtr).fit()
        mutr = np.dot(Xtr, ols.params)
        devtr = -2*np.sum(norm.logpdf(ytr, mutr, 1))
        mute = np.dot(Xte, ols.params)
        #print(mutr.shape, mute.shape)
        devte = -2*np.sum(norm.logpdf(yte, mute, 1))
        #print(k, r, devtr, devte)
        trdevs[r] = devtr
        tedevs[r] = devte
    results_20[k] = (np.mean(trdevs), np.std(trdevs), np.mean(tedevs), np.std(tedevs))
```




```python
import pandas as pd
df = pd.DataFrame(results_20).T
df = df.rename(columns = dict(zip(range(4), ['train', 'train_std', 'test', 'test_std'])))
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
      <th>train</th>
      <th>train_std</th>
      <th>test</th>
      <th>test_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>55.814482</td>
      <td>6.095473</td>
      <td>57.696594</td>
      <td>6.678535</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54.364176</td>
      <td>5.953622</td>
      <td>58.473285</td>
      <td>7.154725</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.736528</td>
      <td>4.831665</td>
      <td>55.947967</td>
      <td>6.708852</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49.812125</td>
      <td>4.638089</td>
      <td>57.396140</td>
      <td>7.605612</td>
    </tr>
    <tr>
      <th>5</th>
      <td>49.088703</td>
      <td>4.480338</td>
      <td>58.820134</td>
      <td>8.512742</td>
    </tr>
  </tbody>
</table>
</div>





```python
import seaborn.apionly as sns
colors = sns.color_palette()
colors
```





    [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
     (1.0, 0.4980392156862745, 0.054901960784313725),
     (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
     (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
     (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
     (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
     (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
     (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
     (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]



We plot the traing and testing deviances



```python
plt.plot(df.index, df.train, 'o', color = colors[0])
plt.errorbar(df.index, df.train, yerr=df.train_std, fmt=None, color=colors[0])
plt.plot(df.index+0.2, df.test, 'o', color = colors[1])
plt.errorbar(df.index+0.2, df.test, yerr=df.test_std, fmt=None, color=colors[1])
plt.xlabel("number of parameters")
plt.ylabel("deviance")
plt.title("N=20");
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:2818: MatplotlibDeprecationWarning: Use of None object as fmt keyword argument to suppress plotting of data values is deprecated since 1.4; use the string "none" instead.



![png](reggen_files/reggen_28_1.png)


This is just an illustration of the training vs testing structure we saw in class, along with the randomness that comes from sampling and noise. (Indeed, here, because of the way we generated the data, the randomness from both the sampling and the noise are explicitly included).

Notice:

- the best fit model may not be the original generating model. Remember that the choice of fit depends on the amount of data you have and the less data you have, the less parameters you should use
- on average, out of sample deviance must be larger than in-sample deviance, through an individual pair may have that order reversed because of sample peculiarity.

If you subtract the average losses between the test and training set, you find something interesting. 



```python
df.test - df.train
```





    1    1.882112
    2    4.109110
    3    5.211439
    4    7.584015
    5    9.731431
    dtype: float64



### Analysis N=100



```python
reps=10000
results_100 = {}
for k in range(1,6):
    trdevs=np.zeros(reps)
    tedevs=np.zeros(reps)
    for r in range(reps):
        Xtr, ytr, Xte, yte = generate_data(100, k)
        ols = sm.OLS(ytr, Xtr).fit()
        mutr = np.dot(Xtr, ols.params)
        devtr = -2*np.sum(norm.logpdf(ytr, mutr, 1))
        mute = np.dot(Xte, ols.params)
        devte = -2*np.sum(norm.logpdf(yte, mute, 1))
        trdevs[r] = devtr
        tedevs[r] = devte
    results_100[k] = (np.mean(trdevs), np.std(trdevs), np.mean(tedevs), np.std(tedevs))
```




```python
df100 = pd.DataFrame(results_100).T
df100 = df100.rename(columns = dict(zip(range(4), ['train', 'train_std', 'test', 'test_std'])))
df100
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
      <th>train</th>
      <th>train_std</th>
      <th>test</th>
      <th>test_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>282.375490</td>
      <td>14.224281</td>
      <td>284.853431</td>
      <td>14.356289</td>
    </tr>
    <tr>
      <th>2</th>
      <td>279.748714</td>
      <td>13.692269</td>
      <td>283.469905</td>
      <td>14.265801</td>
    </tr>
    <tr>
      <th>3</th>
      <td>263.060456</td>
      <td>11.264216</td>
      <td>268.008034</td>
      <td>12.092410</td>
    </tr>
    <tr>
      <th>4</th>
      <td>262.225270</td>
      <td>11.236112</td>
      <td>268.966034</td>
      <td>12.346106</td>
    </tr>
    <tr>
      <th>5</th>
      <td>261.475862</td>
      <td>11.248103</td>
      <td>269.908307</td>
      <td>12.499466</td>
    </tr>
  </tbody>
</table>
</div>





```python
plt.plot(df100.index, df100.train, 'o', color = colors[0])
plt.errorbar(df100.index, df100.train, yerr=df100.train_std, fmt=None, color=colors[0])
plt.plot(df100.index+0.2, df100.test, 'o', color = colors[1])
plt.errorbar(df100.index+0.2, df100.test, yerr=df100.test_std, fmt=None, color=colors[1])
plt.xlabel("number of parameters")
plt.ylabel("deviance")
plt.title("N=100");
```


    //anaconda/envs/py3l/lib/python3.6/site-packages/matplotlib/axes/_axes.py:2818: MatplotlibDeprecationWarning: Use of None object as fmt keyword argument to suppress plotting of data values is deprecated since 1.4; use the string "none" instead.



![png](reggen_files/reggen_36_1.png)




```python
df100.test - df100.train
```





    1    2.477941
    2    3.721191
    3    4.947578
    4    6.740764
    5    8.432444
    dtype: float64


