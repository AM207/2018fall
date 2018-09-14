---
title: Lab 2 - The Frequentist Approach:  A brief overview
shorttitle: freq-lab
notebook: freq-lab.ipynb
noline: 1
summary: ""
keywords: ['frequentist statistics', 'exponential distribution', 'poisson distribution', 'maximum likelihood', 'bootstrap', 'parametric']
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}


## Contents
{:.no_toc}
* 
{: toc}

## Learning Aims

- using scipy.stats
- MLE
- bootstrap
- matplotlib histogram considerations

$\newcommand{\Ex}{\mathbb{E}}$
$\newcommand{\Var}{\mathrm{Var}}$
$\newcommand{\Cov}{\mathrm{Cov}}$
$\newcommand{\SampleAvg}{\frac{1}{N({S})} \sum_{s \in {S}}}$
$\newcommand{\indic}{\mathbb{1}}$
$\newcommand{\avg}{\overline}$
$\newcommand{\est}{\hat}$
$\newcommand{\trueval}[1]{#1^{*}}$
$\newcommand{\Gam}[1]{\mathrm{Gamma}#1}$



```python
# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.
%matplotlib inline

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
```


## Data on the birth of babies

>Forty-four babies -- a new record -- were born in one 24-hour period at
the Mater Mothers' Hospital in Brisbane, Queensland, Australia, on
December 18, 1997.  For each of the 44 babies, _The Sunday Mail_
recorded the time of birth, the sex of the child, and the birth weight
in grams. Also included is the number of minutes since midnight for
each birth.

REFERENCE:
Steele, S. (December 21, 1997), "Babies by the Dozen for Christmas:
24-Hour Baby Boom," _The Sunday Mail_ (Brisbane), p. 7.

"Datasets
and Stories" article "A Simple Dataset for Demonstrating Common
Distributions" in the _Journal of Statistics Education_ (Dunn 1999).

Columns

       1 24hrtime  Time of birth recorded on the 24-hour clock
       2 sex  Sex of the child (1 = girl, 2 = boy)
       3 weight  Birth weight in grams
       4 minutes  Number of minutes after midnight of each birth



```python
df = pd.read_table("data/babyboom.dat.txt", header=None, sep='\s+', 
                   names=['24hrtime','sex','weight','minutes'])
df.head()
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
      <th>24hrtime</th>
      <th>sex</th>
      <th>weight</th>
      <th>minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>1</td>
      <td>3837</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>104</td>
      <td>1</td>
      <td>3334</td>
      <td>64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>118</td>
      <td>2</td>
      <td>3554</td>
      <td>78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>155</td>
      <td>2</td>
      <td>3838</td>
      <td>115</td>
    </tr>
    <tr>
      <th>4</th>
      <td>257</td>
      <td>2</td>
      <td>3625</td>
      <td>177</td>
    </tr>
  </tbody>
</table>
</div>



We find it convenient to group births into the hours of the day during which they occurred.  Let's bin minutes in order to  construct a new feature 'hours' and store in it the number of births in each hour.



```python
df['hours'] = df.minutes // 60
df.head()
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
      <th>24hrtime</th>
      <th>sex</th>
      <th>weight</th>
      <th>minutes</th>
      <th>hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>1</td>
      <td>3837</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>104</td>
      <td>1</td>
      <td>3334</td>
      <td>64</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>118</td>
      <td>2</td>
      <td>3554</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>155</td>
      <td>2</td>
      <td>3838</td>
      <td>115</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>257</td>
      <td>2</td>
      <td>3625</td>
      <td>177</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Frequentist Philosophy:  Samples vs population

We've started out this lab by presenting you with some data.  What are we to do with the data?  What are some reasonable approaches to handling the data in such a way that we can draw insights and render conclusions based on the data.  **Frequentist inference** is one framework which serves to constructively provide answers to these sorts of questions.  According to Wikipedia

> Frequentist inference has been associated with the frequentist interpretation of probability, specifically that any given experiment can be considered as one of an infinite > sequence of possible repetitions of the same experiment, each capable of producing statistically independent results. In this view, the frequentist inference approach to > drawing conclusions from data is effectively to require that the correct conclusion should be drawn with a given (high) probability, among this notional set of repetitions.

The main idea is that given a set of data, frequentism conceives of that data as one **sample** from an existing **population** that comes about from a **data generating process**.  That sample can be used to draw inferences about the population.  This data generating process can be repeated over and over again to create other (slightly different) samples each of which can similarly used conclusions about the population.

This notion is probably clearest to you from elections, where some companies like Zogby or CNN take polls. The sample in these polls may be some finite population (e.g. 1000 people) but that sample is used to  "represent" the electoral population at large. the pollsters can then attempt to draw inferences about how the population will vote based on the samples and use the tools of frequentist intervals to provide guidance about the robustness of their conclusions.

The main purpose of this lab is to introduce **Frequentism** and help familiarize you with some common frequentist tools.

### The Idealized Data-Generating Process

Let's examine the frequentist paradigm under "ideal circumstances".  We'll skip over (or present without explanation) some concepts and terminology as the idea here is to provide you intuition and motivate some of the tools we'll present later.

1. We're provided with our data sample $S = \{X_1, X_2, ..., X_k\}$

2. We calculate our relevant estimate(s) or statistic(s) on that sample

3. In order to generate confidence intervals around our estimate(s) or statistic(s) we generate from the data generating process n additional data samples

4. On each of those n data samples we repeat the calculations of the estimate(s) or statistic(s) 

5. Collectively use the calculated estimate(s)/statistic(s) on the generated samples to construct a confidence interval

Our inference is our original estimate(s)/statistic(s) and the constructed confidence intervals from the estimate(s)/statistic(s) from our n samples from the data generating process

### Example of the Idealized Data Generating Process: Male Heights

Let's say we know that male heights follow a Gaussian distribution with mean 70 inches and standard deviation 4 inches.  Consider the following two data samples drawn from the canonical distribution of male heights.  For each sample use the idealized data generating process to estimate the mean male height using n=100 additional samples to generate confidence intervals.

**We're provided our data sample**



```python
male_heights_sample1 = np.array([65.27156778, 70.54877119, 73.3194762 , 65.39765796, 71.20531282,
       72.60242314, 71.37226406, 68.32409185, 74.53513439, 74.55146588,
       68.66712633, 72.94632781, 67.53369738, 71.91086708, 67.04768347,
       73.93720054, 77.83917981, 73.05772208, 63.5739442 , 70.95335019])
```




```python
male_heights_sample2 = np.array([73.5832399 , 72.20626275, 76.19045706, 67.57730802, 74.77351658,
       85.43682136, 73.38080912, 75.71094703, 70.59903148, 73.5543091 ,
       71.76435781, 69.27927312, 66.54357112, 74.31531105, 76.68361655,
       69.36297994, 75.18045033, 71.16446918, 74.43549175, 68.72731837])
```


**Calculate estimates on data samples**



```python
print("Estimated mean male height from sample 1: ", np.mean(male_heights_sample1))
```


    Estimated mean male height from sample 1:  70.729763208




```python
print("Estimated mean male height from sample 2: ", np.mean(male_heights_sample2))
```


    Estimated mean male height from sample 2:  73.023477081


**Generate n=100 additional data samples from the Data Generating Process**



```python
additional_heights = scipy.stats.norm.rvs(loc=70, scale=4, size=(100,20))
```




```python
additional_heights
```





    array([[74.60822575, 73.59919912, 62.29455565, ..., 69.80969588,
            79.41959974, 68.84566371],
           [71.18942457, 75.10769797, 69.67307501, ..., 74.59932643,
            65.14405227, 64.56041419],
           [76.09159302, 65.80776223, 72.17038548, ..., 71.6139225 ,
            77.34468665, 63.41453257],
           ...,
           [68.99195198, 63.67858316, 67.17857731, ..., 79.80938086,
            72.35722516, 67.11184909],
           [69.94538718, 61.07298655, 67.47832571, ..., 66.21714691,
            75.53423697, 68.95190719],
           [68.93989892, 75.92313866, 66.80171917, ..., 70.84576944,
            71.69745614, 74.5928578 ]])



**Repeat Calculation of estimate on each of the n=100 additional data samples from the Data Generating Process**



```python
additional_heights_estimates = np.mean(additional_heights, axis=1)
```


**Collectively use the calculated estimate to construct confidence intervals**



```python
def viz_estimates(viz_axis, viz_data, viz_label, viz_color,  x_label, viz_title, sample_estimate, estimate_label, bin_count=20):

        
    #plot histogram
    _, viz_bins, _ = viz_axis.hist(viz_data, bins=bin_count, facecolor=viz_color, edgecolor='white', label="Additional Samples", density=True, alpha=0.6)

    viz_axis.axvline(x=sample_estimate, linewidth=4, color='r', label=estimate_label)
    viz_axis.set_xlabel(x_label)
    viz_axis.set_title(viz_title)
    viz_axis.legend(loc='best')
```




```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

viz_estimates(
    ax1,
    additional_heights_estimates,
    'Estimates of Additional Height Samples',
    'salmon',
    'means',
    'Distribution of Height Estimates',
    np.mean(male_heights_sample1),
    'Estimate from Sample 1'
)

viz_estimates(
    ax2,
    additional_heights_estimates,
    'Estimates of Additional Height Samples',
    'salmon',
    'means',
    'Distribution of Height Estimates',
    np.mean(male_heights_sample2),
    'Estimate from Sample 2'
)

```



![png](freq-lab_files/freq-lab_26_0.png)


Are the true value of the parameter for the data generating model in the 95% confidence intervals?



```python
heights_percentile_interval = np.percentile(additional_heights_estimates, [2.5, 97.5])

heights_percentile_interval -= heights_percentile_interval[0]

heights_percentile_interval -= heights_percentile_interval[1]/2
```




```python
print("95% confidence interval sample 1: ", np.mean(male_heights_sample1)+ heights_percentile_interval)
```


    95% confidence interval sample 1:  [68.99507465 72.46445177]




```python
print("95% confidence interval sample 2: ", np.mean(male_heights_sample2)+ heights_percentile_interval)
```


    95% confidence interval sample 2:  [71.28878852 74.75816564]


We happen to know the true parameters of the model for the data generating process and the mean for that model isn't in the 95% confidence interval for male heights sample 2.

## Choosing a model: the exponential distribution

Going back to the baby data that we were given at the beginning of lab, what would a natural choice for the probability distribution for the data generating process for the babies?

One possibility is exponential distribution which comes up naturally when describing the lengths of the inter-arrival times in a homogeneous Poisson process.

It takes the form:
$$
f(x;\lambda) = \begin{cases}
\lambda e^{-\lambda x} & x \ge 0, \\
0 & x < 0.
\end{cases}
$$

From Wikipedia: *In probability theory, a Poisson process is a stochastic process which counts the number of events and the time that these events occur in a given time interval. The time between each pair of consecutive events has an exponential distribution with parameter $\lambda$ and each of these inter-arrival times is assumed to be independent of other inter-arrival times. The process is named after the French mathematician Siméon Denis Poisson and is a good model of radioactive decay, telephone calls and requests for a particular document on a web server, among many other phenomena.*

In our example above, we have the arrival times of the babies. There is no reason to expect any specific clustering in time, so one could think of modelling the arrival of the babies via a poisson process.

Furthermore, the Poisson distribution can be used to model the number of births each hour over the 24-hour period.

### What does the exponential distribution look like?



```python
f = lambda x, l: l*np.exp(-l*x)*(x>0)
xpts=np.arange(-2,3,0.05)
plt.plot(xpts,f(xpts, 2),'.');
plt.xlabel("x")
plt.ylabel("exponential pdf")
```





    Text(0,0.5,'exponential pdf')




![png](freq-lab_files/freq-lab_34_1.png)


Note: **some of the code, and ALL of the visual style for the distribution plots below was shamelessly stolen from https://gist.github.com/mattions/6113437/ **.



```python
from scipy.stats import expon

x = np.linspace(0,4, 100)
colors=sns.color_palette()

lambda_ = [0.5, 1, 2, 4]
plt.figure(figsize=(12,4))
for l,c in zip(lambda_,colors):
    plt.plot(x, expon.pdf(x, scale=1./l), lw=2, 
                color=c, label = "$\lambda = %.1f$"%l)
    plt.fill_between(x, expon.pdf(x, scale=1./l), color=c, alpha = .33)
    
plt.legend()
plt.ylabel("PDF at $x$")
plt.xlabel("$x$")
plt.title("Probability density function of an Exponential random variable;\
 differing $\lambda$");
```



![png](freq-lab_files/freq-lab_36_0.png)


### How would we draw from this distribution?

Lets use the built in machinery in `scipy.stats`:



```python
from scipy.stats import expon
plt.plot(xpts,expon.pdf(xpts, scale=1./2.),'.')
plt.hist(expon.rvs(size=1000, scale=1./2.), density=True, alpha=0.5, bins=50);
plt.xlabel("x")
plt.title("exponential pdf and no. of samples(normalized)");
```



![png](freq-lab_files/freq-lab_38_0.png)


In `scipy.stats`, you can alternatively create a frozen object, which holds values of things like the scale



```python
rv = expon(scale=0.5)
plt.plot(xpts,rv.pdf(xpts),'.')
plt.hist(rv.rvs(size=1000), density=True, alpha=0.5, bins=30);
plt.plot(xpts, rv.cdf(xpts));
plt.xlabel("x")
plt.title("exponential pdf, cdf and samples(normalized)");
```



![png](freq-lab_files/freq-lab_40_0.png)


## The Poisson distribution

The Poisson distribution is a discrete distribution, it expresses the probability of a given number of events occurring in a fixed interval of time (or space, volume, etc.). One assumption made is that these events occur with a known average rate and independently of each other. An example is the number of electrons detected by a sensor in an electron microscope during a time interval, or the number of soldiers in the Prussian army killed accidentally by horse kicks [(see here)](http://en.wikipedia.org/wiki/Poisson_distribution).

The Poisson distribution is defined as:


$$ f(k; \mu)= \frac{\mu^k e^{-\mu}}{k!}, $$

where $k$ is the number of events, $\mu$ is a positive real number, and $e$ is Euler's number ($e = 2.71828 \ldots$).



```python
from scipy.stats import poisson
# generate samples for different values of mu
kpts=np.arange(0,25)
for mu, c in zip([1,2, 4, 6], sns.color_palette()[:4]):
    randomVariates = poisson.rvs(mu, size=1000)
    plt.hist(randomVariates, density=True, color=c, alpha=0.2, bins=range(0,26), label='$\mu=' + np.str(mu) + '$')
    plt.plot(kpts, poisson.pmf(kpts, mu), '.', color=c)

plt.legend()
plt.title("Poisson Distribution")
plt.xlabel("Number of Events")
plt.ylabel("Normed Counts");
```



![png](freq-lab_files/freq-lab_42_0.png)


### Understanding our data using a  histogram-plotted distribution

Lets play with our data a bit to understand it:

The first birth occurred at 0005, and the last birth in the 24-hour period at 2355. Thus the 43 inter-birth times happened over a 1430-minute period, giving a theoretical mean of 1430/43 = 33.26 minutes between births.

Lets plot a histogram of the inter-birth times



```python
timediffs = df.minutes.diff()[1:]
print(timediffs.mean())
timediffs.hist(bins=20, density=True);
```


    33.25581395348837



![png](freq-lab_files/freq-lab_44_1.png)


And do the same for the poisson



```python
poiskcount = df.groupby('hours')['minutes'].count()
poiskcount
```





    hours
    0     1
    1     3
    2     1
    4     4
    7     2
    8     2
    9     1
    10    3
    11    1
    12    2
    13    1
    14    4
    15    1
    16    2
    17    1
    18    3
    19    4
    20    3
    21    2
    22    1
    23    2
    Name: minutes, dtype: int64



Since the Poisson is a discrete distribution, it has a probability mass function. Normalizing a pdf is not what we want here, rather, the values of the pmf (which are probabilities) should sum to 1. So we take matters into our own hands.

We make sure that every point has equal mass, so that the total mass is one. Then the mass at any point depends on how many samples  (the count) we have for it.



```python
weights = np.ones_like(poiskcount)/len(poiskcount)
#poiskcount.hist(weights=weights)

plt.hist(poiskcount)
```





    (array([8., 0., 0., 6., 0., 0., 4., 0., 0., 3.]),
     array([1. , 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4. ]),
     <a list of 10 Patch objects>)




![png](freq-lab_files/freq-lab_48_1.png)




```python
weights
```





    array([0.04761905, 0.04761905, 0.04761905, 0.04761905, 0.04761905,
           0.04761905, 0.04761905, 0.04761905, 0.04761905, 0.04761905,
           0.04761905, 0.04761905, 0.04761905, 0.04761905, 0.04761905,
           0.04761905, 0.04761905, 0.04761905, 0.04761905, 0.04761905,
           0.04761905])



### Maximum Likelihood Estimation

As described in detail above, in frequentist statistics, the data we have in hand, is viewed as a **sample** from a population. So if we want to estimate some parameter of the population, like say the mean, we estimate it on the sample.

This is because we've been given only one sample. Ideally we'd want to see the population, but we have no such luck.

The parameter estimate is computed by applying an estimator $F$ to some data $D$, so $\est{\lambda} = F(D)$. 


**The parameter is viewed as fixed and the data as random, which is the exact opposite of the Bayesian approach which you will learn later in this class. **

In the "ideal circumstances", we knew the true values of the parameters for the probability model for the data generating process and though we were given one sample, we can generate additional samples that we are guaranteed to be from the same distribution as the population at will.  Unfortunately we don't generally reside in ideal circumstances and although we may know the distributional family of the probability model generating the data, we don't know a priori its true parameters.  We need to come up with a way of finding/guessing/estimating those parameters.  One of the techniques used to estimate such parameters in frequentist statistics is **maximum likelihood estimation**. Briefly, the idea behind it is:

The product 

$$
L(\lambda) = \prod_{i=1}^n P(x_i \mid \lambda)
$$

gives us a measure of how likely it is to observe values $x_1,...,x_n$ given the parameters $\lambda$. Maximum likelihood fitting consists of choosing the appropriate "likelihood" function $L=P(X \mid \lambda)$ to maximize for a given set of observations. How likely are the observations if the model is true?

Often it is easier and numerically more stable to maximise the log likelyhood:

$$
\ell(\lambda) = \sum_{i=1}^n ln(P(x_i \mid \lambda))
$$

### Analytic MLE for the Exponential distribution

In the case of the exponential distribution we have:

$$
\ell(lambda) = \sum_{i=1}^n ln(\lambda e^{-\lambda x_i}) = \sum_{i=1}^n \left( ln(\lambda) - \lambda x_i \right).
$$

Maximizing this:

$$
\frac{d \ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0
$$

and thus:

$$
\frac{1}{\est{\lambda_{MLE}}} = \frac{1}{n}\sum_{i=1}^n x_i,
$$

which is identical to the simple estimator we used above. Usually one is not so lucky and one must use numerical optimization techniques.

For the babies, we're assuming that an exponential distribution is a good description of the baby arrival process. When we consider the larger population of babies from our sample of baby arrival data is drawn, there is some true $\trueval{\lambda}$ which defines it. We dont know this true $\trueval{\lambda}$. The best we can do to start with is to estimate a lambda from the data set we have, which we denote $\est{\lambda}$ and in some sense the MLE estimate is the "best" estimate.


This makes intuitive sense: if you get babies at an average rate of 2 per hour, then you can expect to wait half an hour on average for every baby.



```python
lambda_from_mean = 1./timediffs.mean()
print(lambda_from_mean, 1./lambda_from_mean)
```


    0.03006993006993007 33.25581395348837




```python
minutes=np.arange(0, 160, 5)
rv = expon(scale=1./lambda_from_mean)
plt.plot(minutes,rv.pdf(minutes),'.')
timediffs.hist(density=True, alpha=0.5, bins=20);
sns.kdeplot(timediffs)
plt.xlabel("minutes");
plt.xlim([0,200])
plt.title("Normalized data and model for estimated $\hat{\lambda}$");
```



![png](freq-lab_files/freq-lab_54_0.png)


What did we just do? We made a 'point estimate' of the scale or rate parameter as a compression of our data. 

### For Poisson

In the case of the poisson distribution we have:

$$
\ell(lambda) = \sum_{i=1}^n \left( k_i ln(\mu) - \mu - ln(k_i!) \right).
$$

You maximize this now.

*your answer here* 

Maximizing this:

$$
\frac{d \ell}{d\mu} = \frac{1}{\mu}\sum_{i=1}^n k_i - n = 0
$$

and thus:

$$
\est{\mu_{MLE}} = \frac{1}{n}\sum_{i=1}^n k_i
$$

or just the sample mean!

### Sampling Distribution of the Estimator

Now, imagine that I let you peek at the entire population in this way: I gave you some M data sets **drawn** from the population, and you can now find the mean on each such dataset, of which the one we have here is one.
So, we'd have M means. You can think of these means as coming from some fixed parameter by some data drawing process

Thus if we had many replications of this data set: that is, data from other days, an **ensemble** of data sets, for example, we can compute other $\est{\lambda}$, and begin to construct the **sampling distribution** of $\lambda$.  That's the scenario we constructed in the "ideal case".  As you saw above, we can use the sampling distribution to put confidence intervals on the estimation of the parameters. 


So, in the babies case, the uncertainty in the parameter estimate can be measured by computing the **sampling distribution** of the estimator. 
What you are doing is sampling many Data Sets $D_i$ from the true population (which we are not given you will argue, and you are right, but just wait a bit), say M of them, each of size N, from some true model $p(\cdot|\trueval{\lambda})$. We will now calculate M $\est{\lambda}_i$, one for each dataset. As we let $M \rightarrow \infty$, the distribution induced on $\est{\lambda}$ is the sampling distribution of the estimator.

### Bootstrapping

Bootstrap tries to approximate our sampling distribution. If we knew the true parameters of the population, we could generate M fake datasets. Then we could compute the parameter (or another estimator) on each one of these, to get a empirical sampling distribution of the parameter or estimator, and which will give us an idea of how typical our sample is, and thus, how good our parameter estimations from our sample are.
(again from murphy)

But we dont have the true parameter. So we generate these samples, using the parameter we calculated. Or, alteratively, we sample with replacement the X from our original sample D, generating many fake datasets, and then compute the distribution on the parameters as before. 

We do it here for the mean of the time differences. We could also do it for its inverse, $\lambda$.

### Parametric Bootstrapping

Let's start with the parametric case.  It's the one that maps most closely to the ideal case we dealt with above.  We follow the exact same recipe as we did for the ideal data generating process, but since we don't know the true parameters, we use instead the parameters we obtain from the MLE.  In the case of baby arrival, we get an "estimate" of the parameter from our sample, use the exponential distribution to generate many datasets, and then fit the parameter on each one of those datasets. We can then plot the distribution of the mean time-difference.

### The Parametric Bootstrapping Process

1. We're provided with our data sample $S = \{X_1, X_2, ..., X_k\}$

2. We calculate our relevant estimate(s) or statistic(s) on that sample

3. In order to generate confidence intervals around our estimate(s) or statistic(s) we generate from the data generating process n additional data samples **using the MLE parameters instead of the "true parameters" (which we don't know).**

4. On each of those n data samples we repeat the calculations of the estimate(s) or statistic(s) 

5. Collectively use the calculated estimate(s)/statistic(s) on the generated samples to construct a confidence interval

Our inference is our original estimate(s)/statistic(s) and the constructed confidence intervals from the estimate(s)/statistic(s) from our n samples from the data generating process

### Example of Parametric Bootstrapping: Baby Arrival

Let's apply the process above for our baby arrival dataset.

**We're provided our data sample**



```python
baby_arrival_times = timediffs.values

baby_arrival_times
```





    array([ 59.,  14.,  37.,  62.,  68.,   2.,  15.,   9., 157.,  27.,  37.,
             2.,  55.,  86.,  14.,   4.,  40.,  36.,  47.,   9.,  61.,   1.,
            26.,  13.,  28.,  77.,  26.,  45.,  25.,  18.,  29.,  15.,  38.,
             2.,   2.,  19.,  27.,  14.,  13.,  19.,  54.,  70.,  28.])



**Calculate estimates on data samples**



```python
print("Estimated lambda parameter for baby arrival using analytic MLE approximation: ", np.mean(baby_arrival_times))
```


    Estimated lambda parameter for baby arrival using analytic MLE approximation:  33.25581395348837


**Generate n=10000 additional data samples from the Data Generating Process**



```python
additional_arrival_samples = scipy.stats.expon.rvs(scale=np.mean(baby_arrival_times), size=(10000,len(baby_arrival_times)))
```


**Repeat Calculation of estimate on each of the n=10000 additional data samples from the Parametric Bootstrapping**



```python
additional_arrival_estimates = np.mean(additional_arrival_samples, axis=1)
```




```python
additional_arrival_estimates
```





    array([37.13873499, 24.9747385 , 28.45942289, ..., 28.75582071,
           25.1112716 , 35.7755387 ])



**Collectively use the calculated estimate to construct confidence intervals**



```python
fig, ax = plt.subplots()

viz_estimates(
    ax,
    additional_arrival_estimates,
    'Estimates of Additional Baby Arrival Samples',
    'salmon',
    'estimates',
    'Baby Arrivals: Parametric Bootstrapping',
    np.mean(baby_arrival_times),
    'Estimate from Original Sample'
)

```



![png](freq-lab_files/freq-lab_74_0.png)


Are the estimates for the original models in the 95% confidence interval?



```python
arrivals_percentile_interval = np.percentile(additional_arrival_estimates, [2.5, 97.5])

arrivals_percentile_interval -= arrivals_percentile_interval[0]

arrivals_percentile_interval -= arrivals_percentile_interval[1]/2
```




```python
print("95% confidence interval sample 1: ", arrivals_percentile_interval+ np.mean(baby_arrival_times))
```


    95% confidence interval sample 1:  [23.36307171 43.1485562 ]


### Non Parametric Bootstrapping

In the earlier cases we were pretty confident of the underlying statistical model of the data generating process.  We could at least pin it down within a distribution family.  What about the cases where we're not really clear about the distribution of data?  What process can we follow in those cases?  We can still engage in bootstrapping, but instead of using estimated MLE parameters and the purported probability distribution of the data-generating process to construct additional samples, we can create the samples by **resampling directly from the data in the original sample**.  The idea (discovered in 1982 by Zac Effron) is that the data in the original sample is a good representation of the population.  If we sample with replacement from the original, our mewly generated samples should come from the same data generating process as the original.  How do we go about this process?

* Create new samples by sampling with replacement from the original sample
* Each new sample is called **bootsrap replication**
* Each replication should be the same size as the original sample

### The Non-Parametric Bootstrapping Process

1. We're provided with our data sample $S = \{X_1, X_2, ..., X_k\}$

2. We calculate our relevant estimate(s) or statistic(s) on that sample **sampling with replacement with the sample size equal to the original sample.**

4. On each of those n data samples we repeat the calculations of the estimate(s) or statistic(s) 

5. Collectively use the calculated estimate(s)/statistic(s) on the generated samples to construct a confidence interval

Our inference is our original estimate(s)/statistic(s) and the constructed confidence intervals from the estimate(s)/statistic(s) from our n samples from the data generating process

### Example of Non-Parametric Bootstrapping: Baby Arrival

Let's apply the process above for our baby arrival dataset.

**We're provided our data sample (this is the same as the parametric and idealized case)**



```python
# baby_arrival_times = timediffs.values

# baby_arrival_times
```


**Calculate estimates on data samples (this is the same as the parametric and idealized case)**



```python
print("Estimated lambda parameter for baby arrival using analytic MLE approximation: ", np.mean(baby_arrival_times))
```


    Estimated lambda parameter for baby arrival using analytic MLE approximation:  33.25581395348837


**Generate n=10000 additional data samples from the original sample using sampling with replacement**



```python
additional_arrival_samples_np = np.random.choice(timediffs.values, size=(10000,len(baby_arrival_times)), replace=True)
```


**Repeat Calculation of estimate on each of the n=10000 additional data samples from the Non-Parametric Bootstrapping**



```python
additional_arrival_estimates_np = np.mean(additional_arrival_samples_np, axis=1)
```




```python
additional_arrival_estimates_np
```





    array([29.60465116, 37.90697674, 29.93023256, ..., 32.13953488,
           31.93023256, 45.06976744])



**Collectively use the calculated estimate to construct confidence intervals**



```python
fig, ax = plt.subplots()

viz_estimates(
    ax,
    additional_arrival_estimates_np,
    'Estimates of Additional Baby Arrival Samples',
    'salmon',
    'estimates',
    'Baby Arrivals: NP Bootstrapping',
    np.mean(baby_arrival_times),
    'Estimate from Original Sample'
)

```



![png](freq-lab_files/freq-lab_93_0.png)


Are the estimates for the original models in the 95% confidence interval?



```python
arrivals_percentile_interval_np = np.percentile(additional_arrival_estimates_np, [2.5, 97.5])

arrivals_percentile_interval_np -= arrivals_percentile_interval_np[0]

arrivals_percentile_interval_np -= arrivals_percentile_interval_np[1]/2
```




```python
print("95% confidence interval sample: ", arrivals_percentile_interval_np+ np.mean(baby_arrival_times))
```


    95% confidence interval sample:  [24.61627907 41.89534884]


### When does Bootstrapping Do Poorly (and Well)

* Parametric Bootstrapping will converge to the wrong results if you use the wrong data generating process model
* Parametric Bootstrapping will converge faster if you use an appropriately specified model
* Bootstrapping in general assumes that the estimates/statistics that you calculate aren't sensitive to small changes in the bootstrapping process
    * For parametric bootstrapping small changes in the parameters --> small changes in the sampling distribution
    * For non-parametric bootstrapping small changes in the data points (addition or removal) --> small cahanges in the sampling distribution
    * For non-parametric be careful with outliers.
