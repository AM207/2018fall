---
title: "AM207 Fall 2018 edtition"
layout: "default"
noline: 1
---

## Who?

**Instructor:** Rahul Dave (IACS, mailto:rahuldave@gmail.com )

**TFs:**

Patrick Ohiomoba, Will Claybaugh, Peter Chang, Gal Koplewitz

## When?

Tuesday 11.30am - 1pm, Lecture. Compulsory to attend. Maxwell Dworkin G115.

Thursday 11.30am - 1pm, Lecture. Compulsory to attend. Maxwell Dworkin G115.

Fridays 11am - 12.30pm Lab. Compulsory to attend. Pierce 301.

## Office Hours

Always in B125, Maxwell Dworkin.

- anytime by appointment with any of us.
- Rahul: Tue and Thu 1.30 - 2.30pm. ONLINE Thu 2.30 - 3.30pm.
- Will Tuesday 4 - 5pm
- Wed Patrick  4 - 7:30pm. ONLINE Wed 6:30 - 7:30pm
- Thu Patrick 4.30 - 5:30pm, Peter 5:30 - 6:30pm


## Online?

- web site at https://am207.github.io/2018spring/ (you are reading it)
- github repo at https://github.com/am207/2018spring (clone this fresh for each class or fork it and set this original up as a remote)
- discussions at [Piazza](https://piazza.com/class/jcr8h7o7yap73p)

## FAQ

**How onerous is it?**

Quite. This is a hard class.

**What is this class about?**

Develops skills for computational research with focus on stochastic approaches, emphasizing implementation and examples. Stochastic methods make it feasible to tackle very diverse problems when the solution space is too large to explore systematically, or when microscopic rules are known, but not the macroscopic behavior of a complex system. Methods are illustrated with examples from a wide variety of fields, like biology, finance, and physics. We tackle Bayesian methods of data analysis as well as various stochastic optimization methods. Topics include stochastic optimization such as stochastic gradient descent (SGD) and simulated annealing, Bayesian data analysis, Markov chain Monte Carlo (MCMC), and variational analysis.

This course is broadly about learning models from data. To do this, you typically want to solve an optimization problem.

But the problem with optimization problems is that they typically only give you point estimates. These are important. But we'd like to do inference: to learn the variability of our predictions.

Furthermore, functions may have many minima, and we want to explore them. For these reasons we want to do Stochastic Optimization. And since we might want to characterize our variability, we are typically interested in the distributions of our predictions. Bayesian Statistics offers us a principled way of doing this, allowing us to incorporate the regularization of fairly flexible models in a proper way.

A lot of interesting models involve "hidden variables" which are neither observable quantities, nor explicit parameters which we use to model our situations. Examples are unsupervised learning and hidden markov models. Indeed all of Bayesian Stats may be thought of as "marginalizing" over the hidden parameters of nature.

Finally not all data is stationary and IID. Things have time-dependence and memory. How do we deal with these temporal correlations? Or for that matter, spatial correlations as well. Also, parametric models have finite capacity. These days deep networks are interesting because of the large capacity they have, and the generalization of finite to infinite capacity leads us to non-parametric models and stochastic processes.

For these reasons, the main topics for our course are Stochastic optimization techniques, Bayesian Statistics, Hidden Variables, and Stochastic Processes.

Our workhorse will be Monte Carlo algorithms. Monte Carlo methods are a diverse class of algorithms that rely on repeated random sampling to compute the solution to problems whose solution space is too large to explore systematically or whose systemic behavior is too complex to model. This course introduces important principles of Monte Carlo techniques and demonstrates the power of these techniques with simple (but very useful) applications. Starting from the basic ideas of Bayesian analysis and Markov chain Monte Carlo samplers, we move to more recent developments such as slice sampling, and Hamiltonian Monte Carlo.  

 We will be using Python for all programming assignments and projects. All lectures will be recorded and should be available 24 hours after meeting time.

**Expected Learning Outcomes**

After successful completion of this course, you will be able to:

- optimize objectives such as loss functions using Stochastic Gradient Descent and Simulated annealing
- Analyse problems using bayesian approaches
- Perform Bayesian linear and generalized linear regression
- Understand the use of generative models in machine learning; thus be prepared to model complex data in both supervised and especially unsupervised ways
- Perform sampling and MCMC to solve a variety of problems such as integrals and inference
- Understand the philosophy of Machine Learning and Bayesian Stats
- Learn how and when to use parametric and non-parametric stochastic processes


**Who should take this class?**

The prerequisites for this class are programming knowledge, in Python, at the CS 109a level(or above), Data Science (at the level of cs109a or above), and Statistics at the level of Stat 110 (or above). If you are not sure about your preparedness for this class, come talk to us.
