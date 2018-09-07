## Classification

In machine learning, classification is the act of assigning data to one or the other category. Later in this course, we'll see probabilistic algorithms created for this purpose. But currently, let us focus on the "correctness" of these algorithms.

Such correctness is defined with respect to some set where we have the category labels. If we were to hide these labels, and ask our algorithm to predict them, how good would we do? How would we define 'good'?

Lets simplify the problem to classifying something into two categories: positive and negative, or 1 and 0...two choices that are mutually exclusive. The labels we have are called the actual or **observed** values, while the labels predicted by our algorithms are called the **predicted** value.

Perhaps our problem has two **predictors**, also called covariates or features. The algorithm gives us a classification boundary in this two dimensional space, and there are some misclassifications.

![](https://github.com/blueberrymusic/DeepLearningBookFigures-Volume1/raw/master/Chapter03-Probability/Figure-03-017.png)

The left hand panel here shows the actual boundary: greens are 1s or postives, reds are negative. The right panel shows a schematic version of the same diagram.

We can define the fraction of misclassifications as the misclassification rate. Ir the accuracy as the number of correct classifications, or 1 minus the misclassification rate.

Here we have 6 misclassifications out of 20, or a 70% accuracy.

If we wanted to characterize what was being misclassified in more detail, we might want to create a confusion matrix.

![](https://github.com/blueberrymusic/DeepLearningBookFigures-Volume1/raw/master/Chapter03-Probability/Figure-03-018.png)

This construct provides us more detail, showing us which of the actual values are being misclassified more. Here, the actual positives are being falsely classified as negative at a higher rate (called the false negative rate = 4/10) than actual negatives are being classified as positive (false positive rate = 2/10).

The **precision** is defined as the ratio of the number of the samples "properly" labelled positive to the number of samples we labelled positive. That is, how many of the so-called positive predictions were actually positive. This is the ratio of the true positives to the predicted positives: in our case, 6/8. this means that we can only be 75% sure that a any given sample thats been labelled positive has the correct label. A precision less than 1 tells us that there are some samples we reported as positive which were not, as this diagram makes clear.

![](https://github.com/blueberrymusic/DeepLearningBookFigures-Volume1/raw/master/Chapter03-Probability/Figure-03-025.png)



The recall, or the true-positive rate, is the percentage of the actually positive samples that we correctly labelled. Here it is 6/10 or 60%. A recall less than 1 means that we missed some actually positive samples. A similar diagram can be created for recall.

![](https://github.com/blueberrymusic/DeepLearningBookFigures-Volume1/raw/master/Chapter03-Probability/Figure-03-027.png)

## A medical problem

We've set this problem up as a classification problem, but we can calculate these quantities anywhere we have an explicit test, or even type 1 and type 2 errors.

Suppose a new, cheap test can help us figure out if a fetus had Down's Syndrome or not (such as one which measures the thickness of the nuchal membrane). It is not a particularly accurate test, as compared to a more invasive (and risky) amnio test. 

The cheap test has both the types of problems above: sometimes it misses the downs, and comes up negative even when the fetus has downs. And sometimes it comes up positive, when it shouldnt have.

If we label positive as having downs, then a true positive is when a fetus has downs and the cheap test correctly predicts it, while a false positive is the prediction of downs in a non-downs fetus.

Let's say then that the test has a true positive rate of 99%, that is 99% of the time a downs fetus is correctly diagnosed. Thus the False negative rate is 1%.

The test is mildly worse for fetus's that dont have downs (observed negatives). The True negative rate is 98%, which means that the false positive rate is 2%, or that 2% of non-downs fetuses get an incorrect positive diagnosis.

You might this be tempted to write down a confusion matrix which looks like this:

![](https://github.com/blueberrymusic/DeepLearningBookFigures-Volume1/raw/master/Chapter03-Probability/Figure-03-040.png)

This matrix is WRONG. Why?

### We are missing the priors

What we do know, is that historically speaking, most fetus' do not have downs. This is even more true for younger mothers. So we need to ask, what is the known (experimental) prior rate for having downs.

Let us say, then, that for the age cohort of interest, the prior rate for downs is 1%.

That is in 10,000 fetuses, 100 might have Downs, and 9900 dont. Lets use these numbers in the matrix above. We now get:

![](https://github.com/blueberrymusic/DeepLearningBookFigures-Volume1/raw/master/Chapter03-Probability/Figure-03-042.png)

This is now a precision of 99/297, or 33%!

In other words, even given our test has a 99% probability of correctly diagnosing downs, 67% of the time it claims a positive result, the baby does not have downs. Most of our positive Diagnoses are wrong!

One might consider sending all the wrongly predicted positives for an amnio, but that test while much more accurate, has a risk of hurting the fetus.

What about negative diagnosis. Does this need a follow up? A negative prediction, the right hand column, has only 1 in 9700 chance of being wrong!

We chose a particular example here to work out, but the logic here is enshrined in Bayes Theorem. There, we are interested in (O being observed and P being predicted)  $P(O+ | P+)$.

$$P(O+ | P+) P(P+) = P(P+| O+) P(O+)$$

$$P(O+ | P+) = \frac{ P(P+| O+) P(O+)}{P(P+)}$$

$$P(O+ | P+) = \frac{ P(P+| O+) P(O+)}{P(P+|O+)P(O+) + P(P+|O-)P(O-)}$$

$$ = 0.99 \times 0.01 / (0.99 \times 0.01 + 0.02 \times 0.99) = 0.33$$

Suppose the prior probability was not 1%, but rather $x$, which might be a function of mother's age and other factors (a linear regression, if you like)..and you can see how we can combine disparate pieces of information using Bayes theorem..a classifier (perhaps a logistic regression) based on some test (or just a test) with another model for the prior. Then plot as a function of x what the precision will be. At some point you will need to make the following decision...at what value of x do you order the follow up test...