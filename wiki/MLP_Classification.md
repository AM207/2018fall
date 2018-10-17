---
title: Classification using Multi-Layer Perceptrons
shorttitle: MLP_Classification
notebook: MLP_Classification.ipynb
noline: 1
summary: ""
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
import seaborn.apionly as sns
sns.set_context("poster")
```


Two additional imports here, seaborn and tqdm. Install via pip or conda



```python
c0=sns.color_palette()[0]
c1=sns.color_palette()[1]
c2=sns.color_palette()[2]
```




```python
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

def points_plot(ax, Xtr, Xte, ytr, yte, clf_predict, colorscale=cmap_light, cdiscrete=cmap_bold, alpha=0.3, psize=20):
    h = .02
    X=np.concatenate((Xtr, Xte))
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))


    Z = clf_predict(np.c_[xx.ravel(), yy.ravel()])
    ZZ = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, ZZ, cmap=cmap_light, alpha=alpha, axes=ax)
    showtr = ytr
    showte = yte
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=showtr-1, cmap=cmap_bold, s=psize, alpha=alpha,edgecolor="k")
    # and testing points
    ax.scatter(Xte[:, 0], Xte[:, 1], c=showte-1, cmap=cmap_bold, alpha=alpha, marker="s", s=psize+10)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return ax,xx,yy
```


## Create some noisy moon shaped data

In order to illustrate classification by a MLP, we first create some noisy moon shaped data. The *noise level* here and the *amount of data* is the first thing you might want to experiment with to understand the interplay of amount of data, noise level, number of parameters in the model we use to fit, and overfitting as illustrated by jagged boundaries.

We standardize the data so that it is distributed about 0 as well



```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataX, datay = make_moons(noise=0.35, n_samples=400)
dataX = StandardScaler().fit_transform(dataX)
X_train, X_test, y_train, y_test = train_test_split(dataX, datay, test_size=.4)
```




```python
h=.02
x_min, x_max = dataX[:, 0].min() - .5, dataX[:, 0].max() + .5
y_min, y_max = dataX[:, 1].min() - .5, dataX[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.gca()
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.5, s=30)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.5, s=30)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
```





    (-2.5061432692010377, 3.1138567307989669)




![png](MLP_Classification_files/MLP_Classification_7_1.png)




```python
import torch
import torch.nn as nn
from torch.nn import functional as fn
from torch.autograd import Variable
import torch.utils.data
```


## Writing a Multi-Layer Perceptron class

We wrap the construction of our network 



```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity = fn.tanh, additional_hidden_wide=0):
        super(MLP, self).__init__()
        self.fc_initial = nn.Linear(input_dim, hidden_dim)
        self.fc_mid = nn.ModuleList()
        self.additional_hidden_wide = additional_hidden_wide
        for i in range(self.additional_hidden_wide):
            self.fc_mid.append(nn.Linear(hidden_dim, hidden_dim))
        if self.additional_hidden_wide != -1:
            self.fc_final = nn.Linear(hidden_dim, output_dim)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        x = self.fc_initial(x)
        x = self.nonlinearity(x)
        if self.additional_hidden_wide != -1:
            for i in range(self.additional_hidden_wide):
                x = self.fc_mid[i](x)
                x = self.nonlinearity(x)
            x = self.fc_final(x)
        return x
```


We use it to train. Notice the double->float casting. Numpy defautlts to double but torch defaulta to float to enable memory efficient GPU usage.



```python
np.dtype(np.float).itemsize, np.dtype(np.double).itemsize
```





    (8, 8)



But torch floats are 4 byte as can be seen from here: http://pytorch.org/docs/master/tensors.html


### Training the model

Points to note:

- printing a model prints its layers, handy. Note that we implemented layers as functions. The autodiff graph is constructed on the fly on the first forward pass and used in backward.
- we had to cast to float
- `model.parameters` gives us params, `model.named_parameters()` gives us assigned names. You can set your own names when you create a layer
- we create an iterator over the data, more precisely over batches by doing `iter(loader)`. This dispatches to the `__iter__` method of the dataloader. (see https://github.com/pytorch/pytorch/blob/4157562c37c76902c79e7eca275951f3a4b1ef78/torch/utils/data/dataloader.py#L416) Always explore source code to understand what is going on



```python
model2 = MLP(input_dim=2, hidden_dim=3, output_dim=2, nonlinearity=fn.tanh, additional_hidden_wide=1)
print(model2)
criterion = nn.CrossEntropyLoss(size_average=True)
dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
lr, epochs, batch_size = 1e-1 , 1000 , 64
optimizer = torch.optim.SGD(model2.parameters(), lr = lr )
accum=[]
for k in range(epochs):
    localaccum = []
    for localx, localy in iter(loader):
        localx = Variable(localx.float())
        localy = Variable(localy.long())
        output = model2.forward(localx)
        loss = criterion(output, localy)
        model2.zero_grad()
        loss.backward()
        optimizer.step()
        localaccum.append(loss.data[0])
    accum.append(np.mean(localaccum))
plt.plot(accum);                      
```


    MLP(
      (fc_initial): Linear(in_features=2, out_features=3)
      (fc_mid): ModuleList(
        (0): Linear(in_features=3, out_features=3)
      )
      (fc_final): Linear(in_features=3, out_features=2)
    )



![png](MLP_Classification_files/MLP_Classification_14_1.png)


The out put from the foward pass is run on the entire test set. Since pytorch tracks layers upto but before the loss, this handily gives us the softmax output, which we can then use `np.argmax` on.



```python
testoutput = model2.forward(Variable(torch.from_numpy(X_test).float()))
testoutput
```





    Variable containing:
    -1.3997  1.1056
     0.4321 -0.1878
     0.2325 -0.0586
    -1.4748  1.1468
     3.1649 -2.2684
     2.9101 -2.0822
     1.1887 -0.7474
     1.4796 -1.0149
    -3.5469  2.5426
    -3.7733  2.6948
    -0.2148  0.2858
     3.9723 -2.8604
    -1.9199  1.4439
     0.3520 -0.1263
     3.3732 -2.4237
     2.5095 -1.7848
     2.8704 -2.0525
    -1.8319  1.3913
    -0.0806  0.1818
    -0.3542  0.3635
    -0.7936  0.6824
    -0.0041  0.1306
     2.1060 -1.4357
     0.2766 -0.1130
     0.8709 -0.5103
    -1.3461  1.0524
     3.3596 -2.4143
    -0.7062  0.6287
     1.5634 -1.0407
    -0.0492  0.1600
     0.0524  0.0948
     1.8808 -1.3182
    -3.4808  2.4987
    -1.9241  1.4596
     2.3021 -1.5925
    -0.7218  0.6272
    -3.6070  2.5861
    -3.5477  2.5420
    -2.4397  1.7995
    -1.6359  1.2494
     0.9166 -0.5441
    -2.9952  2.1791
    -0.9801  0.8223
    -1.1452  0.9344
     0.1102  0.0475
    -3.6323  2.6063
    -2.7851  2.0277
    -3.1108  2.2546
    -0.5392  0.5021
    -0.8784  0.7372
    -3.9028  2.7874
     2.6638 -1.8992
    -3.0897  2.2326
     2.7642 -1.9722
    -2.2850  1.6921
    -1.7232  1.3218
     4.2042 -3.0399
     1.5363 -1.0084
     1.0733 -0.6613
     4.3506 -3.1474
    -0.8530  0.7152
     4.2824 -3.0974
    -0.0404  0.1513
     0.1457 -0.0290
    -0.9239  0.7579
    -1.3236  1.0476
    -2.5510  1.8800
     1.2802 -0.8159
    -3.4907  2.5036
    -0.2133  0.2611
     1.7668 -1.1816
     3.1958 -2.2577
    -0.7845  0.6720
    -2.2681  1.6901
    -3.7748  2.6956
     2.9403 -2.0981
     1.5907 -1.0496
     1.3040 -0.8349
     0.1481  0.0123
     0.8719 -0.5111
    -0.3822  0.4076
    -1.1132  0.8936
     1.1114 -0.6897
     0.0516  0.0699
    -1.0205  0.8473
    -0.9822  0.8226
    -1.4213  1.1121
     1.3364 -0.8580
    -1.7266  1.3185
     1.8002 -1.2186
    -1.1045  0.8875
     0.3044 -0.0899
    -1.6213  1.2412
    -0.3257  0.3136
    -0.6594  0.5660
    -0.0061  0.1307
    -0.0723  0.1303
     2.2622 -1.5531
     0.6425 -0.3400
     3.5527 -2.5243
     1.4703 -0.9585
     0.4742 -0.2152
    -3.9635  2.8235
    -2.0605  1.5491
    -3.7193  2.6581
     1.3573 -0.9252
     2.8911 -2.0675
    -2.6437  1.9390
    -0.7654  0.6714
     1.1626 -0.7502
     0.3224 -0.1071
    -2.2417  1.6723
     0.5332 -0.3175
     0.6907 -0.3770
     0.0929  0.0572
     2.2506 -1.5903
    -2.1372  1.5943
     2.4597 -1.7284
    -0.9419  0.7867
     1.3708 -0.8835
     0.4274 -0.1803
    -1.8787  1.4144
    -2.8720  2.0867
    -3.2267  2.3252
     3.5142 -2.5284
     0.5442 -0.2674
    -1.0442  0.8467
     0.2991 -0.0894
    -0.5332  0.4947
    -1.9302  1.4491
    -0.3737  0.4014
    -0.2904  0.3255
    -3.1304  2.2694
     0.2306 -0.0791
    -3.0481  2.2154
     0.7065 -0.4357
     0.0990  0.0610
    -1.1890  0.9566
     0.2239 -0.0386
     3.4539 -2.4829
    -0.2503  0.3136
    -3.7716  2.6952
    -3.8336  2.7356
    -0.5169  0.4874
     3.9846 -2.8736
    -2.4657  1.8112
    -1.8777  1.4178
    -1.7955  1.3672
    -0.2697  0.3276
     2.2395 -1.5380
    -2.7763  2.0324
     0.3578 -0.1327
     0.0418  0.0952
     1.6378 -1.0839
     1.5454 -1.0322
    -3.6984  2.6439
     0.4830 -0.2253
    -0.1812  0.2496
     0.9076 -0.5379
    -3.3711  2.4227
    [torch.FloatTensor of size 160x2]





```python
y_pred = testoutput.data.numpy().argmax(axis=1)
y_pred
```





    array([1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
           0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1,
           1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0,
           1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
           0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1,
           0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])



You can write your own but we import some metrics from sklearn



```python
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)
```





    array([[66, 16],
           [ 5, 73]])





```python
accuracy_score(y_test, y_pred)
```





    0.86875000000000002



We can wrap this machinery in a function, and pass this function to `points_plot` to predict on a grid and thus give us a boundary viz



```python
def make_pred(X_set):
    output = model2.forward(Variable(torch.from_numpy(X_set).float()))
    return output.data.numpy().argmax(axis=1)
```




```python
with sns.plotting_context('poster'):
    ax = plt.gca()
    points_plot(ax, X_train, X_test, y_train, y_test, make_pred);
```



![png](MLP_Classification_files/MLP_Classification_23_0.png)


## Making a `scikit-learn` like interface

Since we want to run many experiments, we'll go ahead and wrap our fitting process in a sklearn style interface. Another example of such an interface is [here](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/3_neural_net.py)



```python
from tqdm import tnrange, tqdm_notebook
class MLPClassifier:
    
    def __init__(self, input_dim, hidden_dim, 
                 output_dim, nonlinearity = fn.tanh, 
                 additional_hidden_wide=0):
        self._pytorch_model = MLP(input_dim, hidden_dim, output_dim, nonlinearity, additional_hidden_wide)
        self._criterion = nn.CrossEntropyLoss(size_average=True)
        self._fit_params = dict(lr=0.1, epochs=200, batch_size=64)
        self._optim = torch.optim.SGD(self._pytorch_model.parameters(), lr = self._fit_params['lr'] )
        
    def __repr__(self):
        num=0
        for k, p in self._pytorch_model.named_parameters():
            numlist = list(p.data.numpy().shape)
            if len(numlist)==2:
                num += numlist[0]*numlist[1]
            else:
                num+= numlist[0]
        return repr(self._pytorch_model)+"\n"+repr(self._fit_params)+"\nNum Params: {}".format(num)
    
    def set_fit_params(self, *, lr=0.1, epochs=200, batch_size=64):
        self._fit_params['batch_size'] = batch_size
        self._fit_params['epochs'] = epochs
        self._fit_params['lr'] = lr
        self._optim = torch.optim.SGD(self._pytorch_model.parameters(), lr = self._fit_params['lr'] )
        
    def fit(self, X_train, y_train):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self._fit_params['batch_size'], shuffle=True)
        self._accum=[]
        for k in tnrange(self._fit_params['epochs']):
            localaccum = []
            for localx, localy in iter(loader):
                localx = Variable(localx.float())
                localy = Variable(localy.long())
                output = self._pytorch_model.forward(localx)
                loss = self._criterion(output, localy)
                self._pytorch_model.zero_grad()
                loss.backward()
                self._optim.step()
                localaccum.append(loss.data[0])
            self._accum.append(np.mean(localaccum))
        
    def plot_loss(self):
        plt.plot(self._accum, label="{}".format(self))
        plt.legend()
        plt.show()
        
    def plot_boundary(self, X_train, X_test, y_train, y_test):
        points_plot(plt.gca(), X_train, X_test, y_train, y_test, self.predict);
        plt.text(1, 1, "{}".format(self), fontsize=12)
        plt.show()
        
    def predict(self, X_test):
        output = self._pytorch_model.forward(Variable(torch.from_numpy(X_test).float()))
        return output.data.numpy().argmax(axis=1)
        
```


Some points about this:

- we provide the ability to change the fitting parameters
- by implementing a `__repr__` we let an instance of this class print something useful. Specifically we created a count of the number of parameters so that we can get a comparison of data size to parameter size.

## The simplest model, and a more complex model



```python
logistic = MLPClassifier(input_dim=2, hidden_dim=2, output_dim=2, nonlinearity=lambda x: x, additional_hidden_wide=-1)
logistic.set_fit_params(epochs=1000)
print(logistic)
logistic.fit(X_train,y_train)
```


    MLP(
      (fc_initial): Linear(in_features=2, out_features=2)
      (fc_mid): ModuleList(
      )
    )
    {'lr': 0.1, 'epochs': 1000, 'batch_size': 64}
    Num Params: 6




    




```python
with sns.plotting_context('poster'):
    logistic.plot_loss()
```



![png](MLP_Classification_files/MLP_Classification_29_0.png)




```python
ypred = logistic.predict(X_test)
#training and test accuracy
accuracy_score(y_train, logistic.predict(X_train)), accuracy_score(y_test, ypred)
```





    (0.84583333333333333, 0.80625000000000002)





```python
with sns.plotting_context('poster'):
    logistic.plot_boundary(X_train, X_test, y_train, y_test)
```



![png](MLP_Classification_files/MLP_Classification_31_0.png)




```python
clf = MLPClassifier(input_dim=2, hidden_dim=20, output_dim=2, nonlinearity=fn.tanh, additional_hidden_wide=1)
clf.set_fit_params(epochs=1000)
print(clf)
clf.fit(X_train,y_train)
```


    MLP(
      (fc_initial): Linear(in_features=2, out_features=20)
      (fc_mid): ModuleList(
        (0): Linear(in_features=20, out_features=20)
      )
      (fc_final): Linear(in_features=20, out_features=2)
    )
    {'lr': 0.1, 'epochs': 1000, 'batch_size': 64}
    Num Params: 522




    




```python
with sns.plotting_context('poster'):
    clf.plot_loss()
```



![png](MLP_Classification_files/MLP_Classification_33_0.png)




```python
ypred = clf.predict(X_test)
#training and test accuracy
accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, ypred)
```





    (0.875, 0.875)





```python
with sns.plotting_context('poster'):
    clf.plot_boundary(X_train, X_test, y_train, y_test)
```



![png](MLP_Classification_files/MLP_Classification_35_0.png)


## Experimentation Space

Here is space for you to play. You might want to collect accuracies on the traing and test set and plot on a grid of these parameters or some other visualization. Notice how you might want to adjust number of epochs for convergence.



```python
for additional in [0, 2, 4]:
    for hdim in [2, 10, 100, 1000]:
        print('====================')
        print('Additional', additional, "hidden", hdim)
        clf = MLPClassifier(input_dim=2, hidden_dim=hdim, output_dim=2, nonlinearity=fn.tanh, additional_hidden_wide=additional)
        if additional > 2 and hdim > 50:
            clf.set_fit_params(epochs=1000)
        else:
            clf.set_fit_params(epochs=500)
        print(clf)
        clf.fit(X_train,y_train)
        with sns.plotting_context('poster'):
            clf.plot_loss()
            clf.plot_boundary(X_train, X_test, y_train, y_test)
        print("Train acc", accuracy_score(y_train, clf.predict(X_train)))
        print("Test acc", accuracy_score(y_test, clf.predict(X_test)))

```


    ====================
    Additional 0 hidden 2
    MLP(
      (fc_initial): Linear(in_features=2, out_features=2)
      (fc_mid): ModuleList(
      )
      (fc_final): Linear(in_features=2, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 12




    



![png](MLP_Classification_files/MLP_Classification_37_3.png)



![png](MLP_Classification_files/MLP_Classification_37_4.png)


    Train acc 0.866666666667
    Test acc 0.8375
    ====================
    Additional 0 hidden 10
    MLP(
      (fc_initial): Linear(in_features=2, out_features=10)
      (fc_mid): ModuleList(
      )
      (fc_final): Linear(in_features=10, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 52




    



![png](MLP_Classification_files/MLP_Classification_37_8.png)



![png](MLP_Classification_files/MLP_Classification_37_9.png)


    Train acc 0.870833333333
    Test acc 0.88125
    ====================
    Additional 0 hidden 100
    MLP(
      (fc_initial): Linear(in_features=2, out_features=100)
      (fc_mid): ModuleList(
      )
      (fc_final): Linear(in_features=100, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 502




    



![png](MLP_Classification_files/MLP_Classification_37_13.png)



![png](MLP_Classification_files/MLP_Classification_37_14.png)


    Train acc 0.870833333333
    Test acc 0.8875
    ====================
    Additional 0 hidden 1000
    MLP(
      (fc_initial): Linear(in_features=2, out_features=1000)
      (fc_mid): ModuleList(
      )
      (fc_final): Linear(in_features=1000, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 5002




    



![png](MLP_Classification_files/MLP_Classification_37_18.png)



![png](MLP_Classification_files/MLP_Classification_37_19.png)


    Train acc 0.833333333333
    Test acc 0.8
    ====================
    Additional 2 hidden 2
    MLP(
      (fc_initial): Linear(in_features=2, out_features=2)
      (fc_mid): ModuleList(
        (0): Linear(in_features=2, out_features=2)
        (1): Linear(in_features=2, out_features=2)
      )
      (fc_final): Linear(in_features=2, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 24




    



![png](MLP_Classification_files/MLP_Classification_37_23.png)



![png](MLP_Classification_files/MLP_Classification_37_24.png)


    Train acc 0.875
    Test acc 0.825
    ====================
    Additional 2 hidden 10
    MLP(
      (fc_initial): Linear(in_features=2, out_features=10)
      (fc_mid): ModuleList(
        (0): Linear(in_features=10, out_features=10)
        (1): Linear(in_features=10, out_features=10)
      )
      (fc_final): Linear(in_features=10, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 272




    



![png](MLP_Classification_files/MLP_Classification_37_28.png)



![png](MLP_Classification_files/MLP_Classification_37_29.png)


    Train acc 0.8625
    Test acc 0.90625
    ====================
    Additional 2 hidden 100
    MLP(
      (fc_initial): Linear(in_features=2, out_features=100)
      (fc_mid): ModuleList(
        (0): Linear(in_features=100, out_features=100)
        (1): Linear(in_features=100, out_features=100)
      )
      (fc_final): Linear(in_features=100, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 20702




    



![png](MLP_Classification_files/MLP_Classification_37_33.png)



![png](MLP_Classification_files/MLP_Classification_37_34.png)


    Train acc 0.891666666667
    Test acc 0.85
    ====================
    Additional 2 hidden 1000
    MLP(
      (fc_initial): Linear(in_features=2, out_features=1000)
      (fc_mid): ModuleList(
        (0): Linear(in_features=1000, out_features=1000)
        (1): Linear(in_features=1000, out_features=1000)
      )
      (fc_final): Linear(in_features=1000, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 2007002




    



![png](MLP_Classification_files/MLP_Classification_37_38.png)



![png](MLP_Classification_files/MLP_Classification_37_39.png)


    Train acc 0.8375
    Test acc 0.8125
    ====================
    Additional 4 hidden 2
    MLP(
      (fc_initial): Linear(in_features=2, out_features=2)
      (fc_mid): ModuleList(
        (0): Linear(in_features=2, out_features=2)
        (1): Linear(in_features=2, out_features=2)
        (2): Linear(in_features=2, out_features=2)
        (3): Linear(in_features=2, out_features=2)
      )
      (fc_final): Linear(in_features=2, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 36




    



![png](MLP_Classification_files/MLP_Classification_37_43.png)



![png](MLP_Classification_files/MLP_Classification_37_44.png)


    Train acc 0.508333333333
    Test acc 0.4875
    ====================
    Additional 4 hidden 10
    MLP(
      (fc_initial): Linear(in_features=2, out_features=10)
      (fc_mid): ModuleList(
        (0): Linear(in_features=10, out_features=10)
        (1): Linear(in_features=10, out_features=10)
        (2): Linear(in_features=10, out_features=10)
        (3): Linear(in_features=10, out_features=10)
      )
      (fc_final): Linear(in_features=10, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 492




    



![png](MLP_Classification_files/MLP_Classification_37_48.png)



![png](MLP_Classification_files/MLP_Classification_37_49.png)


    Train acc 0.8625
    Test acc 0.85
    ====================
    Additional 4 hidden 100
    MLP(
      (fc_initial): Linear(in_features=2, out_features=100)
      (fc_mid): ModuleList(
        (0): Linear(in_features=100, out_features=100)
        (1): Linear(in_features=100, out_features=100)
        (2): Linear(in_features=100, out_features=100)
        (3): Linear(in_features=100, out_features=100)
      )
      (fc_final): Linear(in_features=100, out_features=2)
    )
    {'lr': 0.1, 'epochs': 1000, 'batch_size': 64}
    Num Params: 40902




    



![png](MLP_Classification_files/MLP_Classification_37_53.png)



![png](MLP_Classification_files/MLP_Classification_37_54.png)


    Train acc 0.916666666667
    Test acc 0.84375
    ====================
    Additional 4 hidden 1000
    MLP(
      (fc_initial): Linear(in_features=2, out_features=1000)
      (fc_mid): ModuleList(
        (0): Linear(in_features=1000, out_features=1000)
        (1): Linear(in_features=1000, out_features=1000)
        (2): Linear(in_features=1000, out_features=1000)
        (3): Linear(in_features=1000, out_features=1000)
      )
      (fc_final): Linear(in_features=1000, out_features=2)
    )
    {'lr': 0.1, 'epochs': 1000, 'batch_size': 64}
    Num Params: 4009002




    



![png](MLP_Classification_files/MLP_Classification_37_58.png)



![png](MLP_Classification_files/MLP_Classification_37_59.png)


    Train acc 0.866666666667
    Test acc 0.80625

