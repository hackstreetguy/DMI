
import pandas as pd
import numpy as np
import numpy.linalg as npla
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from functools import partial
import pylab
import random


#Pre-Processing of the data
data = pd.read_csv("letter-recognition.csv")
data.columns = ['letter', 'xbox'  , 'ybox' ,'width' ,'height','onpix' ,'xbar'  ,'ybar'  ,'x2bar' , 'y2bar' ,'xybar' , 'x2ybar','xy2bar', 'xedge' ,'xedgey', 'yedge' ,'yedgex']
order = list(np.sort(data['letter'].unique()))
random_state = 100


#Plotting letter vs xbox
plt.figure(figsize=(5, 5))
sns.barplot(x='letter',y='xbox',data=data, palette="deep", order=order)
#Plottin
plt.figure(figsize=(5, 5))
sns.barplot(x='letter',y='xbar', data=data, palette="deep",order=order)

plt.figure(figsize=(5, 5))
sns.barplot(x='letter', y='x2bar', data=data, palette="deep",order=order)

attribute_mean = data.groupby('letter').mean()
# attribute_mean

round(data.drop('letter', axis=1).mean(), 2)

X = data.drop("letter", axis = 1)
Y = data['letter']


#splitting the train test into 7:3
X_scaled = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.3, random_state = random_state)


#Code for Markov Chain Monte Carlo and Gibs sampling by Bruce Walsh

def gaussian_function(x, sigma, sampled=None):
    if sampled is None: 
        L = npla.cholesky(sigma)
        z = np.random.randn(x.shape[0], 1)
        return np.dot(L, z+x)
    else:
        return np.exp(-0.5*np.dot( (x-sampled).T, np.dot(npla.inv(sigma), (x-sampled))))[0,0]


def gaussian_function_1d(x, sigma, sampled=None):
    if sampled is None:
        return sigma*np.random.randn(1)[0]
    else:
        return np.exp(-0.5( (x-sampled)**2)/sigma**2)


def chi_sq(x, sampled = None, n = 0):
    if sampled is None:
        return np.random.chisquare(n)
    else:
        return np.power(sampled,0.5*n - 1)*np.exp(-0.5*sampled)


def inv_chi_sq(theta, n, a):
    return np.power(theta, -0.5*n)*np.exp(-a/(2*theta))


def metropolis(f, proposal, old):
    new = proposal(old)
    input_alpha = np.min([f(new)/f(old), 1])
    u = np.random.uniform()
    cnt = 0
    if (u < input_alpha):
        old = new
        cnt = 1
    return old, cnt


def met_hast(f, proposal, old):
    new = proposal(old)
    input_alpha = np.min([(f(new)*proposal(new, sampled = old))/(f(old) * proposal(old, sampled = new)), 1])
    u = np.random.uniform()
    cnt = 0
    if (u < input_alpha):
        old = new
        cnt = 1
    return old, cnt

def run_chain(chainer, f, proposal, start, n, take=1):
    count = 0
    provided_samples = [start]
    for i in range(n):
        start, c = chainer(f, proposal, start)
        count = count + c
        if i%take == 0:
            provided_samples.append(start)
    return provided_samples, count

def uni_prop(x, frm, to, sampled=None):
    return np.random.uniform(frm, to)


# print(type(X_train))
# print((y_train.to_numpy()))

def markov_sampling_function(X_train, Y_train, k = 5, q = 1.2):

    sample_initialize = np.concatenate((np.vstack(X_train), np.vstack(Y_train.to_numpy())), axis = 1)
    Dtr = random.sample(list(sample_initialize), 2000)

    m = len(Dtr)
    print("M : " , m)
    mneg = 0
    mplus = 0

    index = np.random.choice(len(Dtr), 1, replace=False)  
    Dtra = np.array(Dtr)
    print(type(Dtra))
    zt = Dtra[index][0]

    print("zt : ", zt)
    if m%2 == 0:
        if zt[16] == 'A':
            mplus += 1;
        else:
            mneg += 1
    samp = []

    Pd = 0
    Pdd = 0
    print(zt)
    while(mplus + mneg < m ):
        star_z = Dtra[np.random.choice(len(Dtr), 1, replace=False)][0]
        P = 1
        yt = zt[16]
        zt = star_z
        if P == 1:
            if zt[16] == yt:
                samp.append(star_z) 
            else:
                samp.append(star_z) 

        if len(samp) == k:
            Pdd = q*P
            samp.append(star_z)
            
        ztp1 = star_z
        if yt == 'A':
            mplus += 1
        else:
            mneg += 1

        if P > 1 or Pd > 1 or Pdd > 1:
            samp.append(star_z)
    return samp

taken_sample = np.array(markov_sampling_function(X_train, y_train))
print(taken_sample.shape)
X_train = taken_sample[:, 0:16]
y_train = taken_sample[:, 16]
# print(X_train.shape)
# print(y_train.shape)

Y_train = []
for i in y_train:
    Y_train.append(ord(i))

# print(X_test.shape)


#Linear Kernel
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, Y_train)

y_pred = model_linear.predict(X_test)

Y_test = []
for i in y_test:
    Y_test.append(ord(i))

acc = metrics.accuracy_score(y_true=Y_test, y_pred=y_pred)
print("Accuracy Kernel: Linear :")
print(acc)

#RBF Kernel

non_linear_model = SVC(kernel='rbf')
non_linear_model.fit(X_train, Y_train)
y_pred = non_linear_model.predict(X_test)

acc = metrics.accuracy_score(y_true=Y_test, y_pred=y_pred)
print("Accuracy Kernel:Rbf :")
print(acc)

non_linear_model = SVC(kernel='poly')
non_linear_model.fit(X_train, Y_train)
y_pred = non_linear_model.predict(X_test)

acc = metrics.accuracy_score(y_true=Y_test, y_pred=y_pred)
print("Accuracy Kernel:Polynomial :")
print(acc)
print()

folds = KFold(n_splits = 5, shuffle = True, random_state = random_state)
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]

model = SVC(kernel="rbf")

model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

model_cv.fit(X_train, y_train)

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results['param_C'] = cv_results['param_C'].astype('int')
plt.figure(figsize=(16,8))

plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"],color='#58b970')
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"],color='#58b970')
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"],color='#58b970')
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("Best score :", best_score)
print("Best Hyperparameters:", best_hyperparams)

model = SVC(C=1000, gamma=0.01, kernel="rbf")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy Kernel:hyperplane:")
print(acc)
plt.show()