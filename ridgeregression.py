import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn import linear_model
from inputoutput import *
import time
import datetime
from sklearn import preprocessing
from sklearn.linear_model import Ridge
#Load Data with pandas, and parse the
#first column into datetime
train = pd.read_csv('/Users/AndresEede/Downloads/train.csv', header = 0, error_bad_lines=False)
test = pd.read_csv('/Users/AndresEede/Downloads/test.csv', header = 0, error_bad_lines=False)
features = ['season','year','mnth','hr', 'holiday', 'weekday', 'workingday',
        'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
         ]


#the evaluation metric is the RMSE in the log domain,
#so we should transform the target columns into log domain as well.
for col in ['cnt']:
    train['log-' + col] = train[col].apply(lambda x: np.log1p(x))
    
# there appears to be a general increase in rentals over time, so days from start should be captured
#train['dateDays'] = (train.dteday - train.dteday[0]).astype('timedelta64[D]')

# create a categorical feature for day of the week (0=Monday to 6=Sunday)
#train['dayofweek'] = pd.DatetimeIndex(train.dteday).dayofweek

# create binary features which show if day is Saturday/Sunday
train['Saturday']=0
train.Saturday[train.weekday==6]=1
train['Sunday']=0
train.Sunday[train.weekday==7]=1

# remove old data features
#train = train.drop(['dayofweek'], axis=1)
train.drop(['instant'], axis=1)
# put continuous features into a dictionary
featureConCols = ['temp','atemp','hum','windspeed']
dataFeatureCon = train[featureConCols]
dataFeatureCon = dataFeatureCon.fillna( 'NA' ) #in case I missed any
X_dictCon = dataFeatureCon.T.to_dict().values() 

# put categorical features into a dictionary
featureCatCols = ['season', 'holiday','workingday','weathersit','Saturday', 'Sunday']
dataFeatureCat = train[featureCatCols]
dataFeatureCat = dataFeatureCat.fillna( 'NA' ) #in case I missed any
X_dictCat = dataFeatureCat.T.to_dict().values() 

# vectorize features
vec = DictVectorizer(sparse = False)
X_vec_cat = vec.fit_transform(X_dictCat)
X_vec_con = vec.fit_transform(X_dictCon)

# standardize data - zero mean and unit variance
scaler = preprocessing.StandardScaler().fit(X_vec_con)
X_vec_con = scaler.transform(X_vec_con)

# encode categorical features
enc = preprocessing.OneHotEncoder()
enc.fit(X_vec_cat)
X_vec_cat = enc.transform(X_vec_cat).toarray()

# combine cat & con features
X_vec = np.concatenate((X_vec_con,X_vec_cat), axis=1)

# vectorize targets
Y_vec_cnt = train['cnt'].values.astype(float)

# split into train and cross validation sets for all features
X_train, X_cv, Y_train, Y_cv = train_test_split(X_vec, Y_vec_cnt, test_size=0.2)



n_alphas = 200
alphas = np.logspace(-5, 8, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)
coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X_train, Y_train)
    coefs.append(clf.coef_)
    
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


clf = Ridge(alpha=2.0)
clf.fit(X_train,Y_train)
clf.score(X_cv, Y_cv)
clf.get_params(deep=True)
arreglo = clf.predict(X_train)
readerWriter = ReaderAndWriter()
readerWriter.write_file([arreglo],'answer1.txt')


def date_substraction(date1,date2):
    d1 = date1.split("-")
    d2 = date2.split("-")
    yearDiff = int(d1[0]) -  int(d2[0])
    monthDiff = int(d1[1]) - int(d2[1])
    dayDiff = int(d1[2]) -   int(d2[2])
    return dayDiff + 30 * monthDiff + 365 * yearDiff
    
    
def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

    

for i in range(1, 6495):
    print i
    train.dteday[i] = time.mktime(datetime.datetime.strptime(train.dteday[i], "%Y-%m-%d").timetuple())

















