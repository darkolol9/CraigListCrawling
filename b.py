# %%
#imports and helper functions

from numpy.core.numerictypes import ScalarType
import pandas as pd
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

class labelEncoder:
    def __init__(self):
        self.map = dict()
        self.inverse = dict()
    
    def fit(self,df):
        for col in df:
            self.map[col] = {key:val for val,key in enumerate(df[col].unique())}
            self.inverse[col] = {val:key for val,key in enumerate(df[col].unique())}

    def transform(self,df):
        for col in df:
            if col not in ['Price','Odometer','Model']:
                df[col] = df[col].replace(self.map[col])

        return df

def split_x_y(df):
    
    X = df.drop(['Price'],axis=1)
    y = df.Price

    return X,y

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.40)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def isNumber(a):
    return any(i.isdigit() for i in a)


# %% ##########################################################
# load and cleanse data


df = pd.read_csv('listings_no_dupe.csv')

df = df[:1000]
df.tail()

for i,row in enumerate(df['Brand']):
    if isNumber(row):
        df['Brand'][i] = 'null'


df.Brand = df.Brand.str.upper()

df = df.drop(df[(df['Brand'] == 'NULL')].index)
df = df[df.Price > 500]

df =  remove_outlier(df,'Price')



# %% ###################################################################
# visualize

fig, ax = plt.subplots(3,figsize=(10,15))

ax[0].scatter(df.Model,df.Price,c='red')
ax[1].scatter(df.Model,df.Odometer,c='green')

ax[0].set(xlabel='Model', ylabel='Price')
ax[1].set(xlabel='Model', ylabel='Odometer')

plt.ylim([0,200000])


# %% #######################################################################

le = labelEncoder()
le.fit(df)
le.transform(df)

X,y = split_x_y(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

svr = SVR(kernel='rbf').fit(X_train,y_train)
regressor = RandomForestRegressor(random_state=0).fit(X_train,y_train)
sgd = SGDRegressor().fit(X_train,y_train)
en = ElasticNet().fit(X_train,y_train)
br = BayesianRidge().fit(X_train,y_train)
lr = LinearRegression().fit(X_train,y_train)

ypred_rf  = regressor.predict(X_test)
ypred_svr  = svr.predict(X_test)
ypred_sgd  = sgd.predict(X_test)
ypred_en = en.predict(X_test)
ypred_br = br.predict(X_test)
ypred_lr = lr.predict(X_test)

scores = [metrics.r2_score(y_test,ypred_rf),metrics.r2_score(y_test,ypred_svr),
metrics.r2_score(y_test,ypred_en),metrics.r2_score(y_test,ypred_br),metrics.r2_score(y_test,ypred_lr)]


for i in range(len(scores)):
    scores[i] = round(scores[i],2)


print(regressor.feature_names_in_)
regressor.feature_importances_

