# %%
from numpy.core.numerictypes import ScalarType
import pandas as pd
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# df = pd.read_csv('listings.csv')
df = pd.read_csv('listings_no_dupe.csv')
# df = df.drop_duplicates()

# df['Price'] = df['Price'].str.replace('$','').astype(float)
df = df[:1000]
df.shape
# df.to_csv('listings_no_dupe.csv',index=False)

# df.tail()


# %% 

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

for i,row in enumerate(df['Brand']):
    if isNumber(row):
        df['Brand'][i] = 'null'


df.Brand = df.Brand.str.upper()

df = df.drop(df[(df['Brand'] == 'NULL')].index)
df = df[df.Price > 500]


df =  remove_outlier(df,'Price')



plt.scatter(df.Model,df.Price,c='red')
# plt.plot(df.Price)



# df.describe()








# %%
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



le = labelEncoder()
le.fit(df)
le.transform(df)



X,y = split_x_y(df)

# 2015,Acura,22990.0,32917,black,good,other,other,sedan
data = {'Model':[1976],'Brand':['FORD'],'Odometer':[123456]
,'Paint color':['white']
,'Condition':['fair'],'Fuel':['gas']
,'Transmission':['other'],'Type':['truck']}


data = pd.DataFrame(data)





data = le.transform(data)


plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor["Price"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
print(relevant_features)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)





# svr = SVR(kernel='rbf').fit(X_train,y_train)
regressor = RandomForestRegressor(random_state=0).fit(X_train,y_train)
# sgd = SGDRegressor().fit(X_train,y_train)
# en = ElasticNet().fit(X_train,y_train)
# br = BayesianRidge().fit(X_train,y_train)
# lr = LinearRegression().fit(X_train,y_train)

# param = {'n_estimators':[i for i in range(100,102)],'max_depth':[i for i in range(10)]}
# cv = GridSearchCV(regressor,param,scoring='r2')
# cv.fit(X_train,y_train)

# print(cv.best_params_,cv.best_score_)


ypred_rf  = regressor.predict(X_test)
# ypred_svr  = svr.predict(X_test)
# ypred_sgd  = sgd.predict(X_test)
# ypred_en = en.predict(X_test)
# ypred_br = br.predict(X_test)
# ypred_lr = lr.predict(X_test)

# ypred = regressor.predict(np.array(data))


# pickle.dump(le,open('encoder.pkl','wb'))
# pickle.dump(regressor,open('randomForest.pkl','wb'))




# regressor.score(X,y)
print(metrics.r2_score(y_test,ypred_rf),' random forrest score')
# print(metrics.r2_score(y_test,ypred_svr),' svr score')
# print(metrics.r2_score(y_test,ypred_sgd),' sgd score')
# print(metrics.r2_score(y_test,ypred_en),' elastic net score')
# print(metrics.r2_score(y_test,ypred_br),' bayessian  score')
# print(metrics.r2_score(y_test,ypred_lr),' linear model  score')

# print(regressor.feature_names_in_)
# regressor.feature_importances_


# %%






# %%
