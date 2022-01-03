# %%
import pandas as pd


# df = pd.read_csv('listings.csv')
df = pd.read_csv('listings_no_dupe.csv')
# df = df.drop_duplicates()

# df['Price'] = df['Price'].str.replace('$','').astype(float)
df.shape

# df.to_csv('listings_no_dupe.csv')

df.tail()


# %% 

def isNumber(a):
    return any(i.isdigit() for i in a)

for i,row in enumerate(df['Brand']):
    if isNumber(row):
        df['Brand'][i] = 'null'


df.Brand = df.Brand.str.upper()

df = df.drop(df[(df['Brand'] == 'null')].index)


# with open('brands.txt','w') as f:
#     for b in df['Brand'].unique():
#         f.write(b+'\n')





grp = df['Brand'].value_counts()
grp.plot(kind='bar')








# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np


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

def scale_df_into_XY(df,single=False):
    
    scaleX = StandardScaler()
    scaleY = StandardScaler()

    if single:
        return scaleX.transform(df.drop(['Price'],axis=1))

    X = df.drop(['Price'],axis=1)
    X = scaleX.fit_transform(X) #scale
    y = df.Price
    y = scaleY.fit_transform(np.array(y).reshape(-1,1))

    return X,y,scaleX,scaleY



df = pd.read_csv('fixed.csv')
df = df.drop('Unnamed: 0',axis=1)

le = labelEncoder()
le.fit(df)
le.transform(df)

X,y,scaleX,scaleY = scale_df_into_XY(df)

# 2015,Acura,22990.0,32917,black,good,other,other,sedan
data = {'Model':[2013],'Brand':['Chevy'],'Odometer':[142477]
,'Paint color':['white']
,'Condition':['good'],'Fuel':['gas']
,'Transmission':['automatic'],'Type':['truck']}
data = pd.DataFrame(data)


le.transform(data)
data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
data
data = scaleX.transform(data)


# regressor = SVR(kernel='rbf').fit(X_train,y_train.reshape(-1,1))
regressor = RandomForestRegressor(random_state=0).fit(X_train,y_train)
ypred  = regressor.predict(X_test)

ypred = regressor.predict(np.array(data))
res = scaleY.inverse_transform(ypred.reshape(-1,1))
print(res[0][0])


# regressor.score(X,y)
# metrics.r2_score(y_test,ypred)


# %%






# %%
