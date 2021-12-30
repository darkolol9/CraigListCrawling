# %%
import pandas as pd


df = pd.read_csv('listings_no_dupe.csv')
# df = df.drop_duplicates()

# df['Price'] = df['Price'].str.replace('$','').astype(float)
df.shape

# df.to_csv('listings_no_dupe.csv')

# df.tail()
# %%
grp = df['Brand'].value_counts()
grp.plot(kind='bar')





# %%

col_names = ['Model','Brand','Price','Odometer','Paint color','Condition','Fuel','Transmission','Type']
str_features = ['Brand','Paint color','Condition','Fuel','Transmission','Type']


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

    
    


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
from sklearn.preprocessing import LabelEncoder


def scale_df_into_XY(df,single=False):
    
    scaleX = StandardScaler()
    scaleY = StandardScaler()

    if single:
        return scaleX.transform(df.drop(['Price'],axis=1))

    X = df.drop(['Price'],axis=1)
    X = scaleX.fit_transform(X) #scale
    y = df.Price
    y = scaleY.fit_transform(np.array(y).reshape(-1,1))

    return X,y,scaleY




df = pd.read_csv('fixed.csv')
df = df.drop('Unnamed: 0',axis=1)

le = labelEncoder()
le.fit(df)
le.transform(df)

X,y,scaleY = scale_df_into_XY(df)



# 2015,Acura,22990.0,32917,black,good,other,other,sedan
data = {'Model':[2015],'Brand':['Acura'],'Odometer':[32917],'Paint color':['black'],'Condition':['good'],'Fuel':['other'],'Transmission':['other'],'Type':['sedan']}
data = pd.DataFrame(data)


le.transform(data)
data


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


regressor = SVR(kernel='rbf').fit(X_train,y_train)

ypred = regressor.predict(np.array(data))
scaleY.inverse_transform(ypred.reshape(-1,1))


# metrics.r2_score(y_test,ypred)


# %%






# %%
