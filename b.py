# %%
import pandas as pd


df = pd.read_csv('listings.csv')
df = df.drop_duplicates()
df.shape

df['Price'] = df['Price'].str.replace('$','').astype(int)


df.tail()
# %%
# grp = df.groupby('Paint color').count()
# grp.plot(kind='pie')

colors = df.groupby('Transmission')['Model'].mean()

colors.plot(kind='bar')



# %%
