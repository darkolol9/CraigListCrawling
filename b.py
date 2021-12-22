# %%
import pandas as pd


df = pd.read_csv('listings_no_dupe.csv')
df = df.drop_duplicates()

# df['Price'] = df['Price'].str.replace('$','').astype(int)
df.shape

# df.to_csv('listings_no_dupe.csv')

# df.tail()
# %%
grp = df['Fuel'].value_counts()
grp.plot(kind='bar')

# colors = df.groupby('Transmission')['Model'].mean()

# colors.plot(kind='bar')



# %%
