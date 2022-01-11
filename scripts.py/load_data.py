from numpy.core.numerictypes import ScalarType
import pandas as pd
from scipy import stats
import numpy as np



# df = pd.read_csv('listings.csv')
df = pd.read_csv('listings_no_dupe.csv')
# df = df.drop_duplicates()

# df['Price'] = df['Price'].str.replace('$','').astype(float)
df.shape

# df.to_csv('listings_no_dupe.csv',index=False)

# df.tail()