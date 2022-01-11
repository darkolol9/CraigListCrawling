import pickle
from b import labelEncoder



model = pickle.load(open('randomForest.pkl','rb'))
labelEncoder = pickle.load(open('encoder.pkl','rb'))