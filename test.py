import pickle



model = pickle.load(open('model.pkl','rb'))
print(model.feature_names_in_)