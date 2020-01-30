import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pickle
dataset = pd.DataFrame({
    "experience":[0,0,5,2,7,3,10,11],
    "test_score":[8,8,6,10,9,7,7.857,7],
    "interview_score":[9,6,7,10,6,10,7,8],
    "salary":[50000,45000,60000,65000,70000,62000,72000,80000]
})

dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)

X = dataset.iloc[:,:3]
y = dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)
pickle.dump(regressor,open('model.pkl','wb'))
