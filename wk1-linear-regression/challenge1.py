import pandas as pd 
import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt 

#read data
df = pd.read_csv('challenge_dataset.txt',header=None, names=['x','y'])

#print df.head()
#print df.shape
#print df.info()
#print df.describe()

x = df[['x']]
y = df[['y']]
#print x,y

plt.scatter(x,y)

# train model
reg = linear_model.LinearRegression()
reg.fit(x,y)
pred = reg.predict(x)

#Returns the coefficient of determination R^2 of the prediction
r_square = reg.score(x,y)
print "R^2 is ", r_square

plt.plot(x, pred)
plt.show()

# Mean Squared Error
from sklearn.metrics import mean_squared_error
MSR = mean_squared_error(y,pred)
print "Mean Square Error is ", MSR