import pandas as pd 
import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt 

#read data
df = pd.read_fwf('brain_boday.txt')
x = df[['Brain']]
y = df[['Body']]
#print x

# train model
body_reg = linear_model.LinearRegression()
body_reg.fit(x,y)
#a = np.array([[1],[2],[3],[4]]) # vector
a = np.arange(10).reshape(10,1)
#print a
pred=body_reg.predict(a)
#print pred

plt.scatter(x,y) # independent, dependent
plt.plot(x, body_reg.predict(x))
plt.show()

#plt.plot(a,pred)
#plt.show()
