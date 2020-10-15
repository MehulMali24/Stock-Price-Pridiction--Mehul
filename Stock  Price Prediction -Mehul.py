#!/usr/bin/env python
# coding: utf-8

# In[12]:


pip install quandl


# In[13]:


import quandl

import numpy as np
from sklearn.linear_model import  LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# In[14]:


import quandl
df= quandl.get("WIKI/AMZN")
print(df.head())


# In[15]:


df = df[['Adj. Close']]
print(df.head())


# In[16]:


forecast_out= 30
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

print(df.tail())


# In[17]:


X = np.array(df.drop(['Prediction'],1))
X = X[:-forecast_out]
print(X)


# In[18]:


y = np.array(df['Prediction'])
y = y[:-forecast_out]
print(y)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[20]:


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(X_train, y_train)


# In[21]:


svm_confidence =svr_rbf.score(X_test, y_test)
print("svm confidence: ", svm_confidence)


# In[22]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[23]:


lr_confidence =lr.score(X_test, y_test)
print("lr confidence: ", svm_confidence)


# In[24]:


X_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(X_forecast)


# In[25]:


lr_prediction = lr.predict(X_forecast)
print(lr_prediction)
svm_prediction = svr_rbf.predict(X_forecast)
print(svm_prediction)


# In[26]:


import pickle


# In[31]:


filename= "Stock_Price_Prediction_-Mehul"
pickle.dump(lr,open(filename,'wb'))


# In[33]:


loaded_model = pickle.load(open(filename, 'rb'))
result =loaded_model.score(X_test, y_test)


# In[34]:


filename= "Stock_Price_Prediction_-Mehul"
pickle.dump(svr_rbf,open(filename,'wb'))


# In[35]:


loaded_model = pickle.load(open(filename, 'rb'))
result =loaded_model.score(X_test, y_test)


# In[ ]:




