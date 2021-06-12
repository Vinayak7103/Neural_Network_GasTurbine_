#!/usr/bin/env python
# coding: utf-8

# # Neural Networks 2

# ### TASK: NEURAL NETWORK

# Predicting Turbine Energy Yield (TEY) using ambient variables as features

# IMPORTING LIBRARIES

# In[22]:


import pandas as pd
import numpy as np
#Plot Tools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#Model Building
from sklearn.preprocessing import StandardScaler
import sklearn
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import InputLayer,Dense
import tensorflow as tf
#Model Validation
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error


# ## Load Dataset

# In[2]:


data = pd.read_csv(r"C:/Users/vinay/Downloads/gas_turbines.csv")
data.head()


# In[3]:


data.dtypes


# ## Data Analysis & Data Visualization

# In[4]:


data.columns


# In[5]:


import warnings
warnings.filterwarnings("ignore")
fig, axes = plt.subplots(5, 2, figsize=(20, 15))
fig.suptitle('Univariate Analysis',fontsize=20)
sns.distplot(data['AT'],ax=axes[0,0],color='indigo')
sns.distplot(data['AP'],ax=axes[0,1],color='orange')
sns.distplot(data['AH'],ax=axes[1,0],color='indigo')
sns.distplot(data['AFDP'],ax=axes[1,1],color='orange')
sns.distplot(data['GTEP'],ax=axes[2,0],color='indigo')
sns.distplot(data['TIT'],ax=axes[2,1],color='orange')
sns.distplot(data['TAT'],ax=axes[3,0],color='indigo')
sns.distplot(data['CDP'],ax=axes[3,1],color='orange')
sns.distplot(data['CO'],ax=axes[4,0],color='indigo')
sns.distplot(data['NOX'],ax=axes[4,1],color='orange')


# ### Inferences:
# 
# 1. Left Skewness :  AH | TIT | TAT
# 2. Right Skewness : AFDP |  CO
# 3. AT | NOX  seems to have normally distribution

# In[6]:


sns.distplot(data['TEY'],color='orange')


# In[19]:


import warnings
warnings.filterwarnings("ignore")
fig, axes = plt.subplots(5, 2, figsize=(20, 20))
fig.suptitle('Bivariate Analysis',fontsize=20)
sns.regplot(x="AT",y="TEY",data=data,ax=axes[0,0],color='indigo')
sns.regplot(x="AP",y="TEY",data=data,ax=axes[0,1],color='orange')
sns.regplot(x="AH",y="TEY",data=data,ax=axes[1,0],color='indigo')
sns.regplot(x="AFDP",y="TEY",data=data,ax=axes[1,1],color='orange')
sns.regplot(x="GTEP",y="TEY",data=data,ax=axes[2,0],color='indigo')
sns.regplot(x="TIT",y="TEY",data=data,ax=axes[2,1],color='orange')
sns.regplot(x="TAT",y="TEY",data=data,ax=axes[3,0],color='indigo')
sns.regplot(x="CDP",y="TEY",data=data,ax=axes[3,1],color='orange')
sns.regplot(x="CO",y="TEY",data=data,ax=axes[4,0],color='indigo')
sns.regplot(x="NOX",y="TEY",data=data,ax=axes[4,1],color='orange')


# ### Inferences:
# 
# 1. GTEP | CDP have a perfect Linear Increasing Relation with TEY
# 2. TIT | AFDP has slight Linear Increasing Relation with TEY
# 3. TAT | CO has slight Linear Decreasing Relation with TEY
# 4. AT | AP | AH | NOX have scattered points all around very less relation

# In[21]:


fig, axes = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(),annot=True)


# ### Defining Independent and Dependent Variables

# In[7]:


X = data.loc[:,['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO','NOX']]
y= data.loc[:,['TEY']]


# ### 1. K-Fold Cross Validation

# In[8]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


# In[9]:


def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[10]:


estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=100, verbose=False)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[11]:


estimator.fit(X, y)
prediction = estimator.predict(X)


# In[12]:


prediction


# ### Applying inverse transform on prediction to bring original values

# In[13]:


a=scaler.inverse_transform(prediction)


# In[14]:


b=scaler.inverse_transform(y)


# ### Calculate error of model actual and predicted values

# In[15]:


mean_squared_error(b,a)


# ### 2. Using Train-Test Split Model Validation Technique

# In[23]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[24]:


estimator.fit(X_train, y_train)
prediction = estimator.predict(X_test)


# In[25]:


prediction


# ### Applying inverse transform on prediction to bring original values

# In[26]:


c=scaler.inverse_transform(prediction)


# In[29]:


d=scaler.inverse_transform(y_test)


# In[30]:


mean_squared_error(d,c)

