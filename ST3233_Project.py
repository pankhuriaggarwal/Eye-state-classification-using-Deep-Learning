#!/usr/bin/env python
# coding: utf-8

# Loading and Importing Packages

# In[127]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
import math

import scipy as sp
from sklearn.metrics import mean_squared_error
import sklearn

from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten

from statsmodels.tsa.arima_model import ARIMA
import statsmodels
import statistics as st


# In[143]:


from scipy.io import arff
data = arff.loadarff('EEG Eye State.arff')
df = pd.DataFrame(data[0])


# # Exploratory Data Analysis

# In[139]:


df.head()


# ### Checking the shape/dimensions of the data

# In[130]:


print(df.shape)


# ### Understanding the data through statistical summaries

# In[131]:


print("Summary of Variables \n",df.describe(),"\n\n")
print("Summary for Output Variable\n",df['eyeDetection'].describe())


# ### Checking datatypes

# In[132]:


print(df.dtypes)


# ### Checking for missing Values

# In[133]:


print(df.isnull().sum())


# In[144]:


df['eyeDetection'] = [1 if a == b'1' else 0 for a in df['eyeDetection']]
df.head()
df['eyeDetection'].describe()


# # Creating X and Y Arrays

# In[145]:


# extracting the relevant columns for X (inputs) and Y(output) and converting them to numpy arrays
X = df.drop(columns=['eyeDetection']).values
Y = df[['eyeDetection']].values


# In[146]:


#checks
print(rf" Shape of X: {X.shape}")
print(rf" Shape of Y: {Y.shape}")


# # Standardizing the Data

# In[147]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)


# # Train-Test Split

# In[148]:


#train and test mode. using train size:test size :: 7:3
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=6)


# In[149]:


#checks
print(rf" Shape of X_train: {X_train.shape}")
print(rf" Shape of Y_train: {Y_train.shape}")
print(rf" Shape of X_test:  {X_test.shape}")
print(rf" Shape of Y_test:  {Y_test.shape}")


# # Deep Learning

# ## CNN

# ### Data Prep 

# In[150]:


# converting all arrays to dataframes of appropriate shapes for Deep Learning Models 
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
Y_train = pd.DataFrame(Y_train)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
Y_test = pd.DataFrame(Y_test)
print('Train set shape', X_train.shape)
print('Test set shape', X_test.shape)


# ### CNN Model

# In[162]:


from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense


batch_size = 20 
num_classes = 2
epochs = 10
input_shape=(X_train.shape[1], X_train.shape[2])

model = Sequential()
model.add(Conv1D(128, kernel_size=3,padding ='same',activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(128,kernel_size=3,padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, epochs=epochs, verbose=0)
# demonstrate prediction
yhat = model.predict(X_test, verbose=0)
print(yhat)


# ### CNN Model Evaluation

# In[163]:


model.evaluate(X_train, Y_train, verbose=0)
model.evaluate(X_test, Y_test, verbose=0)

