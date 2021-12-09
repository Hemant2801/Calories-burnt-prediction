#!/usr/bin/env python
# coding: utf-8

# # Importing all the dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


# # Data collection and preprocessing

# In[2]:


calories = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Calories burnt prediction/calories.csv')
exercise = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Calories burnt prediction/exercise.csv')


# Combining the two dataframes

# In[3]:


calories_data = pd.concat([exercise, calories['Calories']], axis = 1)


# In[4]:


# print the first 5 rows of the dataset
calories_data.head()


# In[5]:


# shape of the dataset
calories_data.shape


# In[6]:


# Getting some info about the dataset
calories_data.info()


# In[7]:


# Checking for any missing values
calories_data.isnull().sum()


# #  Data analysis

# In[8]:


calories_data.describe()


# # Data visualization

# In[9]:


sns.set_style(style = 'darkgrid')


# In[10]:


# plotting the gender column in a count plot
sns.countplot(x = 'Gender', data = calories_data)
plt.show


# In[11]:


# distribution plot for age column
sns.displot(calories_data['Age'], kde = True)


# In[12]:


# distribution plot for heart_rate column
sns.displot(calories_data['Heart_Rate'], kde = True)


# In[13]:


# Correlation between the data points
correlation = calories_data.corr()


# In[14]:


correlation


# In[15]:


# Constructing a heatmap to understand the correlation

plt.figure(figsize = (12, 12))
sns.heatmap(correlation, annot = True, annot_kws = {'size' : 10}, square = True)


# # Splitting the data into training and testing data

# In[19]:


# Cconverting the text data into numerical data
calories_data.replace({"Gender":{'female':0,'male':1}}, inplace = True)
calories_data


# In[20]:


# Splitting the various features and labels
X = calories_data.drop(columns = ['User_ID','Calories'], axis = 1)
Y = calories_data['Calories']


# In[21]:


X.info()


# In[22]:


# splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, random_state =2)


# In[23]:


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# # Model training and evaluation

# Model training:
# 
# XGB Regressor

# In[24]:


model = XGBRegressor()


# In[25]:


model.fit(x_train, y_train)


# Model evaluation:
# 
# Mean absolute error

# In[27]:


#on training data
training_prediction = model.predict(x_train)

training_accuracy = metrics.mean_absolute_error(y_train, training_prediction)
print('ACCURACY ON TRAINING DATA : ', training_accuracy)


# In[28]:


#on testing data
testing_prediction = model.predict(x_test)

testing_accuracy = metrics.mean_absolute_error(y_test, testing_prediction)
print('ACCURACY ON TESTING DATA : ', testing_accuracy)


# In[32]:


print(y_test[0:10])


# In[30]:


print(testing_prediction[0:10])


# In[ ]:




