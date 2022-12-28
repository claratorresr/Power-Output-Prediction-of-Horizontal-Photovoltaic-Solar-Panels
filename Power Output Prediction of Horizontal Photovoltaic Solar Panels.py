#!/usr/bin/env python
# coding: utf-8

# # Power Output Prediction of Horizontal Photovoltaic Solar Panels
# ## 12-760 Fall 2022 
# ### Name: Claraestela Torres
# #### Date: December 2022

# This dataset is taken from Kaggle. This data provides the following columns: location, date, time sampled, latitude, longitude, altitude, year and month, month, hour, season, humidity, ambient temperature, power output from the solar panel, wind speed, visibility, pressure, and cloud ceiling for 12 Northern hemisphere sites where the horizontal photovoltaics are installed over 14 months.
# 
# 

# Before starting this project the following libraries will be imported.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#GradientBoostingRegressor will be used in the next submision.


# ## Data Reading and Cleaning

# In[2]:


#Read dataset
solar = pd.read_csv('dataset.csv')
solar.head()


# In[3]:


#Information about the dataset
solar.info()
print(solar.isnull().sum())


# In[4]:


# Dataset shape
print(f'Solar data has {solar.shape[1]} columns and {solar.shape[0]} observations')


# In[5]:


solar.Time.unique()


# In[6]:


#It is important to know that the highest solar generation during day time is usually from 11 am to 4 pm. 
#Since electricity generation is directly proportional to the solar irradiance that hits the panel.
#Therefore, the dataset will be filtered on this timeframe only.
solar = solar[solar['Time'] >= 1100]


# Dataset new shape

# In[7]:


print(f'Solar data has now {solar.shape[1]} columns and {solar.shape[0]} observations')


# As shown before, there are no missing values in this dataset, however `YRMODAHRI` will be dropped since its description is not provided and it lacks of meaning.

# In[8]:


solar = solar.drop(columns = ['YRMODAHRMI'], axis = 1)
solar.head()


# In[9]:


#PolyPwr target variable distribution
sns.displot(solar, x='PolyPwr')


# As shown in the previous histogram, data from 30W and above is limited, however this is not considerable.

# # Feature Engineering
# Since the `Date`, `Time`, `Month`, `Hour` are not relevant in this prediciton, they will be dropped.

# In[10]:


solar = solar.drop(columns = ['Date', 'Time', 'Month', 'Hour'])


# In[11]:


solar.describe()


# In[12]:


#Correlation Matrix
plt.figure(figsize = (10,7))
corr = solar.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask = mask, annot = True)
plt.show()


# From correlation matrix,ambient temperature, followed by cloud ceiling, and humidity are the three most correlated features with solar power output. 
# 
# On the other hand, longitude is the variable with least correlation being it 0.012, therefore this variable will be dropped. As well as altitude since it is perfectly correlated with pressure meaning that the result of one highly enunciate the result of the other one and the latter does not change with location.

# In[13]:


solar = solar.drop(columns = ['Longitude', 'Altitude'], axis=1)
solar.head()


# ## Handling Categorical Variables
# The categorical variables will be handled by creating "dummy" variables. For each categorical variable with `k` levels, it introduces `k-1` columns of boolean data to tell which category that observation belonged to. The "left out" level of the variable is the "default" for the model's parameters. This can be accomplished with `get_dummies()` function from Pandas library. 

# In[1]:


solar = pd.get_dummies(solar, columns = ['Location','Season'], drop_first = True)
solar.head()


# In[15]:


#That data will be splitted as 80% training data and 20% testing data
X = solar.drop(columns = ['PolyPwr'])
y = solar.PolyPwr


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle = True)
X_train


# # KNN Regressor

# In[17]:


#fit the model
knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train, y_train)
print('test R2: %.2f'% knn.score(X_test,y_test))

# predict
knn_y_pred = knn.predict(X_test)
print('test MSE: %.2f'% mean_squared_error(y_test,knn_y_pred))


# # DecisionTree Regressor

# In[62]:


#fit the model
reg_tree = DecisionTreeRegressor(max_leaf_nodes=60,random_state=1)
reg_tree.fit(X_train, y_train)
print('test R2: %.2f'% reg_tree.score(X_test,y_test))

# predict
reg_tree_y_pred = reg_tree.predict(X_test)
print('test MSE: %.2f' % mean_squared_error(y_test, reg_tree_y_pred))


# # Random Forest Regressor

# In[71]:


rf = RandomForestRegressor(max_leaf_nodes=500, random_state=1)
rf.fit(X_train, y_train)
print('test R2: %.2f'% rf.score(X_test,y_test))

#predict
rf_y_pred = rf.predict(X_test)
print('test MSE: %.2f' % mean_squared_error(y_test, rf_y_pred ))


# In[72]:


fimp = pd.DataFrame({'Variables':np.array(X.columns),'Importance':np.array(rf.feature_importances_)})
fimp.sort_values(by=['Importance'], ascending=False,inplace=True)
fimp


# In[73]:


plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Variables',data=fimp)

plt.title('Feature Importance')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Names')

