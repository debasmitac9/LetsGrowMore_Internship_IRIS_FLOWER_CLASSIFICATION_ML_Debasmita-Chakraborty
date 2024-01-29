#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install collections


# In[2]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[3]:


#to read the datafile
df = pd.read_csv('iris.data')
#EDA
df.head()


# In[4]:


df.tail()


# In[5]:


#add header column

# printing data frame
print("Original Data frame")
print(df)

# adding column names
df_new = pd.read_csv("iris.data", names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
print("New Data frame")
print(df_new)

# printing row header
print("Row header")
print(list(df_new.columns))


# In[6]:


df.info()


# In[7]:


#to get summarised statistical info : desc analysis
df.describe()


# In[8]:


#to check the datafile is balanced or not
df_new['Species'].value_counts()


# In[9]:


#to count nul values
df_new.isnull().value_counts() #The above data tells us that there are no missing values in the dataset. So, we can continue with eda and then prediction using scikit-learn.


# In[10]:


type(df_new)


# In[11]:


df_new.isnull().sum()


# In[12]:


#to get summarised statistical info : desc analysis
df_new.describe()


# In[13]:


# Outliers detectios

plt.figure(figsize=(6,4))
sns.boxplot(data=df_new, y='SepalLengthCm')
plt.title("Box Plot of SepalLengthCm Distribution")
plt.ylabel("SepalLengthCm")
plt.show()


# In[14]:


# Outliers detectios

plt.figure(figsize=(6,4))
sns.boxplot(data=df_new, y='SepalWidthCm')
plt.title("Box Plot of SepalWidthCm Distribution")
plt.ylabel("SepalWidthCm")
plt.show() #outliers detected


# In[15]:


# Outliers detectios

plt.figure(figsize=(6,4))
sns.boxplot(data=df_new, y='PetalLengthCm')
plt.title("Box Plot of PetalLengthCm Distribution")
plt.ylabel("PetalLengthCm")
plt.show()


# In[16]:


# Outliers detectios

plt.figure(figsize=(6,4))
sns.boxplot(data=df_new, y='PetalWidthCm')
plt.title("Box Plot of PetalWidthCm Distribution")
plt.ylabel("PetalWidthCm")
plt.show()


# In[17]:


#Histograms
# Distributiion of SepalLengthCm

plt.figure(figsize=(10,6))
sns.histplot(df_new['SepalLengthCm'],bins=100,kde=True)
plt.title("SepalLengthCm Distribution")
plt.xlabel("SepalLengthCm")
plt.ylabel("Frequency")
plt.show()


# In[18]:


# Distributiion of SepalWidthCm

plt.figure(figsize=(10,6))
sns.histplot(df_new['SepalWidthCm'],bins=100,kde=True)
plt.title("SepalWidthCm Distribution")
plt.xlabel("SepalWidthCm")
plt.ylabel("Frequency")
plt.show()


# In[19]:


# Distributiion of PetalLengthCm

plt.figure(figsize=(10,6))
sns.histplot(df_new['PetalLengthCm'],bins=100,kde=True)
plt.title("PetalLengthCm Distribution")
plt.xlabel("PetalLengthCm")
plt.ylabel("Frequency")
plt.show()


# In[20]:


# Distributiion of PetalWidthCm

plt.figure(figsize=(10,6))
sns.histplot(df_new['PetalWidthCm'],bins=100,kde=True)
plt.title("PetalWidthCm Distribution")
plt.xlabel("PetalWidthCm")
plt.ylabel("Frequency")
plt.show()


# In[21]:


#Skewness checking
print("Skewness: %f" % df_new['SepalLengthCm'].skew())
print("Kurtosis: %f" % df_new['SepalLengthCm'].kurt())


# In[22]:


print("Skewness: %f" % df_new['SepalWidthCm'].skew())
print("Kurtosis: %f" % df_new['SepalWidthCm'].kurt()) #around 0 , so normal distribution 


# In[23]:


print("Skewness: %f" % df_new['PetalLengthCm'].skew())
print("Kurtosis: %f" % df_new['PetalLengthCm'].kurt())


# In[24]:


print("Skewness: %f" % df_new['PetalWidthCm'].skew())
print("Kurtosis: %f" % df_new['PetalWidthCm'].kurt())


# In[25]:


sns.pairplot(df_new, hue='Species')
plt.show()


# In[26]:


#removing 'Species' column
df_new.drop('Species', axis = 1, inplace = True)
df_new.head()


# In[27]:


#Correlation matrix
corr = df_new.corr()
sns.heatmap(corr, annot = True)

The data above tells us that both petal parameters are positively correlated with each other unlike the sepal length and width. Sepal length is more correlated to petal length and width than sepal width.