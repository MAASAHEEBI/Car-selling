#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('car data (1).csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df['Seller_Type'].unique()


# In[6]:


df['Transmission'].unique()


# In[7]:


df['Owner'].unique()


# In[8]:


df['Fuel_Type'].unique()


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


df.columns


# In[12]:


final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[13]:


final_dataset.head()


# In[14]:


final_dataset['Current_Year']=2020


# In[15]:


final_dataset.head()


# In[16]:


final_dataset['no_year']=final_dataset['Current_Year']-final_dataset['Year']
final_dataset.head()


# In[17]:


final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.head()


# In[18]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)
final_dataset.head()


# In[19]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()


# In[20]:


final_dataset.corr()


# In[21]:


import seaborn as sns


# In[22]:


sns.pairplot(final_dataset)


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


corrmap=final_dataset.corr()
top_corr_features=corrmap.index
plt.figure(figsize=(10,10))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[25]:


final_dataset.head()


# In[26]:


X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[27]:


X.head()


# In[28]:


y.head()


# In[30]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=sc_x.fit_transform(X)


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[32]:


(x.shape,x_train.shape,x_test.shape)


# In[33]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[34]:


y_pred=model.predict(x_test)


# In[49]:


plt.scatter(y_test,y_pred)


# In[38]:


from sklearn import metrics


# In[39]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[42]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[43]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)


# In[44]:


from sklearn.ensemble import RandomForestRegressor


# In[45]:


from sklearn.datasets import make_regression
x, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


# In[46]:


regr_obj = RandomForestRegressor(max_depth=3, random_state=0)
regr_obj.fit(x, y)


# In[47]:


regr_obj.fit(x, y)
print(regr_obj.predict([[2, 10, 30, 0]]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[54]:




