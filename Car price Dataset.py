#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


data=pd.read_csv(r'C:\Users\ASUS\Downloads\CarPrice_Assignment.csv')
data


# In[3]:


data.dtypes


# In[4]:


data.describe(include='all')


# In[5]:


data=data.drop('CarName', axis=1)


# In[6]:


data.isnull().sum()


# In[7]:


data.head()


# In[8]:


data.corr()['price']


# In[9]:


data=data.drop('car_ID', axis=1)


# In[10]:


data=data.drop('wheelbase', axis=1)


# In[11]:


data=data.drop('carheight', axis=1)


# In[12]:


data=data.drop('boreratio', axis=1)


# In[13]:


data=data.drop('stroke', axis=1)


# In[14]:


data=data.drop('compressionratio', axis=1)


# In[15]:


data=data.drop('peakrpm', axis=1)


# In[16]:


data.head()


# In[17]:


data.columns


# In[18]:


data.dtypes


# In[19]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables=data[['horsepower', 'citympg']]
vif=pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns


# In[20]:


vif


# In[21]:


data.dtypes


# In[22]:


data=data.drop('carlength', axis=1)


# In[23]:


data=data.drop('carwidth', axis=1)


# In[24]:


data=data.drop('curbweight', axis=1)


# In[25]:


data=data.drop('enginesize', axis=1)


# In[26]:


data=data.drop('highwaympg', axis=1)


# In[27]:


data.head()


# In[28]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[29]:


data.dtypes


# In[30]:


for i in data[['horsepower', 'citympg', 'price']]:
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i])


# In[31]:


for i in data[['horsepower', 'citympg', 'price']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[32]:


data=data.reset_index(drop=True)


# In[33]:


data.describe(include='all')


# In[34]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize =(10,5))

ax1.scatter(data['horsepower'],data['price'])
ax1.set_title('Price and Horsepower')
ax1.set_xlabel('Horsepower')
ax1.set_ylabel('Price')
ax2.scatter(data['citympg'],data['price'])
ax2.set_title('Price and Citympg')
ax2.set_xlabel('Citympg')
ax2.set_ylabel('Price')

plt.show()


# In[35]:


data.head()


# In[36]:


log_price=np.log(data['price'])
data['log_price']=log_price


# In[37]:


data=pd.get_dummies(data, drop_first=True)


# In[38]:


data.head()


# In[39]:


data=data.drop('price', axis=True)


# In[40]:


X=data.drop('log_price', axis=1)
y=data['log_price']


# In[41]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)


# In[42]:


X_scaler=scaler.transform(X)


# In[43]:


X_scaler


# In[44]:


data_scaled=pd.DataFrame(X_scaler, columns=X.columns)
data_scaled


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(data_scaled, y, test_size=0.2, random_state=42)
lr=LinearRegression()
lr.fit(X_train, y_train)


# In[46]:


y_pred=lr.predict(X_test)
X_test["Actual price"]=y_test
X_test['Predicted price']=y_pred
X_test


# In[47]:


mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R2', r2)


# In[48]:


y_pred_train=lr.predict(X_train)


# In[49]:


mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
mse_train = metrics.mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = metrics.r2_score(y_train, y_pred_train)
print('MAE:', mae_train)
print('MSE:', mse_train)
print('RMSE:', rmse_train)
print('R2', r2_train)


# In[50]:


X_train.columns


# In[51]:


variables=[]
r2_scores=[]
for i in X_train.columns:
    X_train_single= X_train[[i]]
    X_test_single= X_train[[i]]
    
    lr.fit(X_train_single, y_train)
    y_pred_single=lr.predict(X_test_single)
    
    r2=metrics.r2_score(y_train, y_pred_single)
    variables.append(i)
    r2_scores.append(r2)
    
df = pd.DataFrame({'Variable': variables, 'R2': r2_scores})

df = df.sort_values(by='R2', ascending=False)


# In[52]:


df


# In[53]:


variables= []
train_r2_scores=[]
test_r2_scores=[]

for i in X_train.columns:
    X_train_single=X_train[[i]]
    X_test_single=X_test[[i]]
    
    lr.fit(X_train_single, y_train)
    
    y_pred_train_single=lr.predict(X_train_single)
    
    train_r2=metrics.r2_score(y_train, y_pred_train_single)
    
    lr.fit(X_test_single, y_test)
    y_pred_test_single=lr.predict(X_test_single)
    test_r2=metrics.r2_score(y_test, y_pred_test_single)
    
   
    variables.append(i)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
    

df = pd.DataFrame({'Variable': variables, 'Train R2': train_r2_scores, 'Test R2': test_r2_scores})

df= df.sort_values(by='Test R2', ascending=False)

df

    


# In[54]:


data.head()


# In[55]:


df


# In[56]:


X=data_scaled[['horsepower', 'citympg', 'drivewheel_rwd', 'cylindernumber_four', 'drivewheel_fwd', 'fuelsystem_2bbl']]
y=data['log_price']


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


lr=LinearRegression()
lr.fit(X_train, y_train)


# In[59]:


y_pred=lr.predict(X_test)


# In[65]:


mae_test = metrics.mean_absolute_error(y_test, y_pred)
mse_test = metrics.mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
r2_test = metrics.r2_score(y_test, y_pred)
formatted_r2_test='{:.2f}'.format(r2_test)
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R2', formatted_r2_test)


# In[61]:


y_pred_train=lr.predict(X_train)


# In[62]:


mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
mse_train = metrics.mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = metrics.r2_score(y_train, y_pred_train)
print('MAE:', mae_train)
print('MSE:', mse_train)
print('RMSE:', rmse_train)
print('R2', r2_train)


# In[ ]:





# In[ ]:




