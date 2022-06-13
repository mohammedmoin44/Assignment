#!/usr/bin/env python
# coding: utf-8

# In[8]:


#importing pandas to read the file
import pandas as pd


# In[9]:


#creating a dataframe
df = pd.read_csv(r'C:\Users\Moham\Downloads\training_set.csv')
df


# In[10]:


df.index.names = ['index']


# In[11]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[12]:


df


# In[13]:


df['Y'].value_counts()


#  so we have majority of zeros compare to ones
# 

# In[14]:


df.isna().sum()


# the data doesn't have any null value 

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.title('Output Value count')
sns.histplot(data=df, x="Y")


# In[16]:


#feature selection using correlation
df.corr().style.background_gradient(cmap= 'coolwarm' )


# In[17]:


#selecting only x values which have correlation greater than 0.20
x = df[['X3','X5','X6','X7','X8','X9','X11','X16','X17','X18','X19','X20','X21','X23','X24','X52','X53','X56','X57']]
y = df['Y']
x.head()


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)


# In[19]:


get_ipython().system('pip install xgboost')


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor


# In[21]:


from sklearn.model_selection import cross_val_score
model_a =  cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'),X_train,y_train,cv=5)


# In[22]:


model_a.mean()


# In[23]:


LogisticReg = LogisticRegression(solver='liblinear',multi_class='ovr')
moded_l  = LogisticReg.fit(X_train,y_train)


# In[24]:


pred_l = moded_l.predict(X_test)
moded_l.score(X_test,y_test)


# In[39]:


model_b = cross_val_score(SVC(gamma='auto'),X_train,y_train,cv=5)
model_b.mean()


# In[27]:


svc = SVC(gamma='auto')
moded_s  = svc.fit(X_train,y_train)


# In[28]:


pred_s = moded_l.predict(X_test)
moded_s.score(X_test,y_test)


# In[38]:


model_r =  cross_val_score(RandomForestClassifier(n_estimators=40),X_train,y_train,cv=5)
model_r.mean()


# In[30]:


model = RandomForestClassifier(n_estimators=40)
model_ra = model.fit(X_train,y_train)


# In[34]:


pred_r = model_ra.predict(X_test)
model_ra.score(X_test,y_test)


# In[37]:


model_c = cross_val_score(XGBRegressor(n_estimators=40),X_train,y_train,cv=5)
model_c.mean()


# In[54]:


model_g = XGBRegressor(n_estimators=40)
model_Xg = model_g.fit(X_train,y_train)


# In[55]:


pred_Xg = model_Xg.predict(X_test)
model_Xg.score(X_test,y_test)


# In[44]:


from sklearn.metrics import confusion_matrix , classification_report


# Classification report for LogisticRegression

# In[47]:


print("Classification Report for LogisticRegression: \n", classification_report(y_test,pred_l ))


# Classification report for SVC

# In[48]:


print("Classification Report for SVC: \n", classification_report(y_test,pred_s ))


# Classification report for Random Forest

# In[50]:


print("Classification Report for Random Forest: \n", classification_report(y_test,pred_r))


# In[62]:


df_1 = pd.read_csv(r'C:\Users\Moham\Downloads\test_set.csv')
x_test = df_1[['X3','X5','X6','X7','X8','X9','X11','X16','X17','X18','X19','X20','X21','X23','X24','X52','X53','X56','X57']]


# choosing Random forest becuase it has good accuracy and f1 score

# In[70]:


a = model_ra.predict(x_test)
a


# In[71]:


a.shape

