#!/usr/bin/env python
# coding: utf-8

# In[146]:



# 1.1
import numpy as np
import os
import pandas as pd

# 1.2 For plotting
import matplotlib.pyplot as plt

# 1.3 For modeling
# 1.3 Class for applying multiple data transformation jobs
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

# 1.5 For performance measures
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# 1.6 Plotting metrics related graphs
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve

# 1.7 For data splitting
from sklearn.model_selection import train_test_split


# In[147]:


path = ("D:\\data analyst data\\download data")
os.chdir(path)
os.listdir()


# In[148]:


df = pd.read_csv('credit_card_data.zip')
df


# In[149]:


df.info()


# In[150]:


df.describe()


# In[151]:


df.isnull().sum()


# In[152]:


df.dtypes


# In[153]:


df.nunique() < 5


# In[154]:


y = df.pop('Class').values
y


# In[155]:


X = df


# In[159]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.33,
                                                    shuffle = True
                                                    )


# Classification - Decision Tree

# In[160]:


clf_dt =  DecisionTreeClassifier()  


# In[161]:


clf_dt.fit(X_train,y_train)


# In[162]:


y_pred_dt = clf_dt.predict(X_test)


# In[163]:


y_pred_dt


# In[164]:


y_pred_dt_prob = clf_dt.predict_proba(X_test)


# In[165]:


accuracy_score(y_test,y_pred_dt)


# Classification - Random Forest 

# In[166]:


clf_rf =  RandomForestClassifier(n_estimators=100)


# In[167]:


clf_rf.fit(X_train,y_train)


# In[168]:


y_pred_rf = clf_rf.predict(X_test)


# In[169]:


y_pred_rf_prob = clf_rf.predict_proba(X_test)


# In[170]:


accuracy_score(y_test,y_pred_rf)


# Classification -  Gradient Boosting

# In[171]:


clf_gbm = GradientBoostingClassifier()


# In[172]:


clf_gbm.fit(X_train,y_train)


# In[173]:


y_pred_gbm= clf_gbm.predict(X_test)


# In[174]:


y_pred_gbm_prob= clf_gbm.predict_proba(X_test)


# In[175]:


accuracy_score(y_test,y_pred_gbm)


# Classification - XGBClassifier

# In[176]:


clf_xg =  XGBClassifier(learning_rate=0.5,
                        reg_alpha= 5,
                        reg_lambda= 0.1
                        )


# In[177]:


clf_xg.fit(X_train,y_train)


# In[178]:


y_pred_xg= clf_xg.predict(X_test)


# In[179]:


y_pred_xg_prob = clf_xg.predict_proba(X_test)


# In[180]:


accuracy_score(y_test,y_pred_xg)


# In[181]:


fig = plt.figure()
ax = fig.subplots()
plot_roc_curve(clf_dt,  X_test, y_test, ax =ax)
plot_roc_curve(clf_rf,  X_test, y_test, ax =ax)
plot_roc_curve(clf_gbm, X_test, y_test, ax =ax)
plot_roc_curve(clf_xg,  X_test, y_test, ax =ax)
plt.show()


# In[182]:


fig = plt.figure()
ax = fig.subplots()
plot_precision_recall_curve(clf_dt,  X_test, y_test, ax =ax)
plot_precision_recall_curve(clf_rf,  X_test, y_test, ax =ax)
plot_precision_recall_curve(clf_gbm, X_test, y_test, ax =ax)
plot_precision_recall_curve(clf_xg,  X_test, y_test, ax =ax)
plt.show()

