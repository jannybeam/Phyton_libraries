#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

pd.options.display.max_columns = 100

from sklearn.metrics import roc_auc_score

from sklearn.datasets import load_wine


# ЗАДАНИЕ 1

# In[23]:


boston = load_boston()
data = boston['data']
target = boston['target']
feature_names = boston['feature_names']


# In[24]:


X = pd.DataFrame(data, columns = feature_names)
X.head()


# In[25]:


X.info()


# In[26]:


y = pd.DataFrame(target, columns = ['price'])
y.info()


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[28]:


lr = LinearRegression()


# In[29]:


lr.fit(X_train, y_train)


# In[30]:


y_pred = lr.predict(X_test)
y_pred.shape


# In[31]:


r2_score(y_test, y_pred)


# ЗАДАНИЕ 2

# In[38]:


model = RandomForestRegressor(n_estimators = 1000, max_depth = 12, random_state = 42)


# In[39]:


model.fit(X_train, y_train.values[:, 0])


# In[41]:


y_train_new = np.array(y_train.values[:, 0], dtype = int)


# In[37]:


y_pred2 = model.predict(X_test)
y_pred2.shape


# In[42]:


r2_score(y_test, y_pred2)


# ЗАДАНИЕ 3

# In[44]:


sum(model.feature_importances_)


# In[48]:


feature_importance = pd.DataFrame({'name':X.columns, 'feature_importance': model.feature_importances_}, columns = ['feature_importance', 'name'])
print(feature_importance)
feature_importance.nlargest(2, 'feature_importance')


# ЗАДАНИЕ 4

# In[50]:


DATASET_PATH = 'C:/PythonDataFiles/creditcard.csv'


# In[51]:


df = pd.read_csv(DATASET_PATH, sep = ',')
df.head(4)


# In[52]:


cl = df['Class'].value_counts(normalize = True)
cl


# In[53]:


df.info()


# In[55]:


df.head(10)


# In[56]:


X = df.drop('Class', axis = 1)
X.info()


# In[57]:


df.iloc[:, :-1]
X


# In[58]:


df.info()


# In[59]:


y = df['Class']
y


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100, stratify = y)


# In[61]:


print('X_train ', X_train.shape)
print('X_test ', X_test.shape)
print('y_train ', y_train.shape)
print('y_test ', y_test.shape)


# In[63]:


parameters = [{'n_estimators': [10, 15], 'max_features': np.arange(3, 5), 'max_depth': np.arange(4, 7)}]

clf = GridSearchCV(estimator=RandomForestClassifier(random_state=100), param_grid=parameters, scoring='roc_auc', cv=3)

clf.fit(X_train, y_train)


# In[64]:


clf.best_params_


# In[69]:


clf = RandomForestClassifier(max_depth=6, max_features=3, n_estimators=15, random_state = 100)

clf.fit(X_train, y_train)


# In[70]:


y_pred = clf.predict_proba(X_test)

y_pred_proba = y_pred[:, 1]

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred_proba)


# In[ ]:




