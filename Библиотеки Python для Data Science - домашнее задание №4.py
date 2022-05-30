#!/usr/bin/env python
# coding: utf-8

# ЗАДАНИЕ 1

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


# In[5]:


boston = load_boston()

data = boston['data']
feature_names = boston['feature_names']
X = pd.DataFrame(data, columns = feature_names)
X.head()


# In[7]:


target = boston['target']
y = pd.DataFrame(target, columns = ['price'])
y.info()


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[11]:


scaler = StandardScaler()


# In[12]:


X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)


# In[14]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(n_components = 2, learning_rate = 250, random_state = 42)\nX_train_tsne = tsne.fit_transform(X_train_scaled)')


# In[15]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])
plt.show()


# ЗАДАНИЕ 2

# In[17]:


kmeans = KMeans(n_clusters = 3, max_iter = 100, random_state = 42)
labels_train = kmeans.fit_predict(X_train_scaled)
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c = labels_train)
plt.show()


# In[18]:


print('Claster 0: {}'.format(X_train.loc[labels_train == 0, 'CRIM'].mean()))
print('Claster 1: {}'.format(X_train.loc[labels_train == 1, 'CRIM'].mean()))
print('Claster 2: {}'.format(X_train.loc[labels_train == 2, 'CRIM'].mean()))


# In[20]:


print('Claster 0: {}'.format(y_train.loc[labels_train == 0, 'price'].mean()))
print('Claster 1: {}'.format(y_train.loc[labels_train == 1, 'price'].mean()))
print('Claster 2: {}'.format(y_train.loc[labels_train == 2, 'price'].mean()))


# ЗАДАНИЕ 3

# In[22]:


labels_test = kmeans.predict(X_test_scaled)
X_test_tsne = tsne.fit_transform(X_test_scaled)
plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c = labels_test)
plt.show()


# In[23]:


print('Claster 0: {}'.format(X_train.loc[labels_train == 0, 'CRIM'].mean()))
print('Claster 1: {}'.format(X_train.loc[labels_train == 1, 'CRIM'].mean()))
print('Claster 2: {}'.format(X_train.loc[labels_train == 2, 'CRIM'].mean()))


# In[24]:


print('Claster 0: {}'.format(y_train.loc[labels_train == 0, 'price'].mean()))
print('Claster 1: {}'.format(y_train.loc[labels_train == 1, 'price'].mean()))
print('Claster 2: {}'.format(y_train.loc[labels_train == 2, 'price'].mean()))

