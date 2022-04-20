#!/usr/bin/env python
# coding: utf-8

# ### Вычисления с помощью Numpy

# ЗАДАНИЕ 1

# In[4]:


import numpy as np
import pickle
import pandas as pd


# In[5]:


a = np.array([[1, 6], [2, 8], [3, 11], [3, 10], [1, 7]])
a


# In[6]:


mean_a = a.mean(axis = 0)
mean_a


# ЗАДАНИЕ 2

# In[7]:


a_centered = a - mean_a
a_centered


# ЗАДАНИЕ 3

# In[10]:


a_centered_sp = a_centered[:, 0] @ a_centered[:, 1]
a_centered_sp


# ЗАДАНИЕ 4

# In[12]:


np_cov = np.cov(m = a.T)[0, 1]
np_cov == a_centered_sp


# ### Работа с данными в Pandas

# ЗАДАНИЕ 1

# In[13]:


import numpy as np
import pickle
import pandas as pd


# In[16]:


authors = pd.DataFrame({'author_id': [1, 2, 3], 'author_name': ['Тургенев', 'Чехов', 'Островский']})
authors


# In[18]:


book = pd.DataFrame({'author_id': [1, 1, 1, 2, 2, 3, 3], 'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'], 'price': [450, 300, 350, 500, 450, 370, 290]})
book


# ЗАДАНИЕ 2

# In[20]:


authors_price = pd.merge(authors, book, on = 'author_id', how = 'outer')
authors_price


# ЗАДАНИЕ 3

# In[22]:


top5 = authors_price.sort_values(by = 'price', inplace = False)[::-1][:5]
top5.reset_index(drop = True, inplace = True)
top5


# ЗАДАНИЕ 4

# In[24]:


gb = authors_price.groupby('author_name')
authors_stat = pd.DataFrame({'min_price': gb['price'].min(), 'max_proce': gb['price'].max(), 'mean_price':gb['price'].mean()})
authors_stat


# ЗАДАНИЕ 5

# In[25]:


authors_price['cover'] = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
authors_price


# In[26]:


book_info = pd.pivot_table(data = authors_price, values = 'price', index = 'author_name', columns = 'cover', aggfunc = 'sum', fill_value = 0, dropna = False)
book_info


# In[27]:


book_info.to_pickle('book_info.pkl')


# In[28]:


book_info2 = pd.read_pickle('book_info.pkl')


# In[29]:


book_info == book_info2

