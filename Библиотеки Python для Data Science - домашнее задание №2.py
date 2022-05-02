#!/usr/bin/env python
# coding: utf-8

# ЗАДАНИЕ 1

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBekend.figure_format = 'svg'")

import warnings
warnings.filterwarnings('ignore')

x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]

plt.plot(x, y)
plt.show

plt.scatter(x, y)
plt.show()

# Можно было записать plt.plot и plt.scatter в 2 разные ячейки чтобы получить 2 разных графика: один с прямой, второй с точками. В этом задании я решила попробовать записать их в одной ячейке.


# ЗАДАНИЕ 2

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 51)
t


# In[5]:


f = np.cos(t)
f


# In[13]:


plt.plot(t, f, color = 'green')
plt.title('График f(t)', color = 'darkred')
plt.xlabel('t - значения', color = 'darkred')
plt.ylabel('f - значения', color = 'darkred')
plt.axis([0.5, 9.5, -2.5, 2.5])
plt.show


# ЗАДАНИЕ 3

# In[14]:


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 51)
x


# In[16]:


y1 = x**2
y2 = 2 * x + 0.5
y3 = -3 * x - 0.5
y4 = np.sin(x)

fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax1, ax2, ax3, ax4 = ax.flatten()
ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y4)

ax1.set_title('График y1')
ax2.set_title('График y2')
ax3.set_title('График y3')
ax4.set_title('График y4')

ax1.set_xlim([-5, 5])
fig.set_size_inches(8, 6)
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
plt.show()


# ЗАДАНИЕ 4

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
dataset_path = 'C:\PythonDataFiles\creditcard.csv'

df = pd.read_csv(dataset_path, sep = ',')
df.head(4)


# In[20]:


cl = df['Class'].value_counts()
cl


# In[21]:


cl.plot(kind = 'bar')
plt.show()


# In[25]:


plt.bar(['Class 0', 'Class 1'], df['Class'].value_counts(dropna = False))
plt.title(label = 'Value count (linear)')


# In[24]:


cl.plot(kind = 'bar', logy = True)
plt.show()


# In[27]:


plt.hist(df[df['Class'] == 1]['V1'], density = True, bins = 20, alpha = 0.5, color = 'red')
plt.hist(df[df['Class'] == 0]['V1'], density = True, bins = 20, alpha = 0.5, color = 'grey')
plt.legend(labels = ['Class 1', 'Class 0'])
plt.xlabel('v1')
plt.show()

