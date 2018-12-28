
# coding: utf-8

# In[68]:


# for data analysis
import numpy as np # perform math operations (matrix math)
import pandas as pd 
from pandas import Series, DataFrame

# for plotting charts
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

#for the image display
from IPython.display import Image

#statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats

import os
os.chdir(r"C:\Users\prisc\code\Intro to DA\Final Group Project\Data file")


# In[69]:


# Main 3 libraries
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Additional libraries
import matplotlib.pyplot as plt 
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D


# In[70]:


df = pd.read_csv("Intelligent_Couponing.csv")


# In[71]:


#############################
### 2.EDA before cleaning ###
#############################

#Table of Content 

#2.1 Variable Indetification
    #2.1.1 How does my data looks like?
    #2.1.2 Data type & basic statistics (central tendency and spread for continous; count for categorical)
    #2.1.3 How does my variable look like?(predictor & target, continous & categorical, datatypes)
#2.2 Uni-variant Analyisis
    #2.2.1 countinous variable - do we have outliers?
    #2.2.2 categorical variable -what is the distribution of each category?
#2.3 Bi-variant Analysis 
    # my target variable to other categorical variable
    # Chi square test - are there any significate association between the two variables?
#2.4 Correlation


# In[72]:


###2.1 Variable Identification ###
#2.1.1 How does my data looks like?
print(df.head())


# In[73]:


#2.1.2 Data Types & Basic Description 
print(df.info())
print(df.describe())


# In[74]:


# 2.1.3 How does my variable looks like?
# my predictor variables & target variables 
Image(filename='1.png',width=750, height=750)


# In[75]:


# what are my variable data type and variable category
Image(filename='2.png',width=750, height=750)


# In[76]:


###2.2 Uni-Variant Analysis ###
#2.2.1 Continuous Variable
# weight
sns.distplot(df['weight'], bins = 30, color = "blue")

#Question: why we have weight which equals to 0 ?


# In[77]:


# no.of items
sns.distplot(df['numberitems'], bins = 30, color = "purple")


# In[84]:


#2.2.2 Categorical Data - understand the distribution of each category
#data which has more than one categories - count
variables = ['salutation','paymenttype','case','domain']
print("Data Distribution Analysis")
for v in variables:
    df = df.sort_values(by=[v])
    df[v].value_counts().plot(kind = 'bar')
    plt.title(v)
    plt.show()


# In[85]:


#data which has either yes or no entry
variables = ['newsletter','deliverytype','voucher','entry','shippingcosts']
print("Data Distribution Analysis")
for v in variables:
    df = df.sort_values(by=[v])
    df[v].value_counts().plot(kind = 'bar')
    plt.title(v)
    plt.show()


# In[45]:


### 2.3 Bi-Variant Analysis ###
## Target 90 against other Catgorical Variables
## Chi-squred to test aossiciation(if knowing A will help you to predict B)


# In[14]:


##2.3.1 Target90 against salutation (female, male or company)
df_group1 = df.groupby(['target90','salutation'])['target90'].count().unstack('target90').fillna(99)
df_group1.plot(kind='bar', figsize=(10,10), stacked = True )
df_group1

# null-hypothesis: they are indenpent
# chi-square test for the independance between target90 and gender
scipy.stats.chi2_contingency(df_group1)

# results shows that they are dependent


# In[15]:


#2.3.2 Target90 against newsletter
df_group1 = df.groupby(['target90','newsletter'])['target90'].count().unstack('target90').fillna(99)
df_group1.plot(kind='bar', figsize=(10,10), stacked = True )
df_group1

# chi-square test for the independance between target90 and gender
scipy.stats.chi2_contingency(df_group1)

#result: dependent


# In[16]:


#2.3.3 Target90 against domain (assumptio: domain may be an indicator that if you are a sophisticated user)
df_group1 = df.groupby(['target90','domain'])['target90'].count().unstack('target90').fillna(99)
df_group1.plot(kind='bar', figsize=(10,10), stacked = True )
plt.legend
df_group1

scipy.stats.chi2_contingency(df_group1)
# result: not dependent - drop this column 


# In[17]:


#2.3.4 Target90 against payment type
df_group1 = df.groupby(['target90','paymenttype'])['target90'].count().unstack('target90').fillna(99)
df_group1.plot(kind='bar', figsize=(10,10), stacked = True )
df_group1

scipy.stats.chi2_contingency(df_group1)

# result: dependent


# In[18]:


# 2.3.5 target90 againset deliverytype
df_group1 = df.groupby(['target90','deliverytype'])['target90'].count().unstack('target90').fillna(99)
df_group1.plot(kind='bar', figsize=(10,10), stacked = True )
df_group1

scipy.stats.chi2_contingency(df_group1)

#result: dependent


# In[81]:


# 2.3.6 target90 againset voucher (if previous vouncher cashed or not)
df_group1 = df.groupby(['target90','voucher'])['target90'].count().unstack('target90').fillna(99)
df_group1.plot(kind='bar', figsize=(10,10), stacked = True )
df_group1

scipy.stats.chi2_contingency(df_group1)

#result: dependent


# In[86]:


# 2.3.7 target90 againset case
df_group1 = df.groupby(['target90','case'])['target90'].count().unstack('target90').fillna(99)
df_group1.plot(kind='bar', figsize=(10,10), stacked = True )
df_group1

scipy.stats.chi2_contingency(df_group1)
# result: dependent


# In[22]:


# 2.3.8 target90 against entry
df_group1 = df.groupby(['target90','entry'])['target90'].count().unstack('target90').fillna(99)
df_group1.plot(kind='bar', figsize=(10,10), stacked = True )
df_group1

scipy.stats.chi2_contingency(df_group1)
# result: dependent


# In[92]:


# 2.3.9 target90 against shippingcosts
df_group1 = df.groupby(['target90','shippingcosts'])['target90'].count().unstack('target90').fillna(99)
df_group1.plot(kind='bar', figsize=(10,10), stacked = True )
df_group1

scipy.stats.chi2_contingency(df_group1)
# result: dependent


# In[23]:


# Conclunsion (Can my variables predict the target 90?):
# 1. domain and target90 are independent,so our assumption is not right, drop the column
# 2. among all categorical varirable: newsletter, paymenttype, deliverytype, case, voucher, entry, shippingcosts are dependent with the target 90


# In[108]:


### 2.4 getting the correlation ### 
#correlation of 11 product category and target 90
df_heatmap = df.iloc[:,26:38]
corr = df_heatmap.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr, vmax=.8,annot_kws={'size': 5}, annot=True, cmap = 'BuPu');


# In[109]:


# correlation of other variable and target 90
df_heatmeat2 = df.loc [:,['salutation', 'newsletter', 'paymenttype','deliverytype','case','voucher', 'entry', 'shippingcosts','numberitems', 'weight', 'target90']]
corr = df_heatmeat2.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr, vmax=.8,annot_kws={'size': 5}, annot=True, cmap = "YlGnBu");


# In[ ]:


# conclusion from heatmap correlations: 
# The heatmap gives us a visualization of the correlation matix, the correlation between weight & number of item is 0.77, 
# thus, we think they are highly correlcted, we need to remove one


# In[ ]:


# limitation: more through consideration of multicolinearity, maybe use PCA to remove features

