#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col 
import scipy as sp
from linearmodels import OLS
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


# In[2]:


df = pd.read_stata("FRED-QD.dta")
df.head()


# In[6]:


df["oilpricex"].isna().sum()


# In[3]:


df.info()


# In[4]:


## Transform the series by taking first differences

oilPrice = df["oilpricex"]
oilPrice_diff = oilPrice.diff(1).dropna()
oilPrice_diff.head()


# In[5]:


## Create a dataframe for lagged oil prices

exog = [oilPrice_diff.shift(1), oilPrice_diff.shift(2), oilPrice_diff.shift(3), oilPrice_diff.shift(4)]
exog = pd.concat([oilPrice_diff.shift(1), oilPrice_diff.shift(2), oilPrice_diff.shift(3), oilPrice_diff.shift(4)], axis=1)
exog.columns = ['oilpricex_d1', 'oilpricex_d2', 'oilpricex_d3', 'oilpricex_d4']
exog["const"] = 1
exog = exog.iloc[:, [4, 0, 1, 2, 3]]

## Estimate an AR(4) by OLS with heterokesdacity covarance matrix

mod = sm.OLS(endog=oilPrice_diff, exog=exog, missing="drop")
result1 = mod.fit(cov_type='HAC', cov_kwds={'maxlags':4})
print(result1.summary())


# In[6]:


## Estimate an AR(4) by Conditional MLE with heterokesdacity covarance matrix

mod = AutoReg(oilPrice_diff, 4, old_names=False)
result2 = mod.fit(cov_type='HAC', cov_kwds={'maxlags':4})
print(result2.summary())


# In[7]:


## Do the Wald-Test for the coefficients (OLS & Conditional MLE)

hypothesis1 = 'oilpricex_d1 = oilpricex_d2 = oilpricex_d3 = oilpricex_d4 = 0'
hypothesis2 = 'oilpricex.L1 = oilpricex.L2 = oilpricex.L3 = oilpricex.L4 = 0'


test1 = result1.wald_test(hypothesis1)
test2 = result2.wald_test(hypothesis2)


# In[8]:


## Test resut for OLS
print(test1)


# In[9]:


## Test resut for Conditional MLE
print(test2)


# In[ ]:




