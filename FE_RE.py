#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col 
from linearmodels import RandomEffects
from linearmodels import PanelOLS
import scipy as sp


# In[3]:


df = pd.read_stata("jtrain.dta")
df.head()


# In[4]:


df.info()


# In[5]:


df1 = df[["fcode", "hrsemp", "d88", "d89", "lemploy", "grant", "grant_1"]]
df1 = df1.dropna()

## Drop the firms if they don't have data for all three time periods 
df1 = df1.groupby("fcode").filter(lambda x: len(x)==3)
df1.info()


# In[6]:


## Compute the first difference
df1_diff = df1.groupby("fcode").diff(1).dropna()
df1_diff["const"] = 1


# In[7]:


## Run the first-difference OLS regression

exog_var = ["const", "d89", "lemploy", "grant", "grant_1"]

result1 = sm.OLS(df1_diff["hrsemp"], df1_diff[exog_var]).fit() 
print(result1.summary())


# In[8]:


df = pd.read_stata("murder.dta")
df.head()


# In[9]:


df.info()


# In[10]:


df2 = df[["id", "mrdrte", "d90", "d93", "exec", "unem"]]
df2 = df2.dropna()
df2.info()


# In[11]:


df2["const"] = 1

## Run Pooled OLS regression
result2 = sm.OLS(df2["mrdrte"],
                   df2[["const", "d90", "d93", "exec", "unem"]]).fit() 
print(result2.summary())


# In[12]:


## Compute the first difference
df2_diff = df2.groupby("id").diff(1).dropna()
df2_diff["const"] = 1
df2_diff.info()


# In[13]:


## Run the first-difference OLS regression

result2 = sm.OLS(df2_diff["mrdrte"],
                   df2_diff[["const", "d93", "exec", "unem"]]).fit() 
print(result2.summary())


# In[14]:


df = pd.read_stata("cornwell.dta")
df.head()


# In[15]:


df.info()


# In[16]:


year = pd.Categorical(df.year)
df3 = df.set_index(['county', 'year'])
df3['year'] = year
df3['const'] = 1


# In[17]:


## Estimate the model by random effects

exog_var = ['const', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87',
            'lprbarr', 'lprbconv', 'lprbpris', 'lavgsen', 'lpolpc']
exog = df3[exog_var]
model_RE = RandomEffects(df3.lcrmrte, exog)
result_RE = model_RE.fit()
print(result_RE)


# In[18]:


## Estimate the model by fixed effects

model_FE = PanelOLS(df3.lcrmrte, exog, entity_effects=True)
result_FE = model_FE.fit()
print(result_FE)


# In[19]:


## Compute W_i by averaging all explanatory variables across time

time_average = df3.groupby("county").mean()

## Merge the dataset

df3_merged = df3.merge(time_average, how='left', 
                       left_index=True, right_index=True, suffixes=('', '_avg'))


# In[20]:


## Estimate the model by random effects

exog_var = ['const', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87',
            'lprbarr', 'lprbconv', 'lprbpris', 'lavgsen', 'lpolpc',
            'lprbarr_avg', 'lprbconv_avg', 'lprbpris_avg', 'lavgsen_avg', 'lpolpc_avg']
exog = df3_merged[exog_var]
model_RE2 = RandomEffects(df3_merged.lcrmrte, exog)
result_RE2 = model_RE2.fit()
print(result_RE2)


# In[21]:


## Do the Wald-Test for the coefficients

formula = 'lprbarr_avg = lprbconv_avg = lprbpris_avg  = lavgsen_avg = lpolpc_avg = 0'
result_RE2.wald_test(formula=formula)


# In[22]:


## Add logarithms of the nine wage variables

exog_var = ['const', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87',
            'lprbarr', 'lprbconv', 'lprbpris', 'lavgsen', 'lpolpc',
            'lwcon', 'lwtuc', 'lwtrd', 'lwfir', 'lwser', 'lwmfg', 'lwfed', 'lwsta', 'lwloc']
exog = df3[exog_var]

## Estimate the model by fixed effects

model_FE2 = PanelOLS(df3.lcrmrte, exog, entity_effects=True)
result_FE2 = model_FE2.fit()
print(result_FE2)


# In[23]:


## Do the Wald-Test for the coefficients

formula = 'lwcon = lwtuc = lwtrd = lwfir = lwser = lwmfg = lwfed = lwsta = lwloc = 0'
result_FE2.wald_test(formula=formula)


# In[24]:


df3_diff = df3.groupby("county").diff(1)
df3_diff.info()


# In[25]:


## Compute the first difference

df3_diff = df3.groupby("county").diff(1)

df3_diff['const'] = 1 
exog_var = ['const', 'd83', 'd84', 'd85', 'd86', 'd87',
            'lprbarr', 'lprbconv', 'lprbpris', 'lavgsen', 'lpolpc']
df3_diff = df3_diff[["lcrmrte", 'const', 'd83', 'd84', 'd85', 'd86', 'd87',
            'lprbarr', 'lprbconv', 'lprbpris', 'lavgsen', 'lpolpc']].dropna()


## Estimate the model by first difference

result_diff = sm.OLS(df3_diff["lcrmrte"], exog=df3_diff[exog_var]).fit() 
print(result_diff.summary())


# In[28]:


## Extract the residual and run the AR(1) regression

resid = result_diff.resid
resid_lag = resid.groupby("county").shift(1)
result_AR_test = sm.OLS(resid, resid_lag, missing = 'drop').fit() 
print(result_AR_test.summary())

