#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

from scipy import stats
import re

import acquire
import prepare


# In[2]:


df = acquire.get_telco_data()


# In[3]:


#df.info()


# In[4]:


train, validate, test = prepare.prep_telco_data(df)


# In[5]:


#train.info()


# ## multiple lines/churn

# In[6]:


#multiple lines and churn chi2
def get_chi2_mult(train):
    observed = pd.crosstab(train.multiple_lines_Yes, train.churn_Yes)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')


# ## dependents/churn

# In[7]:


#dependents and churn chi2
def get_chi2_dep(train):
    observed = pd.crosstab(train.dependents_encoded, train.churn_Yes)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
#get_chi2_dep(train)


# ## partner/churn

# In[8]:


#partner and churn chi2
def get_chi2_part(train):
    observed = pd.crosstab(train.partner, train.churn_Yes)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')


# ## senior/churn

# In[9]:


#senior_citizen and churn chi2
def get_chi2_sen(train):
    observed = pd.crosstab(train.senior_citizen, train.churn_Yes)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')


# ## internet/churn

# In[10]:


#internet_service_type_None and churn chi2
def get_chi2_int(train):
    observed = pd.crosstab(train.internet_service_type_None, train.churn_Yes)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')


# In[11]:



#observed6 = pd.crosstab(train.phone_service, train.churn_Yes)
#chi2, p, degf, expected = stats.chi2_contingency(observed6)
#print(f'chi^2 = {chi2:.4f}')
#print(f'p     = {p:.4f}')


# ## customer tenure/churn

# In[12]:


#churn_sample=train[train.churn_Yes==1].tenure
#overall_mean = train.tenure.mean()

#t, p = stats.ttest_1samp(churn_sample, overall_mean)
#print(f't     = {t:.4f}')
#print(f'p     = {p:.4f}')


# In[13]:


def get_ind_ttest_tenure(train):

    churn_sample=train[train.churn_Yes==1].tenure
    no_churn_sample=train[train.churn_Yes==0].tenure
    overall_mean = train.tenure.mean()

    t, p = stats.ttest_ind(no_churn_sample, churn_sample, equal_var=False)
    print(f't     = {t:.4f}')
    print(f'p     = {p:.4f}')
    
#get_ind_ttest_tenure(train)


# In[ ]:




