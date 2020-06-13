#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import CRM_base as crm

# %% import sys

# %% from CRM_base import base_CRM

data_prod = pd.read_csv('init/prod_data.csv', index_col = 0, header = 0)
data_prod.index = data_prod['Date']
data_prod.index = pd.to_datetime(data_prod.index)
del data_prod['Date']
days_of_work = np.array((data_prod.index - data_prod.first_valid_index()).days)

# %%data_inj = #isert your code
data_inj = pd.read_csv('init/inj_data.csv', index_col = 0, header = 0)
data_inj.index = data_inj['Date']
del data_inj['Date']
#data_press = #isert your code
data_press = pd.read_csv('init/press_data.csv', index_col = 0, header = 0)
data_press.index = data_press['Date']
del data_press['Date']

# %%
Qt_target = data_prod.values[1:,:]
Qt = data_prod.values[:-1,:]
# %%
Inj = data_inj.values[1:,:]
dP = data_press.values[1:,:] - data_press.values[:-1,:]
time_delta = (days_of_work[1:] - days_of_work[:-1]).reshape(-1,1)
# %%
importlib.reload(crm)
#%%
test_crm = crm.CRMP()
#%%
X = np.concatenate((Qt, time_delta, Inj), axis = 1)
well_on = np.ones_like(Qt_target) * (Qt_target > 0)
#%%
test_crm.fit(X, Qt_target )
# %%
test_crm.test_func()
#%%
