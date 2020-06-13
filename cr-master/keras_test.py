#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import CRM_base as crm
import plot_funcs as crm_plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import seaborn as sns
#%%
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, SimpleRNN
from keras.models import Sequential
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
Inj = data_inj.values[1:,:]
dP = data_press.values[1:,:] - data_press.values[:-1,:]
time_delta = (days_of_work[1:] - days_of_work[:-1]).reshape(-1,1)
# %%
#%%
X = np.concatenate((Qt, Inj), axis = 1)
#%%
X_norm = X.reshape(-1,1)
X_scaler = MinMaxScaler()
X_scaler.fit(X_norm)
X_norm = X_scaler.transform(X_norm)
X_norm = X_norm.reshape(X.shape)

#%%
plt.plot(X_norm[:,:5])
#%%
x_train = X_norm[100:300,:]
y_train = X_scaler.transform(Qt_target[100:300,:])
x_test = X_norm[300:,:]
y_test = X_scaler.transform(Qt_target[300:,:])
#%%
print(x_test[1,0])
print(y_test[0,0])
#%%
#X = np.concatenate((Qt, time_delta, Inj), axis = 1)
#well_on = np.ones_like(Qt_target) * (Qt_target > 0)
#%%
plt.plot(x_test[:,1],color = 'g')
plt.plot(y_test[:,1], color = 'r')
#%%
#%%
# %%

#%%
plt.plot(Qt_pred[:,1])
plt.plot(Qt_target[:,1])
#%%
#x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#%%
def reshape_for_lstm(data, y, window_size = 1):
    transformed_data = np.zeros((data.shape[0]-window_size+1, window_size, data.shape[1]))
    for i in range(data.shape[0] - window_size+1):
        transformed_data[i,:,:] = data[i:i+window_size,:]
    if y is not None:
        return transformed_data, y[window_size-1:,:]
    else:
        return transformed_data
#%%
window_size = 20
batch_size = x_train.shape[0] - window_size +1
x_train_trans, y_train_trans = reshape_for_lstm(x_train, y_train, window_size = window_size)

x_test_trans, y_test_trans = reshape_for_lstm(x_test, y_test, window_size = window_size)
#%%
y_test_trans.shape
#%%
def smae(y_target, y_predicted):
    return np.average(np.absolute(y_target - y_predicted) / (np.absolute(y_target + y_predicted)/2), axis=0)
#%%
def set_lstm(loss_func = 'mse', num_tsteps = 1, num_units = 5):
    model = Sequential()
    model.add(LSTM(units = num_units, input_shape = (num_tsteps,9)))
    #model.add(Dense(18, input_shape=(9, ), activation = 'tanh'))
    #model.add(Dense(72, activation = 'tanh'))
    #model.add(Dense(90, activation = 'tanh'))
    #model.add(Dense(72, activation = 'tanh'))
    #model.add(Dense(36, activation = 'tanh'))
    model.add(Dense(5, activation = 'linear'))
        #model.add(Dropout(0.2))

        #model.add(LSTM(layers[2], return_sequences=False))
        #model.add(Dropout(0.2))
        #   model.add(Dense(output_dim=layers[3]))
        #model.add(Activation("linear"))

    model.compile(loss=loss_func, optimizer="rmsprop", metrics = ['mean_absolute_percentage_error'])
    model.summary()
    print("> model compiled : ")
    return model
#%%
x_test_null = x_test.copy()
x_test_null[1:,:5] = 0
#%%
#%%
model_mse = set_lstm(loss_func ='mse', num_tsteps = window_size, num_units = 18)


#%%
model_mse.fit(x_train_trans, y_train_trans, batch_size=batch_size, epochs=1000)
#%%
model_mse.evaluate(x_test_trans, y_test_trans)
#%%
y_pred_test_step = model_mse.predict(x_test_trans)
y_pred_train_step = model_mse.predict(x_train_trans)
Qt_pred_test_step = X_scaler.inverse_transform(y_pred_test_step)
Qt_pred_train_step = X_scaler.inverse_transform(y_pred_train_step)
Qt_pred = np.concatenate((Qt_target[:100+window_size-1,:], Qt_pred_train_step,Qt_target[300:300+window_size-1,:], Qt_pred_test_step), axis = 0)
#%%
Qt_pred_train_step.shape
#%%
def predict_sequence_full(model, X, num_p, lstm = False, window_size=1):
    #predicted = []
    if lstm:
        for i in range(window_size, X.shape[0]):
                X[i,:num_p] = model.predict(reshape_for_lstm(X[i-window_size:i,:], None, window_size))
    else:
        for i in range(1, X.shape[0]):
                X[i,:num_p] = model.predict(X[[i-1],:])
    return X[:,:num_p]

#%%
y_pred_train_seq = predict_sequence_full(model_mse, x_train, 5, lstm = True, window_size=window_size)
y_pred_test_seq = predict_sequence_full(model_mse, x_test_null, 5, lstm = True, window_size=window_size)
Qt_pred_test_seq = X_scaler.inverse_transform(y_pred_test_seq)
Qt_pred_train_seq = X_scaler.inverse_transform(y_pred_train_seq)
Qt_pred_seq = np.concatenate((Qt_target[:100,:], Qt_pred_train_seq, Qt_pred_test_seq), axis = 0)
#%%
plt.plot(Qt_pred[:,3], color='y')
plt.plot(Qt_pred_seq[:,3], color='r')
plt.plot(Qt_target[:,3],  color='g')
#%%
plt.plot(Qt_target[:,0] - Qt_pred_seq[:,0])
#%%
max_rate = Qt_target.max()
sns.color_palette("GnBu_r")
g2 = (sns.jointplot( Qt_target[300:,:], Qt_pred_seq[300:,:], kind = 'scatter',
              size = 8, xlim=(0,max_rate),  ylim=(0,max_rate), dropna = True))#.plot_joint(sns.kdeplot, zorder=0, n_levels=6))
g2.ax_joint.set(xlabel = 'История, Дебит жидкости, м3/сут', ylabel = 'Прогноз, Дебит жидкости, м3/сут', title = 'Тренировочная выборка')
#%%
smae(Qt_target[300:,:], Qt_pred_seq[300:,:])
#%%
r2_score(Qt_target[300:,:], Qt_pred_seq[300:,:], multioutput='raw_values')
