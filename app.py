import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import streamlit as st
import pandas as pd

def datamodel():
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(2, 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    return model
model = datamodel()
loaded_model = keras.models.load_model('bestmodel.h5')
st.title("Fetch Rewards Take Home Expercise")
st.header("Prediciting Receipt Counts for 2022")
st.write("Choose the month for prediction of Receipt Counts")
month_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
month_dict = {}
for idx, month_val in enumerate(month_list):
    month_dict[month_val] = idx + 1
user_selection = str(st.selectbox("Choose one", month_list))

if user_selection:
    selected_month = month_dict[user_selection]
else:
    selected_month = 1

def reshape_conversion(data):
    numpyarray = np.array([data])
    numpyarray_dtype = numpyarray.astype(np.float32)
    numpyarray_reshape = numpyarray_dtype.reshape(-1, 1)
    tensor_array = torch.tensor(numpyarray_reshape)
    tensor_array = Variable(tensor_array)
    return tensor_array

input_val = reshape_conversion(selected_month)
input_val= input_val.reshape(1, 1, 1)
input_val=tf.convert_to_tensor(input_val, dtype=tf.float32)

res = loaded_model(input_val)
res = res.numpy()[0][0]
output = int(res * (234507746 - 211813448) + 211813448)
st.write("The predicted count for the month of", user_selection, "is", output)