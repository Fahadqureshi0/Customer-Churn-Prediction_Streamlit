import numpy as np
import streamlit as st
import pickle

# Loading Model

loaded_model = pickle.load(open('F:/Machine Learning Projects/Customer_Churn_Prediction/Trained_model.sav', 'rb'))

def customer_churn_prediction():
    input_data = (23, 1, 3, 5, 9, 25, 0, 0, 120, 29)

    # Change input data into numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
       return('Customer Did not Churn')
    else:
       return('Customer Churn')


result = customer_churn_prediction()
print(result)
