import numpy as np
import streamlit as st
import pickle

# Loading Model

loaded_model = pickle.load(open('/Customer_Churn_Prediction/Trained_model.sav', 'rb'))

def customer_churn_prediction():
    input_data = (23, 1, 3, 5, 9, 25, 0, 0, 120, 29)

    # Change input data into numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
       return(' 📉 Customer Did not Churn')
    else:
       return(' 🚶‍♀️ Customer Churn')


result = customer_churn_prediction()
print(result)


# ---------------Streamlit---------------#

def main():

   st.title(' 👤 Customer Churn Prediction')

   Age = st.text_input('Person Age')
   Gender = st.text_input('Person Gender')
   Tenure = st.text_input('Tenure Value')
   Usage_Frequency = st.text_input('Usage Frequency Value')
   Support_Calls = st.text_input('Support Call Value')	 
   Payment_Delay = st.text_input('Payment Delay Value')
   Subscription_Type = st.text_input('Subscription Type')
   Contract_Length = st.text_input('Contract Length')
   Total_Spend = st.text_input('Total Spend Value')
   Last_Interaction = st.text_input('Last Interaction')


   if st.button('Predict'):

      result = customer_churn_prediction([
         float(Age),
         float(Gender),
         float(Tenure),
         float(Usage_Frequency),
         float(Support_Calls),
         float(Payment_Delay),
         float(Subscription_Type),
         float(Contract_Length),
         float(Total_Spend),
         float(Last_Interaction)
      ])

      st.success(result)


if __name__ == "__main__":
   main()

