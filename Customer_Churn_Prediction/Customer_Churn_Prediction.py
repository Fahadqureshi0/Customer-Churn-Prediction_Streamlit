import numpy as np
import streamlit as st
import pickle

# Load model
loaded_model = pickle.load(open('Trained_model.sav', 'rb'))

def customer_churn_prediction(input_data):

    input_data = np.asarray(input_data).reshape(1, -1)

    prediction = loaded_model.predict(input_data)

    if prediction[0] == 0:
        return "📉 Customer Did not Churn"
    else:
        return "🚶‍♀️ Customer Churn"


def main():

    st.title("👤 Customer Churn Prediction")

    Age = st.text_input("Age")
    Gender = st.text_input("Gender")
    Tenure = st.text_input("Tenure")
    Usage_Frequency = st.text_input("Usage Frequency")
    Support_Calls = st.text_input("Support Calls")
    Payment_Delay = st.text_input("Payment Delay")
    Subscription_Type = st.text_input("Subscription Type")
    Contract_Length = st.text_input("Contract Length")
    Total_Spend = st.text_input("Total Spend")
    Last_Interaction = st.text_input("Last Interaction")

    if st.button("Predict"):

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
