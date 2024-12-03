import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model and the LabelEncoder
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a LabelEncoder for categorical variables (if required)
lr = LabelEncoder()

# You might need to create the LabelEncoder for categorical columns that were encoded during training
# Example:
# company_encoder = lr.fit_transform(['Apple', 'Acer', 'HP', 'Asus'])
# Update these based on how you encoded columns during training

# Sidebar Inputs for the Dashboard
st.title('Laptop Price Prediction')

# Dropdowns for each feature
company = st.selectbox('Select Company', ['Apple', 'Acer', 'HP', 'Asus', 'Dell', 'Lenovo', 'Samsung', 'Other'])
type_name = st.selectbox('Select Type', ['Ultrabook', 'Notebook', 'Gaming', 'Convertible'])
inches = st.selectbox('Select Inches', ['13.3', '14.0', '15.6', '17.3'])
screen_resolution = st.selectbox('Select Screen Resolution', ['Full HD', 'Quad HD+', 'WQXGA', '4K Ultra HD', 'HD Ready', 'WQHD+'])
cpu = st.selectbox('Select CPU', ['Intel Core i5', 'Intel Core i7', 'AMD A9-Series', 'Intel Core i3', 'Intel Core i9'])
ram = st.selectbox('Select RAM (GB)', ['4', '8', '16', '32'])
memory = st.selectbox('Select Memory (GB)', ['128', '256', '512', '1TB'])
gpu = st.selectbox('Select GPU', ['Intel Iris Plus Graphics', 'Intel HD Graphics', 'AMD Radeon R5', 'Nvidia GeForce MX150', 'AMD Radeon Pro 455', 'Intel UHD Graphics 620'])
os = st.selectbox('Select Operating System', ['macOS', 'Windows 10', 'No OS'])
storage_type = st.selectbox('Select Storage Type', ['SSD', 'Other'])

# Convert user input to the required format for the model
# Convert numerical inputs to float
inches = float(inches)
ram = int(ram)
memory = int(memory)
price = 0  # Since price is what we want to predict, don't include it as input

# Encode categorical inputs
company_encoded = lr.fit_transform([company])[0]
type_name_encoded = lr.fit_transform([type_name])[0]
screen_resolution_encoded = lr.fit_transform([screen_resolution])[0]
cpu_encoded = lr.fit_transform([cpu])[0]
gpu_encoded = lr.fit_transform([gpu])[0]
os_encoded = lr.fit_transform([os])[0]
storage_type_encoded = lr.fit_transform([storage_type])[0]

# Prepare the input for prediction
input_data = np.array([[company_encoded, type_name_encoded, inches, screen_resolution_encoded, cpu_encoded, ram, memory, gpu_encoded, os_encoded, storage_type_encoded]])
input_df = pd.DataFrame(input_data, columns=['Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Storage Type'])

# Predict button
if st.button('Predict Laptop Price'):
    predicted_price = model.predict(input_df)
    st.write(f"The predicted price for the selected laptop is: ₹{predicted_price[0]:,.2f}")

