import streamlit as st
import pandas as pd
import numpy as np
import pickle

#LOAD DATASET
data = pd.read_csv("traineddata.csv")
file = open("pipe.pkl", "rb")
pipe = pickle.load(file)
file.close()
st.title("Laptop Price Predictor")

#TAKE USER INPUT
#BRAND
company = st.selectbox('Brand', ["Please select an option..."] + list(data['Company'].unique()))
#LAPTOP TYPE
laptop_type = st.selectbox('Type', ["Please select an option..."] + list(data['TypeName'].unique()))
#RAM
ram = st.selectbox('RAM (GB)', ["Please select an option..."] + [2,4,6,8,12,16,24,32,64])
#OPERATING SYSTEM
os = st.selectbox('OS', ["Please select an option..."] + list(data['OpSys'].unique()))
#WEIGHT
weight = st.number_input("Weight of the Laptop (Kg)")
#TOUCH SCREEN
touchscreen = st.selectbox('Touchscreen', ["Please select an option...", 'No', 'Yes'])
#IPS
ips = st.selectbox('IPS', ["Please select an option...", 'No', 'Yes'])
#SCREEN SIZE
screen_size = st.number_input('Screen Size (inc):')
#RESOLUTION
resolution = st.selectbox('Screen Resolution', ["Please select an option...", '1920x1080', '1366x900', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
#CPU
cpu = st.selectbox('CPU', ["Please select an option..."] + list(data['CPU_Name'].unique()))
#HDD
hdd = st.selectbox('HDD (GB)', ["Please select an option..."] + [0,128,256,512,1024,2048])
#SSD
ssd = st.selectbox('SSD (GB)', ["Please select an option..."] + [0,8,128,256,512,1024])
#GPU
gpu = st.selectbox('GPU', ["Please select an option..."] + list(data['Gpu_brand'].unique()))

inputBox = [company,laptop_type,ram,os,touchscreen,ips,resolution,cpu,hdd,ssd,gpu]
inputNumber = [weight,screen_size]

#PREDICTION
if st.button('Predict Price'):
  if all(value != "Please select an option..." for value in inputBox) and all(inputNumber):
    if touchscreen =='Yes':
      touchscreen = 1
    else:
      touchscreen = 0
    if ips =='Yes':
      ips = 1
    else:
      ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2)+(Y_res**2))**0.5/(screen_size)

    query = np.array([company,laptop_type,ram,os,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu])
    query = query.reshape(1,12)
    prediction = float(np.exp(pipe.predict(query)[0]))
    price = f"$ {prediction:,.2f}"
    st.success(f"Prediction price of the configuration laptop is around {price}")

  else:
    st.warning("Please complete all required fields with accurate data to proceed")