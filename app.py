# import libraries
import streamlit as st
import numpy as np
import pickle

# import models
laptops_df = pickle.load(open("data/laptops_df.pkl", "rb"))
pipe = pickle.load(open("data/pipe.pkl", "rb"))

# app title
st.title("AI LAPTOP PRICE PREDICTOR")

# select brand
brand = st.selectbox("BRANDS", laptops_df["Company"].unique())

# os
os = st.selectbox("OS", laptops_df["OS"].unique())

# select laptop type
laptop_type = st.selectbox("LAPTOP TYPE", laptops_df["TypeName"].unique())

# select RAM
ram = st.selectbox("RAM", sorted(laptops_df["Ram"].unique())[1:])

# cpu
cpu = st.selectbox("CPU", laptops_df["Cpu Brand"].unique())

# gpu
gpu = st.selectbox("GPU", laptops_df["Gpu Brand"].unique())

# hdd
hdd = st.selectbox("HDD", [0, 128, 256, 512, 1024,
                   1024*2, 1024*3, 1024*4, 1024*5])

# ssd
ssd = st.selectbox("SSD", [0, 128, 256, 512, 1024,
                   1024*2, 1024*3, 1024*4, 1024*5])

# touchscreen
touchscreen = st.selectbox("TOUCHSCREEN", ["YES", "NO"])

# screensize
screensize = st.number_input("SCREEN SIZE (In)")

# ips panel
ips = st.selectbox("IPS", ["Yes", "No"])

# screen resolution
screen_resolution = st.selectbox("SCREEN RESOLUTION", ["1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800",
                                                       "2880x1800", "2304x1440", "2560x1600", "2560x1440"])

# weight
weight = st.number_input("LAPTOP WEIGHT (Kg)")

print(laptops_df.columns)
# predict price
if st.button("PREDICT PRICE"):

    # since ips is Yes-No
    if ips == "Yes":
        ips = 1
    else:
        ips = 0

    # since touchscreen is Yes-No
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0

    # extracting x_res & y_res from resolution input
    x_res = int(screen_resolution.split("x")[0])
    y_res = int(screen_resolution.split("x")[1])

    # ppi
    ppi = np.sqrt((x_res**2) + (y_res**2))/screensize

    # query : order of the column matters for prediction
    query = np.array([brand, laptop_type, ram, weight,
                     touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    print(query)

    # reshape query to 1x12
    query = query.reshape(1, 12)

    # predict
    prediction = pipe.predict(query)  # in log form

    # convert from log form to actual form
    predicted_price = int(np.exp(prediction)[0])

    # display predicted price
    st.title(f"Rs {round(predicted_price, 2)}")
