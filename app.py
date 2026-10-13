import streamlit as st
import numpy as np
import pickle

# Load models from notebooks
clustering_model = pickle.load(open("clustering_model.pkl", "rb"))
regression_model = pickle.load(open("regression_model.pkl", "rb"))
classification_model = pickle.load(open("classification_model.pkl", "rb"))
centres = clustering_model.cluster_centers_

# Clustering function
def clustering_n_activity(input_data):
    lst = np.array([0, 0, 0])
    for i in range(3):
        index = np.argmin(np.abs(np.array(centres[:, i]) - input_data[i]))
        lst[index] += 1
    nb=np.argmax(lst)
    return centres[nb][3]


# Title
st.title("Nature Harmony - AI Dashboard")

# Clustering section
st.header("Activity prediction")
age = st.slider("Age", 18, 80)
nb_group = st.number_input("Group number", value=1)
nb_kids = st.number_input("Children", value=0)

if st.button("Predict Activity"):
    result = clustering_n_activity([age, nb_group, nb_kids])
    st.write(f"Activity type: {result}")

# Regression & Classification section
st.header("Enter day conditions:")
ferie = st.selectbox("Holiday?", [0, 1])
weekend = st.selectbox("Weekend?", [0, 1])
temperature = st.slider("Temperature (°C)", 0, 40)
precipitation = st.slider("Precipitation", 0, 40)
vacance = st.selectbox("Vacance?", [0, 1])

if st.button("Predict"):
    input_data = np.array([weekend, ferie, temperature, precipitation, vacance]).reshape(1, -1)

    visitors = regression_model.predict(input_data)
    if int(visitors[0]<0): visitors[0]='0'
    st.write(f"Expected visitors: {int(visitors[0])}")

    input_data = np.array([weekend, ferie, temperature, precipitation, vacance, int(visitors)]).reshape(1, -1)
    activity = classification_model.predict(input_data)
    st.write(f"Place: {'Outdoor 🌿' if activity[0] == 1 else 'Indoor 🏠'}")