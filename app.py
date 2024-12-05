import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
@st.cache_data
def load_data():
    # Ensure 'heart_disease.csv' is in the same directory as this script
    data = pd.read_csv('heart_disease.csv')
    return data

# Train the model
@st.cache_resource
def train_model(data):
    # Use only the three selected variables
    X = data[['thalach', 'age', 'cp']]  # Features
    y = data['target']  # Target variable ('1' = disease present, '0' = no disease)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Streamlit app setup
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# App styling for simple interface
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f8ff;  /* Light background color */
        }
        .custom-header {
            color: black;  /* Header text color */
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .stMarkdown, .stTitle, .stHeader, .stTextInput, .stSelectbox label, .stNumberInput label {
            color: black; /* Set all text to black */
        }
        .stButton>button {
            font-size: 16px;
            color: black;  /* Button text color */
        }
        .stError, .stSuccess {
            color: black; /* Set result text to black */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Heading for the app
st.markdown("<div class='custom-header'>Heart Disease Predictor</div>", unsafe_allow_html=True)

# Load dataset and train model
data = load_data()
model = train_model(data)

# Layout for image and input fields
col1, col2 = st.columns([1, 2])  # Divide page into two sections

# Add image in the first column (no curved edges)
with col1:
    st.markdown("### Your Heart Health Matters")
    st.image("heart_image.png", caption="Heart Image", use_container_width=True)

# Add input fields in the second column
with col2:
    st.markdown("### Enter Your Health Details Below")
    
    # User input with descriptive names
    st.markdown("**Age**")
    st.markdown("_Enter the age of the individual_")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    st.markdown("**Chest Pain Type (cp)**")
    st.markdown("_Select the chest pain type_")
    cp = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"])

    st.markdown("**Maximum Heart Rate Achieved (thalach)**")
    st.markdown("_Enter the maximum heart rate achieved_")
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=150)

    # Map chest pain type to numeric values (to match the dataset)
    cp_mapping = {"Type 1": 1, "Type 2": 2, "Type 3": 3, "Type 4": 4}
    cp_numeric = cp_mapping[cp]

    # Predict button
    if st.button("Predict"):
        input_data = np.array([[thalach, age, cp_numeric]])  # Arrange inputs in a 2D array
        prediction = model.predict(input_data)
        
        # Display result in black
        if prediction[0] == 1:
            st.markdown("<h3 style='color: black;'>Heart Disease Present. Please consult a doctor.</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: black;'>Heart Disease Not Present. Keep up the good work!</h3>", unsafe_allow_html=True)
