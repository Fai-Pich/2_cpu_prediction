
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained models
lr_model_filename = "linear_regression_model.pkl"
rf_model_filename = "random_forest_model.pkl"

# Function to load models safely
def load_model(filename):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"❌ Model file '{filename}' not found. Please upload it to the GitHub repo.")
        return None

# Load models
lr_model = load_model(lr_model_filename)
rf_model = load_model(rf_model_filename)

# Streamlit UI
st.title("🔬 SYN Attack CPU Usage Predictor")
st.write("Compare **Linear Regression** and **Random Forest** models in predicting CPU usage under SYN attacks.")

# Sidebar for User Inputs
incoming_syn_rate = st.sidebar.slider("Incoming SYN Rate (packets/sec)", min_value=0, max_value=5000, value=350, step=1)
network_traffic = st.sidebar.slider("Network Traffic (KB/s)", min_value=0, max_value=5000, value=350, step=1)

# Convert input into DataFrame
input_features = pd.DataFrame([[incoming_syn_rate, network_traffic]], columns=['incoming_syn_rate', 'network_traffic'])

# Dropdown to select model
model_choice = st.radio("Select a model for prediction:", ["Linear Regression", "Random Forest"])

# Perform prediction
predicted_cpu_usage = None
if model_choice == "Linear Regression" and lr_model:
    predicted_cpu_usage = lr_model.predict(input_features)[0]
elif model_choice == "Random Forest" and rf_model:
    predicted_cpu_usage = rf_model.predict(input_features)[0]

# Ensure the prediction stays between 0 and 100
if predicted_cpu_usage is not None:
    predicted_cpu_usage = max(0, min(100, predicted_cpu_usage))
    st.subheader(f"🖥️ Predicted CPU Usage ({model_choice}): **{predicted_cpu_usage:.2f}%**")
else:
    st.error("❌ Model not loaded. Please check the model files.")

# Comparison of Both Models
if lr_model and rf_model:
    lr_prediction = max(0, min(100, lr_model.predict(input_features)[0]))
    rf_prediction = max(0, min(100, rf_model.predict(input_features)[0]))

    st.write("### 🔍 Model Comparison:")
    st.write(f"📈 **Linear Regression Prediction:** {lr_prediction:.2f}%")
    st.write(f"🌲 **Random Forest Prediction:** {rf_prediction:.2f}%")

    # ✅ Side-by-side bar chart using Matplotlib with fixed scale (0-100%)
    fig, ax = plt.subplots(figsize=(6, 4))

    models = ["Linear Regression", "Random Forest"]
    predictions = [lr_prediction, rf_prediction]

    ax.bar(models, predictions, color=["blue", "green"], width=0.4)
    ax.set_ylabel("Predicted CPU Usage (%)")
    ax.set_title("CPU Usage Prediction Comparison")
    ax.set_ylim(0, 110)  # ✅ Increased to 110% for extra spacing

    # Show values on top of bars (smaller text, better positioning)
    for i, v in enumerate(predictions):
        ax.text(i, v + 6, f"{v:.2f}%", ha='center', fontsize=10, fontweight='bold')  # ✅ Increased gap (v + 6)

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

# Footer
st.markdown("Developed for SYN Attack Analysis in a controlled environment. 🚀 ")
