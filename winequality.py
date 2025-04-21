import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("Wine Quality Prediction using KNN")

# Load dataset directly
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
st.write("## Dataset Preview", data.head())

# Split features and label
X = data.drop('quality', axis=1)
y = data['quality']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create sliders for user input in sidebar
st.sidebar.header("Input Chemical Properties")
input_data = []
for col in data.columns[:-1]:
    val = st.sidebar.slider(
        label=col,
        min_value=float(data[col].min()),
        max_value=float(data[col].max()),
        value=float(data[col].mean())
    )
    input_data.append(val)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Predict on user input
input_scaled = scaler.transform([input_data])
prediction = model.predict(input_scaled)

st.write("## Prediction Results")
st.write(f"*Predicted Wine Quality:* {prediction[0]}")
st.write(f"*Model Accuracy on Test Set:* {accuracy:.2f}")