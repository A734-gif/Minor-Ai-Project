# Artificial Intelligence Minor Project
# ANN Model for Customer Churn Prediction

# ----------------- Import Libraries -----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ----------------- Load Dataset -----------------
# Change path if needed
file_path = "Churn_Modelling.csv"  # rename dataset if different
dataset = pd.read_csv(file_path)

print("Dataset Shape:", dataset.shape)
print("Columns:", dataset.columns)

# ----------------- Preprocessing -----------------
# Drop unnecessary fields
X = dataset.drop(["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)
y = dataset["Exited"]

# Encode categorical variables
le_gender = LabelEncoder()
X["Gender"] = le_gender.fit_transform(X["Gender"])  # Male=1, Female=0

# One-Hot Encode Geography
X = pd.get_dummies(X, columns=["Geography"], drop_first=True)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------- ANN Model -----------------
model = Sequential()

# Input + First Hidden Layer
model.add(Dense(units=16, activation="relu", input_dim=X_train.shape[1]))
model.add(Dropout(0.3))

# Second Hidden Layer
model.add(Dense(units=8, activation="relu"))
model.add(Dropout(0.3))

# Output Layer (binary classification)
model.add(Dense(units=1, activation="sigmoid"))

# Compile Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ----------------- Train Model -----------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32, epochs=50, verbose=1
)

# ----------------- Model Evaluation -----------------
# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# ----------------- Plot Training History -----------------
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()
