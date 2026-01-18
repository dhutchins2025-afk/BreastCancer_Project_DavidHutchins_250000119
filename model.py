# Breast Cancer Prediction Using Feed-Forward Neural Network (ANN)

# This notebook demonstrates a feed-forward neural network to predict whether a tumor is benign or malignant using the `sklearn.datasets.load_breast_cancer`.

# We will cover:
#   - Dataset inspection
#   - Data preprocessing
#   - ANN building
#   - Model evaluation (accuracy, precision, recall, F1-score, ROC-AUC)
#   - Visualization: confusion matrix, ROC curve, Precision-Recall curve
#   - Model saving & loading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Load dataset
data = load_breast_cancer(as_frame=True)
data = data.frame  # Already a DataFrame with features + target

# Show first 5 rows
print(data.head())
print(data.info())

# Features
X = data.drop("target", axis=1)

# Target (already numeric: 0 = malignant, 1 = benign)
y = data["target"]

# Inspect shape
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
# Create ANN model
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation="tanh"),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")  # Output layer for binary classification
])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Model summary
model.summary()

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=1
)

# Predict probabilities and class labels
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC-ROC:", auc_roc)


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

model.save("model/model_cancer_predictor.h5")

# Load the saved model
model = load_model("model/model_cancer_predictor.h5")

# Example new data (replace with actual values)
new_data = np.array([X_test.iloc[0]])  # Using first test sample as example
new_data_scaled = scaler.transform(new_data)

# Predict
y_pred_prob = model.predict(new_data_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Predicted Tumor Type (0 = Malignant, 1 = Benign):", y_pred[0][0])
