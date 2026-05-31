from data_loader import load_data
import joblib
import numpy as np

# Load dataset
df = load_data()

# Features only
X = df.drop("MEDV", axis=1)

# Convert to NumPy array
sample_data = np.array([
    [
        0.02731,  # CRIM
        0.0,      # ZN
        7.07,     # INDUS
        0,        # CHAS
        0.469,    # NOX
        6.421,    # RM
        78.9,     # AGE
        4.9671,   # DIS
        2,        # RAD
        242,      # TAX
        17.8,     # PTRATIO
        396.90,   # B
        9.14      # LSTAT
    ]
])

# Load saved model
model = joblib.load(
    "saved_models/Decision Tree Regressor.pkl"
)

# Predict
prediction = model.predict(sample_data)

print("Decision Tree Regressor Prediction:")
print(prediction)