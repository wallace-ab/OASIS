import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

# Sample test data - larger dataset to ensure LightGBM can learn
# We use a dataset where class 1 is clearly separated from class 0
X_raw = np.array([
    [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
    [10, 20], [10, 20], [10, 20], [10, 20], [10, 20],
    [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Convert to DataFrame to ensure feature names are consistent
feature_names = ['feature1', 'feature2']
X = pd.DataFrame(X_raw, columns=feature_names)

# Create a LightGBM model
model = lgb.LGBMClassifier(
    random_state=42,
    min_child_samples=1,
    n_estimators=10,
    min_child_weight=0,
    verbose=-1
)

# Fit the model
model.fit(X, y)

# Test if the model predicts correctly

def test_model_prediction():
    # Test on the same data
    predictions = model.predict(X)
    
    # Check specific indices that represent the classes
    # Index 0-4: Class 0
    # Index 5-9: Class 1
    # Index 10-14: Class 0
    
    assert predictions[2] == 0, f"Sample 2 should be class 0, but got {predictions[2]}"
    assert predictions[7] == 1, f"Sample 7 should be class 1, but got {predictions[7]}"
    assert predictions[12] == 0, f"Sample 12 should be class 0, but got {predictions[12]}"

# Run the test
if __name__ == '__main__':
    pytest.main(["-s"])
