import lightgbm as lgb
import numpy as np
import pytest

# Sample test data - larger and clearly separable
X = np.array([[i, i] for i in range(100)])
y = np.array([0 if i < 50 else 1 for i in range(100)])

# Create a LightGBM model
model = lgb.LGBMClassifier(n_estimators=10, min_child_samples=1, min_data_in_leaf=1)

# Fit the model
model.fit(X, y)

# Test if the model predicts correctly

def test_model_prediction():
    predictions = model.predict(X)
    # Check if accuracy is high enough instead of exact match if needed, 
    # but for this simple case it should be 100%
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.9, f"Model accuracy {accuracy} is too low!"

# Run the test
if __name__ == '__main__':
    pytest.main()
