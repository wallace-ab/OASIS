import lightgbm as lgb
import numpy as np
import pytest

# Sample test data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

# Create a LightGBM model
model = lgb.LGBMClassifier()

# Fit the model
model.fit(X, y)

# Test if the model predicts correctly

def test_model_prediction():
    predictions = model.predict(X)
            expected_predictions = np.array([0, 0, 0])  # Expected outputs for the sample data
    assert np.array_equal(predictions, expected_predictions), "Model predictions do not match expected outputs!"

# Run the test
if __name__ == '__main__':
    pytest.main()