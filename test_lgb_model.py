import lightgbm as lgb
import numpy as np
import pytest

# Sample test data - increased size to ensure model can learn
X = np.array([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0])

# Create a LightGBM model
model = lgb.LGBMClassifier(min_child_samples=1, min_data_in_leaf=1)

# Fit the model
model.fit(X, y)

# Test if the model predicts correctly

def test_model_prediction():
    predictions = model.predict(X)
    expected_predictions = y  # Expected outputs for the sample data
    assert np.array_equal(predictions, expected_predictions), f"Model predictions {predictions} do not match expected outputs {expected_predictions}!"

if __name__ == '__main__':
    pytest.main()
import lightgbm as lgb
import numpy as np
import pytest
from sklearn.metrics import accuracy_score

# Sample data - use different train/test sets
X_train = np.array([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]])
y_train = np.array([0, 1, 0, 0, 1, 0])

X_test = np.array([[1, 2], [3, 4], [5, 6]])
y_test = np.array([0, 1, 0])

model = lgb.LGBMClassifier(num_leaves=3, min_child_samples=1, min_data_in_leaf=1)
model.fit(X_train, y_train)

def test_model_accuracy():
    predictions = model.predict(X_test)
    # Check accuracy is reasonable (allowing for some error)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy >= 0.66, f"Accuracy {accuracy} is too low"
    
    # Or check specific predictions
    np.testing.assert_array_equal(predictions, y_test)
