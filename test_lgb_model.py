import lightgbm as lgb
import numpy as np
import pytest
import warnings

# Sample test data - increased size to ensure model can learn
X = np.array([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0])

# Create a LightGBM model with parameters suitable for small datasets
model = lgb.LGBMClassifier(
    n_estimators=10,
    min_child_samples=1,
    min_data_in_leaf=1,
    verbosity=-1
)

# Fit the model
model.fit(X, y)

# Test if the model predicts correctly
def test_model_prediction():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        predictions = model.predict(X)
    expected_predictions = y  # Expected outputs for the sample data
    assert np.array_equal(predictions, expected_predictions), f"Model predictions {predictions} do not match expected outputs {expected_predictions}!"

# Run the test
if __name__ == '__main__':
    pytest.main()
