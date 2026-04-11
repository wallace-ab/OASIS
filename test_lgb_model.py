import lightgbm as lgb
import numpy as np
import pytest
from sklearn.datasets import make_classification

# Generate a synthetic dataset that is easy to classify
X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)

# Create a LightGBM model
model = lgb.LGBMClassifier(n_estimators=10, random_state=42)

# Fit the model
model.fit(X, y)

def test_model_prediction():
    # Test if the model can achieve a reasonable accuracy on the training set
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.8, f"Model accuracy {accuracy} is too low!"

def test_model_output_shape():
    # Test if the model output shape is correct
    predictions = model.predict(X)
    assert predictions.shape == (100,), f"Expected shape (100,), got {predictions.shape}"

# Run the test
if __name__ == '__main__':
    pytest.main()
