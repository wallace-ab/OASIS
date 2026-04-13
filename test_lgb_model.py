import lightgbm as lgb
import numpy as np
import pytest
from sklearn.datasets import make_classification

# Generate a synthetic dataset that is sufficiently large for LightGBM
# Using 100 samples ensures the model can learn and splits are meaningful.
X, y = make_classification(
    n_samples=100, 
    n_features=4, 
    n_informative=2, 
    n_redundant=0, 
    random_state=42
)

# Create a LightGBM model with parameters optimized for small datasets
# min_child_samples=1 and min_data_in_leaf=1 allow splits on small data.
model = lgb.LGBMClassifier(
    min_child_samples=1,
    min_data_in_leaf=1,
    n_estimators=10,
    random_state=42,
    verbose=-1
)

# Fit the model
model.fit(X, y)

def test_model_prediction():
    """Test if the model achieves reasonable accuracy on the training data."""
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    
    # Assert that the model has learned something (accuracy > 80%)
    assert accuracy > 0.8, f"Model accuracy is too low: {accuracy:.2f}"

if __name__ == '__main__':
    pytest.main()
