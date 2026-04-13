import lightgbm as lgb
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_model_prediction():
    # Generate a synthetic dataset that is easy to learn
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use a LightGBM model with deterministic settings
    model = lgb.LGBMClassifier(
        n_estimators=20,
        random_state=42,
        verbosity=-1,
        min_child_samples=1
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Assert that accuracy is high enough (should be > 0.8 for this simple task)
    assert accuracy > 0.7, f"Model accuracy {accuracy} is below the threshold of 0.7!"

if __name__ == '__main__':
    pytest.main()
