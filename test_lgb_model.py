import lightgbm as lgb
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_model_prediction():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.5, f'Accuracy is {accuracy}, expected greater than 0.5'

if __name__ == '__main__':
    pytest.main()
