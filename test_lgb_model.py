import lightgbm as lgb
import numpy as np
import pytest
import warnings
from sklearn.metrics import accuracy_score

# --- Data ---
# Increased data to help the model learn the pattern
X_train = np.array([[1, 2], [3, 4], [5, 6],
                    [1, 2], [3, 4], [5, 6],
                    [1, 2], [3, 4], [5, 6],
                    [1, 2], [3, 4], [5, 6]])
y_train = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])

X_test = np.array([[1, 2], [3, 4], [5, 6]])
y_test  = np.array([0, 1, 0])


# --- Fixture: one model, shared across tests ---
@pytest.fixture(scope="module")
def trained_model():
    model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=3,
        min_child_samples=1,
        min_data_in_leaf=1,
        random_state=42,
        verbosity=-1,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    return model


# --- Tests ---
def test_model_prediction(trained_model):
    """Exact prediction check on known toy data."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        predictions = trained_model.predict(X_test)

    np.testing.assert_array_equal(
        predictions, y_test,
        err_msg=f"Got {predictions}, expected {y_test}"
    )


def test_model_accuracy(trained_model):
    """Accuracy gate — useful when exact match is too strict."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        predictions = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy >= 0.66, f"Accuracy {accuracy:.2f} is below threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
