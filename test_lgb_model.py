import lightgbm as lgb
import numpy as np
import warnings


# Sample test data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])


def test_model_prediction():
    # Configure a deterministic tiny model that can fit this toy dataset.
    model = lgb.LGBMClassifier(
        n_estimators=50,
        min_data_in_leaf=1,
        min_data_in_bin=1,
        num_leaves=7,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X, y)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        predictions = model.predict(X)
    expected_predictions = np.array([0, 1, 0])
    assert np.array_equal(predictions, expected_predictions), "Model predictions do not match expected outputs!"
