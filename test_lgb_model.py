import lightgbm as lgb
import numpy as np
import warnings

X = np.array([
    [1, 2], [3, 4], [5, 6],
    [1, 2], [3, 4], [5, 6],
    [1, 2], [3, 4], [5, 6],
])
y = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0])


def test_model_prediction():
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
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        predictions = model.predict(X)

    expected = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0])
    assert np.array_equal(predictions, expected), (
        f"Predictions {predictions} do not match expected {expected}"
    )
