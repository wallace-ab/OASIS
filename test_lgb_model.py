import pandas as pd
import lightgbm as lgb
from sklearn.datasets import make_classification

def get_model_and_data():
    X, y = make_classification(n_samples=100, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    X_df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    model = lgb.LGBMClassifier(n_estimators=20, random_state=42, verbosity=-1, min_child_samples=1)
    model.fit(X_df, y)
    return model, X_df

model, _ = get_model_and_data()

def test_model_runs():
    X = pd.DataFrame(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        columns=["feature1", "feature2", "feature3"],
    )

    preds = model.predict(X)

    assert len(preds) == 3
    assert set(preds).issubset({0, 1})
