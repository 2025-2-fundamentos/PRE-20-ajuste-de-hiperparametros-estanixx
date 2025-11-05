"""Train a regressor on the red wine quality dataset and save the best estimator.

This script writes `estimator.pickle` to the current working directory.
"""

from pprint import pprint
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score
from scipy.stats import randint


def load_data():
    url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
        "winequality-red.csv"
    )
    df = pd.read_csv(url, sep=";")
    y = df["quality"]
    X = df.drop(columns=["quality"])
    return X, y


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Random forest with a small randomized search. The dataset is small so this is quick.
    estimator = RandomForestRegressor(random_state=0)

    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        # sklearn accepts 'sqrt' and 'log2' as options for max_features
        "max_features": ["sqrt", "log2", 0.8],
    }

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="r2",
        cv=5,
        random_state=0,
        n_jobs=-1,
        verbose=1,
    )

    print("Fitting RandomizedSearchCV on training data...")
    search.fit(X_train, y_train)

    print("Best params:")
    pprint(search.best_params_)

    best = search.best_estimator_

    y_pred = best.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Test R2: {r2:.4f}")

    out_path = os.path.join(os.getcwd(), "estimator.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(best, f)

    print(f"Saved estimator to {out_path}")


if __name__ == "__main__":
    main()
