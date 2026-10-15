from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import FEATURE_COLUMNS


def train_model(train_df: pd.DataFrame) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=42,
    )
    clf.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    return clf


def predict_proba_up(model: HistGradientBoostingClassifier, df: pd.DataFrame) -> pd.Series:
    proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    return pd.Series(proba, index=df.index)


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
