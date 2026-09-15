import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# DO NOT MODIFY - same categorical features as the pipeline
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def _get_sample_data():
    """Helper: load a small slice of the census data for testing."""
    data = pd.read_csv("data/census.csv")
    sample = data.sample(n=500, random_state=42)
    X, y, encoder, lb = process_data(
        sample,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    return X, y


def test_train_model_returns_random_forest():
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    X, y = _get_sample_data()
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_array():
    """
    Test that inference returns a numpy array with one prediction per row.
    """
    X, y = _get_sample_data()
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)


def test_compute_model_metrics_returns_floats():
    """
    Test that compute_model_metrics returns three float values.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
