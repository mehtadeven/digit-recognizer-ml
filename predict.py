"""
Predict a single digit from an 8x8 pixel grid.
Edit the grid below (values 0-16, like the training data) and run:
    python predict.py

If model.joblib exists (created by train.py), it is loaded directly.
Otherwise a fresh SVM is trained on the spot, like before.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

MODEL_PATH = Path(__file__).with_name("model.joblib")

# Draw a "7" on an 8x8 grid (0 = blank, 16 = full ink)
GRID = np.array([
    0,  0, 12, 14, 15, 13,  0, 0,
    0,  0,  0,  0,  0, 15,  0, 0,
    0,  0,  0,  0, 13,  0,  0, 0,
    0,  0,  0, 11,  0,  0,  0, 0,
    0,  0,  0, 12,  0,  0,  0, 0,
    0,  0,  0, 10,  0,  0,  0, 0,
    0,  0,  0, 10,  0,  0,  0, 0,
    0,  0,  0,  9,  0,  0,  0, 0,
], dtype=float)


def get_model():
    """Load the saved model if there is one, otherwise train a new SVM."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    digits = load_digits()
    X_train, _, y_train, _ = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )
    return SVC(gamma=0.001).fit(X_train, y_train)


def predict_digit(grid) -> int:
    """Return the digit (0-9) the model sees in one 8x8 grid."""
    return int(get_model().predict([grid])[0])


def main() -> None:
    print(f"The model reads this digit as: {predict_digit(GRID)}")


if __name__ == "__main__":
    main()
