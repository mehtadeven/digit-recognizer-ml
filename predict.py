"""
Predict a single digit from an 8x8 pixel grid.
Edit the grid below (values 0-16, like the training data) and run:
    python predict.py
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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


def main() -> None:
    digits = load_digits()
    X_train, _, y_train, _ = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )
    model = SVC(gamma=0.001).fit(X_train, y_train)
    print(f"The model reads this digit as: {model.predict([GRID])[0]}")


if __name__ == "__main__":
    main()
