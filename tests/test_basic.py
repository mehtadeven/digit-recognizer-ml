"""Basic tests for the digit recognizer.

Run from the project root:  python -m pytest tests/
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import predict


def test_dataset_shape():
    digits = load_digits()
    assert digits.data.shape == (1797, 64)  # 1,797 images, 8x8 = 64 pixels
    assert digits.target.shape == (1797,)


def test_svm_scores_above_95_percent():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )
    model = SVC(gamma=0.001).fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.95


def test_predict_grid_returns_a_digit():
    result = predict.predict_digit(predict.GRID)
    assert isinstance(result, int)
    assert 0 <= result <= 9


def test_per_digit_accuracy_covers_all_digits():
    import evaluate

    scores = evaluate.per_digit_accuracy()
    assert sorted(scores) == list(range(10))  # one score per digit 0-9
    for acc in scores.values():
        assert 0.8 <= acc <= 1.0
