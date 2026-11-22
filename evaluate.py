"""
Per-digit evaluation for the Digit Recognizer.

Trains the same SVM as train.py, then prints the accuracy for each
digit (0-9) separately, so you can see which digits are hardest.

Run:  python evaluate.py
"""

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def per_digit_accuracy() -> dict:
    """Train the SVM and return {digit: accuracy} on the test split."""
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target,
        test_size=0.2, random_state=42, stratify=digits.target,
    )
    model = SVC(gamma=0.001).fit(X_train, y_train)
    predicted = model.predict(X_test)

    scores = {}
    for digit in range(10):
        mask = y_test == digit  # only the test images of this digit
        scores[digit] = accuracy_score(y_test[mask], predicted[mask])
    return scores


def main() -> None:
    scores = per_digit_accuracy()
    print("Per-digit accuracy (SVM, held-out test split):\n")
    for digit, acc in scores.items():
        print(f"  Digit {digit}: {acc:.2%}")
    hardest = min(scores, key=scores.get)
    print(f"\nHardest digit: {hardest} ({scores[hardest]:.2%})")


if __name__ == "__main__":
    main()
