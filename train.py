"""
Handwritten Digit Recognizer
-----------------------------
Trains two classifiers on scikit-learn's built-in digits dataset
(1,797 images of handwritten digits, each 8x8 pixels) and compares them.

Run:  python train.py
Outputs: accuracy comparison + saves results/confusion_matrix.png
         and the best model to model.joblib (used by predict.py)
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main() -> None:
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Dataset: {X.shape[0]} images, each {int(X.shape[1] ** 0.5)}x{int(X.shape[1] ** 0.5)} pixels")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Support Vector Machine": SVC(gamma=0.001),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    best_name, best_acc, best_model = "", 0.0, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name:<26} accuracy: {acc:.2%}")
        if acc > best_acc:
            best_name, best_acc, best_model = name, acc, model

    print(f"\nBest model: {best_name} ({best_acc:.2%})")

    # Save the best model so predict.py can load it instead of retraining
    model_path = Path(__file__).with_name("model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path.name}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    ConfusionMatrixDisplay(cm).plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=120)
    print(f"Confusion matrix saved to {out / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
