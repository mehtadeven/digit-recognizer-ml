"""
Generate the README result chart for the Digit Recognizer.

Trains both models exactly like train.py, measures their real test
accuracy, and saves a bar chart to assets/result.png.

Run:  python generate_charts.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

NAVY = "#0B3C5D"
GOLD = "#E3A82B"
CREAM = "#F7F1E1"
OUT = Path(__file__).parent / "assets" / "result.png"


def main() -> None:
    # Same split and models as train.py, so the numbers match.
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target,
        test_size=0.2, random_state=42, stratify=digits.target,
    )
    models = {
        "SVM": SVC(gamma=0.001),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    names, scores = [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        names.append(name)
        scores.append(acc * 100)
        print(f"{name}: {acc:.2%}")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    bars = ax.bar(names, scores, color=[NAVY, GOLD], width=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.7,
                f"{score:.2f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Digit Recognizer — model comparison (1,797 images, 80/20 split)",
                 fontsize=12, fontweight="bold", color=NAVY)
    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"Chart saved to {OUT}")


if __name__ == "__main__":
    main()
