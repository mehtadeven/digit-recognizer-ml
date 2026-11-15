# ✍️ Handwritten Digit Recognizer

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/status-working-brightgreen)

A machine learning project that recognises **handwritten digits (0–9)** from tiny 8×8 pixel images — the classic first step into computer vision, built with scikit-learn.

## 📊 Results

Trained and tested on scikit-learn's built-in digits dataset (1,797 images, 80/20 split):

| Model | Accuracy |
|---|---|
| **Support Vector Machine** | **98.89%** |
| Random Forest | 96.39% |

Running `train.py` also saves a confusion matrix of the best model to `results/confusion_matrix.png`, so you can see exactly which digits get confused (usually 8 ↔ 1 and 9 ↔ 7).

## 🚀 Run it

```bash
pip install -r requirements.txt
python train.py     # trains both models and compares them
python predict.py   # reads a hand-drawn 8x8 digit grid and predicts it
```

`predict.py` contains an 8×8 grid drawn as numbers (0 = blank, 16 = full ink). Edit the grid to draw your own digit and see what the model reads.

## 🧠 What I learned

- How image data becomes numbers a model can learn from (each image = 64 features)
- Why SVMs work so well on small, clean image datasets
- How to read a confusion matrix instead of trusting one accuracy number
- Train/test splits with stratification, so every digit is fairly represented

## 🔮 Next steps

- [ ] Try a neural network (MLP) and compare
- [ ] Move to the full MNIST dataset (28×28 images)
- [ ] Add a small web demo where you draw with the mouse

---
Built by [Mehta Deven](https://github.com/mehtadeven) — IT student learning ML/AI in public.
