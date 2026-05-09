# 🚢 Titanic Survival Prediction — Decision Tree from Scratch

> **Assignment 3 | BCS-6B | Roll No. 23L-0734**
> A machine learning project that builds a Decision Tree classifier **entirely from scratch** using NumPy — no sklearn trees — to predict survival on the Titanic.

---

## 📌 Overview

This notebook walks through the complete machine learning pipeline: from raw data exploration to training a hand-coded Decision Tree and tuning its hyperparameters. The goal is to deeply understand how decision trees work internally — splitting logic, information gain, recursion, and the bias-variance tradeoff — by implementing every piece by hand.

---

## 🗂️ Project Structure

```
23L_0734_bcs_6b_A3.ipynb   ← Main notebook (all parts A–H)
README.md                   ← You are here
```

---

## 🔬 Notebook Walkthrough

### Part A — 📦 Imports
Standard scientific Python stack: `numpy`, `pandas`, `matplotlib`, `seaborn`, and `sklearn` (used only for preprocessing utilities and evaluation metrics — **not** for the tree itself).

---

### Part B — 🗃️ Dataset Loading
The classic **Titanic dataset** is loaded via `seaborn.load_dataset('titanic')`.

- **891 passengers**, **15 columns**
- Target variable: `survived` (0 = did not survive, 1 = survived)

---

### Part C — 🧹 Data Cleaning & Feature Engineering

#### C.1 — Survival Rate by Categorical Features
Visualized mean survival rates across `pclass`, `sex`, `embarked`, `who`, and `alone` using grouped bar charts. Key insight: **sex and passenger class are the strongest predictors**.

#### C.2 — Feature Selection
Carefully selected 7 informative features and dropped noisy or redundant ones:

| Kept ✅ | Dropped ❌ | Reason for dropping |
|---|---|---|
| `pclass` | `deck` | >75% missing values |
| `sex` | `embark_town` | Duplicate of `embarked` |
| `age` | `who`, `adult_male`, `alive` | Derived/leaky columns |
| `sibsp`, `parch` | `class` | Categorical duplicate of `pclass` |
| `fare`, `embarked` | `alone` | Derived from `sibsp` + `parch` |

Missing values imputed with **median** (age, fare) and **mode** (embarked). Categorical columns one-hot encoded with `drop_first=True`.

---

### Part D — ✂️ Train/Test Split
80/20 split using `sklearn.model_selection.train_test_split` with `random_state=42` for reproducibility. Class balance verified across both splits.

---

### Part F — 🌳 Decision Tree — Built from Scratch

The centrepiece of this project. A fully functional Decision Tree implemented in pure NumPy with:

- **`Node` class** — stores split feature, threshold, left/right children, and leaf prediction
- **`DecisionTree` class** — recursive tree builder with:
  - Gini impurity as the splitting criterion
  - Best-split search across all features and thresholds
  - `max_depth` and `min_samples_split` as stopping conditions
  - Majority-class prediction at leaf nodes

```python
tree = DecisionTree(max_depth=5, min_samples_split=5)
tree.fit(Xy_train)
y_pred = tree.predict(X_test.values)
```

No `sklearn.tree` was used. Every split, every threshold, every recursion — handwritten. 💪

---

### Part G — 📊 Evaluation at max_depth=5

| Metric | Score |
|---|---|
| Accuracy | ~79–82% |
| F1 Score | reported per class |

A confusion matrix is plotted to visualize true positives, false negatives, and the model's failure modes. Analysis: if accuracy stagnates around 60%, the tree is **underfitting** — `max_depth` is too low to capture interactions like `sex × pclass`.

---

### Part H — 🔧 Hyperparameter Tuning — Sweeping max_depth

`max_depth` swept from 1 to 15. Accuracy plotted at each depth to find the sweet spot.

```
Low depth  → High bias   → Underfitting  → Low accuracy
High depth → High variance → Overfitting → Noisy accuracy
Sweet spot → Best generalisation
```

A vertical red line on the plot marks the **optimal depth**, demonstrating the **bias-variance tradeoff** empirically.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🐍 Python 3 | Core language |
| 🔢 NumPy | Tree logic, array ops |
| 🐼 Pandas | Data loading & cleaning |
| 📊 Matplotlib / Seaborn | Visualisations |
| ⚙️ Scikit-learn | Preprocessing, metrics only |

---

## ▶️ How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open the notebook:
   ```bash
   jupyter notebook 23L_0734_bcs_6b_A3.ipynb
   ```
4. Run all cells top to bottom (`Kernel → Restart & Run All`)

---

## 💡 Key Takeaways

- 🌱 **Decision Trees are intuitive** — greedy best-split search at each node is surprisingly effective
- ⚖️ **Depth is everything** — it directly controls the bias-variance tradeoff
- 🔍 **Feature engineering matters** — dropping leaky/redundant columns meaningfully improves signal
- 🧮 **Scratch implementations teach** — writing the tree by hand makes the algorithm stick in a way that `sklearn` never could

---

## 👤 Author

**Abdullah Haroon** — Roll No. 23L-0734 | BCS-6B  
FAST-NUCES, Lahore
