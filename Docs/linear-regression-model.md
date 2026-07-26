# Linear Regression Model & Standard Notation

> **Core Concept**: **Linear Regression** is a Supervised Learning algorithm that fits a straight line to historical data to predict continuous numerical outputs ($y$). It is the foundational model for predictive analytics in Machine Learning.

---

## 📌 1. Overview & Problem Definition

* **Model Objective**: Fit a straight line through dataset points to map input features ($x$) to output values ($y$).
* **Problem Type**: **Regression Problem** — predicts a number from an infinite set of continuous real numbers ($\mathbb{R}$).
* **Contrast with Classification**: 
  * **Regression**: Output $y$ is continuous (e.g., $\$220,000$, $\$150.50$, $-33.2$).
  * **Classification**: Output $y$ belongs to a small, discrete set (e.g., Cat vs. Dog, Benign vs. Malignant).

---

## 🏠 2. Case Study: Portland Housing Dataset

* **Dataset**: Historical house sales in Portland, Oregon ($m = 47$ houses).
* **Inputs ($x$)**: Size of house in square feet.
* **Outputs ($y$)**: Sale price in thousands of dollars ($1000\text{'s}$).
* **Goal**: Predict the price for a new, unsold house ($x = 1250\text{ sq ft}$).

![Linear Regression Model](./assets/images/linear_regression_model.svg)

### Inference Step:
1. Locate $x = 1250\text{ sq ft}$ on the horizontal axis.
2. Move vertically up to intersect the fitted linear regression line.
3. Move horizontally left to read the predicted price on the vertical axis ($\hat{y} \approx \$220,000$).

---

## 📖 3. Standard Machine Learning Notation Reference

The dataset used to train a model is called the **Training Set**. Standard mathematical notation used across AI/ML to describe training data includes:

| Symbol / Notation | Term | Definition | Portland Dataset Example |
| :--- | :--- | :--- | :--- |
| **$x$** | Input Variable / Feature | The input attribute provided to the model | House size (e.g., $2104\text{ sq ft}$) |
| **$y$** | Output / Target Variable | The true value we want the model to predict | House price (e.g., $\$400,000$) |
| **$m$** | Total Training Examples | Total number of rows / data points in the dataset | $m = 47$ training examples |
| **$(x, y)$** | Single Training Example | A single ordered pair of input and output | $(2104, 400)$ |
| **$(x^{(i)}, y^{(i)})$** | $i^{\text{th}}$ Training Example | The specific pair at row index $i$ in the data table | Row 1: $(x^{(1)}, y^{(1)}) = (2104, 400)$ |

---

### ⚠️ Critical Notation Rule: Index vs. Exponent

```
+-----------------------------------------------------------------------------------+
| NOTATION DISTINCTION:                                                             |
|                                                                                   |
|   x⁽ⁱ⁾  <-- Parentheses in superscript denotes INDEX (the i-th training example)  |
|   xⁱ    <-- NO parentheses in superscript denotes EXPONENT (x to the power of i)  |
|                                                                                   |
| Example: x⁽²⁾ refers to the 2nd house in the dataset (NOT x squared).             |
+-----------------------------------------------------------------------------------+
```

---

## 🎯 Summary Checklist

- [x] **Linear Regression**: Fits a straight line to continuous training data.
- [x] **Training Set**: Historical data used by the algorithm to learn $x \to y$ mappings.
- [x] **Notation**:
  * $x$ = Feature / Input
  * $y$ = Target / Output
  * $m$ = Number of training examples
  * $(x^{(i)}, y^{(i)})$ = $i^{\text{th}}$ training example (Row $i$)
