# Supervised Learning

> **Core Concept**: Supervised Learning algorithms learn input-to-output ($X \longrightarrow Y$) mappings. The algorithm is provided with training data containing both the input $X$ and the correct target label $Y$ ("the right answer"). Once trained, it predicts output $Y$ for new, unseen input $X$.

---

## 📌 1. Fundamentals of Supervised Learning

* **Economic Value**: ~99% of the economic value created by AI/ML today is driven by Supervised Learning.
* **Mechanism**:

$$\text{Training: } \text{Input } X + \text{Label } Y \longrightarrow \text{Model } f(X)$$

$$\text{Inference: } \text{New Input } X_{\text{new}} \xrightarrow{\quad f(X_{\text{new}}) \quad} \text{Predicted Label } \hat{Y}$$

---

## 🌐 2. Industry Input-to-Output ($X \to Y$) Applications

| Input ($X$) | Output Label ($Y$) | Application Domain | Economic/Business Impact |
| :--- | :--- | :--- | :--- |
| Email content & metadata | Spam (1) / Inbox (0) | Email Spam Filter | Security & user productivity |
| Audio clip | Text transcript | Speech Recognition | Voice assistants, captioning |
| Source language text (e.g. English) | Target language text (e.g. Spanish) | Machine Translation | Cross-border communication |
| User profile & Ad features | Click (1) / No Click (0) | Online Advertising | Ad platform revenue engine |
| Camera feeds + Radar data | Surrounding vehicle coordinates | Self-Driving Cars | Autonomous navigation |
| Manufactured product photo | Scratch/Defect (1) or Clean (0) | Visual Inspection (Manufacturing) | Quality assurance & defect prevention |
| House size (sq ft) | Price ($) | Housing Price Prediction | Real estate valuation |

---

## 📈 3. Sub-type 1: Regression (Continuous Prediction)

**Regression** algorithms predict a continuous numerical value from **infinitely many possible numbers**.

$$\text{Output Space } Y \in \mathbb{R} \quad (\text{e.g., } \$150,000, \$183,420, \$200,000)$$

### 🏠 Case Study: Housing Price Prediction

* **Input ($X$)**: House size (sq ft)
* **Target ($Y$)**: House price ($k)

```
  Price ($k)
    ^
200 |                       *  <-- (Fitted Curve Prediction: ~$200k)
150 |             *       /
    |     *            --'    <-- (Fitted Line Prediction: ~$150k)
100 |  *     _..---'
 50 |     .-'
  0 +-----------------------------------> Size (sq ft)
    0     500    750    1000   1250
```

* **Linear Regression**: Fits a straight line ($y = wx + b$). For $750\text{ sq ft}$, predicts $\sim\$150,000$.
* **Non-Linear Regression**: Fits a curve. For $750\text{ sq ft}$, predicts $\sim\$200,000$.

---

## 🏷️ 4. Sub-type 2: Classification (Discrete Prediction)

**Classification** algorithms predict a discrete category or class label from a small, finite set of possibilities.

$$\text{Output Space } Y \in \{0, 1, \dots, K-1\} \quad (\text{Discrete Classes / Categories})$$

> 💡 *Note*: The terms **Output Classes** and **Output Categories** are used interchangeably.

### Classification Types

1. **Binary Classification**: Only 2 possible classes.
   * Example: $0 = \text{Benign (Non-cancerous)}$, $1 = \text{Malignant (Cancerous)}$.
   * Example: $0 = \text{Not Spam}$, $1 = \text{Spam}$.
2. **Multi-class Classification**: 3 or more possible classes.
   * Example: $0 = \text{Benign}$, $1 = \text{Type 1 Cancer}$, $2 = \text{Type 2 Cancer}$.
   * Example: Predict image labels $\in \{\text{Cat}, \text{Dog}, \text{Bird}\}$.

---

### 🩺 Case Study: Breast Cancer Diagnosis

#### A) Single Input Feature ($X = \text{Tumor Size}$)

Data points plotted along a 1D line using symbols: $\mathbf{O}$ for Benign ($0$) and $\mathbf{X}$ for Malignant ($1$).

```
Diagnosis (Y)
    1 (Malignant) |                  X   X   X   X
                  |
    0 (Benign)    |    O   O   O
                  +-----------------------------------> Tumor Size (X)
```

#### B) Multiple Input Features ($X_1 = \text{Tumor Size}, X_2 = \text{Age}$)

When multiple features are provided, the classification algorithm fits a **Decision Boundary** line to separate the categories in multi-dimensional feature space.

```
   Age (X2)
     ^
     |      O      O    |    X      X
     |    O      O      |  X      X    (Malignant X)
     |       O          |     X
     |    O      O      |  X      X
     |                  | (Decision Boundary Line)
     |  (Benign O)      |
     +-----------------------------------------> Tumor Size (X1)
```

> 🧬 **Multi-Feature Vectors**: Industrial medical AI systems use high-dimensional feature vectors: 
> $$X = [\text{Tumor Size}, \text{Patient Age}, \text{Clump Thickness}, \text{Cell Size Uniformity}, \text{Cell Shape Uniformity}]$$

---

## 📊 5. Summary Matrix: Regression vs. Classification

| Feature | Regression | Classification |
| :--- | :--- | :--- |
| **Output Type** | Continuous numerical values ($\mathbb{R}$) | Discrete classes / categories |
| **Output Space** | Infinitely many possible numbers | Small, finite set ($\{0,1\}$ or $\{0,1,2\}$) |
| **Target Examples** | House price, temperature, stock price | Spam/Not Spam, Benign/Malignant, Cat/Dog |
| **Model Goal** | Fit a trend line or curve to data | Fit a decision boundary separating classes |

---

- [x] **Supervised Learning**: $X \to Y$ input-to-output mappings.
- [x] **Regression**: Predicts continuous numbers.
- [x] **Classification**: Predicts discrete categories using decision boundaries across single or multiple features.
