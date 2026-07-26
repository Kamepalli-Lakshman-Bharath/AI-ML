# Supervised and Unsupervised Learning

## 📌 1. Definition of Machine Learning

> **Arthur Samuel (1959)**: *"The field of study that gives computers the ability to learn without being explicitly programmed."*

* **Core Principle**: Algorithms improve performance by observing data rather than following hardcoded rules.
* **Key Axiom**: 

$$\text{More Data / Experience } (E) \implies \text{Higher Accuracy } (P)$$

---

## 🟢 2. Supervised Learning ($X \longrightarrow Y$)

Learns a mapping function $f(X) = Y$ from training pairs of inputs $X$ and correct target labels $Y$.

$$\text{Input } X + \text{Target Label } Y \longrightarrow \text{Learned Model } f(X)$$

### Input-to-Output ($X \to Y$) Industry Mappings

| Input ($X$) | Output ($Y$) | Application |
| :--- | :--- | :--- |
| Email content | Spam (1) / Inbox (0) | Spam Filtering |
| Audio clip | Text transcript | Speech Recognition |
| English text | Spanish text | Machine Translation |
| User & Ad features | Click (1) / No Click (0) | Online Advertising |
| Camera image & Radar | Surrounding vehicle coordinates | Autonomous Driving |
| Product photo | Scratch / Defect (1) or Clean (0) | Visual Inspection |
| House size (sq ft) | Estimated price ($) | Housing Price Prediction |

---

## 📐 3. Types of Supervised Learning

```
                       ┌───────────────────────────┐
                       │    Supervised Learning    │
                       └─────────────┬─────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │       Regression        │             │     Classification      │
    │  Predicts Continuous    │             │   Predicts Discrete     │
    │  Numbers (Infinite)     │             │   Categories (Finite)   │
    └─────────────────────────┘             └─────────────────────────┘
```

### A) Regression
* **Output ($Y$)**: Continuous numerical value from infinitely many possible numbers.
* **Objective**: Fit a line ($y = wx + b$) or curve through data points.
* **Examples**: Predicting house prices, temperature, stock values.

### B) Classification
* **Output ($Y$)**: Discrete category or class label from a small, finite set.
* **Binary Classification**: 2 classes (e.g., $0 = \text{Benign}$, $1 = \text{Malignant}$).
* **Multi-class Classification**: $K$ classes (e.g., Cat, Dog, Bird).
* **Decision Boundary**: Boundary in feature space (e.g., Tumor Size vs. Patient Age) that separates classes.

---

## 🔵 4. Unsupervised Learning ($X \text{ Only}$)

Learns hidden structures, patterns, or groupings from unlabeled data inputs $X$.

$$\text{Unlabeled Data } (X) \longrightarrow \text{Discovered Structure / Clusters}$$

* **Clustering**: Groups similar data points (e.g., Google News article grouping, customer segmentation).
* **Anomaly Detection**: Flags unusual data points (e.g., fraud detection).
* **Dimensionality Reduction**: Compresses feature spaces (e.g., PCA, t-SNE).

---

## 📊 5. Summary Matrix: Regression vs. Classification

| Feature | Regression | Classification |
| :--- | :--- | :--- |
| **Output Type** | Continuous number ($\mathbb{R}$) | Discrete class label |
| **Output Space** | Infinitely many numbers | Small, finite set |
| **Target Example** | Price ($\$250\text{k}$), Temperature ($72^\circ\text{F}$) | Spam (0/1), Medical diagnosis |
| **Goal** | Fit a trend line/curve | Find a decision boundary |
