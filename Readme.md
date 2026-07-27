# 🤖 Machine Learning & Artificial Intelligence Study Notes

Welcome to the **AI/ML Learning Hub**! This documentation collection contains structured, easy-to-understand notes, diagrams, real-world analogies, and practical code examples to master Machine Learning.

---

## 📚 Table of Contents

### 1. Fundamentals & Core Concepts
* 📄 **[01. Introduction to Machine Learning](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/01-introduction-to-ai-ml.md)**
  * What is Machine Learning?
  * Traditional Programming vs. Machine Learning
  * Everyday & Industrial Applications
  * AI vs. ML vs. AGI
  * Economic Impact & Value Creation

* 📄 **[02. Supervised Learning](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/02-supervised-learning.md)**
  * $X \longrightarrow Y$ Input-to-Output Mappings
  * 7 Industry $X \to Y$ Applications Table
  * **Regression**: Continuous numerical output prediction
  * **Classification**: Discrete category prediction (Breast Cancer case study, Decision Boundaries)

* 📄 **[03. Unsupervised Learning](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/03-unsupervised-learning.md)**
  * Unlabeled Data ($X \text{ only}$, no $y$ labels)
  * **Clustering**: Definition & 3 Case Studies (Google News, Genetics, Customer Segmentation)
  * **Anomaly Detection** & **Dimensionality Reduction**
  * Supervised vs Unsupervised Quiz Matrix

* 📄 **[04. Linear Regression Model & Notation](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/04-linear-regression-model.md)**
  * Straight Line Model Fitting
  * Portland Housing Dataset Case Study ($x = 1250 \to \hat{y} \approx \$220k$)
  * Standard ML Notation Reference ($x, y, m, (x^{(i)}, y^{(i)})$)
  * Index vs Exponent Superscript Notation Rule

* 📄 **[05. Cost Function & Intuition](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/05-cost-function.md)**
  * Squared Error Cost Function Formula ($J(w,b) = \frac{1}{2m}\sum (f(x^{(i)}) - y^{(i)})^2$)
  * Why $\frac{1}{2m}$? (Averaging & derivative calculus simplification)
  * Visualizing 1D Parameter $w$: Tracing out the U-shaped Parabola
  * 3D Surface Plots ("Soup Bowl" / Hammock shape)
  * 2D Contour Plots (Level curves & Concentric ellipses)
  * Mapping Contour Points $(w,b)$ to Model Fits
  * Transition to Automated Optimization via Gradient Descent

* 📄 **[06. Gradient Descent Algorithm](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/06-gradient-descent.md)**
  * Automated optimization of parameters $w$ and $b$ to minimize cost $J(w,b)$.
  * Derivative slope intuition, learning rate dynamics, and automatic step-size shrinking.

* 📄 **[07. Multiple Linear Regression & Vectorization](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/07-multiple-linear-regression.md)**
  * Multiple features notation ($n, x_j^{(i)}, \vec{x}^{(i)}$) & parameter interpretation
  * Hypothesis in vector dot product form $f(\vec{x}) = \vec{w} \cdot \vec{x} + b$
  * Vectorization with NumPy (`np.dot(w, x) + b`) and SIMD hardware parallelization

* 📄 **[08. Feature Scaling, Learning Rate Tuning & Feature Engineering](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/08-feature-scaling-and-polynomial-regression.md)**
  * Vectorized Gradient Descent updates & Normal Equation $(X^T X)^{-1} X^T \vec{y}$
  * Feature Scaling techniques (Divide by Max, Mean Normalization, Z-score Standardization)
  * Learning Curve diagnostics ($J$ vs iterations) & systematic $\alpha$ search
  * Feature Engineering & Polynomial Regression ($x^2, x^3, \sqrt{x}$) with mandatory scaling

* 📄 **[09. Logistic Regression Model, Decision Boundaries & Loss Function](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/09-logistic-regression-model.md)**
  * Binary classification ($y \in \{0, 1\}$) & why linear regression fails
  * Sigmoid activation $g(z) = \frac{1}{1 + e^{-z}}$ & probability output interpretation
  * Decision boundaries ($z=0 \implies \vec{w} \cdot \vec{x} + b = 0$) & linear vs. polynomial shapes
  * Non-convexity of MSE for classification & Binary Cross-Entropy Loss derivation

* 📄 **[10. Overfitting, Underfitting & Regularization](file:///Users/lakshmanbharathkomepalli/Lakshman/my_space/AI-ML/Docs/10-overfitting-and-regularization.md)**
  * Generalization definition, Goldilocks analogy & High Bias vs. High Variance
  * 3 Remedies for Overfitting (More data, Feature selection, Regularization)
  * Regularized Cost Function formula & $\lambda$ parameter trade-offs
  * Weight Decay factor $(1 - \alpha \frac{\lambda}{m})$ in gradient descent updates
  * Decision boundary smoothing & Course 1 completion milestone

---

## 💡 How to Use These Notes
1. **Read Conceptually**: Follow the structured breakdown and real-world analogies.
2. **Review Diagrams**: Check the flow diagrams to visualize how data transforms into models.
3. **Practice Implementation**: Work through the practical code snippets provided in subsequent sections.
