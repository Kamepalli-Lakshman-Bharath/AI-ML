# 10. Overfitting, Underfitting & Regularization

> **Core Concept**: **Overfitting (High Variance)** occurs when a model fits training data almost *too* well (training error $\approx 0$) by creating erratic/wiggly fits that fail to **generalize** to unseen test data. **Regularization** shrinks parameter weights $w_j$ toward zero using a penalty parameter $\lambda$, smoothing out regression curves and classification decision boundaries.

---

## 📌 1. The Core Spectrum: Underfitting vs. Overfitting

| Model State | Technical Term | Description & Characteristics | Real-World Intuition |
| :--- | :--- | :--- | :--- |
| **Underfitting** | **High Bias** | Model is too simple (e.g. straight line on curved data). Fails on both training and test data. Strong prior preconception. | **"Too Cold"** |
| **Just Right** | **Balanced** | Optimal model complexity (e.g. quadratic curve or elliptical decision boundary). Low training and test error. | **"Just Right"** |
| **Overfitting** | **High Variance** | Model is overly complex ($4^{\text{th}}$-order polynomial). Passes through all training points ($J=0$) but wiggles erratically; fails on new data. | **"Too Hot"** |

![Comparison of Regression Model Fitting](./assets/images/overfitting_underfitting_regression.png)
![Comparison of Classification Decision Boundaries](./assets/images/overfitting_underfitting_classification.png)

> ⚠️ **Technical Bias vs Social Bias**:
> - **Technical Model Bias**: Underfitting caused by model simplicity.
> - **Social/Ethical Bias**: Algorithmic discrimination based on sensitive attributes (race, gender). Auditing models for social bias is essential for responsible AI deployment.

---

## 🛠️ 2. Three Strategies to Address Overfitting

```
                  +----------------------------------------------+
                  |    Strategies to Address Overfitting         |
                  +----------------------------------------------+
                                         |
     +-----------------------------------+-----------------------------------+
     |                                   |                                   |
     v                                   v                                   v
1. Collect More Data           2. Feature Selection           3. Regularization
   (More m examples ->            (Use a smaller subset          (Shrink parameters w_j
    smoother fit curve)            of features)                   without setting to 0)
```

1. **Collect More Training Data 📊**: #1 primary defense. High-degree models perform well if provided with sufficient training data $m$.
2. **Feature Selection ✂️**: Drop unhelpful or redundant features ($n$) manually or algorithmically. Disadvantage: discards potentially useful information.
3. **Regularization ⚖️**: Keep all features, but gently penalize large weight parameters $w_j$ to prevent any single feature from exerting an erratic, dominant influence.

---

## 📐 3. Regularized Cost Function & $\lambda$ Dynamics

### Mathematical Formula:

$$J(\vec{w},b) = \text{Original Cost} + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

- **Linear Regression**: $J(\vec{w},b) = \frac{1}{2m} \sum_{i=1}^{m} (f(\vec{x}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$
- **Logistic Regression**: $J(\vec{w},b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log f + (1-y^{(i)}) \log(1-f)] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$

![Regularization Lambda Dynamics](./assets/images/regularization_lambda_dynamics.png)

| Regularization Parameter $\lambda$ | Model State | Behavior |
| :--- | :--- | :--- |
| **$\lambda = 0$** | **Overfitting (High Variance)** | Unregularized. Wiggly curves fit noise. |
| **$\lambda \to \infty$ (e.g. $10^{10}$)** | **Underfitting (High Bias)** | Heavy penalty forces all $w_j \approx 0 \implies f(\vec{x}) \approx b$ (flat horizontal line). |
| **$\lambda$ Optimal** | **Balanced / Just Right** | Smooth, balanced curve achieving high generalization. |

> 💡 **Bias $b$ Convention**: By standard convention, we regularize weight parameters $w_1 \dots w_n$ and omit $b$.

---

## ⚙️ 4. Regularized Gradient Descent & Weight Decay

### Simultaneous Update Rules ($j = 1 \dots n$):

$$w_j = w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j \right]$$

$$b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)$$

### Deep Intuition: The Weight Decay Factor
Rearranging the weight update formula reveals an explicit shrinkage factor:

$$w_j = w_j \left( 1 - \alpha \frac{\lambda}{m} \right) - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

![Weight Decay Intuition](./assets/images/weight_decay_intuition.png)

On **every single iteration**, $w_j$ is multiplied by a factor slightly less than 1 (e.g., $0.9998$), gently decaying its magnitude before applying the standard gradient update step!

---

## 🖼️ 5. Decision Boundary Smoothing in Regularized Classification

![Regularized Decision Boundary](./assets/images/regularized_decision_boundary.png)

In regularized logistic regression, weight shrinkage prevents complex polynomial features from forming contorted decision boundaries around noisy outliers, smoothing them into elegant ellipses with high test accuracy.

---

## 🎓 Course 1 Graduation Milestone: Bridge to Deep Learning

Mastering **Linear Regression, Logistic Regression, Feature Scaling, Gradient Descent, and Regularization** completes the foundational Supervised Learning course!

- **Silicon Valley & Real-World Value**: Powers production decision engines, fraud detectors, and recommendation systems across top tech companies.
- **Bridge to Neural Networks**: Deep learning algorithms build directly on these foundations—stacking logistic/sigmoid neurons into layers, optimizing cross-entropy loss with backpropagation, and applying L2 weight decay regularization.

---

## 🎯 Summary Checklist

- [x] **Generalization**: Model accuracy on brand-new, unseen data.
- [x] **High Bias vs High Variance**: Underfitting (overly simple) vs Overfitting (overly complex).
- [x] **3 Remedies**: More data, Feature selection, Regularization.
- [x] **Regularized Cost**: Adds $\frac{\lambda}{2m} \sum w_j^2$ weight penalty.
- [x] **Weight Decay**: $(1 - \alpha \frac{\lambda}{m})$ factor shrinks weights toward zero on every iteration.
- [x] **Course 1 Completed**: Prepared for Neural Networks & Deep Learning!
