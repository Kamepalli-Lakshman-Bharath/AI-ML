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

<svg viewBox="0 0 620 330" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background:#1e1e2e; border-radius:8px; padding:12px;">
  <!-- Grid Lines -->
  <line x1="80" y1="240" x2="570" y2="240" stroke="#45475a" stroke-dasharray="4"/>
  <line x1="80" y1="180" x2="570" y2="180" stroke="#45475a" stroke-dasharray="4"/>
  <line x1="80" y1="120" x2="570" y2="120" stroke="#45475a" stroke-dasharray="4"/>
  <line x1="80" y1="60"  x2="570" y2="60"  stroke="#45475a" stroke-dasharray="4"/>

  <!-- Axes -->
  <line x1="80" y1="260" x2="570" y2="260" stroke="#cdd6f4" stroke-width="2"/>
  <line x1="80" y1="260" x2="80"  y2="40"  stroke="#cdd6f4" stroke-width="2"/>

  <!-- Labels -->
  <text x="325" y="300" fill="#cdd6f4" font-family="sans-serif" font-size="14" text-anchor="middle" font-weight="bold">House Size (sq ft)</text>
  <text x="25" y="150" fill="#cdd6f4" font-family="sans-serif" font-size="14" transform="rotate(-90 25,150)" text-anchor="middle" font-weight="bold">Price ($ in thousands)</text>

  <!-- Axis Ticks & Numbers -->
  <text x="80"  y="280" fill="#a6adc8" font-size="12" text-anchor="middle">0</text>
  <text x="195" y="280" fill="#a6adc8" font-size="12" text-anchor="middle">500</text>
  <text x="310" y="280" fill="#a6adc8" font-size="12" text-anchor="middle" font-weight="bold">750</text>
  <text x="425" y="280" fill="#a6adc8" font-size="12" text-anchor="middle">1000</text>
  <text x="540" y="280" fill="#a6adc8" font-size="12" text-anchor="middle">1250</text>

  <text x="65" y="244" fill="#a6adc8" font-size="12" text-anchor="end">$50k</text>
  <text x="65" y="184" fill="#a6adc8" font-size="12" text-anchor="end">$100k</text>
  <text x="65" y="124" fill="#a6adc8" font-size="12" text-anchor="end">$150k</text>
  <text x="65" y="64"  fill="#a6adc8" font-size="12" text-anchor="end">$200k</text>

  <!-- Data Points -->
  <circle cx="150" cy="225" r="5" fill="#f38ba8"/>
  <circle cx="220" cy="195" r="5" fill="#f38ba8"/>
  <circle cx="280" cy="155" r="5" fill="#f38ba8"/>
  <circle cx="360" cy="115" r="5" fill="#f38ba8"/>
  <circle cx="440" cy="85"  r="5" fill="#f38ba8"/>
  <circle cx="510" cy="75"  r="5" fill="#f38ba8"/>

  <!-- Linear Fit Line (Blue) -->
  <line x1="80" y1="245" x2="540" y2="85" stroke="#89b4fa" stroke-width="2.5" stroke-dasharray="6,4"/>

  <!-- Curve Fit Path (Green) -->
  <path d="M 80 250 Q 280 200 540 65" fill="none" stroke="#a6e3a1" stroke-width="3"/>

  <!-- Vertical Reference Line at 750 sq ft -->
  <line x1="310" y1="260" x2="310" y2="50" stroke="#fab387" stroke-width="1.5" stroke-dasharray="4,4"/>

  <!-- Prediction Points -->
  <!-- Linear Fit @ 750 sq ft (~$150k) -->
  <circle cx="310" cy="165" r="6" fill="#89b4fa"/>
  <text x="322" y="170" fill="#89b4fa" font-size="12" font-weight="bold">Linear Fit (~$150,000)</text>

  <!-- Curve Fit @ 750 sq ft (~$200k) -->
  <circle cx="310" cy="115" r="6" fill="#a6e3a1"/>
  <text x="322" y="112" fill="#a6e3a1" font-size="12" font-weight="bold">Curve Fit (~$200,000)</text>
</svg>

* **Linear Regression**: Fits a straight line ($y = wx + b$). For $750\text{ sq ft}$, predicts $\approx \$150,000$.
* **Non-Linear Regression**: Fits a curve (e.g. polynomial). For $750\text{ sq ft}$, predicts $\approx \$200,000$.

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

Data points plotted along a 1D line using symbols: **O** for Benign ($0$) and **X** for Malignant ($1$).

```
Diagnosis (Y)
    1 (Malignant) |                  X   X   X   X
                  |
    0 (Benign)    |    O   O   O
                  +-----------------------------------> Tumor Size (X)
```

#### B) Multiple Input Features ($X_1 = \text{Tumor Size}, X_2 = \text{Age}$)

When multiple features are provided, the classification algorithm fits a **Decision Boundary** line to separate the categories in multi-dimensional feature space.

<svg viewBox="0 0 600 300" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background:#1e1e2e; border-radius:8px; padding:12px;">
  <!-- Axes -->
  <line x1="70" y1="240" x2="540" y2="240" stroke="#cdd6f4" stroke-width="2"/>
  <line x1="70" y1="240" x2="70"  y2="30"  stroke="#cdd6f4" stroke-width="2"/>

  <!-- Labels -->
  <text x="300" y="280" fill="#cdd6f4" font-family="sans-serif" font-size="14" text-anchor="middle" font-weight="bold">Tumor Size (X1)</text>
  <text x="25" y="135" fill="#cdd6f4" font-family="sans-serif" font-size="14" transform="rotate(-90 25,135)" text-anchor="middle" font-weight="bold">Patient Age (X2)</text>

  <!-- Benign Data Points (Circles - O) -->
  <circle cx="120" cy="200" r="8" fill="none" stroke="#89b4fa" stroke-width="2.5"/>
  <circle cx="160" cy="170" r="8" fill="none" stroke="#89b4fa" stroke-width="2.5"/>
  <circle cx="140" cy="120" r="8" fill="none" stroke="#89b4fa" stroke-width="2.5"/>
  <circle cx="210" cy="190" r="8" fill="none" stroke="#89b4fa" stroke-width="2.5"/>
  <circle cx="200" cy="140" r="8" fill="none" stroke="#89b4fa" stroke-width="2.5"/>
  <circle cx="240" cy="80"  r="8" fill="none" stroke="#89b4fa" stroke-width="2.5"/>

  <!-- Malignant Data Points (Crosses - X) -->
  <path d="M 330 65 L 344 79 M 344 65 L 330 79" stroke="#f38ba8" stroke-width="3"/>
  <path d="M 380 110 L 394 124 M 394 110 L 380 124" stroke="#f38ba8" stroke-width="3"/>
  <path d="M 350 150 L 364 164 M 364 150 L 350 164" stroke="#f38ba8" stroke-width="3"/>
  <path d="M 420 80 L 434 94 M 434 80 L 420 94" stroke="#f38ba8" stroke-width="3"/>
  <path d="M 440 160 L 454 174 M 454 160 L 440 174" stroke="#f38ba8" stroke-width="3"/>
  <path d="M 480 120 L 494 134 M 494 120 L 480 134" stroke="#f38ba8" stroke-width="3"/>

  <!-- Decision Boundary Line -->
  <line x1="280" y1="30" x2="300" y2="240" stroke="#a6e3a1" stroke-width="3.5" stroke-dasharray="6,4"/>

  <!-- Region Labels -->
  <text x="160" y="50" fill="#89b4fa" font-size="13" font-weight="bold">Benign Region (O)</text>
  <text x="380" y="50" fill="#f38ba8" font-size="13" font-weight="bold">Malignant Region (X)</text>
  <text x="315" y="210" fill="#a6e3a1" font-size="12" font-weight="bold">Decision Boundary</text>
</svg>

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
