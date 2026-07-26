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

### 🏠 Case Study: Housing Price Prediction (Mental Model)

* **Input ($X$)**: House size in feet²
* **Target ($Y$)**: House price in $1000's

<svg viewBox="0 0 650 360" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; border:1px solid #e0e0e0; border-radius:8px; padding:16px; font-family: sans-serif;">
  <!-- Title -->
  <text x="30" y="40" fill="#0070f3" font-size="24" font-weight="bold">Regression:<tspan fill="#222222" font-weight="normal"> Housing price prediction</tspan></text>

  <!-- Y axis -->
  <line x1="120" y1="270" x2="120" y2="80" stroke="#000000" stroke-width="3"/>
  <!-- Y axis ticks -->
  <line x1="110" y1="270" x2="120" y2="270" stroke="#000000" stroke-width="3"/>
  <line x1="110" y1="222.5" x2="120" y2="222.5" stroke="#000000" stroke-width="3"/>
  <line x1="110" y1="175" x2="120" y2="175" stroke="#000000" stroke-width="3"/>
  <line x1="110" y1="127.5" x2="120" y2="127.5" stroke="#000000" stroke-width="3"/>
  <line x1="110" y1="80" x2="120" y2="80" stroke="#000000" stroke-width="3"/>

  <!-- Y axis tick labels -->
  <text x="100" y="275" fill="#000000" font-size="14" text-anchor="end">0</text>
  <text x="100" y="227" fill="#000000" font-size="14" text-anchor="end">100</text>
  <text x="100" y="180" fill="#000000" font-size="14" text-anchor="end">200</text>
  <text x="100" y="132" fill="#000000" font-size="14" text-anchor="end">300</text>
  <text x="100" y="85" fill="#000000" font-size="14" text-anchor="end">400</text>

  <!-- Y axis Title -->
  <text x="50" y="180" fill="#000000" font-size="15" font-weight="bold" text-anchor="middle">Price in</text>
  <text x="50" y="200" fill="#000000" font-size="15" font-weight="bold" text-anchor="middle">$1000's</text>
  <line x1="20" y1="210" x2="80" y2="210" stroke="#70a800" stroke-width="3"/>

  <!-- X axis -->
  <line x1="120" y1="270" x2="550" y2="270" stroke="#000000" stroke-width="3"/>
  <!-- X axis ticks -->
  <line x1="120" y1="270" x2="120" y2="280" stroke="#000000" stroke-width="3"/>
  <line x1="206" y1="270" x2="206" y2="280" stroke="#000000" stroke-width="3"/>
  <line x1="292" y1="270" x2="292" y2="280" stroke="#000000" stroke-width="3"/>
  <line x1="378" y1="270" x2="378" y2="280" stroke="#000000" stroke-width="3"/>
  <line x1="464" y1="270" x2="464" y2="280" stroke="#000000" stroke-width="3"/>
  <line x1="550" y1="270" x2="550" y2="280" stroke="#000000" stroke-width="3"/>

  <!-- X axis tick labels -->
  <text x="120" y="300" fill="#000000" font-size="14" text-anchor="middle">0</text>
  <text x="206" y="300" fill="#000000" font-size="14" text-anchor="middle">500</text>
  <text x="292" y="300" fill="#000000" font-size="14" text-anchor="middle">1000</text>
  <text x="378" y="300" fill="#000000" font-size="14" text-anchor="middle">1500</text>
  <text x="464" y="300" fill="#000000" font-size="14" text-anchor="middle">2000</text>
  <text x="550" y="300" fill="#000000" font-size="14" text-anchor="middle">2500</text>

  <!-- X axis Title -->
  <text x="335" y="325" fill="#000000" font-size="15" font-weight="bold" text-anchor="middle">House size in feet²</text>
  <line x1="260" y1="335" x2="410" y2="335" stroke="#70a800" stroke-width="3"/>

  <!-- Data Points: Red Crosses (X) -->
  <path d="M 184 231 L 194 241 M 194 231 L 184 241" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 218 208 L 228 218 M 228 208 L 218 218" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 227 165 L 237 175 M 237 165 L 227 175" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 261 170 L 271 180 M 271 170 L 261 180" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 278 132 L 288 142 M 288 132 L 278 142" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 321 141 L 331 151 M 331 141 L 321 151" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 364 118 L 374 128 M 374 118 L 364 128" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 399 130 L 409 140 M 409 130 L 399 140" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 425 103 L 435 113 M 435 103 L 425 113" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 450 118 L 460 128 M 460 118 L 450 128" stroke="#ff0000" stroke-width="3.5"/>
  <path d="M 485 101 L 495 111 M 495 101 L 485 111" stroke="#ff0000" stroke-width="3.5"/>

  <!-- Blue Regression Line -->
  <line x1="140" y1="230" x2="520" y2="85" stroke="#1890ff" stroke-width="4"/>

  <!-- Orange Prediction Marker for 750 sq ft -->
  <line x1="249" y1="270" x2="249" y2="203" stroke="#ff9c6e" stroke-width="2.5"/>
  <polygon points="249,198 244,208 254,208" fill="#ff9c6e"/>
  <circle cx="249" cy="198.4" r="5" fill="none" stroke="#ff9c6e" stroke-width="2.5"/>
  <text x="249" y="325" fill="#d97706" font-size="18" font-weight="bold" text-anchor="middle">750</text>

  <!-- Pink Prediction Line to Y-axis ($150k) -->
  <line x1="249" y1="198.4" x2="120" y2="198.4" stroke="#eb2f96" stroke-width="2.5"/>
  <circle cx="120" cy="198.4" r="4" fill="none" stroke="#eb2f96" stroke-width="2.5"/>
  <text x="110" y="203" fill="#eb2f96" font-size="16" font-weight="bold" text-anchor="end">150k</text>
</svg>

#### How to Read This Mental Model:
1. **Training Data ($X, Y$)**: Each red cross ($\mathbf{X}$) represents a historical house sale (Size on X-axis, Price on Y-axis).
2. **Model Training**: The regression algorithm fits a blue straight line ($y = wx + b$) through the data points.
3. **Inference**: For a new house of size $750\text{ feet}^2$:
   * Follow the orange arrow up from $750$ to the regression line.
   * Trace the pink line horizontally left to the Y-axis to read the predicted price: **$150,000**.

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

<svg viewBox="0 0 600 300" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; border:1px solid #e0e0e0; border-radius:8px; padding:12px; font-family: sans-serif;">
  <!-- Axes -->
  <line x1="70" y1="240" x2="540" y2="240" stroke="#000000" stroke-width="2.5"/>
  <line x1="70" y1="240" x2="70"  y2="30"  stroke="#000000" stroke-width="2.5"/>

  <!-- Labels -->
  <text x="300" y="275" fill="#000000" font-size="14" text-anchor="middle" font-weight="bold">Tumor Size (X1)</text>
  <text x="25" y="135" fill="#000000" font-size="14" transform="rotate(-90 25,135)" text-anchor="middle" font-weight="bold">Patient Age (X2)</text>

  <!-- Benign Data Points (Circles - O) -->
  <circle cx="120" cy="200" r="8" fill="none" stroke="#0070f3" stroke-width="2.5"/>
  <circle cx="160" cy="170" r="8" fill="none" stroke="#0070f3" stroke-width="2.5"/>
  <circle cx="140" cy="120" r="8" fill="none" stroke="#0070f3" stroke-width="2.5"/>
  <circle cx="210" cy="190" r="8" fill="none" stroke="#0070f3" stroke-width="2.5"/>
  <circle cx="200" cy="140" r="8" fill="none" stroke="#0070f3" stroke-width="2.5"/>
  <circle cx="240" cy="80"  r="8" fill="none" stroke="#0070f3" stroke-width="2.5"/>

  <!-- Malignant Data Points (Crosses - X) -->
  <path d="M 330 65 L 344 79 M 344 65 L 330 79" stroke="#ff0000" stroke-width="3"/>
  <path d="M 380 110 L 394 124 M 394 110 L 380 124" stroke="#ff0000" stroke-width="3"/>
  <path d="M 350 150 L 364 164 M 364 150 L 350 164" stroke="#ff0000" stroke-width="3"/>
  <path d="M 420 80 L 434 94 M 434 80 L 420 94" stroke="#ff0000" stroke-width="3"/>
  <path d="M 440 160 L 454 174 M 454 160 L 440 174" stroke="#ff0000" stroke-width="3"/>
  <path d="M 480 120 L 494 134 M 494 120 L 480 134" stroke="#ff0000" stroke-width="3"/>

  <!-- Decision Boundary Line -->
  <line x1="280" y1="30" x2="300" y2="240" stroke="#70a800" stroke-width="3.5" stroke-dasharray="6,4"/>

  <!-- Region Labels -->
  <text x="160" y="50" fill="#0070f3" font-size="13" font-weight="bold">Benign Region (O)</text>
  <text x="380" y="50" fill="#ff0000" font-size="13" font-weight="bold">Malignant Region (X)</text>
  <text x="315" y="210" fill="#70a800" font-size="12" font-weight="bold">Decision Boundary</text>
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
