# Unsupervised Learning: Clustering, Anomaly Detection & Dimensionality Reduction

> **Core Concept**: Unsupervised Learning algorithms process data containing **inputs $X$ only (no target output labels $y$)**. The algorithm's objective is to autonomously discover hidden structures, patterns, anomalies, or compressed representations in the data without human supervision.

---

## 📌 1. Formal Definition: Supervised vs. Unsupervised

$$\text{Supervised Learning: } \{(X_1, y_1), (X_2, y_2), \dots, (X_m, y_m)\} \implies \text{Learns } f(X) \approx y$$

$$\text{Unsupervised Learning: } \{X_1, X_2, \dots, X_m\} \implies \text{Discovers Structure / Patterns in } X$$

```
+-----------------------------------------------------------------------------------+
| SUPERVISED DATA (Labeled)                 | UNSUPERVISED DATA (Unlabeled)         |
|                                           |                                       |
|  Age (X2)                                 |  Age (X2)                             |
|    ^                                      |    ^                                  |
|    |   O   O   |   X   X                  |    |   *   *       *   *              |
|    | O   O     | X   X                    |    | *   *       *   *                |
|    +----------------------> Size (X1)     |    +----------------------> Size (X1) |
|    Labels: Benign (O), Malignant (X)      |    Labels: NONE (Raw inputs X only)   |
+-----------------------------------------------------------------------------------+
```

<svg viewBox="0 0 620 260" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; border:1px solid #e0e0e0; border-radius:8px; padding:12px; font-family: sans-serif;">
  <!-- Supervised Plot -->
  <g transform="translate(10,0)">
    <text x="140" y="30" fill="#0070f3" font-size="14" font-weight="bold" text-anchor="middle">Supervised Data (Labeled Y)</text>
    <line x1="40" y1="210" x2="240" y2="210" stroke="#000" stroke-width="2"/>
    <line x1="40" y1="210" x2="40" y2="50" stroke="#000" stroke-width="2"/>
    
    <!-- Benign O -->
    <circle cx="80" cy="170" r="6" fill="none" stroke="#0070f3" stroke-width="2"/>
    <circle cx="100" cy="140" r="6" fill="none" stroke="#0070f3" stroke-width="2"/>
    <circle cx="120" cy="180" r="6" fill="none" stroke="#0070f3" stroke-width="2"/>

    <!-- Malignant X -->
    <path d="M 170 80 L 182 92 M 182 80 L 170 92" stroke="#ff0000" stroke-width="2.5"/>
    <path d="M 190 110 L 202 122 M 202 110 L 190 122" stroke="#ff0000" stroke-width="2.5"/>
    <path d="M 210 70 L 222 82 M 222 70 L 210 82" stroke="#ff0000" stroke-width="2.5"/>
    
    <line x1="145" y1="50" x2="155" y2="210" stroke="#70a800" stroke-width="2" stroke-dasharray="4,4"/>
  </g>

  <!-- Unsupervised Plot -->
  <g transform="translate(320,0)">
    <text x="140" y="30" fill="#70a800" font-size="14" font-weight="bold" text-anchor="middle">Unsupervised Data (No Labels Y)</text>
    <line x1="40" y1="210" x2="240" y2="210" stroke="#000" stroke-width="2"/>
    <line x1="40" y1="210" x2="40" y2="50" stroke="#000" stroke-width="2"/>

    <!-- Cluster 1 (Unlabeled dots) -->
    <ellipse cx="100" cy="160" rx="35" ry="30" fill="#89b4fa" fill-opacity="0.2" stroke="#0070f3" stroke-dasharray="3,3"/>
    <circle cx="80" cy="170" r="5" fill="#333"/>
    <circle cx="100" cy="140" r="5" fill="#333"/>
    <circle cx="115" cy="175" r="5" fill="#333"/>

    <!-- Cluster 2 (Unlabeled dots) -->
    <ellipse cx="195" cy="95" rx="35" ry="30" fill="#a6e3a1" fill-opacity="0.2" stroke="#70a800" stroke-dasharray="3,3"/>
    <circle cx="180" cy="90" r="5" fill="#333"/>
    <circle cx="195" cy="115" r="5" fill="#333"/>
    <circle cx="210" cy="80" r="5" fill="#333"/>
  </g>
</svg>

---

## 🎯 2. Three Major Types of Unsupervised Learning

```
                           ┌───────────────────────────┐
                           │   Unsupervised Learning   │
                           └─────────────┬─────────────┘
                                         │
     ┌───────────────────────────────────┼───────────────────────────────────┐
     ▼                                   ▼                                   ▼
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│        Clustering        │   │    Anomaly Detection     │   │ Dimensionality Reduction │
│  Group similar data      │   │  Detect unusual events   │   │  Compress data features  │
│  points together         │   │  (e.g., Financial Fraud) │   │  without losing info     │
└──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
```

### 1️⃣ Clustering
* **Goal**: Partition unlabeled data into groups based on feature similarity.
* **Real-World Applications**:
  1. **Google News Aggregation**: Crawls hundreds of thousands of daily articles and automatically clusters stories sharing words like `"panda"`, `"twin"`, `"zoo"`.
  2. **Genetic Microarray Analysis**: Groups DNA profiles across thousands of genes into genetic subtypes ($\text{Type 1}$, $\text{Type 2}$, $\text{Type 3}$).
  3. **Customer Market Segmentation**: Clusters customer databases by underlying motivations (e.g. Skill Seekers, Career Advancers, AI Enthusiasts).

---

### 2️⃣ Anomaly Detection
* **Goal**: Detect rare, unusual events or data points that deviate significantly from standard behavior.
* **Real-World Applications**:
  * **Financial Fraud Detection**: Flagging unusual credit card transactions or banking activity.
  * **System Health & Monitoring**: Detecting unexpected cloud server memory/CPU spikes before a system outage occurs.

---

### 3️⃣ Dimensionality Reduction
* **Goal**: Compress a high-dimensional dataset (e.g., 100+ features) into a smaller feature space (e.g., 2 or 3 features) while retaining maximum information.
* **Real-World Applications**:
  * Data compression & fast model processing.
  * 2D/3D visualization of complex high-dimensional datasets.

---

## 🧩 3. Self-Assessment: Supervised vs. Unsupervised Problem Identification

| Problem Scenario | Paradigm | Reasoning |
| :--- | :--- | :--- |
| **Spam Filtering** (Emails labeled Spam / Not Spam) | **Supervised** (Classification) | Uses labeled $y$ outputs (Spam=1, Inbox=0). |
| **Google News Aggregation** | **Unsupervised** (Clustering) | Groups articles automatically with no human topic labels. |
| **Customer Market Segmentation** | **Unsupervised** (Clustering) | Discovers customer persona groups from raw behavior data $X$. |
| **Diagnosing Diabetes** (Patient records labeled Diabetes / No Diabetes) | **Supervised** (Classification) | Uses labeled medical diagnosis $y$ (Diabetes=1, Healthy=0), identical to breast cancer classification. |

---

## 🎯 Summary Checklist

- [x] **Unsupervised Learning**: Inputs $X$ only (no target output labels $y$).
- [x] **Three Core Types**:
  1. **Clustering**: Grouping similar data points.
  2. **Anomaly Detection**: Spotting unusual events (Fraud detection).
  3. **Dimensionality Reduction**: Data compression & feature reduction.
