# Computational Method for Ovarian Cancer Subtype Classification

An advanced translational bioinformatics framework that applies bio-inspired meta-heuristics to high-dimensional genomic data for high-precision molecular subtyping of ovarian cancer.

## 📌 Project Overview
Ovarian cancer is a highly heterogeneous malignancy traditionally categorized by histopathology. However, distinct transcriptomic profiles often dictate varying therapeutic responses. This project establishes a robust machine learning pipeline utilizing **TCGA RNA-Seq data** to automate the classification of primary High-Grade Serous Ovarian Carcinoma (HGSOC) into four molecular subtypes: **Differentiated, Immunoreactive, Mesenchymal, and Proliferative**.

To overcome the "curse of dimensionality" inherent to transcriptomic datasets (20,530 gene features), this study evaluates and contrasts two primary dimensionality reduction approaches:
1. **Ant Colony Optimization (ACO):** A bio-inspired feature selection heuristic.
2. **Principal Component Analysis (PCA):** A classic linear feature transformation technique.

---

## 🔬 Core Methodology
The framework leverages an optimized **ACO-Logistic Regression (ACO-LR)** pipeline that preserves biological gene identity while executing extreme noise filtration.

* **Data Source:** TCGA Ovarian Cancer (TCGA-OV) dataset via the UCSC Xena platform (308 patient samples, 20,530 features).
* **Data Preprocessing:** Matrix transposition (samples as rows, genes as columns) and feature-label integration.
* **Feature Selection Strategy:** ACO meta-heuristic optimization successfully **filtered out 88.4% of feature noise**, retaining a highly discriminative consensus subset of **2,395 features**.
* **Evaluated Models:** Random Forest (RF), Support Vector Machine (SVM), Logistic Regression (LR), and XGBoost.
* **Validation Protocol:** 90:10 Train-Test split integrated with a 5-Fold Cross-Validation `GridSearchCV` hyperparameter tuning mechanism.

---

## 📈 Key Results & Performance
The bio-inspired feature selection strategy significantly outperformed linear mathematical projection, establishing **Logistic Regression via ACO** as the project's **Champion Model**.

### Global Performance Matrix
| Metric | Value (Weighted Average) |
| :--- | :--- |
| **Mean CV Accuracy** | `0.9675 (+/- 0.0273)` |
| **Precision** | `0.9703` |
| **Recall** | `0.9675` |
| **F1-Score** | `0.9675` |
| **Permutation Test Accuracy (Shuffled Labels)** | `0.2258` (Confirms genuine pattern learning over noise) |

### Class-Specific Diagnostics
* **Proliferative Subtype:** Achieved absolute separation (**1.00 Precision, 1.00 Recall**).
* **Immunoreactive Subtype:** Attained high sensitivity (**1.00 Recall, 0.89 Precision**).
* **Mesenchymal Subtype:** Attained high confidence (**1.00 Precision, 0.86 Recall**).
* **Differentiated Subtype:** Proved the most plastic boundary (**0.86 Precision, 0.86 Recall**), representing the primary source of subtle cross-contamination.

### Top 5 Influential Diagnostic Biomarkers
Rather than relying on a single "silver bullet" gene, the champion model establishes stable linear decision boundaries based on a consensus of highly weighted features:
1. `MURC` (Weight: 0.0190) — Primary driver distinguishing the proliferative subtype.
2. `UCHL1` (Weight: 0.0182)
3. `PCDHAC1` (Weight: 0.0181)
4. `HSPA13` (Weight: 0.0180)
5. `FAM84A` (Weight: 0.0177)

---

## ⚙️ Project Structure
```text
├── tb_4.py                # Main execution script for PCA/ACO pipeline and auditing
├── Figure1.png            # 2D scatter plot visualization mapping & Winning model multi-class confusion matrix
├── TB_ProjectBIM-2024-3,5&10.docx    # Detailed translational bioinformatics report
└── README.md              # Repository documentation
