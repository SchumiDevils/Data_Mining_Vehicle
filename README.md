# Data Mining 1 (DM1) — Project Plan

**Dataset:** Seller-Level Vehicle Dataset (`dm1_26_27_dataset.zip`)  
**Course:** Data Mining 1 (a.a. 2026/2027) — University of Pisa

---

## 1. Project Overview & Dataset

The project analyzes the **Seller-Level Vehicle Dataset**, where each record corresponds to a single vehicle seller/dealer identified by `sp_id` and aggregates information across all listings observed for that seller.

### Dataset Composition
- **Identifier:** `sp_id` (excluded from modeling; used only for indexing/deduplication checks).
- **Vehicle Characteristic Statistics:** Seller-level aggregations of price, mileage, age, engine specifications, fuel economy, and vehicle dimensions.
- **Inventory Composition Indicators:** Proportions of `new`, `cab`, `salvage`, and `theft-title` vehicles.
- **Listing Activity & Dealer Profile:** Listing activity metrics, franchise status, and new/used inventory profile.
- **Dominant Categorical Features:** Most common vehicle characteristics within each seller's inventory.
- **Mandatory Classification Target:** `seller_rating_class` (`low`, `medium`, `high`).

> **Core Evaluation Principle:** `seller_rating_class` is intentionally a **hard classification task**. Lower-than-expected predictive scores are acceptable as long as the **underlying methodology, experimental rigor, and critical discussion are sound**.

---

## 2. Report Specifications & Mandatory Standards

**Only the written PDF report is evaluated** (source code and notebooks are not assessed).

### Format Constraints
- **Format:** Single PDF document (recommended: LaTeX / Overleaf).
- **Page Limit:** **Maximum 20 pages** including all text, tables, and figures. **Title page and Table of Contents (Index) do not count** toward the 20-page limit.
- **Typography & Layout:** Font size strictly **greater than 9pt** (recommended: `10pt`). **Multi-column layouts are allowed** (2-column LaTeX is recommended to organize figures and tables cleanly within 20 pages).

### Mandatory Content Rules
1. **Justify every choice:** Explain variable retention/elimination, cleaning, imputation, transformations, feature selection, distance metrics, and hyperparameter settings.
2. **Report all tested configurations:** Never report only the winning model/configuration; document the parameter ranges tested and justify why the final configuration was chosen.
3. **Discuss every result honestly:** Analyze weak or unconvincing outcomes fairly and attribute limitations appropriately (e.g., dataset noise/overlap vs. algorithm assumptions).
4. **No uncommented visuals:** Every figure and table must be crisp, readable, explicitly referenced, and analyzed in the text.
5. **Non-lecture methodologies:** Additional tools, feature selection techniques, plots, or algorithms not covered in lectures are permitted, provided prior approval is requested from the instructor.

### Grading Structure
Each of the 4 analytical modules is graded out of **30 points**, averaged together, plus bonus points:

$$\text{Project Mark} = \frac{\text{M1} + \text{M2} + \text{M3} + \text{M4}}{4} + \text{Bonus (up to 3 pts)}$$

---

## 3. Report Blueprint & 20-Page Budget

| Section | Counted? | Target Pages | Core Scope |
| :--- | :---: | :---: | :--- |
| Title Page & Table of Contents | No | 2 | Metadata, group members, navigation |
| Executive Summary / Introduction | Yes | 0.5 | Dataset scope, pipeline overview, high-level findings |
| **Module 1: Data Understanding & Preparation** | Yes | **4.5** | Semantics, distributions, quality audit, transformations, correlations |
| **Module 2: Clustering** | Yes | **4.5** | K-Means, DBSCAN, Hierarchical (+ optional method), comparison |
| **Module 3: Classification & Regression** | Yes | **5.5** | Mandatory classification, single & multiple regression, insights |
| **Module 4: Pattern Mining** | Yes | **4.0** | Frequent patterns, association rules, sweeps/histograms, exploitation |
| Overall Conclusions & Limitations | Yes | 1.0 | Cross-module synthesis connecting segments, models, and rules |
| **Total Counted Pages** | **Yes** | **≤ 20.0** | Hard page cap |

---

## 4. Module-by-Module Technical Plan

### Module 1: Data Understanding & Preparation (30 pts)

1. **Data Semantics**
   - Introduce every variable with its semantic meaning, data type (continuous, discrete, bounded proportion `[0, 1]`, nominal, ordinal), unit/range, and modeling role (`sp_id` vs. feature vs. target).
   - Organize variables cleanly in a grouped reference table.

2. **Distributions & Quantitative Statistics**
   - Compute descriptive statistics (count, mean, std, median, IQR, min/max, skewness) for numerical attributes and frequency tables for categorical variables and `seller_rating_class` (class balance analysis).
   - Explore univariate and bivariate distributions via readable histograms, boxplots, and scatter/grouped plots highlighting skewness, multimodality, and relationships with `seller_rating_class`.

3. **Data Quality Assessment & Cleaning**
   - **Missing values:** Detect explicit (`NaN`) and implicit (`0`, `-1`, `"Unknown"`, empty strings) missing values; analyze missingness rates and patterns; justify drop vs. imputation strategies (ensuring imputation statistics are fit strictly on training folds during Module 3).
   - **Duplicates:** Check exact row duplicates and duplicate `sp_id` identifiers.
   - **Syntactic & semantic inconsistencies:** Flag impossible values (e.g., negative prices/mileage/age, proportions outside `[0, 1]`, contradictions between new/used status and mileage/age, invalid engine or dimension values) and justify corrections.
   - **Outliers:** Identify univariate (IQR/z-score) and multivariate anomalies; distinguish real extreme sellers (e.g., luxury or high-volume franchises) from corrupted entries; justify retaining, capping, or removing them per downstream task.

4. **Variable Transformations (Module-Aware)**
   - Evaluate whether transformed variables improve downstream modeling (e.g., `log1p` for right-skewed price/mileage/listing counts; `StandardScaler` vs. `MinMaxScaler` vs. `RobustScaler`).
   - Maintain clearly documented dataset views derived from the same clean base:
     - **Scaled/transformed numeric view:** Clustering (M2), KNN & Regression (M3).
     - **Clean raw/encoded view:** Interpretable Decision Trees (M3).
     - **Discretized/transactional view:** Pattern Mining (M4).

5. **Pairwise Correlations & Feature Elimination**
   - Compute Pearson and Spearman correlation matrices and visualize them with a legible heatmap.
   - Analyze strongly correlated pairs (e.g., $|r| > 0.85$ or domain-redundant specs) and explicitly justify which variables to eliminate vs. retain.

---

### Module 2: Clustering (30 pts)

*Rule: Exclude `sp_id` and `seller_rating_class` during clustering. Reserve `seller_rating_class` strictly for post-hoc external interpretation (checking whether discovered dealer segments align with rating classes).*

1. **Attribute Selection & Preprocessing**
   - Justify the selected low-collinearity, scaled/transformed numeric feature set and explain how categorical or proportion attributes are used (either excluded and used for profiling, or appropriately encoded).
   - Run sensitivity checks (e.g., impact of log-transform or extreme outlier handling).

2. **Centroid-Based Clustering**
   - **K-Means (Mandatory):**
     - Sweep `k` across a well-justified range (e.g., `k = 2..15`) with fixed seeds and multiple initializations (`n_init`).
     - Plot Elbow (SSE/inertia) and Silhouette score vs. `k` (optionally complemented by Davies-Bouldin / Calinski-Harabasz).
     - Justify the chosen `k` and profile clusters in original interpretable units (size distribution, centroid profiles, cross-tabulation with franchise status and `seller_rating_class`).
   - **Bisecting K-Means / X-Means (Optional):** Include at least one to compare top-down divisive clustering or BIC-driven automatic `k` selection against standard K-Means.

3. **Density-Based Clustering**
   - **DBSCAN (Mandatory):**
     - Justify the feature subset (accounting for distance concentration in high dimensions).
     - Plot **k-distance graphs** across candidate `MinPts` values to systematically choose the `eps` search range.
     - Report a grid table of tested `(eps, MinPts)` configurations showing cluster count, noise percentage, and Silhouette score (computed on core/non-noise points, always reporting noise % alongside it).
     - Profile both the dense clusters and the **noise dealers** (interpreting what makes them anomalous).
   - **OPTICS (Optional):** Useful if the dataset exhibits varying densities; inspect the reachability plot to explain density structure.

4. **Hierarchical Clustering**
   - Justify chosen attributes and **distance functions** (e.g., Euclidean vs. Manhattan or Cosine).
   - Compute and analyze **several dendrograms** comparing linkage criteria (**Single, Complete, Average, Ward**).
   - Discuss structural phenomena (e.g., chaining in Single linkage vs. compact clusters in Ward/Complete), justify chosen cut thresholds, and profile resulting clusters.

5. **Final Clustering Discussion**
   - Summarize all methods in a comparative table covering internal metrics (Silhouette, SSE where applicable, Davies-Bouldin), cluster size balance, noise handling, external alignment with `seller_rating_class`, and domain interpretability.
   - Conclude which algorithm is **"best"** with respect to aggregate statistics, cluster distributions, and algorithm typology vs. the geometry of this dataset.

---

### Module 3: Classification and Regression (30 pts)

#### Part A: Classification (Mandatory Target: `seller_rating_class`)

1. **Experimental Design (Leak-Free)**
   - Stratified train/test split (e.g., 80/20) with stratified $k$-fold cross-validation (e.g., 5- or 10-fold) on the training set for hyperparameter tuning.
   - Wrap imputation, scaling, and encoding inside CV pipelines so no test/validation statistics leak into training.
   - Justify attribute selection, categorical encoding, and evaluation emphasis given class balance.

2. **Mandatory Classifiers & Parameter Sweeps** *(list all tested configurations and justify the best)*
   - **Decision Trees:** Tune splitting criterion (`gini`, `entropy`), `max_depth`, `min_samples_split`, `min_samples_leaf`, and pruning (`ccp_alpha`). Compare unpruned vs. pruned performance to demonstrate overfitting control.
   - **K-Nearest Neighbors (KNN):** Tune `n_neighbors` ($k$), distance metric (`euclidean`, `manhattan`, `minkowski`), and weighting (`uniform`, `distance`). Plot CV performance vs. $k$.
   - **Naïve Bayes:** Select appropriate variant(s) for the feature types (e.g., `GaussianNB` on continuous/transformed features, `CategoricalNB`/`MultinomialNB` on categorical/discretized features) and tune smoothing (`var_smoothing` / `alpha`). Discuss how well conditional independence and distributional assumptions hold.
   - *(Optional Extra Task):* Classify a second meaningful categorical target (e.g., franchise status or inventory profile) if page budget permits.

3. **Mandatory Classification Metrics & Insights**
   - Evaluate all final models on the held-out test set using:
     - **Confusion Matrix** (`low` / `medium` / `high`)
     - **Accuracy**, **Precision**, **Recall**, **$F_1$-score** (per-class + macro/weighted averages)
     - **ROC Curves & AUC** (multiclass One-vs-Rest curves per class + macro/micro averages)
   - **Interpretability & Error Analysis:** Visualize and interpret the pruned **Decision Tree** (top decision rules, feature importance) and analyze systematic misclassifications (e.g., confusion around the `medium` class).

#### Part B: Regression (Single & Multiple)

1. **Task Setup**
   - Choose **one meaningful continuous target** (motivated by Module 1 analysis, e.g., average seller price) and justify the choice.
   - Use the **exact same target variable** for both single and multiple regression.

2. **Single Linear Regression**
   - Choose **one independent variable** justified via correlation and domain relevance.
   - Fit OLS linear regression, visualize the fitted line and residuals, and evaluate **MSE** and **$R^2$** on train and test sets.

3. **Multiple Regression (2+ Independent Variables)**
   - Select **2+ predictors** (avoiding severe multicollinearity from Module 1) predicting the **same target**.
   - Compare:
     1. **Multiple Linear Regression**
     2. **At least 2 non-linear approaches**, e.g.:
        - **Decision Tree Regressor** (tuned depth/leaf size)
        - **KNN Regressor** (tuned $k$, metric, weights)
        - *(Alternative/extra: Polynomial Regression, SVR, or Random Forest Regressor — confirm with instructor if outside lecture scope)*
   - List all tested parameter configurations and compare every model (including Single Linear baseline) on train and test **MSE** and **$R^2$**, accompanied by residual/predicted-vs-actual diagnostics.

#### Part C: Module 3 Discussion
- Identify the **best** classification algorithm and the **best** regression algorithm from multiple perspectives: quantitative metrics, generalization gap, interpretability, robustness to noise/skew, and computational cost.

---

### Module 4: Pattern Mining (30 pts)

1. **Transactional Encoding & Discretization**
   - Convert seller records into itemsets combining categorical features, binary indicators, `seller_rating_class`, and discretized key numeric variables.
   - Justify binning choices (e.g., quantile vs. domain-defined bins) and keep attribute-prefixed item names (e.g., `price=high`, `mileage=low`, `rating=high`).

2. **Frequent Pattern Extraction & Discussion**
   - Extract frequent itemsets (using Apriori / FP-Growth) across varying `min_sup` values (optionally comparing frequent, closed, and maximal itemsets).
   - **Quantitative analysis:** Plot how the **number of frequent patterns varies w.r.t. `min_sup` and itemset length $k$**; justify the selected `min_sup`.
   - **Qualitative analysis:** Interpret the most insightful frequent dealer profiles.

3. **Association Rules Extraction & Discussion**
   - Extract association rules across varying `min_conf` values and filter/rank using **lift** (to separate genuinely interesting associations from trivial high-support consequents).
   - **Quantitative analysis:**
     - Plot how the **number of rules varies w.r.t. `min_conf` and rule length $k$**.
     - Plot **histograms of rules' confidence and lift** (plus optional support–confidence–lift scatter plot).
   - **Qualitative analysis:** Present and discuss top high-lift, domain-meaningful rules describing dealer inventories and ratings.

4. **Rule Exploitation (Mandatory)**
   - Apply and quantitatively evaluate the most useful extracted rules in at least one (ideally both) downstream tasks:
     1. **Target Prediction (`seller_rating_class`):** Build a rule-based classifier using rules with `seller_rating_class` as consequent (trained only on the Module 3 training split), evaluate Accuracy, Macro-$F_1$, and coverage on the Module 3 test split, and compare against DT/KNN/NB.
     2. **Missing Value Imputation:** Mask known values (e.g., 10% of a categorical/binned attribute) or impute actual missing entries via top-confidence/lift matching rules; evaluate imputation accuracy and coverage vs. a mode baseline.

---

## 5. Quality & Bonus Optimization Criteria

To maximize project quality and target the content-based bonus points (out of the 3 total bonus points):

- **Innovation (+0.5 pts):** Connect modules into a cohesive narrative (e.g., profiling Module 2 clusters using Module 4 rules, benchmarking rule-based classification against Module 3 models, or adding instructor-approved analyses).
- **Experimentation (+0.5 pts):** Include optional algorithms (Bisecting K-Means or X-Means, OPTICS), systematic ablation/sensitivity checks (raw vs. log-scaled, with vs. without outliers), and well-documented parameter grids.
- **Performance (+0.5 pts):** Maintain strictly leak-free pipelines, thoughtful domain feature engineering, proper handling of skew/imbalance, and disciplined hyperparameter tuning.
- **Appearance, Summary & Organization (+0.5 pts):** Polished 2-column LaTeX design (>9pt), consistent colorblind-safe vector plots, clear tables, strong Executive Summary and Conclusions, and strict adherence to ≤20 counted pages.

---

## 6. Repository Structure & Technical Standards

### Recommended Workspace Layout
```text
.
├── README.md
├── data/
│   ├── raw/                  # Original dm1_26_27_dataset files (read-only)
│   └── processed/            # Cleaned & module-specific dataset views (versioned)
├── notebooks/
│   ├── 01_data_understanding_prep.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_classification_regression.ipynb
│   └── 04_pattern_mining.ipynb
├── src/                      # Shared preprocessing, metrics, and plotting utilities
└── figures/                  # Exported vector figures (PDF/SVG) for LaTeX report
```

### Engineering & Oral Exam Discipline
- **Reproducibility:** Fix `random_state=42` across all splits, CV folds, and stochastic algorithms.
- **Zero Data Leakage:** Fit all imputers, scalers, encoders, discretizers, and rule extractors strictly on training partitions whenever evaluating predictive performance.
- **Unified Visual Style:** Export high-resolution vector plots (`PDF`/`SVG`) with consistent fonts, legible axis labels, and unified class/cluster color palettes.
- **Shared Understanding:** Every team member should understand all 4 modules, including algorithms, formulas, and every methodological choice made in the report.

---

## 7. Project Quality & Compliance Checklist

- [ ] **Report Constraints:** ≤20 counted pages (excluding Title Page & Index); font size >9pt; PDF only; every plot/table is legible, referenced, and commented; every choice and tested parameter range is documented and justified.
- [ ] **Module 1:** Variable semantics table; univariate & bivariate stats/plots; missing values, duplicates, inconsistencies, and outliers audited and resolved; transformations justified; correlation heatmap and feature elimination explained.
- [ ] **Module 2:** K-Means (mandatory), DBSCAN (mandatory), Hierarchical with *multiple dendrograms* (mandatory); attributes, distances, and parameters justified; full comparative "best algorithm" discussion across aggregate metrics, cluster distributions, and dataset geometry.
- [ ] **Module 3:** `seller_rating_class` classified with Decision Trees, KNN, and Naïve Bayes; Confusion Matrices, Accuracy, Precision, Recall, $F_1$-score, and ROC curves reported; Single Linear Regression + Multiple Regression on the **same continuous target** with Linear + **≥2 non-linear** models evaluated by **MSE** and **$R^2$**; tree/model insights and "best" discussion included.
- [ ] **Module 4:** Frequent patterns across `min_sup` (quantitative vs. $k$/`min_sup` + qualitative); Association rules across `min_conf` (quantitative vs. $k$/`min_conf`, **confidence & lift histograms**, + qualitative); concrete rule exploitation (target prediction and/or missing-value imputation) quantitatively evaluated.
