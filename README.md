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

### Logistics, Deadlines & Submission
- **Group:** min 2, max 3 people; register the group via the link on the course page.
- **Deadlines (bonus track):**
  - **15/11/2026:** mid-draft (Module 1 Data Understanding & Preparation + Module 2 Clustering) → **+0.5**
  - **31/12/2026:** complete project → **+0.5**
  - **No-bonus track:** complete project delivered **1 week before** the chosen oral exam date.
  - Deadlines may change; the course main page is authoritative.
- **Submission:** email the PDF to **both** `alessio.cascione@phd.unipi.it` and `riccardo.guidotti@unipi.it`.
  - Subject: `[DM1 Project 26/27]` (from the 26/27 guidelines; the course exam page still shows the stale `[DM1 2025-2026] Project`; confirm with the instructor if in doubt).
  - File name: `Project_Surname1_Surname2[_Surname3].pdf`.
- **Oral exam:** registration on https://esami.unipi.it/; only possible after project delivery. It covers (1) theory (formulas/pseudocode on paper), (2) pen-and-paper exercises on the course algorithms, (3) discussion of the project's questionable steps/choices. An insufficient project must be redone on a new or substantially updated dataset.

### Format Constraints
- **Format:** Single PDF document (LaTeX / Overleaf suggested; no .doc, notebooks, or code).
- **Page Limit:** **Maximum 20 pages** including all text, tables, and figures. **Title page and Table of Contents (Index) do not count** toward the 20-page limit.
- **Typography & Layout:** Guidelines recommend a font size **above 9pt**; we use `10pt`. **Multi-column layouts are allowed** (2-column LaTeX is recommended to organize figures and tables cleanly within 20 pages).

### Mandatory Content Rules
1. **Justify every choice:** Explain variable retention/elimination, cleaning, imputation, transformations, feature selection, distance metrics, and hyperparameter settings.
2. **Report all tested configurations:** Never report only the winning model/configuration; document the parameter ranges tested and justify why the final configuration was chosen.
3. **Discuss every result honestly:** Analyze weak or unconvincing outcomes fairly and attribute limitations appropriately (e.g., dataset noise/overlap vs. algorithm assumptions).
4. **No uncommented visuals:** Every figure and table must be crisp, readable, explicitly referenced, and analyzed in the text.
5. **Non-lecture methodologies:** Additional tools, feature selection techniques, plots, or algorithms not covered in lectures are permitted; the instructor suggests emailing them **beforehand** (we will do so for anything outside the lectures).
6. **Free organization:** The suggested analyses are guidance; content may be organized freely, but `seller_rating_class` classification is mandatory. Extra classification targets are optional; the regression target is free.

### Grading Structure
Each of the 4 analytical modules is graded out of **30 points**, averaged together, plus bonus points:

$$\text{Project Mark} = \frac{\text{M1} + \text{M2} + \text{M3} + \text{M4}}{4} + \text{Bonus (up to 3 pts)}$$

Bonus breakdown: Innovation 0.5 + Experimentation 0.5 + Performance 0.5 + Appearance/Summary/Organization 0.5 + on-time delivery (mid-draft 0.5 + final 0.5) = **3.0**.

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
   - **Missing values:** Detect explicit (`NaN`) and implicit (`0`, `-1`, `"Unknown"`, empty strings) missing values; analyze missingness rates and patterns (including whether missingness depends on `seller_rating_class`); justify drop vs. imputation strategies.
     - **Leakage rule:** Module 1 saves a **clean but un-imputed base** (`NaN` kept). Imputation may be fit on the full data only for the unsupervised views (M2 clustering, M4 frequent patterns); for M3 and the M4 rule-based classifier it is re-fit inside the training folds. Row drops in M1 may only use row-local criteria (corrupted/impossible entries), never dataset-wide statistics tied to the target.
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
   - Elimination here is **unsupervised** (feature–feature redundancy only). Feature–target associations are reported descriptively; any target-driven feature selection happens inside the M3 CV pipeline.

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
   - **Ward is defined only with Euclidean distance** (it minimizes the within-cluster variance increase); Manhattan/Cosine are paired only with Single/Complete/Average.
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
   - **K-Nearest Neighbors (KNN):** Tune `n_neighbors` ($k$), the Minkowski order `p` (`p=1` Manhattan, `p=2` Euclidean; listing `minkowski` as a third metric is redundant since it defaults to `p=2`), and weighting (`uniform`, `distance`). Features must be scaled (KNN is distance-based). Plot CV performance vs. $k$.
   - **Naïve Bayes:** Select appropriate variant(s) for the feature types: `GaussianNB` on continuous/transformed features (tune `var_smoothing`) and `CategoricalNB` on categorical/discretized features (tune `alpha`; bins fit on training folds only). `MultinomialNB` is for counts, not for one-hot discretized attributes. Discuss how well conditional independence and distributional assumptions hold.
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
   - Exclude predictors that are near-deterministic derivations of the target (e.g., another statistic of the same quantity, such as median vs. mean price), which would inflate $R^2$ without insight.
   - Use the same train/test split for single and multiple regression so the models can be compared directly.

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
   - For the rule-based classifier (step 4), bin edges and rules are computed on the **M3 training split only**, and `rating=*` items are excluded from antecedents.

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

To maximize project quality and target the content-based bonus points (2 of the 3 total bonus points; the remaining 1 point is on-time delivery, see Section 2):

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
│   ├── 04_pattern_mining.ipynb
│   └── 05_quality_extensions.ipynb   # Section 5 extras; RECOMPUTE = False loads saved results
├── src/                      # Shared preprocessing, metrics, and plotting utilities (plot_style.py)
├── figures/                  # Exported vector figures (PDF + PNG preview) for LaTeX report
├── tables/                   # CSV tables backing the report's tables
└── report/                   # LaTeX report (main.tex, sections/, make_tables.py; build with latexmk -pdf -outdir=build main.tex)
```

### Environment
- Python 3.9 conda env `tf` (`C:\Users\crist\miniconda3\envs\tf`): pandas 2.1, numpy 1.25, scipy 1.13, scikit-learn 1.6, matplotlib 3.9, seaborn 0.13, jupyter, mlxtend 0.23.4 (Apriori, association rules).

### Module 1 outputs (consumed by Modules 2–4)
- `data/processed/clean_base.csv`: cleaned, `sp_id` index, **NaN kept** (impute downstream; M3 inside CV folds).
- `data/processed/feature_sets.json`: modelling numeric/categorical features, `log1p` variables, eliminated variables and reasons, category groupings.
- Multivariate outliers are not removed in M1: they are analysed as DBSCAN noise in Module 2.

### Module 2 outputs
- Clustering view: 7 seller-profile features (`n_listings`, `price`, `savings`, `daysonmarket` log1p; `mileage`, `pct_is_new`, `listing_span_days` raw), median-imputed and standardized.
- `data/processed/cluster_labels.csv`: `kmeans4` (chosen segmentation), `kmeans2`, `ward4`, `dbscan` (−1 = noise) per `sp_id`, for M4 cluster profiling and the cross-module discussion.
- Key results: K-Means k=4 is the best method (silhouette 0.34, stable across seeds); DBSCAN (eps=0.6, MinPts=15) finds 2 cores + 22% noise; no clustering aligns with `seller_rating_class` (ARI ≤ 0.025).

### Module 3 outputs
- `data/processed/split.csv`: the stratified 80/20 train/test split (`set` = train/test per `sp_id`); **Module 4's rule-based classifier must reuse it**.
- `src/m3_utils.py`: `PriorCorrectedKNN` (KNN with a class-prior-corrected decision rule; also works around a scikit-learn 1.6 bug with string labels).
- Classification (test macro-F1 / macro AUC): Decision Tree 0.437 / 0.63 (pruned, depth 5, balanced), KNN 0.433 / 0.64 (k=151, p=1, prior-corrected), CategoricalNB 0.414, GaussianNB 0.407; majority baseline 0.236. Best overall: pruned DT.
- Regression on log1p(price): single OLS on mileage R² 0.75; multiple (mileage, horsepower, width, pct_is_new): OLS 0.876, DT 0.903, KNN 0.913.

### Module 4 outputs
- Encoding: one item per attribute (equal-frequency tertiles for skewed numerics, domain bins for savings, presence items for salvage/theft, Module 1 category groupings); items with support > 0.9 dropped (`trans=a`). Algorithm: Apriori (mlxtend FP-Growth ~70× slower on this dense data).
- Descriptive mining: `min_sup` = 0.05 (6,835 itemsets, 1,042 maximal), `min_conf` = 0.6 (34,233 rules, 99.8% with lift > 1).
- Rule-based rating classifier (trained on the M3 train split, CV-tuned): "most discriminative matching antecedent" aggregation, s = 0.005, L = 4 → test macro-F1 0.433 (ties DT/KNN); CBA-style per-class best-rule aggregation fails (≈ 0.24–0.28).
- Imputation of masked values: price tertile 75.2% vs 35.1% mode baseline; body type 65.0% vs 64.2% (its determinants are not encoded).
- Segment rules describe the M2 K-Means clusters (confidence 0.80–1.00).

### Section 5 outputs (notebook 05, report)
- `notebooks/05_quality_extensions.ipynb` + `src/m5_utils.py`: cluster null model, stability and X-Means; M3 ablations; 12 engineered features; tunable prior correction P(c|x)/P(c)^τ; nested CV; bootstrap CIs; ordinal, learning-curve and agreement analyses; binning and regression ablations. Modules 1–4 outputs are unchanged.
- **Do not set `RECOMPUTE = True` on a laptop without care:** the full run needs several GB of RAM and ~15 min of CPU (keep `N_JOBS` ≤ 2). The default loads `tables/m5_*.csv` and takes seconds.
- Extended classifiers (test macro-F1, 95% CI): KNN τ = 0.8 + engineered **0.464** [0.443, 0.483]; DT τ = 0.7 (16 leaves) 0.444; statistically level with the random forest (0.475) and gradient boosting (0.466), used only as references. Cost: lower `low` recall.
- **Simpson's paradox:** odds ratio of `high` for franchise vs independent is 0.53 crude, 1.73 adjusted for size deciles. Seller size, not dealer type, drives the rating.
- Random forest, gradient boosting, nested CV, bootstrap, ordinal decomposition and Mantel–Haenszel may be outside the lectures: **email the instructor before submitting** (Section 2, rule 5).
- Report: 18 counted pages. Fill in `report/authors.tex` and rename the PDF before submission.

### Engineering & Oral Exam Discipline
- **Reproducibility:** Fix `random_state=42` across all splits, CV folds, and stochastic algorithms.
- **Zero Data Leakage:** Fit all imputers, scalers, encoders, discretizers, and rule extractors strictly on training partitions whenever evaluating predictive performance.
- **Unified Visual Style:** Export high-resolution vector plots (`PDF`/`SVG`) with consistent fonts, legible axis labels, and unified class/cluster color palettes.
- **Shared Understanding:** Every team member should understand all 4 modules, including algorithms, formulas, and every methodological choice made in the report.

---

## 7. Project Quality & Compliance Checklist

- [ ] **Submission:** mid-draft by 15/11/2026, final by 31/12/2026; emailed to both instructors with the correct subject and `Project_Surname1_Surname2.pdf` file name.
- [ ] **Report Constraints:** ≤20 counted pages (excluding Title Page & Index); font size >9pt; PDF only; every plot/table is legible, referenced, and commented; every choice and tested parameter range is documented and justified.
- [ ] **Module 1:** Variable semantics table; univariate & bivariate stats/plots; missing values, duplicates, inconsistencies, and outliers audited and resolved; transformations justified; correlation heatmap and feature elimination explained.
- [ ] **Module 2:** K-Means (mandatory), DBSCAN (mandatory), Hierarchical with *multiple dendrograms* (mandatory); attributes, distances, and parameters justified; full comparative "best algorithm" discussion across aggregate metrics, cluster distributions, and dataset geometry.
- [ ] **Module 3:** `seller_rating_class` classified with Decision Trees, KNN, and Naïve Bayes; Confusion Matrices, Accuracy, Precision, Recall, $F_1$-score, and ROC curves reported; Single Linear Regression + Multiple Regression on the **same continuous target** with Linear + **≥2 non-linear** models evaluated by **MSE** and **$R^2$**; tree/model insights and "best" discussion included.
- [ ] **Module 4:** Frequent patterns across `min_sup` (quantitative vs. $k$/`min_sup` + qualitative); Association rules across `min_conf` (quantitative vs. $k$/`min_conf`, **confidence & lift histograms**, + qualitative); concrete rule exploitation (target prediction and/or missing-value imputation) quantitatively evaluated.
