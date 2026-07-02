# Section 3.2 Project Description (Revised for Phase 2)

## Copy the content below to replace the current Section 3.2 in your Databricks notebook:

---

### 3.2 Project Description:

This report documents the development of our machine learning pipeline for predicting flight departure delays across three project phases, with Phase 2 delivering a complete, production-ready feature engineering pipeline and baseline modeling results using the full 2015 calendar year (5.7M flights, 108 features).

**Phase 2 Accomplishments:**

We have successfully transformed raw flight and weather data through a five-checkpoint pipeline that eliminates data leakage, handles missing values through domain-informed imputation, and engineers 79 features spanning temporal patterns, geographic relationships, weather conditions, and historical performance metrics. Our custom flights-to-weather join respects the T-2h prediction window using UTC-aligned as-of logic, ensuring no future information contaminates training data. Baseline classification and regression models have been trained and evaluated using time-series cross-validation, with Gradient Boosted Trees outperforming Random Forest on both F₀.₅-score and PR-AUC metrics.

**Report Structure:**

- **Section 3.3 - The Data**: Details our production dataset (5,704,114 flights, 108 features from the complete 2015 calendar year), the five-stage checkpoint pipeline from raw ingestion to modeling-ready format, comprehensive data quality analysis achieving 0% missing data, and entity-relationship diagrams documenting the custom flights-weather-airport join architecture. Includes target variable analysis (18.39% delayed, 4.44:1 imbalance ratio), class imbalance mitigation strategies, and complete feature dictionary organized by engineering method and family.

- **Section 3.3.6 - Exploratory Data Analysis (embedded)**: Presents key findings on temporal delay patterns (6.2% at 6AM to 27.3% by 11PM), quarterly seasonality (16.0% Q1 to 20.4% Q4), carrier performance spread (7.3% Hawaiian to 27.3% Envoy Air), geographic patterns (Newark 29.3% vs. Phoenix 11.5%), and weather correlations. Each pattern includes explicit implications for feature engineering and modeling.

- **Section 4 - Extended EDA**: Provides detailed visualizations and analysis of hourly delay accumulation, day-of-week effects, airport-specific delay patterns, airline performance comparisons, and weather impact assessments including temperature, wind speed, precipitation, and visibility effects on departure delays.

- **Section 5 - Machine Learning Algorithms and Modeling Strategy**: Describes our two-stage pipeline architecture where Stage 1 implements ensemble binary classification (Logistic Regression baseline, Random Forest, Gradient Boosted Trees) to predict delay occurrence, while Stage 2 applies regression models to estimate delay duration only for flights predicted as delayed. Includes mathematical formulations for each model, discussion of class weighting and threshold tuning, and future-proofing for Phase 3 requirements (time-series analysis, graph features, MLP/neural network models).

- **Section 6 - Evaluation Metrics**: Establishes PR-AUC as primary classification metric (appropriate for imbalanced datasets) and F₀.₅-score as secondary metric (prioritizing precision over recall since false delay predictions cause more operational disruption than missed delays). Includes mathematical formulations, per-segment confusion analysis by carrier and airport, regression metrics (MAE, RMSE) for Stage 2, and operational/domain-level evaluation views.

- **Section 7 - Project Timeline**: Presents Gantt chart tracking parallel workstreams across three phases including data engineering tasks, feature development activities, baseline and advanced modeling efforts, and report preparation milestones.

- **Section 8 - Machine Learning Pipeline**: Illustrates end-to-end flow from data ingestion through leakage-aware cleaning, time-series split and feature engineering (using only T-2h information), feature selection via correlation analysis and ANOVA testing, Stage 1 classification with SMOTE undersampling, evaluation using time-ordered validation, Stage 2 regression for delayed flights, and checkpoint saving for reproducibility.

- **Section 9 - Open Issues and Next Steps**: Discusses remaining challenges including scaling to multi-year data (target: 28M flights), hyperparameter optimization, advanced feature engineering (graph-based metrics, network centrality), neural network experimentation, and production deployment considerations.

- **Appendix A - Complete Feature Inventory**: Provides comprehensive listing of all 108 features organized by category (Temporal, Geographic, Weather, Binary Indicators, Indexed Categorical, RFM, Rolling Aggregations, Cyclic Encoded, Congestion, Interaction Terms, Aircraft, Network, Breiman, Historical, and Target).

- **Appendix - Code Notebooks and References**: Links to supporting Databricks notebooks for data cleaning, EDA, custom joins, and modeling pipelines, plus academic references on flight delay prediction methodologies.

**Key Technical Achievements:**

Throughout this project, we emphasize reproducibility through checkpoint-based workflows, scalability using Apache Spark's distributed computing (tested on 5.7M records with projections to 28M), and strict adherence to temporal validation protocols that prevent look-ahead bias. Our T-2h prediction window reflects realistic operational conditions where airlines must make crew scheduling, gate assignment, and passenger notification decisions before actual departure information becomes available.

---

## Notes on Changes from Phase 1:

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Dataset Size | 1.4M flights (3-month sample) | 5.7M flights (full 2015 year) |
| Features | 216 raw | 108 final (29 raw + 79 engineered) |
| Missing Data | ~10% | 0.0% (complete imputation) |
| Data Leakage | Identified | Eliminated (15 features removed) |
| Join Logic | Pre-joined OTPW | Custom T-2h aligned join |
| Baseline Models | Planned | Implemented (LR, RF, GBT) |
| Evaluation | Framework design | Actual results with time-series CV |
| Primary Metric | F₂-score (originally) | PR-AUC + F₀.₅-score |

