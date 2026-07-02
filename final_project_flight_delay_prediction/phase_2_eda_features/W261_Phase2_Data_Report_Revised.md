# 3.3 The Data

## 3.3.1 Data Sources

Our flight delay prediction system integrates five primary data sources, each serving a distinct purpose in building a comprehensive modeling dataset.

**Table 3.1: Data Sources Overview**

| Source | Description | Purpose | Reference |
|--------|-------------|---------|-----------|
| **Flights (BTS TranStats)** | U.S. DOT On-Time Performance data with scheduled/actual times, cancellations, diversions, and delay indicators | Factual flight backbone with row-level timing and delay labels | [transtats.bts.gov](https://transtats.bts.gov/DatabaseInfo.asp?DB_URL=&QO_VQ=EFD) |
| **Weather (NOAA ISD)** | Global Hourly observations from NOAA NCEI's Integrated Surface Database | As-of weather enrichment at T-2h before departure | [ncei.noaa.gov](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) |
| **Station Metadata (ISD)** | Station identifiers, coordinates, elevation, and period-of-record details | Airport-to-station bridge for weather joins | [ncei.noaa.gov](https://www.ncei.noaa.gov/products/land-based-station/station-histories) |
| **Airport Codes** | IATA/ICAO identifiers with timezone and coordinates | Time conversion (local → UTC) for weather join alignment | [datahub.io](https://datahub.io/core/airport-codes) |
| **OTPW (Pre-joined)** | Databricks-provided joined ATP and Weather dataset | Phase 1 baseline and pipeline prototyping | dbfs:/mnt/mids-w261/ |

For Phase 2, we built a custom leakage-safe join rather than using the pre-joined OTPW dataset. This approach enables precise T-2h weather alignment, better airport metadata integration, and complete control over the join logic to prevent data leakage.

---

## 3.3.2 Dataset Scope and Dimensions

### Phase 2 Analysis Dataset

For Phase 2, we conduct comprehensive analysis on the **complete 2015 calendar year** dataset, representing one full year of domestic U.S. flight operations. This provides sufficient temporal coverage to capture seasonal patterns, operational dynamics, and weather variability while remaining computationally tractable for iterative development.

**Table 3.2: Final Dataset Dimensions (Checkpoint 5)**

| Dimension | Value | Details |
|-----------|-------|---------|
| **Total Flights** | 5,704,114 | 98.0% retention from raw 5.8M records |
| **Features** | 108 | 29 raw + 79 engineered |
| **Time Period** | Jan 1 – Dec 31, 2015 | 365 days, 100% coverage |
| **Airlines** | 14 | Unique carriers |
| **Airports** | 319 origin / 321 destination | Unique airport codes |
| **States** | 53 | Origin and destination states |
| **Missing Data** | 0.0% | Complete imputation achieved |
| **Target Variable** | DEP_DEL15 | Binary: 1=delayed ≥15min, 0=on-time |
| **Class Distribution** | 81.61% on-time / 18.39% delayed | 4.44:1 imbalance ratio |
| **Storage Size** | ~2.97 GB | Parquet format |

**Quarterly Distribution:**
- Q1 2015: 1,354,285 flights (23.7%)
- Q2 2015: 1,459,259 flights (25.6%)
- Q3 2015: 1,477,899 flights (25.9%)
- Q4 2015: 1,412,671 flights (24.8%)

This represents a significant scale-up from our Phase 1 prototype (3-month sample) to a production-ready dataset with 5.7M flight records—sufficient statistical power to capture rare events, seasonal patterns, and complex interactions.

---

## 3.3.3 Data Processing Pipeline

Our pipeline implements a systematic, five-stage transformation workflow designed to convert raw flight and weather data into a production-ready modeling dataset. Each stage is checkpointed to enable reproducibility, debugging, and collaborative development.

```
display(Image(filename="/dbfs/student-groups/Group_4_4/comprehensive_pipeline_analysis.png"))
```

**Table 3.3: Pipeline Stage Summary**

| Stage | Description | Key Operations | Rows | Features |
|-------|-------------|----------------|------|----------|
| **CP1: Initial Joined** | Raw OTPW flights combined with geographic data | Join flights + weather + geographic enrichment | 5,819,079 | 75 |
| **CP2: Cleaned & Imputed** | Data quality improvements applied | Remove leakage features, filter cancelled/diverted, 3-tier weather imputation | 5,704,114 | 60 |
| **CP3: Basic Features** | Foundational feature engineering | Temporal, distance, weather severity, rolling 24h metrics | 5,704,114 | 87 |
| **CP4: Advanced Features** | Complex derived features | Aircraft lag, RFM, network centrality, interactions, cyclic encodings, Breiman meta-features | 5,704,114 | 156 |
| **CP5: Final Clean** | Production-ready optimization | Remove 46 correlated features, index 12 categoricals, verify 0% missing | 5,704,114 | 108 |

### Checkpoint Strategy

All checkpoints are stored at `dbfs:/student-groups/Group_4_4/` with the following benefits:

- **Resumability**: Team members can start from any intermediate stage without re-running earlier steps
- **Debugging**: Errors can be traced to specific pipeline stages
- **Collaboration**: Multiple team members can work on different stages simultaneously
- **Computational Efficiency**: Expensive operations (joins, aggregations) are computed once and cached

---

## 3.3.4 Data Quality and Missing Data Analysis

### Quality Metrics Across Pipeline

Our systematic data processing pipeline achieved comprehensive quality improvement, eliminating all missing data while preserving 98% of flight records and ensuring zero data leakage.

**Table 3.4: Data Quality Metrics Across Pipeline Stages**

| Metric | CP1: Initial | CP2: Cleaned | CP3: Basic | CP4: Advanced | CP5: Final | Improvement |
|--------|--------------|--------------|------------|---------------|------------|-------------|
| **Total Rows** | 5,819,079 | 5,704,114 | 5,704,114 | 5,704,114 | **5,704,114** | 98.0% retained |
| **Total Features** | 75 | 60 | 87 | 156 | **108** | +33 net features |
| **Missing Data %** | 10.16% | 0.00% | 0.00% | 0.55% | **0.00%** | 100% reduction |
| **Target Nulls** | 86,153 | 0 | 0 | 0 | **0** | Complete |
| **Data Leakage Features** | 15 | 0 | 0 | 0 | **0** | All removed |
| **Categorical (unindexed)** | 16 | 14 | 14 | 14 | **0** | All indexed |

### Missing Data Analysis

```
display(Image(filename="/dbfs/student-groups/Group_4_4/missing_data_comprehensive_analysis.png"))
```

**Initial State (CP1):**
- Overall missing: 10.16% across 51 features
- Most affected: Weather features (HourlyWindGustSpeed: 78%, HourlyPresentWeatherType: 62%)
- Least affected: Core identifiers and scheduled times (<0.01% missing)

**CP4 Temporary Missing (0.55%):**
During advanced feature engineering, 6 derived RFM features temporarily showed missing values due to sparse historical data for new routes (e.g., `days_since_last_delay_route`, `route_delays_30d`). These were resolved in CP5 with appropriate defaults (999 days for "no prior delay", 0 for "no delays counted", global median for rates).

**Final State (CP5):**
- Missing data: 0.00% across all 108 features
- Target variable completeness: 100%
- Production status: Ready for model training

### 3-Tier Weather Imputation Strategy

Our imputation strategy prioritizes temporal relevance while preventing information leakage:

| Tier | Method | Coverage | Rationale |
|------|--------|----------|-----------|
| **Tier 1** | Actual observed values | ~90% | No transformation required |
| **Tier 2** | 24-hour rolling average at same airport | ~8% | Weather exhibits strong temporal autocorrelation; uses only historical observations (1-24 hours prior) |
| **Tier 3** | Global median | ~2% | Neutral baseline for new airports or extended data gaps |

**Features Imputed (12 total):** HourlyDryBulbTemperature, HourlyDewPointTemperature, HourlyWetBulbTemperature, HourlyPrecipitation, HourlyWindSpeed, HourlyWindDirection, HourlyWindGustSpeed, HourlyVisibility, HourlyRelativeHumidity, HourlyStationPressure, HourlySeaLevelPressure, HourlyAltimeterSetting

---

## 3.3.5 Target Variable and Class Balance

### Target Definition

Our binary classification task predicts **DEP_DEL15**, where:
- **DEP_DEL15 = 1**: Flight delayed ≥15 minutes from scheduled departure
- **DEP_DEL15 = 0**: Flight on-time (delayed <15 minutes)

This 15-minute threshold aligns with the U.S. Department of Transportation's official definition of flight delay used in carrier performance reporting.

**Table 3.5: Target Variable Distribution**

| Class | Count | Percentage | Description |
|-------|-------|------------|-------------|
| **On-Time (0)** | 4,655,123 | 81.61% | Flights departing <15min late |
| **Delayed (1)** | 1,048,991 | 18.39% | Flights departing ≥15min late |
| **Total** | 5,704,114 | 100.00% | Complete 2015 dataset |

**Imbalance Ratio:** 4.44:1 (on-time : delayed)

### Class Imbalance Mitigation Strategies

The 4.44:1 imbalance is substantial but manageable through multiple complementary approaches:

1. **Class Weighting:** Apply inverse frequency weights during model training (weight_0 = 1.0, weight_1 = 4.44)
2. **SMOTE Oversampling:** Synthetic Minority Over-sampling Technique applied to training data only
3. **Threshold Tuning:** Adjust decision threshold from default 0.5 to optimize F₀.₅-score
4. **Ensemble Methods:** Tree-based models naturally handle imbalance via sample weighting
5. **Evaluation Metrics:** Prioritize F₀.₅-score, precision, and precision-recall AUC over accuracy

**Business Justification:** From a business perspective, false negatives (predicting on-time when actually delayed) are more costly than false positives. Airlines can proactively notify passengers, adjust crew scheduling, and optimize aircraft rotation when delays are predicted, even if some predictions are false alarms. Therefore, we optimize for high recall on the delayed class.

---

## 3.3.6 Exploratory Data Analysis

Our EDA reveals critical patterns in temporal dynamics, carrier performance, and geographic distributions that inform feature engineering and modeling strategies.

### Temporal Patterns

```
display(Image(filename="/dbfs/student-groups/Group_4_4/temporal_patterns_analysis.png"))
```

**Key Temporal Insights:**

| Pattern | Finding | Implication |
|---------|---------|-------------|
| **Quarterly Seasonality** | Delay rates increase from 16.0% (Q1) to 20.4% (Q4) | Winter weather and holiday travel compound operational challenges |
| **Day-of-Week Effects** | Thursday (19.5%) and Friday (19.0%) highest; Sunday lowest (16.3%) | Business travel concentration and cascading mid-week delays |
| **Within-Day Accumulation** | Delay rates increase from 6.2% at 6AM to 27.3% by 11PM | 21 percentage-point increase validates cascading delay hypothesis |
| **Weekend vs. Weekday** | Weekday delays (18.8%) exceed weekend (16.9%) by 1.9pp | Higher operational tempo during business days |

**Feature Engineering Implications:** Time-of-day features are essential; rolling 24h metrics capture cascading effects; day-of-week indicators distinguish business vs. leisure patterns; cyclic encodings preserve hour-to-hour continuity.

### Carrier Performance

```
display(Image(filename="/dbfs/student-groups/Group_4_4/carrier_performance_analysis.png"))
```

**Key Carrier Insights:**

| Metric | Finding |
|--------|---------|
| **Performance Range** | Delay rates range from 7.3% (Hawaiian) to 27.3% (Envoy Air)—a 20pp spread |
| **Volume vs. Performance** | Large carriers (WN, DL, AA) maintain moderate delay rates (15-18%) despite high volume (>800k flights), suggesting economies of scale |
| **Best Performers** | Hawaiian (7.3%), Alaska (11.4%), Virgin America (13.3%) |
| **Worst Performers** | Envoy Air (27.3%), Frontier (24.5%), ExpressJet (23.5%) |
| **Market Share** | Top 5 carriers represent 72% of flights; Southwest dominates at 20.9% |

### Geographic Patterns

```
display(Image(filename="/dbfs/student-groups/Group_4_4/geographic_patterns_analysis.png"))
```

**Key Geographic Insights:**

| Pattern | Finding |
|---------|---------|
| **Highest Delay Airports** | Newark (EWR, 29.3%), LaGuardia (LGA, 25.7%), JFK (23.4%) |
| **Best Performing Airports** | Phoenix (PHX, 11.5%), Salt Lake City (SLC, 13.2%), Seattle (SEA, 14.6%) |
| **Volume-Delay Relationship** | No simple correlation—Phoenix handles 200k+ flights at 11.5% delay rate while Newark has moderate volume but 29.3% delays |
| **State-Level Distribution** | Mean delay rate 17.9%; range 11% to 25% with slight right skew |

**Feature Engineering Implications:** Airport-indexed features capture infrastructure efficiency; state-level features aggregate regional patterns; network centrality captures hub congestion; rolling airport-level metrics capture real-time congestion state.

### Weather Correlations

Preliminary correlation analysis between weather features and delays:

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| **HourlyVisibility** | -0.15 | Lower visibility associated with higher delays |
| **HourlyPrecipitation** | +0.08 | Precipitation increases delay likelihood |
| **HourlyWindSpeed** | +0.05 | High winds increase delays |
| **HourlyDryBulbTemperature** | ~0 overall | U-shaped pattern—extreme temperatures (hot and cold) show slight delay increase |

These moderate correlations suggest weather is a contributing but not dominant factor, emphasizing the need for comprehensive feature engineering that captures operational factors alongside weather conditions.

---

## 3.3.7 Custom Flights × Weather Join

### Data Integration Challenges

Our exploratory review identified key blockers in the lookup tables rather than the flights themselves:

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Missing Timezones** | Original airport codes file lacks timezones; flight local times cannot align to UTC weather | Build master airport dimension by joining GitHub timezone data with existing codes |
| **Coordinate Parsing** | Airport geolocation packed as single text field | Parse and standardize lat/lon for all airports |
| **Station-Based Weather** | NOAA data is station-based, not airport-based; no native key for airport-to-weather connection | Create airport → station bridge using nearest-station matching |
| **Time Alignment** | Weather is UTC while flights are local time; some flights depart at odd minutes (e.g., 01:59) where no weather row exists | Convert to UTC using airport timezone; floor to hour with fallback |
| **Destination Leakage** | Destination weather can leak future information | Only keep observations where obs_time_utc ≤ prediction_utc |

### As-Of Join Logic

Our custom join pipeline:

1. **Normalize flight times** using master airports dimension (IATA, timezone, lat/lon)
2. **Convert to UTC** for weather alignment
3. **Link airports to nearest 3 NOAA stations** via haversine distance
4. **Filter to hourly report types** (FM-15, FM-16, FM-12)
5. **Apply as-of rule**: obs_utc ≤ prediction_utc with 6-hour lookback
6. **Select latest qualifying observation**, preferring station rank 1→3
7. **Preserve provenance fields**: station ID, observation timestamp, station distance, as-of minutes
8. **Exclude leakage features**: actual times, taxi times, delay causes

**Output Location:** `dbfs:/student-groups/Group_4_4/checkpoint_1_initial_joined_2015.parquet`

---

## 3.3.8 Feature Engineering

### Feature Transformation Overview

**Table 3.6: Feature Transformation Methods**

| Transformation | Features Affected | Method | Rationale | Stage |
|----------------|-------------------|--------|-----------|-------|
| **String Indexing** | Carrier, airports, states, weather types (12 features) | StringIndexer | Converts categoricals to numeric indices for ML | CP5 |
| **Cyclic Encoding** | Time and direction (9 features) | Sin/cos transformation | Preserves periodicity (23:59 → 00:01 = 2 min, not 1438) | CP4 |
| **Binning** | Distance, time-of-day (19 features) | Category-based binning | Captures non-linear effects (e.g., extreme weather, long distances) | CP3-4 |
| **Rolling Window Aggregation** | Delay rates, congestion (8 features) | Window functions with temporal constraints | Captures temporal patterns without data leakage | CP3-4 |
| **Interaction Terms** | Distance×weather, time×congestion (5 features) | Multiplicative interaction | Captures non-linear relationships between feature pairs | CP4 |
| **Dimensionality Reduction** | 46 features removed (correlation >0.8) | Correlation-based selection | Removes redundancy, reduces multicollinearity | CP5 |
| **Meta-Feature Generation** | rf_prob_delay, rf_prob_delay_binned | Random Forest probability predictions | Breiman's method—captures complex non-linear patterns | CP4 |

### Feature Engineering Progression

**Table 3.7: Feature Count Evolution by Category**

| Stage | Temporal | Distance | Weather | Rolling | Aircraft Lag | Network | RFM | Interactions | Cyclic | Breiman | Indexed Cat | Total Change |
|-------|----------|----------|---------|---------|--------------|---------|-----|--------------|--------|---------|-------------|--------------|
| **CP3** | +15 | +4 | +2 | +8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **+27** |
| **CP4** | 0 | +3 | +2 | 0 | +5 | +4 | +8 | +13 | +10 | +2 | 0 | **+69** |
| **CP5** | -1 | -4 | -6 | -2 | -3 | -2 | -4 | -8 | -1 | 0 | +12 | **-48** |
| **Net** | +14 | +3 | -2 | +6 | +2 | +2 | +4 | +5 | +9 | +2 | +12 | **+48** |

### Key Transformation Highlights

1. **Weather Imputation Strategy:** Our 3-tier imputation preserves temporal autocorrelation by preferring recent values over global statistics, reducing information loss from 10.16% missing to 0% while maintaining realistic weather patterns.

2. **Cyclic Encoding Rationale:** Time and direction are circular variables where the distance between 23:59 and 00:01 should be 2 minutes, not 1,438 minutes. Sin/cos transformations preserve this topology for 9 temporal and directional features.

3. **Breiman's Method:** Following Leo Breiman's stacked generalization approach, we use Random Forest probability predictions as meta-features, allowing linear models to leverage complex non-linear decision boundaries learned by tree ensembles.

4. **Correlation-Based Selection:** We removed 46 features with Pearson correlation >0.8, prioritizing features with higher target correlation and domain interpretability.

5. **No Data Leakage:** All transformations respect the T-2h prediction cutoff. Rolling windows use `RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '2' HOUR PRECEDING` to exclude same-flight information.

---

## 3.3.9 Feature Families and Data Dictionary

### Feature Family Distribution

```
display(Image(filename="/dbfs/student-groups/Group_4_4/feature_family_summary.png"))
```

**Table 3.8: Feature Families Summary**

| Feature Family | Count | Description |
|----------------|-------|-------------|
| **Temporal** | 49 | Dates, hours, days, quarters, seasonal indicators, time-of-day categories, cyclic encodings |
| **Geographic** | 32 | Airport identifiers, coordinates, states, distances, network centrality |
| **Weather** | 20 | Raw measurements (temperature, wind, visibility, precipitation, pressure) and derived features |
| **Binary Indicators** | 19 | Boolean flags for conditions (weekend, peak times, extreme weather, distance categories) |
| **Indexed Categorical** | 12 | String features converted to numeric indices |
| **RFM Features** | 12 | Recency, Frequency, Monetary-proxy features for routes, carriers, airports |
| **Rolling Aggregations** | 8 | 24h and 30-day windowed delay rates and volumes |
| **Cyclic Encoded** | 9 | Sin/cos transformations preserving periodicity |
| **Congestion Metrics** | 6 | Airport traffic density, flight counts, congestion indicators |
| **Interaction Terms** | 5 | Multiplicative features (distance×weather, time×volume) |
| **Aircraft Features** | 4 | Previous flight information, turnaround time, first-flight indicator |
| **Network Features** | 3 | Graph-based centrality metrics, carrier consistency |
| **Breiman Features** | 2 | Meta-features from Random Forest predictions |
| **Historical Features** | 2 | Prior-day delay rates, same-day accumulated patterns |
| **Target** | 1 | DEP_DEL15 |
| **Total** | **108** | 29 raw + 79 engineered |

### Complete Data Dictionary

The comprehensive data dictionary is provided in **Appendix A**. Below we summarize the key feature categories:

**Raw Features (29):**
- **Temporal identifiers:** FL_DATE, prediction_utc, origin_obs_utc, YEAR, QUARTER, DAY_OF_MONTH, DAY_OF_WEEK, CRS_ARR_TIME, asof_minutes
- **Flight identifiers:** OP_CARRIER_FL_NUM
- **Airport identifiers:** ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID
- **Weather measurements (12):** HourlyDryBulbTemperature, HourlyPrecipitation, HourlyWindDirection, HourlyWindGustSpeed, HourlyVisibility, HourlyRelativeHumidity, HourlyStationPressure, HourlyAltimeterSetting, etc.
- **Geographic coordinates (6):** origin_airport_lat/lon, dest_airport_lat/lon, origin_station_dis, dest_station_dis
- **Target and reference:** DEP_DEL15 (target), DEP_DELAY (reference), CANCELLATION_CODE

**Engineered Features (79) - By Method:**
1. **Aircraft Lag (4):** Previous flight delay status, turnaround time, first-flight indicator
2. **Breiman Meta-Features (2):** Random forest probability predictions
3. **Congestion Metrics (6):** Airport traffic density, real-time delay counts
4. **Cyclic Encodings (9):** Sin/cos for time and wind direction
5. **Distance Features (4):** Log distance, categorical bins
6. **Historical Patterns (2):** Prior-day delay rates, same-day accumulation
7. **Interaction Terms (5):** Distance×weather, time×volume, weather×delays
8. **Indexed Categoricals (12):** Carrier, airport, state, weather condition indices
9. **Network Features (3):** Degree centrality, carrier consistency
10. **RFM Features (9):** Recency/frequency/monetary for routes and carriers
11. **Rolling Aggregations (5):** 24h and 30-day windowed metrics
12. **Temporal Features (8):** Time-of-day categories, weekend flags, seasonal indicators
13. **Weather Features (8):** Weather lag, extreme conditions, rapid changes, anomalies

---

## 3.3.10 Dataset Sizes and Storage

### Storage Metrics Across Pipeline

**Table 3.9: Dataset Sizes and Storage Requirements**

| Checkpoint | File Name | Rows | Columns | Total Cells | Size (GB) | Avg Cell (Bytes) |
|------------|-----------|------|---------|-------------|-----------|------------------|
| **CP1** | checkpoint_1_initial_joined_2015.parquet | 5,819,079 | 75 | 436,430,925 | 2.85 | 6.86 |
| **CP2** | checkpoint_2_cleaned_imputed_2015.parquet | 5,704,114 | 60 | 342,246,840 | 2.21 | 6.78 |
| **CP3** | checkpoint_3_basic_features_2015.parquet | 5,704,114 | 87 | 496,257,918 | 3.12 | 6.60 |
| **CP4** | checkpoint_4_advanced_features_2015.parquet | 5,704,114 | 156 | 889,841,784 | 5.58 | 6.58 |
| **CP5** | **checkpoint_5_final_clean_2015.parquet** | **5,704,114** | **108** | **616,044,312** | **2.97** | **5.06** |
| **Total** | All checkpoints | — | — | — | **16.73** | — |

### Storage Insights

**Parquet Compression Performance:**
- Average cell size: 5-7 bytes demonstrates excellent columnar compression
- Compression ratio: ~10x vs. raw CSV format
- String indexing in CP5 reduces storage by 46% (5.58 GB → 2.97 GB from CP4)

**Scalability Projections:**
- Current 1-year dataset (2015): 2.97 GB, 5.7M flights, 108 features
- Projected 5-year dataset (2015-2019): ~15 GB, ~28.5M flights, 108 features
- Feasibility: Manageable on Databricks cluster with 32GB+ RAM per node

**Storage Location:**
- All checkpoints: `dbfs:/student-groups/Group_4_4/`
- Primary modeling dataset: `checkpoint_5_final_clean_2015.parquet`

---

## 3.3.11 Final Dataset Validation and Production Readiness

```
display(Image(filename="/dbfs/student-groups/Group_4_4/checkpoint5_final_analysis.png"))
```

### Production Readiness Checklist

| Validation | Status | Details |
|------------|--------|---------|
| **All features numeric** | ✓ | ML-compatible (except reference columns) |
| **Zero data leakage** | ✓ | T-2h compliance verified; 15 post-departure features removed |
| **Complete imputation** | ✓ | 0.00% missing (from initial 10.16%) |
| **Target completeness** | ✓ | 100% (zero nulls in DEP_DEL15) |
| **No duplicates** | ✓ | Verified zero duplicate records |
| **Categorical encoding** | ✓ | 12 string features indexed to numeric |
| **Optimal storage** | ✓ | 2.97 GB Parquet format |
| **Checkpointed** | ✓ | Reproducible across 5 stages |

### Critical Features Validation (All 0% Missing)

- **DEP_DEL15** (target variable)
- **prev_flight_dep_del15** (aircraft lag—top predictor, correlation: 0.373)
- **dep_delay15_24h_rolling_avg_by_origin_weighted** (rolling metric)
- **rf_prob_delay** (Breiman meta-feature)
- **extreme_weather_score** (weather composite)
- **origin_degree_centrality** (network metric)
- **OP_UNIQUE_CARRIER_indexed, ORIGIN_indexed, DEST_indexed** (categorical identifiers)

**This comprehensive validation confirms our dataset is production-ready for Phase 2 modeling experiments, including baseline model development, cross-validation with temporal splits, and hyperparameter tuning.**

---

# Appendix A: Complete Feature Inventory

This appendix provides a comprehensive listing of all 108 features in the final dataset (Checkpoint 5), organized by category.

## A.1 Target Variable (1 feature)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| DEP_DEL15 | Binary | Flight departure delay indicator (1=delayed ≥15min, 0=on-time) | FINAL |

## A.2 Temporal Features (26 features in final dataset)

### Core Temporal Identifiers (9 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| FL_DATE | Date | Flight date | FINAL |
| prediction_utc | Timestamp | Prediction timestamp (T-2h before scheduled departure) | FINAL |
| origin_obs_utc | Timestamp | Origin weather observation timestamp | FINAL |
| YEAR | Integer | Year of flight | FINAL |
| QUARTER | Integer | Quarter of year (1-4) | FINAL |
| DAY_OF_MONTH | Integer | Day of month (1-31) | FINAL |
| DAY_OF_WEEK | Integer | Day of week (1=Monday, 7=Sunday) | FINAL |
| CRS_ARR_TIME | Integer | Scheduled arrival time (HHMM format) | FINAL |
| asof_minutes | Numeric | Minutes between weather observation and prediction time | FINAL |

### Engineered Temporal Features (10 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| departure_dayofweek | Integer | Derived day of week | FINAL |
| departure_hour_weighted | Numeric | Importance-weighted departure hour | FINAL |
| is_weekend | Binary | Weekend indicator (Saturday/Sunday) | FINAL |
| is_peak_month | Binary | Peak travel month indicator (Jun-Aug) | FINAL |
| is_holiday_month | Binary | Holiday travel month indicator (Nov-Dec) | FINAL |
| time_of_day_early_morning | Binary | Early morning departure (5-8am) | FINAL |
| time_of_day_morning | Binary | Morning departure (9-11am) | FINAL |
| time_of_day_afternoon | Binary | Afternoon departure (12-5pm) | FINAL |
| time_of_day_evening | Binary | Evening departure (6-10pm) | FINAL |
| time_of_day_night | Binary | Night departure (after 10pm or before 5am) | FINAL |

### Cyclic Temporal Encodings (7 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| dep_time_sin | Numeric | Sine encoding of departure time (preserves 24h periodicity) | FINAL |
| dep_time_cos | Numeric | Cosine encoding of departure time | FINAL |
| arr_time_sin | Numeric | Sine encoding of arrival time | FINAL |
| day_of_week_sin | Numeric | Sine encoding of day of week | FINAL |
| day_of_week_cos | Numeric | Cosine encoding of day of week | FINAL |
| month_sin | Numeric | Sine encoding of month | FINAL |
| month_cos | Numeric | Cosine encoding of month | FINAL |

## A.3 Geographic Features (16 features in final dataset)

### Airport Identifiers (6 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| ORIGIN_AIRPORT_ID | Integer | Origin airport unique ID | FINAL |
| DEST_AIRPORT_ID | Integer | Destination airport unique ID | FINAL |
| ORIGIN_indexed | Numeric | Origin airport IATA code (indexed, 319 categories) | FINAL |
| DEST_indexed | Numeric | Destination airport IATA code (indexed, 321 categories) | FINAL |
| ORIGIN_STATE_ABR_indexed | Numeric | Origin state abbreviation (indexed, 53 categories) | FINAL |
| DEST_STATE_ABR_indexed | Numeric | Destination state abbreviation (indexed, 53 categories) | FINAL |

### Geographic Coordinates (6 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| origin_airport_lat | Numeric | Origin airport latitude | FINAL |
| origin_airport_lon | Numeric | Origin airport longitude | FINAL |
| dest_airport_lat | Numeric | Destination airport latitude | FINAL |
| dest_airport_lon | Numeric | Destination airport longitude | FINAL |
| origin_station_dis | Numeric | Distance from origin airport to weather station (km) | FINAL |
| dest_station_dis | Numeric | Distance from destination airport to weather station (km) | FINAL |

### Network Centrality Features (2 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| origin_degree_centrality | Numeric | Origin airport network connectivity (0-1) | FINAL |
| dest_degree_centrality | Numeric | Destination airport network connectivity (0-1) | FINAL |

### Indexed Airport Categories (2 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| origin_type_indexed | Numeric | Origin airport type (indexed, 3 categories) | FINAL |
| season_indexed | Numeric | Season (indexed, 4 categories) | FINAL |

## A.4 Weather Features (20 features in final dataset)

### Raw Weather Measurements (8 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| HourlyDryBulbTemperature | Numeric | Ambient temperature (°F) | FINAL |
| HourlyPrecipitation | Numeric | Precipitation amount (inches) | FINAL |
| HourlyWindDirection | Numeric | Wind direction (degrees) | FINAL |
| HourlyWindGustSpeed | Numeric | Wind gust speed (mph) | FINAL |
| HourlyVisibility | Numeric | Visibility distance (miles) | FINAL |
| HourlyRelativeHumidity | Numeric | Relative humidity (%) | FINAL |
| HourlyStationPressure | Numeric | Station pressure (inHg) | FINAL |
| HourlyAltimeterSetting | Numeric | Altimeter setting (inHg) | FINAL |

### Engineered Weather Features (7 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| weather_lag_hours | Numeric | Hours since last weather observation | FINAL |
| rapid_weather_change | Binary | Rapid temperature or wind change in 3-hour window | FINAL |
| temp_anomaly | Numeric | Deviation from monthly average temperature | FINAL |
| extreme_precipitation | Binary | Extreme precipitation (>95th percentile) | FINAL |
| extreme_wind | Binary | Extreme wind speed (>95th percentile) | FINAL |
| extreme_temperature | Binary | Extreme temperature (<5th or >95th percentile) | FINAL |
| extreme_weather_score | Numeric | Weighted extreme weather composite score (0-1) | FINAL |

### Cyclic Weather Encodings (2 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| wind_direction_sin | Numeric | Sine encoding of wind direction | FINAL |
| wind_direction_cos | Numeric | Cosine encoding of wind direction | FINAL |

### Indexed Weather Categories (3 features)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| HourlyPresentWeatherType_indexed | Numeric | Present weather codes (indexed, 620 categories) | FINAL |
| sky_condition_parsed_indexed | Numeric | Sky condition category (indexed, 6 categories) | FINAL |
| weather_condition_category_indexed | Numeric | Weather severity category (indexed, 3 categories) | FINAL |

## A.5 Distance Features (4 features in final dataset)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| log_distance | Numeric | Log-transformed flight distance | FINAL |
| distance_medium | Binary | Medium distance flight (500-1000 miles) | FINAL |
| distance_long | Binary | Long distance flight (1000-2000 miles) | FINAL |
| distance_very_long | Binary | Very long distance flight (>2000 miles) | FINAL |

## A.6 Rolling/Temporal Aggregation Features (8 features in final dataset)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| rolling_origin_num_delays_24h | Numeric | Number of delays at origin in past 24 hours | FINAL |
| dep_delay15_24h_rolling_avg_by_origin_dayofweek | Numeric | Rolling 24h delay rate by origin and day of week | FINAL |
| dep_delay15_24h_rolling_avg_by_origin_weighted | Numeric | Importance-weighted rolling 24h delay rate | FINAL |
| dep_delay15_24h_rolling_avg_by_origin_carrier_weighted | Numeric | Importance-weighted rolling 24h delay rate by origin-carrier | FINAL |
| rolling_30day_volume | Numeric | 30-day flight volume at origin | FINAL |
| route_delay_rate_30d | Numeric | 30-day delay rate for specific route | FINAL |
| carrier_delays_at_origin_30d | Numeric | Carrier delays at origin in past 30 days | FINAL |
| route_delays_30d | Numeric | Delays on route in past 30 days | FINAL |

## A.7 Aircraft Lag Features (4 features in final dataset)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| prev_flight_dep_del15 | Numeric | Previous flight delay status (same aircraft) | FINAL |
| prev_flight_crs_elapsed_time | Numeric | Previous flight scheduled duration | FINAL |
| hours_since_prev_flight | Numeric | Aircraft turnaround time in hours | FINAL |
| is_first_flight_of_aircraft | Binary | First flight of aircraft today | FINAL |

## A.8 RFM Features (12 features in final dataset)

### Recency, Frequency, Monetary-Proxy Features

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| last_delay_date | Date | Most recent delay date for route | FINAL |
| days_since_last_delay_route | Numeric | Days since route last had delay | FINAL |
| days_since_carrier_last_delay_at_origin | Numeric | Days since carrier had delay at origin | FINAL |
| route_delays_30d | Numeric | Number of delays on route in past 30 days | FINAL |
| carrier_delays_at_origin_30d | Numeric | Number of carrier delays at origin in past 30 days | FINAL |
| route_1yr_volume | Numeric | 1-year flight volume on route | FINAL |
| origin_1yr_delay_rate | Numeric | 1-year historical delay rate at origin | FINAL |
| dest_1yr_delay_rate | Numeric | 1-year historical delay rate at destination | FINAL |
| route_delay_rate_30d | Numeric | 30-day delay rate for route | FINAL |

## A.9 Congestion Features (6 features in final dataset)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| airport_traffic_density | Numeric | Percentage of daily flights in this hour | FINAL |
| carrier_flight_count | Numeric | Total flights by carrier | FINAL |
| num_airport_wide_delays | Numeric | Delays at airport in 2-hour window | FINAL |
| oncoming_flights | Numeric | Arrivals at origin in 2-hour window | FINAL |
| prior_flights_today | Numeric | Flights at origin so far today | FINAL |
| time_based_congestion_ratio | Numeric | Current vs historical traffic ratio | FINAL |

## A.10 Network Features (3 features in final dataset)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| origin_degree_centrality | Numeric | Origin airport network connectivity (0-1) | FINAL |
| dest_degree_centrality | Numeric | Destination airport network connectivity (0-1) | FINAL |
| carrier_delay_stddev | Numeric | Carrier delay consistency metric | FINAL |

## A.11 Interaction Terms (5 features in final dataset)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| distance_x_weather_severity | Numeric | Distance × extreme_weather_score | FINAL |
| weekend_x_route_volume | Numeric | Weekend × route_1yr_volume | FINAL |
| weather_x_airport_delays | Numeric | Weather × num_airport_wide_delays | FINAL |
| temp_x_holiday | Numeric | Temperature × is_holiday_month | FINAL |
| route_delay_rate_x_peak_hour | Numeric | Route delay rate × peak hour | FINAL |

## A.12 Cyclic Encoded Features (9 features in final dataset)

Time and direction features with sin/cos encodings to preserve periodicity (listed in Temporal and Weather sections above).

## A.13 Breiman Meta-Features (2 features in final dataset)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| rf_prob_delay | Numeric | Random Forest predicted probability of delay | FINAL |
| rf_prob_delay_binned | Numeric | Binned RF delay probability (5 bins) | FINAL |

## A.14 Indexed Categorical Features (12 features in final dataset)

| Feature | Original | Cardinality | Status |
|---------|----------|-------------|--------|
| OP_UNIQUE_CARRIER_indexed | OP_UNIQUE_CARRIER | 14 | FINAL |
| ORIGIN_indexed | ORIGIN | 319 | FINAL |
| DEST_indexed | DEST | 321 | FINAL |
| ORIGIN_STATE_ABR_indexed | ORIGIN_STATE_ABR | 53 | FINAL |
| DEST_STATE_ABR_indexed | DEST_STATE_ABR | 53 | FINAL |
| HourlyPresentWeatherType_indexed | HourlyPresentWeatherType | 620 | FINAL |
| origin_type_indexed | origin_type | 3 | FINAL |
| season_indexed | season | 4 | FINAL |
| weather_condition_category_indexed | weather_condition_category | 3 | FINAL |
| turnaround_category_indexed | turnaround_category | 4 | FINAL |
| day_hour_interaction_indexed | day_hour_interaction | 168 | FINAL |
| sky_condition_parsed_indexed | sky_condition_parsed | 6 | FINAL |

## A.15 Historical Pattern Features (2 features in final dataset)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| prior_day_delay_rate | Numeric | Previous day's delay rate at origin | FINAL |
| same_day_prior_delay_percentage | Numeric | Percentage of flights delayed so far today | FINAL |

## A.16 Reference/Operational Features (3 features retained)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| OP_CARRIER_FL_NUM | Integer | Flight number | FINAL |
| DEP_DELAY | Numeric | Actual departure delay (reference only) | FINAL |
| CANCELLATION_CODE | String | Cancellation reason code | FINAL |

## A.17 Removed Features Summary

| Category | Features Removed | Stage | Reason |
|----------|------------------|-------|--------|
| **Data Leakage** | 15 features | CP2 | Actual times, post-departure operations, delay cause breakdowns, arrival outcomes |
| **High Correlation** | 46 features | CP5 | Pearson correlation >0.8 with retained features |
| **Low Utility** | 3 features | Various | CANCELLED, DIVERTED (filtered), HourlySkyConditions (115K cardinality), flight_id (unique identifier) |

**Final Feature Count:** 108 total (29 raw + 79 engineered)
