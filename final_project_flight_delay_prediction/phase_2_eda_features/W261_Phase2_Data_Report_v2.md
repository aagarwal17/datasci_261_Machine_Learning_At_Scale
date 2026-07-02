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

### Scalability Considerations

We read data directly into Apache Spark using `spark.read.csv()` or `spark.read.parquet()` depending on the source format. This approach leverages Spark's lazy evaluation, allowing us to work with large datasets efficiently without loading everything into memory at once. For our initial development, we used the 3-month CSV sample; for production scaling, we converted to Parquet format for improved performance and compression.

**Dataset Scale Progression:**
- 3-month sample: ~1.4M flights (Phase 1 prototyping)
- 6-month sample: ~2.8M flights
- 1-year sample: ~5.7M flights (Phase 2 production)
- Full 5-year dataset: ~28M flights (Phase 3 target)

We monitor Spark job performance, partition data appropriately, and adjust cluster resources as needed to maintain reasonable processing times.

---

## 3.3.3 Data Processing Pipeline

Our pipeline implements a systematic, five-stage transformation workflow designed to convert raw flight and weather data into a production-ready modeling dataset. Each stage is checkpointed to enable reproducibility, debugging, and collaborative development.

```
display(Image(filename="/dbfs/student-groups/Group_4_4/comprehensive_pipeline_analysis.png"))
```

**Table 3.3: Pipeline Stage Summary**

| Stage | File Name | Description | Key Operations | Rows | Features |
|-------|-----------|-------------|----------------|------|----------|
| **CP1: Initial Joined** | `checkpoint_1_initial_joined_2015.parquet` | Initial joined dataset combining OTPW flights with weather station geographic data. Includes raw weather measurements, flight schedules, and geographic coordinates. | Join flights + weather + geographic enrichment | 5,819,079 | 75 |
| **CP2: Cleaned & Imputed** | `checkpoint_2_cleaned_imputed_2015.parquet` | Cleaned dataset after data quality improvements: removed 15 leakage features, filtered cancelled/diverted flights, applied 3-tier weather imputation, converted data types. | Remove leakage, filter invalid flights, impute weather, type conversion | 5,704,114 | 60 |
| **CP3: Basic Features** | `checkpoint_3_basic_features_2015.parquet` | Enhanced dataset with basic feature engineering: temporal features (hour, day, season), distance categorization, weather severity scoring, rolling 24h delay statistics. | Temporal (15), distance (4), weather severity (2), rolling metrics (8) | 5,704,114 | 87 |
| **CP4: Advanced Features** | `checkpoint_4_advanced_features_2015.parquet` | Comprehensive feature-engineered dataset including: aircraft lag features, RFM features, network centrality metrics, interaction terms, cyclic encodings, Breiman tree-based meta-features. | Aircraft lag (5), RFM (8), network (4), interactions (13), cyclic (10), Breiman (2), congestion (6) | 5,704,114 | 156 |
| **CP5: Final Clean** | `checkpoint_5_final_clean_2015.parquet` | Production-ready dataset after feature selection and optimization: removed 46 high-correlation features, indexed 12 categorical features, dropped string columns, verified 0% missing data. | Correlation-based selection, string indexing, final validation | 5,704,114 | 108 |

### Key Transformations by Checkpoint

**CP1 → CP2: Data Cleaning and Leakage Removal**
- Removed 15 data leakage features (actual times, delay breakdowns, post-flight operations)
- Eliminated: DEP_TIME, ARR_TIME, WHEELS_OFF, WHEELS_ON, TAXI_OUT, TAXI_IN, ACTUAL_ELAPSED_TIME, AIR_TIME
- Eliminated: CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY, ARR_DEL15, ARR_DELAY
- Filtered cancelled flights (28,812 records, 0.49%) and diverted flights (eliminated as fundamentally different operational scenarios)
- Applied 3-tier weather imputation strategy
- Result: 10.16% missing → 0.00% missing

**CP2 → CP3: Basic Feature Engineering (+27 features)**
- Focus: Temporal patterns, distance categorization, rolling metrics
- Key Additions: Time-of-day categories (5), distance bins (4), rolling 24h delay metrics (8), weather severity indicators (2), temporal identifiers (8)

**CP3 → CP4: Advanced Feature Engineering (+69 features)**
- Focus: Complex interactions, historical patterns, meta-features
- Key Additions: Aircraft lag features (5), RFM features (8), network centrality (4), interaction terms (13), cyclic encodings (10), Breiman meta-features (2), additional congestion metrics (6)

**CP4 → CP5: Feature Selection and Optimization (-48 features)**
- Correlation threshold: Pearson >0.8
- String indexing: 13 categorical features converted to numeric indices
- Dropped: 60 original string columns after indexing
- Result: 156 features → 108 features (31% reduction for optimal modeling)

### Checkpoint Strategy

All checkpoints are stored at `dbfs:/student-groups/Group_4_4/` with the following benefits:

- **Resumability**: Team members can start from any intermediate stage without re-running earlier steps
- **Debugging**: Errors can be traced to specific pipeline stages
- **Collaboration**: Multiple team members can work on different stages simultaneously
- **Version Control**: Each checkpoint represents a stable state of data transformation
- **Computational Efficiency**: Expensive operations (joins, aggregations) are computed once and cached

---

## 3.3.4 Data Quality and Missing Data Analysis

### Quality Metrics Across Pipeline

Our systematic data processing pipeline achieved comprehensive quality improvement, eliminating all missing data while preserving 98% of flight records and ensuring zero data leakage.

**Table 3.4: Data Quality Metrics Across Pipeline Stages**

| Metric | CP1: Initial | CP2: Cleaned | CP3: Basic | CP4: Advanced | CP5: Final | Improvement |
|--------|--------------|--------------|------------|---------------|------------|-------------|
| **File Name** | checkpoint_1_initial_joined_2015.parquet | checkpoint_2_cleaned_imputed_2015.parquet | checkpoint_3_basic_features_2015.parquet | checkpoint_4_advanced_features_2015.parquet | checkpoint_5_final_clean_2015.parquet | — |
| **Total Rows** | 5,819,079 | 5,704,114 | 5,704,114 | 5,704,114 | **5,704,114** | 98.0% retained |
| **Rows Removed** | — | 114,965 | 0 | 0 | 0 | Cancelled/diverted/null-target only |
| **Data Retention** | 100% | 98.0% | 98.0% | 98.0% | **98.0%** | High retention maintained |
| **Total Features** | 75 | 60 | 87 | 156 | **108** | +33 net features |
| **Missing Data %** | 10.16% | 0.00% | 0.00% | 0.55% | **0.00%** | 100% reduction |
| **Target Nulls** | 86,153 | 0 | 0 | 0 | **0** | Complete |
| **Data Leakage Features** | 15 | 0 | 0 | 0 | **0** | All removed |
| **Categorical (unindexed)** | 16 | 14 | 14 | 14 | **0** | All indexed |

### Cancelled and Diverted Flights

Cancelled and diverted flights were filtered from the dataset during CP2 cleaning:

- **Cancelled flights**: 28,812 records (0.49% of total)
- **Diverted flights**: Included in the 114,965 rows removed

**Why Remove These Flights?**
Cancelled and diverted flights represent fundamentally different operational scenarios than delayed departures. A cancelled flight never departs, so predicting its departure delay is meaningless. Similarly, diverted flights involve mid-flight decisions that cannot be predicted at the T-2h departure window. Including these records would introduce noise and potentially mislead the model, as the target variable DEP_DEL15 is undefined or irrelevant for these cases. By filtering them out, we ensure our model focuses exclusively on flights that actually departed, making predictions actionable for operational decision-making.

### Missing Data Analysis

```
display(Image(filename="/dbfs/student-groups/Group_4_4/missing_data_comprehensive_analysis.png"))
```

**Initial State (CP1) — 10.16% Overall Missing:**

| Category | Features Affected | Missing Rate | Reason for Missingness |
|----------|-------------------|--------------|------------------------|
| **Weather (High)** | HourlyWindGustSpeed | 78% | Gusts only recorded during high-wind events; calm conditions have no gust measurement |
| **Weather (High)** | HourlyPresentWeatherType | 62% | Weather type codes only populated when specific conditions (rain, fog, snow) are present; clear weather often left blank |
| **Weather (Moderate)** | HourlyPrecipitation | ~15% | Precipitation only recorded when measurable; dry periods have null entries |
| **Delay Breakdowns** | CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY | ~80% | By design—only populated when delays actually occur and causes are attributed |
| **Target Variable** | DEP_DEL15 | 1.5% (86,153 nulls) | Cancelled/diverted flights have no departure delay to measure |
| **Core Identifiers** | FL_DATE, CRS_DEP_TIME, ORIGIN, DEST | <0.01% | Essential flight identifiers rarely missing |

**Why So Much Weather Data is Missing:**
Weather observations are event-driven rather than continuous. NOAA stations report certain conditions only when they occur:
- Wind gusts are only measured when gusts exceed sustained wind speed thresholds
- Present weather types (rain, fog, thunderstorm codes) are only logged during active weather events
- Precipitation amounts are null during dry periods rather than recorded as zero
- Some remote airports have limited weather station coverage, leading to temporal gaps

This pattern means that missing weather data is informative—it often indicates benign conditions rather than data quality issues.

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
| **Tier 2** | 24-hour rolling average at same airport | ~8% | Weather exhibits strong temporal autocorrelation; uses only historical observations (1-24 hours prior) via `ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING` |
| **Tier 3** | Global median | ~2% | Neutral baseline for new airports or extended data gaps; avoids introducing temporal bias |

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

1. **SMOTE Undersampling (Current Approach):** We apply undersampling of the majority class (on-time flights) to training data only, creating a more balanced training set while preserving the original test distribution for realistic evaluation.

2. **Class Weighting (Previously Tested):** We initially experimented with inverse frequency weights during model training (weight_0 = 1.0, weight_1 = 4.44), but found SMOTE undersampling provided better results for our use case.

3. **Threshold Tuning:** Adjust decision threshold from default 0.5 to optimize F₀.₅-score based on precision-recall trade-offs.

4. **Ensemble Methods:** Tree-based models (Random Forest, Gradient Boosting) naturally handle imbalance via sample weighting and bagging.

5. **Evaluation Metrics:** Prioritize F₀.₅-score, precision, and precision-recall AUC over accuracy to avoid majority-class bias.

**Business Justification:** From a business perspective, false negatives (predicting on-time when actually delayed) are more costly than false positives. Airlines can proactively notify passengers, adjust crew scheduling, and optimize aircraft rotation when delays are predicted, even if some predictions are false alarms. Therefore, we optimize for high recall on the delayed class.

---

## 3.3.6 Exploratory Data Analysis

Our EDA reveals critical patterns in temporal dynamics, carrier performance, geographic distributions, and distance effects that inform feature engineering and modeling strategies.

### Temporal Patterns

```
display(Image(filename="/dbfs/student-groups/Group_4_4/temporal_patterns_analysis.png"))
```

**Table 3.6: Key Temporal Insights**

| Pattern | Finding | Implication for Modeling |
|---------|---------|--------------------------|
| **Quarterly Seasonality** | Delay rates increase from 16.0% (Q1) to 20.4% (Q4) | Winter weather and holiday travel compound operational challenges; seasonal features and holiday indicators are essential for capturing these patterns |
| **Day-of-Week Effects** | Thursday (19.5%) and Friday (19.0%) highest; Sunday lowest (16.3%) | Business travel concentration creates mid-week congestion; day-of-week features help model distinguish business vs. leisure travel patterns |
| **Within-Day Accumulation** | Delay rates increase from 6.2% at 6AM to 27.3% by 11PM (21pp increase) | Delays cascade through the day due to aircraft rotation and airport congestion; time-of-day features and rolling delay metrics are critical for capturing this accumulation effect |
| **Flight Volume Pattern** | Peak departures 7AM-8PM (>250k flights/hour); lower overnight | Volume correlates with delay accumulation; congestion features should capture hourly traffic density relative to airport capacity |
| **Weekend vs. Weekday** | Weekday delays (18.8%) exceed weekend (16.9%) by 1.9pp | Higher operational tempo during business days creates more opportunities for cascading delays; weekend indicator helps model adjust expectations |
| **Cumulative Distribution** | 50% of total delays occur by hour 16 (4PM) | Morning and afternoon operations are critical for overall delay management; models should weight early-day predictions appropriately |

### Carrier Performance

```
display(Image(filename="/dbfs/student-groups/Group_4_4/carrier_performance_analysis.png"))
```

**Table 3.7: Key Carrier Insights**

| Metric | Finding | Implication for Modeling |
|--------|---------|--------------------------|
| **Performance Range** | Delay rates range from 7.3% (Hawaiian) to 27.3% (Envoy Air)—a 20pp spread | Carrier identity is a strong predictor; carrier-indexed features capture systematic operational efficiency differences between airlines |
| **Volume vs. Performance** | Large carriers (WN, DL, AA) maintain moderate delay rates (15-18%) despite high volume (>800k flights) | Economies of scale exist in airline operations; carrier features should be combined with volume metrics to capture this interaction |
| **Best Performers** | Hawaiian (7.3%), Alaska (11.4%), Virgin America (13.3%) | These carriers may have operational practices worth studying; their routes and hub strategies differ from high-delay carriers |
| **Worst Performers** | Envoy Air (27.3%), Frontier (24.5%), ExpressJet (23.5%) | Regional carriers and ultra-low-cost carriers face different operational constraints; carrier type features help distinguish these patterns |
| **Market Share** | Top 5 carriers represent 72% of flights; Southwest dominates at 20.9% | Model will be heavily influenced by large carrier patterns; ensure adequate representation of smaller carriers in training |
| **Performance-Consistency Matrix** | Four quadrants: Green (low delay, high consistency), Red (high delay, low consistency), Blue (consistent high delay), Orange (variable low delay) | Carrier_delay_stddev feature captures consistency—important for business travelers who value predictability |

### Geographic Patterns

```
display(Image(filename="/dbfs/student-groups/Group_4_4/geographic_patterns_analysis.png"))
```

**Table 3.8: Key Geographic Insights**

| Pattern | Finding | Implication for Modeling |
|---------|---------|--------------------------|
| **Highest Delay Airports** | Newark (EWR, 29.3%), LaGuardia (LGA, 25.7%), JFK (23.4%) | New York area airports face unique congestion and weather challenges; airport-indexed features capture these infrastructure and operational constraints |
| **Best Performing Airports** | Phoenix (PHX, 11.5%), Salt Lake City (SLC, 13.2%), Seattle (SEA, 14.6%) | These airports demonstrate that high volume doesn't require high delays; their operational efficiency should be captured by congestion ratio features |
| **Volume-Delay Relationship** | No simple correlation—Phoenix handles 200k+ flights at 11.5% while Newark has moderate volume but 29.3% | Airport delay rates depend on infrastructure, weather exposure, and operational practices, not just volume; network centrality and congestion features capture these dynamics |
| **Hub Effects** | Atlanta (ATL) handles highest volume (440k flights) with moderate 17.1% delays | Hub airports have different delay dynamics than spoke airports; origin_type_indexed distinguishes hub/spoke operations |
| **State-Level Distribution** | Mean delay rate 17.9%; range 11% to 25% with slight right skew | Regional weather patterns and regulatory environments create state-level effects; state-indexed features aggregate these patterns |
| **High Volume, High Delay States** | Illinois (19.6%), New York (20.8%), New Jersey (24.1%) | Northeastern states face weather and congestion challenges; geographic coordinates enable distance-based and regional weather features |

### Distance and Flight Duration

**Table 3.9: Distance and Duration Analysis**

| Metric | Value | Implication for Modeling |
|--------|-------|--------------------------|
| **Average Distance** | 920 miles | Provides baseline for distance categorization |
| **Distance Range** | 31 miles (shortest) to 4,983 miles (longest) | Wide range requires non-linear distance features (log transformation, categorical bins) |
| **Average Scheduled Duration** | 130 minutes | Short-haul vs. long-haul distinction affects operational dynamics |
| **Short-Haul Delay Pattern** | Shorter flights show slightly higher delay rates proportionally | 15-minute delays represent larger percentage of total flight time for short routes; tighter turnarounds create more delay propagation risk |
| **Distance-Weather Interaction** | Long-distance flights cross multiple weather systems | Distance × weather_severity interaction term captures compounding effects of weather across route length |

### Weather Correlations

Preliminary correlation analysis between weather features and delays:

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| **HourlyVisibility** | -0.15 | Lower visibility associated with higher delays; fog and low clouds directly impact airport operations |
| **HourlyPrecipitation** | +0.08 | Precipitation increases delay likelihood through ground operations slowdowns and de-icing requirements |
| **HourlyWindSpeed** | +0.05 | High winds increase delays through crosswind limitations and turbulence-related ground holds |
| **HourlyDryBulbTemperature** | ~0 overall | U-shaped pattern—extreme temperatures (hot and cold) show slight delay increase due to de-icing (cold) or aircraft performance limits (hot) |

These moderate correlations suggest weather is a contributing but not dominant factor, emphasizing the need for comprehensive feature engineering that captures operational factors (aircraft history, airport congestion, time-of-day) alongside weather conditions.

---

## 3.3.7 Custom Flights × Weather Join

### Entity-Relationship Model

The following diagram illustrates the relationships between our core data entities used for enriching flight records with meteorological data:

```
erDiagram
    %% LEGEND
    %% PK_... = primary key (or business key)
    %% FK_... = foreign key to another table

    FLIGHTS {
        string PK_flight_row
        date FL_DATE
        string FK_origin_iata_code
        string FK_dest_iata_code
        string OP_UNIQUE_CARRIER
        int OP_CARRIER_FL_NUM
        int CRS_DEP_TIME
        int CRS_ARR_TIME
        int YEAR
        int MONTH
        int DAY_OF_MONTH
    }

    MASTER_AIRPORTS {
        string PK_iata_code
        string ident
        string name
        string municipality
        string iso_country
        string iso_region
        string airport_timezone
        float lat
        float lon
    }

    WEATHER {
        string PK_station_id
        datetime PK_obs_datetime
        float LATITUDE
        float LONGITUDE
        string NAME
        float HourlyDryBulbTemperature
        float HourlyVisibility
        float HourlyWindSpeed
    }

    NOAA_STATIONS {
        string PK_station_id_norm
        float lat
        float lon
        string neighbor_id
        float distance_to_neighbor
    }

    AIRPORT_WEATHER_STATION {
        string PK_iata_code
        string PK_station_id
        int PK_rank
        float dist_km
    }

    CHECKPOINTS {
        string checkpoint_name
        string file_path
        int rows
        int columns
        string description
    }

    %% RELATIONSHIPS
    FLIGHTS }o--|| MASTER_AIRPORTS : "origin (FK_origin_iata_code)"
    FLIGHTS }o--|| MASTER_AIRPORTS : "destination (FK_dest_iata_code)"
    MASTER_AIRPORTS ||--o{ AIRPORT_WEATHER_STATION : "airport → nearest stations"
    WEATHER ||--o{ AIRPORT_WEATHER_STATION : "station in bridge"
    WEATHER }o--|| NOAA_STATIONS : "normalize/enrich station"
    FLIGHTS ||--o{ CHECKPOINTS : "pipeline stages"
```

### Data Integration Challenges

Our exploratory review identified key blockers in the lookup tables rather than the flights themselves:

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Missing Timezones** | Original airport codes file lacks timezones; flight local times cannot align to UTC weather | Build master airport dimension by joining GitHub timezone data with existing codes; coalesce coordinates so every IATA has timezone, lat/lon, and name |
| **Coordinate Parsing** | Airport geolocation packed as single text field ("lon, lat") | Parse and standardize lat/lon for all airports appearing as origin/destination |
| **Station-Based Weather** | NOAA data is station-based, not airport-based; no native key for airport-to-weather connection | Compute airport → nearest-(1..3)-station pairs using haversine distance; store in airport_weather_station bridge table |
| **Station ID Normalization** | Stations come from two slightly different sources (weather.csv vs stations.csv) | Normalize station identifiers across sources |
| **Time Alignment** | Weather is UTC while flights are local time; some flights depart at odd minutes (e.g., 01:59) where no weather row exists | Convert flight times to UTC using airport timezone; floor to hour with 1-hour fallback |
| **Odd Minute Departures** | Flights at 01:55 or 01:59 have no matching hourly weather observation | Date-trunc to hour and/or fallback 1 hour to avoid nulls |
| **Destination Leakage** | Destination weather can leak future information | Only keep observations where obs_time_utc ≤ prediction_utc |

### As-Of Join Logic

Our custom, leakage-safe join combines DOT on-time performance with NOAA Global Hourly weather at the origin airport as-of T-2h:

1. **Normalize flight times** using master airports dimension (IATA, timezone, lat/lon)
2. **Convert to UTC** for weather alignment
3. **Link airports to nearest 3 NOAA stations** via haversine distance
4. **Filter to hourly report types** (FM-15, FM-16, FM-12)
5. **Apply as-of rule**: obs_utc ≤ prediction_utc with 6-hour lookback
6. **Select latest qualifying observation**, preferring station rank 1→3
7. **Preserve provenance fields**: station ID, observation timestamp, station distance, as-of minutes
8. **Select only features valid at T-2h**: visibility, wind, gusts, precipitation, temperature, plus compact weather flags
9. **Exclude leakage features**: actual times, taxi times, delay causes, cancelled/diverted flights

**Output Locations:**
- Phase 2 modeling dataset: `dbfs:/student-groups/Group_4_4/checkpoint_1_initial_joined_2015.parquet`
- Scalable to full 2015-2021 range using same pipeline

---

## 3.3.8 Feature Engineering

### Feature Transformation Overview

**Table 3.10: Feature Transformation Methods**

| Transformation | Features Affected | Method | Rationale | Applied Stage |
|----------------|-------------------|--------|-----------|---------------|
| **String Indexing** | Carrier, airports, states, weather types (12 features) | StringIndexer | Converts categoricals to numeric indices for ML algorithms | CP5 |
| **Cyclic Encoding** | Time and direction (9 features) | Sin/cos transformation | Preserves periodicity (23:59 → 00:01 = 2 min, not 1438) | CP4 |
| **Binning** | Distance, time-of-day (19 features) | Category-based binning | Captures non-linear effects (e.g., extreme weather, long distances) | CP3-4 |
| **Rolling Window Aggregation** | Delay rates, congestion (8 features) | Window functions with temporal constraints | Captures temporal patterns without data leakage | CP3-4 |
| **Interaction Terms** | Distance×weather, time×congestion (5 features) | Multiplicative interaction | Captures non-linear relationships between feature pairs | CP4 |
| **Dimensionality Reduction** | 46 features removed (correlation >0.8) | Correlation-based selection | Removes redundancy, reduces multicollinearity | CP5 |
| **Meta-Feature Generation** | rf_prob_delay, rf_prob_delay_binned | Random Forest probability predictions | Breiman's method—captures complex non-linear patterns | CP4 |
| **Normalization/Standardization** | All numeric features (pre-modeling) | StandardScaler (applied during modeling) | Ensures features on same scale for distance-based algorithms | Modeling phase |

### Dimensionality Reduction Approach

Our feature selection process employed multiple statistical techniques to identify and remove redundant or uninformative features:

**Table 3.11: Dimensionality Reduction Techniques**

| Technique | Purpose | Implementation | Features Affected |
|-----------|---------|----------------|-------------------|
| **Pearson Correlation** | Identify linear relationships between features | Correlation matrix with threshold >0.8 | Removed 46 highly correlated features (e.g., HourlyWetBulbTemperature vs HourlyDewPointTemperature) |
| **Spearman Correlation** | Identify monotonic (including non-linear) relationships | Rank-based correlation comparison | Validated Pearson findings; identified features with hidden non-linear patterns |
| **Correlation Heatmaps** | Visualize feature relationships | Top-K correlation pairs (25-100) filtered heatmap | Guided manual review of feature redundancy |
| **ANOVA (F-test)** | Test categorical feature significance | f_oneway for each categorical vs. DEP_DEL15 | Dropped categoricals with p ≥ 0.05 (not statistically significant) |
| **Chi-Squared Test** | Statistical feature importance | ChiSqSelector with numTopFeatures | Confirmed importance rankings (limited applicability to continuous features) |
| **High Cardinality Analysis** | Identify problematic categoricals | Distinct value counts per feature | Dropped HourlySkyConditions (115K cardinality), flight_id (unique identifier) |

**Features Dropped Due to:**
- **High Collinearity**: Relative humidity and wet bulb temperature (r > 0.95)
- **Repetitiveness**: DISTANCE and DISTANCE_GROUP (redundant information)
- **Data Leakage**: Features only available after the T-2h prediction window (arrival time, actual delays)
- **Low Utility**: Constant columns after filtering (CANCELLED, DIVERTED after row removal)

### Key Transformation Highlights

1. **Weather Imputation Strategy:** Our 3-tier imputation preserves temporal autocorrelation by preferring recent values over global statistics, reducing information loss from 10.16% missing to 0% while maintaining realistic weather patterns.

2. **Cyclic Encoding Rationale:** Time and direction are circular variables where the distance between 23:59 and 00:01 should be 2 minutes, not 1,438 minutes. Sin/cos transformations preserve this topology for 9 temporal and directional features.

3. **Breiman's Method:** Following Leo Breiman's stacked generalization approach, we use Random Forest probability predictions as meta-features, allowing linear models to leverage complex non-linear decision boundaries learned by tree ensembles.

4. **Correlation-Based Selection:** We removed 46 features with Pearson correlation >0.8 to reduce multicollinearity, prioritizing features with higher target correlation and domain interpretability (e.g., keeping dep_delay15_24h_rolling_avg_by_origin_weighted over simple rolling averages).

5. **No Data Leakage:** All transformations respect the T-2h prediction cutoff. Rolling windows use `RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '2' HOUR PRECEDING` to exclude same-flight information.

### Feature Engineering Progression

**Table 3.12: Feature Count Evolution by Category**

| Stage | Temporal | Distance | Weather | Rolling | Aircraft Lag | Network | RFM | Interactions | Cyclic | Breiman | Indexed Cat | Total Change |
|-------|----------|----------|---------|---------|--------------|---------|-----|--------------|--------|---------|-------------|--------------|
| **CP3** | +15 | +4 | +2 | +8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **+27** |
| **CP4** | 0 | +3 | +2 | 0 | +5 | +4 | +8 | +13 | +10 | +2 | 0 | **+69** |
| **CP5** | -1 | -4 | -6 | -2 | -3 | -2 | -4 | -8 | -1 | 0 | +12 | **-48** |
| **Net** | +14 | +3 | -2 | +6 | +2 | +2 | +4 | +5 | +9 | +2 | +12 | **+48** |

---

## 3.3.9 Feature Families and Data Dictionary

### Feature Family Distribution

```
display(Image(filename="/dbfs/student-groups/Group_4_4/feature_family_summary.png"))
```

**Table 3.13: Feature Families Summary**

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

This 73% engineered feature composition reflects our hypothesis that predictive power for flight delays emerges primarily from capturing operational patterns, temporal dependencies, and complex interactions rather than raw measurements alone.

---

## 3.3.10 Dataset Sizes and Storage

### Storage Metrics Across Pipeline

**Table 3.14: Dataset Sizes and Storage Requirements**

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

| Dataset | Size (GB) | Rows | Features | Feasibility |
|---------|-----------|------|----------|-------------|
| Current 1-year (2015) | 2.97 | 5.7M | 108 | ✓ Production-ready |
| Projected 5-year (2015-2019) | ~15 | ~28.5M | 108 | Manageable on Databricks cluster |

**Resource Requirements:**
- Distributed processing: Optimal with 4-8 worker nodes (8GB RAM each)
- In-memory operations: Feasible with medium-sized compute nodes
- Iterative modeling: Fast I/O from Parquet enables rapid experimentation

**Storage Location:**
- All checkpoints: `dbfs:/student-groups/Group_4_4/`
- Primary modeling dataset: `checkpoint_5_final_clean_2015.parquet`
- Total pipeline storage: 16.73 GB enables rollback, debugging, and reproducibility

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

**This comprehensive validation confirms our dataset is production-ready for Phase 2 modeling experiments, including baseline model development, cross-validation with temporal splits, and hyperparameter tuning.**

---

# Appendix A: Complete Feature Inventory

This appendix provides a comprehensive listing of all 108 features in the final dataset (Checkpoint 5), organized by category.

## A.1 Target Variable

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| DEP_DEL15 | Binary | Flight departure delay indicator (1=delayed ≥15min, 0=on-time) | FINAL |

## A.2 Temporal Features

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

## A.3 Geographic Features

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

## A.4 Weather Features

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

## A.5 Distance Features

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| log_distance | Numeric | Log-transformed flight distance | FINAL |
| distance_medium | Binary | Medium distance flight (500-1000 miles) | FINAL |
| distance_long | Binary | Long distance flight (1000-2000 miles) | FINAL |
| distance_very_long | Binary | Very long distance flight (>2000 miles) | FINAL |

## A.6 Rolling/Temporal Aggregation Features

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

## A.7 Aircraft Lag Features

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| prev_flight_dep_del15 | Numeric | Previous flight delay status (same aircraft) | FINAL |
| prev_flight_crs_elapsed_time | Numeric | Previous flight scheduled duration | FINAL |
| hours_since_prev_flight | Numeric | Aircraft turnaround time in hours | FINAL |
| is_first_flight_of_aircraft | Binary | First flight of aircraft today | FINAL |

## A.8 RFM Features

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

## A.9 Congestion Features

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| airport_traffic_density | Numeric | Percentage of daily flights in this hour | FINAL |
| carrier_flight_count | Numeric | Total flights by carrier | FINAL |
| num_airport_wide_delays | Numeric | Delays at airport in 2-hour window | FINAL |
| oncoming_flights | Numeric | Arrivals at origin in 2-hour window | FINAL |
| prior_flights_today | Numeric | Flights at origin so far today | FINAL |
| time_based_congestion_ratio | Numeric | Current vs historical traffic ratio | FINAL |

## A.10 Network Features

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| origin_degree_centrality | Numeric | Origin airport network connectivity (0-1) | FINAL |
| dest_degree_centrality | Numeric | Destination airport network connectivity (0-1) | FINAL |
| carrier_delay_stddev | Numeric | Carrier delay consistency metric | FINAL |

## A.11 Interaction Terms

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| distance_x_weather_severity | Numeric | Distance × extreme_weather_score | FINAL |
| weekend_x_route_volume | Numeric | Weekend × route_1yr_volume | FINAL |
| weather_x_airport_delays | Numeric | Weather × num_airport_wide_delays | FINAL |
| temp_x_holiday | Numeric | Temperature × is_holiday_month | FINAL |
| route_delay_rate_x_peak_hour | Numeric | Route delay rate × peak hour | FINAL |

## A.12 Cyclic Encoded Features

Time and direction features with sin/cos encodings to preserve periodicity (listed in Temporal and Weather sections above).

## A.13 Breiman Meta-Features

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| rf_prob_delay | Numeric | Random Forest predicted probability of delay | FINAL |
| rf_prob_delay_binned | Numeric | Binned RF delay probability (5 bins) | FINAL |

## A.14 Indexed Categorical Features

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

## A.15 Historical Pattern Features

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| prior_day_delay_rate | Numeric | Previous day's delay rate at origin | FINAL |
| same_day_prior_delay_percentage | Numeric | Percentage of flights delayed so far today | FINAL |

## A.16 Reference/Operational Features

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
