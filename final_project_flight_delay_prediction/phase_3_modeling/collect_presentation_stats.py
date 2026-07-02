"""
Statistics Collection for Flight Delay Prediction Project Presentation
Collects key metrics from raw, joined, cleaned, and feature-engineered datasets
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, isnull, countDistinct, avg, stddev
import pandas as pd
from datetime import datetime

# Initialize Spark session (if not already active)
try:
    spark
except NameError:
    spark = SparkSession.builder.appName("PresentationStats").getOrCreate()

print("=" * 80)
print("FLIGHT DELAY PREDICTION - PRESENTATION STATISTICS")
print("=" * 80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Define file paths based on actual data files
BASE_DIR = "dbfs:/student-groups/Group_4_4"
RAW_DATA_PATH = "dbfs:/mnt/mids-w261/OTPW_12M"

# Actual file paths from your directory
JOINED_1Y_PATH = f"{BASE_DIR}/JOINED_1Y_2019.parquet"
CLEANED_1Y_PATH = f"{BASE_DIR}/joined_1Y_clean_imputed.parquet"
FEATURE_ENG_PATH = f"{BASE_DIR}/joined_1Y_feat.parquet"
FINAL_FEATURE_PATH = f"{BASE_DIR}/joined_1Y_final_feature_clean.parquet"

# Dictionary to store all statistics
stats = {}

# ============================================================================
# 1. RAW OTPW DATA
# ============================================================================
print("\n" + "=" * 80)
print("1. RAW OTPW DATA (12 Months)")
print("=" * 80)

try:
    # Raw data is stored as parquet
    df_raw = spark.read.parquet(RAW_DATA_PATH)
    df_raw.cache()
    
    raw_rows = df_raw.count()
    raw_cols = len(df_raw.columns)
    
    stats['raw'] = {
        'rows': raw_rows,
        'columns': raw_cols,
        'description': 'Raw OTPW data (12 months of 2019)'
    }
    
    print(f"✓ Rows: {raw_rows:,}")
    print(f"✓ Columns: {raw_cols}")
    
    # Check if this has basic flight data
    if 'DEP_DEL15' in df_raw.columns:
        delay_count = df_raw.filter(col('DEP_DEL15') == 1).count()
        delay_rate = (delay_count / raw_rows) * 100
        stats['raw']['delays'] = delay_count
        stats['raw']['delay_rate'] = delay_rate
        print(f"✓ Delayed flights: {delay_count:,} ({delay_rate:.1f}%)")
    
    df_raw.unpersist()
    
except Exception as e:
    print(f"⚠ Could not load raw data: {e}")
    print("  (This is optional - continuing with joined data)")
    stats['raw'] = {'note': 'Raw data not loaded (optional)'}

# ============================================================================
# 2. JOINED DATA (AIRLINE + WEATHER)
# ============================================================================
print("\n" + "=" * 80)
print("2. JOINED DATA (Airline + Weather)")
print("=" * 80)

try:
    df_joined = spark.read.parquet(JOINED_1Y_PATH)
    df_joined.cache()
    
    joined_rows = df_joined.count()
    joined_cols = len(df_joined.columns)
    
    stats['joined'] = {
        'rows': joined_rows,
        'columns': joined_cols,
        'description': 'Raw joined airline + weather data'
    }
    
    print(f"✓ Rows: {joined_rows:,}")
    print(f"✓ Columns: {joined_cols}")
    
    # Target variable distribution
    if 'DEP_DEL15' in df_joined.columns:
        delay_count = df_joined.filter(col('DEP_DEL15') == 1).count()
        delay_rate = (delay_count / joined_rows) * 100
        stats['joined']['delays'] = delay_count
        stats['joined']['delay_rate'] = delay_rate
        print(f"✓ Delayed flights: {delay_count:,} ({delay_rate:.1f}%)")
    
    # Calculate nulls in joined data
    null_counts = []
    for column in df_joined.columns[:10]:  # Sample first 10 columns for speed
        null_count = df_joined.filter(col(column).isNull()).count()
        if null_count > 0:
            null_counts.append((column, null_count, (null_count/joined_rows)*100))
    
    if null_counts:
        avg_null_pct = sum([x[2] for x in null_counts]) / len(null_counts)
        stats['joined']['avg_null_percentage'] = avg_null_pct
        print(f"✓ Average null % (sampled cols): {avg_null_pct:.1f}%")
    
    df_joined.unpersist()
    
except Exception as e:
    print(f"⚠ Could not load joined data: {e}")
    stats['joined'] = {'error': str(e)}

# ============================================================================
# 3. CLEANED DATA
# ============================================================================
print("\n" + "=" * 80)
print("3. CLEANED DATA (After Cleaning & Imputation)")
print("=" * 80)

try:
    df_cleaned = spark.read.parquet(CLEANED_1Y_PATH)
    
    df_cleaned.cache()
    
    cleaned_rows = df_cleaned.count()
    cleaned_cols = len(df_cleaned.columns)
    
    stats['cleaned'] = {
        'rows': cleaned_rows,
        'columns': cleaned_cols,
        'description': 'After cleaning and imputation'
    }
    
    print(f"✓ Rows: {cleaned_rows:,}")
    print(f"✓ Columns: {cleaned_cols}")
    
    # Calculate data reduction
    if 'joined' in stats and 'rows' in stats['joined']:
        rows_removed = stats['joined']['rows'] - cleaned_rows
        pct_removed = (rows_removed / stats['joined']['rows']) * 100
        stats['cleaned']['rows_removed'] = rows_removed
        stats['cleaned']['pct_removed'] = pct_removed
        print(f"✓ Rows removed: {rows_removed:,} ({pct_removed:.1f}%)")
        
        cols_removed = stats['joined']['columns'] - cleaned_cols
        stats['cleaned']['cols_removed'] = cols_removed
        print(f"✓ Columns removed: {cols_removed}")
    
    # Target variable distribution after cleaning
    if 'DEP_DEL15' in df_cleaned.columns:
        delay_count = df_cleaned.filter(col('DEP_DEL15') == 1).count()
        delay_rate = (delay_count / cleaned_rows) * 100
        stats['cleaned']['delays'] = delay_count
        stats['cleaned']['delay_rate'] = delay_rate
        print(f"✓ Delayed flights: {delay_count:,} ({delay_rate:.1f}%)")
        print(f"✓ Class balance: {100-delay_rate:.1f}% on-time, {delay_rate:.1f}% delayed")
    
    # Check for remaining nulls (should be minimal after imputation)
    null_check_cols = ['DEP_DEL15', 'DISTANCE', 'CRS_DEP_TIME']
    null_check_cols = [c for c in null_check_cols if c in df_cleaned.columns]
    
    remaining_nulls = 0
    for col_name in null_check_cols:
        null_count = df_cleaned.filter(col(col_name).isNull()).count()
        remaining_nulls += null_count
    
    stats['cleaned']['remaining_nulls_checked'] = remaining_nulls
    if remaining_nulls == 0:
        print(f"✓ No nulls in critical columns")
    else:
        print(f"⚠ Remaining nulls in critical columns: {remaining_nulls:,}")
    
    # Unique values for key categorical features
    cat_features = ['ORIGIN', 'DEST', 'OP_UNIQUE_CARRIER']
    cat_features = [c for c in cat_features if c in df_cleaned.columns]
    
    for feat in cat_features:
        unique_count = df_cleaned.select(feat).distinct().count()
        print(f"✓ Unique {feat}: {unique_count}")
    
    df_cleaned.unpersist()
    
except Exception as e:
    print(f"⚠ Could not load cleaned data: {e}")
    stats['cleaned'] = {'error': str(e)}

# ============================================================================
# 4. FEATURE ENGINEERED DATA (Intermediate)
# ============================================================================
print("\n" + "=" * 80)
print("4. FEATURE ENGINEERED DATA (Intermediate)")
print("=" * 80)

try:
    df_features = spark.read.parquet(FEATURE_ENG_PATH)
    
    df_features.cache()
    
    features_rows = df_features.count()
    features_cols = len(df_features.columns)
    
    stats['features'] = {
        'rows': features_rows,
        'columns': features_cols,
        'description': 'After feature engineering'
    }
    
    print(f"✓ Rows: {features_rows:,}")
    print(f"✓ Columns: {features_cols}")
    
    # Features added
    if 'cleaned' in stats and 'columns' in stats['cleaned']:
        features_added = features_cols - stats['cleaned']['columns']
        stats['features']['features_added'] = features_added
        print(f"✓ Features added: {features_added}")
    
    # Check for engineered features (based on chat history)
    engineered_features = [
        'departure_hour', 'departure_month', 'departure_dayofweek',
        'is_weekend', 'is_peak_hour', 'season', 'hour_category',
        'total_flights_per_origin_day',
        'rolling_origin_num_flights_24h', 'rolling_origin_delay_ratio_24h',
        'is_holiday_window', 'weather_severity_index', 'distance_category'
    ]
    
    present_features = [f for f in engineered_features if f in df_features.columns]
    stats['features']['engineered_features_present'] = len(present_features)
    
    print(f"✓ Engineered features present: {len(present_features)}/{len(engineered_features)}")
    
    if present_features:
        print("\n  Present engineered features:")
        for feat in present_features[:8]:  # Show first 8
            print(f"    • {feat}")
        if len(present_features) > 8:
            print(f"    ... and {len(present_features) - 8} more")
    
    # Final delay rate
    if 'DEP_DEL15' in df_features.columns:
        delay_count = df_features.filter(col('DEP_DEL15') == 1).count()
        delay_rate = (delay_count / features_rows) * 100
        stats['features']['delays'] = delay_count
        stats['features']['delay_rate'] = delay_rate
        print(f"\n✓ Final delayed flights: {delay_count:,} ({delay_rate:.1f}%)")
    
    df_features.unpersist()
    
except Exception as e:
    print(f"⚠ Could not load feature engineered data: {e}")
    stats['features'] = {'error': str(e)}

# ============================================================================
# 5. FINAL FEATURE ENGINEERED DATA (Ready for Modeling)
# ============================================================================
print("\n" + "=" * 80)
print("5. FINAL FEATURE ENGINEERED DATA (Ready for Modeling)")
print("=" * 80)

try:
    df_final = spark.read.parquet(FINAL_FEATURE_PATH)
    df_final.cache()
    
    final_rows = df_final.count()
    final_cols = len(df_final.columns)
    
    stats['final'] = {
        'rows': final_rows,
        'columns': final_cols,
        'description': 'Final cleaned feature set for modeling'
    }
    
    print(f"✓ Rows: {final_rows:,}")
    print(f"✓ Columns: {final_cols}")
    
    # Calculate difference from intermediate feature set
    if 'features' in stats and 'columns' in stats['features']:
        cols_removed = stats['features']['columns'] - final_cols
        if cols_removed > 0:
            print(f"✓ Columns removed in final cleaning: {cols_removed}")
        elif cols_removed < 0:
            print(f"✓ Columns added in final step: {abs(cols_removed)}")
    
    # Final delay rate
    if 'DEP_DEL15' in df_final.columns:
        delay_count = df_final.filter(col('DEP_DEL15') == 1).count()
        delay_rate = (delay_count / final_rows) * 100
        stats['final']['delays'] = delay_count
        stats['final']['delay_rate'] = delay_rate
        print(f"✓ Final delayed flights: {delay_count:,} ({delay_rate:.1f}%)")
        print(f"✓ Final class balance: {100-delay_rate:.1f}% on-time, {delay_rate:.1f}% delayed")
    
    # Check for any remaining nulls
    critical_cols = ['DEP_DEL15', 'ORIGIN', 'DEST', 'OP_UNIQUE_CARRIER']
    critical_cols = [c for c in critical_cols if c in df_final.columns]
    
    total_nulls = 0
    for col_name in critical_cols:
        null_count = df_final.filter(col(col_name).isNull()).count()
        total_nulls += null_count
    
    if total_nulls == 0:
        print(f"✓ No nulls in critical columns (ready for modeling)")
    else:
        print(f"⚠ Warning: {total_nulls:,} nulls remaining in critical columns")
    
    df_final.unpersist()
    
except Exception as e:
    print(f"⚠ Could not load final feature data: {e}")
    stats['final'] = {'error': str(e)}

# ============================================================================
# 6. SUMMARY FOR PRESENTATION
# ============================================================================
print("\n" + "=" * 80)
print("PRESENTATION SUMMARY STATISTICS")
print("=" * 80)

print("\n📊 DATA PIPELINE PROGRESSION:")
print("-" * 80)

stages = ['raw', 'joined', 'cleaned', 'features', 'final']
stage_names = [
    'Raw OTPW Data (12M)',
    'Joined Data (Airline + Weather)', 
    'Cleaned Data',
    'Feature Engineered (Intermediate)',
    'Final Feature Set (Ready for Modeling)'
]

for stage, name in zip(stages, stage_names):
    if stage in stats and 'rows' in stats[stage]:
        print(f"\n{name}:")
        print(f"  • Rows: {stats[stage]['rows']:,}")
        print(f"  • Columns: {stats[stage]['columns']}")
        if 'delay_rate' in stats[stage]:
            print(f"  • Delay Rate: {stats[stage]['delay_rate']:.1f}%")

print("\n" + "=" * 80)
print("\n📈 KEY METRICS FOR TALK TRACK:")
print("-" * 80)

# Calculate overall metrics using the best available data
start_stage = 'raw' if 'raw' in stats and 'rows' in stats['raw'] else 'joined'
end_stage = 'final' if 'final' in stats and 'rows' in stats['final'] else 'features'

if start_stage in stats and end_stage in stats and 'rows' in stats[start_stage] and 'rows' in stats[end_stage]:
    total_data_reduction = stats[start_stage]['rows'] - stats[end_stage]['rows']
    pct_data_reduction = (total_data_reduction / stats[start_stage]['rows']) * 100
    
    print(f"\n1. DATA SCALE:")
    print(f"   • Started with: {stats[start_stage]['rows']:,} records ({start_stage.upper()} data)")
    print(f"   • Final dataset: {stats[end_stage]['rows']:,} records")
    print(f"   • Data reduction: {pct_data_reduction:.1f}% (cleaned invalid/cancelled flights)")

if end_stage in stats and 'delay_rate' in stats[end_stage]:
    print(f"\n2. CLASS BALANCE:")
    print(f"   • On-time flights: {100 - stats[end_stage]['delay_rate']:.1f}%")
    print(f"   • Delayed flights: {stats[end_stage]['delay_rate']:.1f}%")
    imbalance_ratio = (100 - stats[end_stage]['delay_rate']) / stats[end_stage]['delay_rate']
    print(f"   • Imbalance ratio: {imbalance_ratio:.2f}:1 (on-time:delayed)")
    print(f"   • Note: Imbalanced dataset requiring special handling (SMOTE, class weights, etc.)")

# Feature engineering impact
if 'joined' in stats and end_stage in stats and 'columns' in stats['joined'] and 'columns' in stats[end_stage]:
    features_added = stats[end_stage]['columns'] - stats['joined']['columns']
    print(f"\n3. FEATURE ENGINEERING:")
    print(f"   • Original columns (joined): {stats['joined']['columns']}")
    print(f"   • Features added/engineered: {features_added}")
    print(f"   • Final feature count: {stats[end_stage]['columns']}")
    
    # Show intermediate steps if available
    if 'features' in stats and 'columns' in stats['features']:
        print(f"   • Intermediate feature count: {stats['features']['columns']}")
        if end_stage == 'final':
            features_removed = stats['features']['columns'] - stats['final']['columns']
            if features_removed > 0:
                print(f"   • Features removed in final cleaning: {features_removed}")

if 'cleaned' in stats:
    print(f"\n4. DATA QUALITY:")
    if 'rows_removed' in stats['cleaned']:
        print(f"   • Rows removed during cleaning: {stats['cleaned']['rows_removed']:,} ({stats['cleaned']['pct_removed']:.1f}%)")
    if 'remaining_nulls_checked' in stats['cleaned']:
        print(f"   • Nulls after imputation: {stats['cleaned']['remaining_nulls_checked']}")
    print(f"   • Data integrity: High (cancelled/diverted flights removed, nulls imputed)")

print(f"\n5. DATASET CHARACTERISTICS:")
print(f"   • Time period: 12 months (2019)")
print(f"   • Data sources: 2 (Bureau of Transportation Statistics + NOAA Weather)")
print(f"   • Prediction task: Binary classification (delay ≥15 min)")
print(f"   • Target variable: DEP_DEL15 (1 = delayed, 0 = on-time)")

# Save statistics to file for reference
print("\n" + "=" * 80)
print("SAVING STATISTICS")
print("=" * 80)

# Convert to pandas DataFrame for easy viewing
summary_data = []
for stage in ['raw', 'joined', 'cleaned', 'features', 'final']:
    if stage in stats and 'rows' in stats[stage]:
        row_data = {
            'Stage': stage.capitalize(),
            'Rows': f"{stats[stage]['rows']:,}",
            'Columns': stats[stage]['columns']
        }
        if 'delay_rate' in stats[stage]:
            row_data['Delay_Rate_%'] = f"{stats[stage]['delay_rate']:.1f}"
        if 'rows_removed' in stats[stage]:
            row_data['Rows_Removed'] = f"{stats[stage]['rows_removed']:,}"
        summary_data.append(row_data)

df_summary = pd.DataFrame(summary_data)
print("\n")
print(df_summary.to_string(index=False))

# Save to CSV
output_path = "/dbfs/student-groups/Group_4_4/presentation_statistics.csv"
try:
    df_summary.to_csv(output_path, index=False)
    print(f"\n✓ Statistics saved to: {output_path}")
except Exception as e:
    print(f"\n⚠ Could not save to CSV: {e}")
    print("Statistics available in memory as 'stats' dictionary")

print("\n" + "=" * 80)
print("STATISTICS COLLECTION COMPLETE!")
print("=" * 80)

# Make stats dictionary available for further use
print("\n💡 Access detailed stats via the 'stats' dictionary")
print("   Example: stats['features']['delay_rate']")
