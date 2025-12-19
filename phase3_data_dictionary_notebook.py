"""
Phase 3 Data Dictionary Generator
Creates comprehensive data dictionary for Final Clean Dataset (Checkpoint 5a - Production Ready)

Run this in a Databricks notebook after the main statistics collection.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, min, max, mean, stddev
import pandas as pd
from datetime import datetime

print("=" * 100)
print("PHASE 3: DATA DICTIONARY GENERATOR")
print("=" * 100)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# Paths
BASE_PATH = "dbfs:/student-groups/Group_4_4/"
OUTPUT_PATH = "/dbfs/student-groups/Group_4_4/images/"

# Load final dataset (Checkpoint 5a - Production Ready)
print("\nLoading Checkpoint 5a: Final Clean Dataset (Production Ready)...")
df_final = spark.read.parquet(f"{BASE_PATH}checkpoint_5a_comprehensive_all_features_2015-2019.parquet")

print(f"✓ Dataset loaded: {df_final.count():,} rows x {len(df_final.columns)} columns")

# =============================================================================
# FEATURE DESCRIPTIONS
# =============================================================================

# Known feature descriptions
feature_descriptions = {
    # Target
    'DEP_DEL15': 'Binary target: 1 if departure delayed ≥15 minutes, 0 otherwise',
    
    # Flight Identifiers
    'FL_DATE': 'Flight date (YYYY-MM-DD)',
    'YEAR': 'Year of flight',
    'QUARTER': 'Quarter of year (1-4)',
    'MONTH': 'Month of flight (1-12)',
    'DAY_OF_MONTH': 'Day of month (1-31)',
    'DAY_OF_WEEK': 'Day of week (1=Monday, 7=Sunday)',
    'OP_UNIQUE_CARRIER': 'Unique carrier code (airline)',
    'OP_CARRIER_FL_NUM': 'Flight number',
    'TAIL_NUM': 'Aircraft tail number',
    
    # Airport Information
    'ORIGIN': 'Origin airport code (IATA)',
    'ORIGIN_AIRPORT_ID': 'Origin airport ID',
    'ORIGIN_CITY_NAME': 'Origin city name',
    'ORIGIN_STATE_ABR': 'Origin state abbreviation',
    'DEST': 'Destination airport code (IATA)',
    'DEST_AIRPORT_ID': 'Destination airport ID',
    'DEST_CITY_NAME': 'Destination city name',
    'DEST_STATE_ABR': 'Destination state abbreviation',
    
    # Timing
    'CRS_DEP_TIME': 'Scheduled departure time (HHMM format)',
    'CRS_ARR_TIME': 'Scheduled arrival time (HHMM format)',
    'CRS_ELAPSED_TIME': 'Scheduled elapsed time of flight (minutes)',
    
    # Distance
    'DISTANCE': 'Distance between airports (miles)',
    'DISTANCE_GROUP': 'Distance group category',
    
    # Temporal Features
    'departure_hour': 'Hour of scheduled departure (0-23)',
    'departure_month': 'Month of departure (1-12)',
    'departure_dayofweek': 'Day of week (0=Monday, 6=Sunday)',
    'is_weekend': 'Binary: 1 if weekend (Sat/Sun), 0 otherwise',
    'is_peak_hour': 'Binary: 1 if peak travel hour (6-9am, 4-7pm), 0 otherwise',
    'season': 'Season of year (Spring/Summer/Fall/Winter)',
    'hour_category': 'Time of day category (Early Morning/Morning/Afternoon/Evening/Night)',
    'is_holiday_window': 'Binary: 1 if within major holiday period, 0 otherwise',
    
    # Distance Categories
    'distance_category': 'Flight distance category (Short <500mi / Medium 500-1500mi / Long >1500mi)',
    
    # Weather Features (common ones)
    'HourlyAltimeterSetting': 'Altimeter setting in inches of mercury',
    'HourlyDewPointTemperature': 'Dew point temperature in Fahrenheit',
    'HourlyDryBulbTemperature': 'Dry bulb temperature in Fahrenheit',
    'HourlyPrecipitation': 'Precipitation amount in inches',
    'HourlyPresentWeatherType': 'Present weather type code',
    'HourlyPressureChange': 'Pressure change in last 3 hours',
    'HourlyRelativeHumidity': 'Relative humidity percentage',
    'HourlySeaLevelPressure': 'Sea level pressure in inches of mercury',
    'HourlyStationPressure': 'Station pressure in inches of mercury',
    'HourlyVisibility': 'Visibility in miles',
    'HourlyWetBulbTemperature': 'Wet bulb temperature in Fahrenheit',
    'HourlyWindDirection': 'Wind direction in degrees',
    'HourlyWindSpeed': 'Wind speed in miles per hour',
    'weather_severity_index': 'Composite weather severity index (0-100, higher = more severe)',
}

def categorize_feature(feature_name):
    """Categorize a feature based on its name"""
    name_lower = feature_name.lower()
    
    if feature_name == 'DEP_DEL15':
        return 'target'
    elif feature_name in ['FL_DATE', 'YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
                          'OP_UNIQUE_CARRIER', 'OP_CARRIER_FL_NUM', 'TAIL_NUM']:
        return 'flight_identifiers'
    elif any(x in name_lower for x in ['origin', 'dest']) and not any(x in name_lower for x in ['rolling', 'delay', 'ratio']):
        return 'airport_info'
    elif any(x in name_lower for x in ['crs_dep', 'crs_arr', 'crs_elapsed']):
        return 'timing_scheduled'
    elif 'distance' in name_lower:
        return 'distance'
    elif any(x in name_lower for x in ['hourly', 'daily', 'sky', 'precipitation', 'visibility', 'pressure', 'humidity']):
        return 'weather_observations'
    elif any(x in name_lower for x in ['departure_hour', 'departure_month', 'departure_day', 'is_weekend', 
                                       'is_peak', 'season', 'hour_category', 'is_holiday']):
        return 'temporal_features'
    elif any(x in name_lower for x in ['rolling', '_24h', '_ratio', '_avg', '_median', '_std']):
        return 'rolling_aggregates'
    elif any(x in name_lower for x in ['total_flights', 'congestion', 'delay_rate', 'delay_ratio']) and 'rolling' not in name_lower:
        return 'airport_carrier_metrics'
    elif 'weather_severity' in name_lower or 'weather_category' in name_lower:
        return 'weather_derived'
    elif any(x in name_lower for x in ['route', 'pair']):
        return 'route_features'
    else:
        return 'other'

def get_description(feature_name, category):
    """Get or generate description for a feature"""
    
    # Check if we have a predefined description
    if feature_name in feature_descriptions:
        return feature_descriptions[feature_name]
    
    # Generate description based on patterns
    name_lower = feature_name.lower()
    
    if 'rolling_' in name_lower and '_24h' in name_lower:
        base = feature_name.replace('rolling_', '').replace('_24h', '')
        return f"24-hour rolling window statistic: {base}"
    elif 'rolling_' in name_lower:
        base = feature_name.replace('rolling_', '')
        return f"Rolling window statistic: {base}"
    elif name_lower.startswith('hourly'):
        return f"Hourly weather observation: {feature_name.replace('Hourly', '')}"
    elif name_lower.startswith('daily'):
        return f"Daily weather summary: {feature_name.replace('Daily', '')}"
    elif 'total_flights_per' in name_lower:
        return "Total number of flights for this origin/destination/day combination"
    elif 'delay_rate' in name_lower or 'delay_ratio' in name_lower:
        if 'origin' in name_lower:
            return "Historical delay rate for origin airport"
        elif 'dest' in name_lower:
            return "Historical delay rate for destination airport"
        elif 'carrier' in name_lower:
            return "Historical delay rate for carrier"
        else:
            return "Historical delay rate metric"
    elif 'congestion' in name_lower:
        return "Airport congestion index"
    elif 'performance' in name_lower:
        return "Performance metric"
    
    # Generic description based on category
    category_descriptions = {
        'target': 'Target variable',
        'flight_identifiers': 'Flight identification information',
        'airport_info': 'Airport location information',
        'timing_scheduled': 'Scheduled timing information',
        'distance': 'Flight distance metric',
        'weather_observations': 'Weather observation data',
        'temporal_features': 'Engineered time-based feature',
        'rolling_aggregates': 'Rolling window aggregate statistic',
        'airport_carrier_metrics': 'Airport or carrier performance metric',
        'weather_derived': 'Derived weather feature',
        'route_features': 'Route characteristic feature',
        'other': 'Additional feature'
    }
    
    return category_descriptions.get(category, 'Feature')

# =============================================================================
# BUILD DATA DICTIONARY
# =============================================================================

print("\n" + "=" * 100)
print("BUILDING DATA DICTIONARY")
print("=" * 100)

data_dict_rows = []

for field in df_final.schema.fields:
    feature_name = field.name
    data_type = str(field.dataType)
    
    # Categorize
    category = categorize_feature(feature_name)
    description = get_description(feature_name, category)
    
    print(f"Processing: {feature_name}...")
    
    try:
        # Get basic statistics
        if 'int' in data_type.lower() or 'double' in data_type.lower() or 'float' in data_type.lower():
            # Numerical feature
            stats = df_final.select(
                count(col(feature_name)).alias('count'),
                countDistinct(col(feature_name)).alias('distinct'),
                min(col(feature_name)).alias('min'),
                max(col(feature_name)).alias('max'),
                mean(col(feature_name)).alias('mean'),
                stddev(col(feature_name)).alias('std')
            ).first()
            
            dict_row = {
                'Feature_Name': feature_name,
                'Category': category,
                'Data_Type': 'Numerical',
                'Description': description,
                'Non_Null_Count': stats['count'] if stats else 0,
                'Distinct_Values': stats['distinct'] if stats else 0,
                'Min': f"{stats['min']:.2f}" if stats and stats['min'] is not None else 'N/A',
                'Max': f"{stats['max']:.2f}" if stats and stats['max'] is not None else 'N/A',
                'Mean': f"{stats['mean']:.2f}" if stats and stats['mean'] is not None else 'N/A',
                'Std': f"{stats['std']:.2f}" if stats and stats['std'] is not None else 'N/A'
            }
        else:
            # Categorical feature
            stats = df_final.select(
                count(col(feature_name)).alias('count'),
                countDistinct(col(feature_name)).alias('distinct')
            ).first()
            
            dict_row = {
                'Feature_Name': feature_name,
                'Category': category,
                'Data_Type': 'Categorical',
                'Description': description,
                'Non_Null_Count': stats['count'] if stats else 0,
                'Distinct_Values': stats['distinct'] if stats else 0,
                'Min': 'N/A',
                'Max': 'N/A',
                'Mean': 'N/A',
                'Std': 'N/A'
            }
    
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        dict_row = {
            'Feature_Name': feature_name,
            'Category': category,
            'Data_Type': 'Unknown',
            'Description': description,
            'Non_Null_Count': 'Error',
            'Distinct_Values': 'Error',
            'Min': 'N/A',
            'Max': 'N/A',
            'Mean': 'N/A',
            'Std': 'N/A'
        }
    
    data_dict_rows.append(dict_row)

# Create DataFrame
df_dict = pd.DataFrame(data_dict_rows)

# Sort by category then feature name
df_dict = df_dict.sort_values(['Category', 'Feature_Name'])

# =============================================================================
# DISPLAY DATA DICTIONARY BY CATEGORY
# =============================================================================

print("\n" + "=" * 100)
print("DATA DICTIONARY BY CATEGORY")
print("=" * 100)

for category in sorted(df_dict['Category'].unique()):
    cat_features = df_dict[df_dict['Category'] == category]
    print(f"\n{category.upper().replace('_', ' ')} ({len(cat_features)} features)")
    print("=" * 100)
    
    for _, row in cat_features.iterrows():
        print(f"\n  {row['Feature_Name']}")
        print(f"    Type: {row['Data_Type']}")
        print(f"    Description: {row['Description']}")
        print(f"    Non-Null: {row['Non_Null_Count']}, Distinct: {row['Distinct_Values']}")
        if row['Mean'] != 'N/A':
            print(f"    Range: [{row['Min']}, {row['Max']}], Mean: {row['Mean']}, Std: {row['Std']}")

# =============================================================================
# SAVE DATA DICTIONARY
# =============================================================================

print("\n" + "=" * 100)
print("SAVING DATA DICTIONARY")
print("=" * 100)

# Save full dictionary
dict_path = f"{OUTPUT_PATH}phase3_data_dictionary.csv"
df_dict.to_csv(dict_path, index=False)
print(f"✓ Full data dictionary saved to: {dict_path}")
print(f"  Total features documented: {len(df_dict)}")

# Create and save category summary
category_summary = df_dict.groupby('Category').agg({
    'Feature_Name': 'count',
    'Data_Type': lambda x: f"{sum(x=='Numerical')} numerical, {sum(x=='Categorical')} categorical"
}).rename(columns={'Feature_Name': 'Count', 'Data_Type': 'Type_Breakdown'})

print("\n" + "=" * 100)
print("FEATURE SUMMARY BY CATEGORY")
print("=" * 100)
print(category_summary.to_string())

summary_path = f"{OUTPUT_PATH}phase3_feature_category_summary.csv"
category_summary.to_csv(summary_path)
print(f"\n✓ Category summary saved to: {summary_path}")

# Create data type summary
print("\n" + "=" * 100)
print("DATA TYPE DISTRIBUTION")
print("=" * 100)

type_summary = df_dict['Data_Type'].value_counts()
print(type_summary.to_string())

print("\n" + "=" * 100)
print("DATA DICTIONARY GENERATION COMPLETE!")
print("=" * 100)

print(f"\n📁 Generated Files:")
print(f"  • {dict_path}")
print(f"  • {summary_path}")

print("\n💡 Data dictionary DataFrame available as 'df_dict' for further analysis")
print("=" * 100)
