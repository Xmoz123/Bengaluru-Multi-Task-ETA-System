"""
GPS Trajectory Data Preprocessing Pipeline
Author: Pratheek Shanbhogue
Competition: IISc Bengaluru Last Mile Challenge 2025 - Task 1

Production-quality pipeline for processing 60GB of GPS trajectory data
for bus ETA prediction. Handles data validation, outlier detection,
and quality control at scale.
"""

import numpy as np
import pandas as pd
import polars as pl
from geopy.distance import great_circle
import math
from datetime import datetime
from tqdm.auto import tqdm
import gc
import os

# ===== CONFIGURATION =====
class PreprocessingConfig:
    """Configuration for GPS trajectory preprocessing"""
    MIN_POINTS = 5  # Minimum points per trajectory
    MAX_POINTS = 50000  # Filter extreme outliers
    
    # Bengaluru GPS bounds (prevent invalid coordinates)
    LAT_MIN, LAT_MAX = 12.0, 14.0
    LON_MIN, LON_MAX = 77.0, 78.5
    
    ESSENTIAL_COLS = [
        "trip_id", "vehicle_timestamp", "latitude", 
        "longitude", "vehicle_id"
    ]


# ===== UTILITY FUNCTIONS =====
def bearing(pointA: tuple, pointB: tuple) -> float:
    """
    Calculate the bearing between two GPS points.
    
    Args:
        pointA: Tuple of (lat, lon) in degrees
        pointB: Tuple of (lat, lon) in degrees
        
    Returns:
        Bearing in degrees (0-360)
    """
    if pointA == pointB:
        return 0.0
        
    lat1, lon1 = math.radians(pointA[0]), math.radians(pointA[1])
    lat2, lon2 = math.radians(pointB[0]), math.radians(pointB[1])
    
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    
    initial_bearing = math.atan2(x, y)
    compass_bearing = (math.degrees(initial_bearing) + 360) % 360
    return compass_bearing


# ===== MAIN PREPROCESSING PIPELINE =====
def preprocess_trajectories(
    data_dir: str, 
    date: str, 
    config: PreprocessingConfig = None
) -> pl.DataFrame:
    """
    Process raw GPS trajectory data with production-quality validation.
    
    Key features:
    - GPS coordinate validation (Bengaluru bounds)
    - Outlier detection and filtering
    - Trajectory reconstruction with metadata
    - Date range validation
    
    Args:
        data_dir: Root directory containing parquet files
        date: Date to process (YYYY-MM-DD format)
        config: Preprocessing configuration
        
    Returns:
        Polars DataFrame with cleaned trajectories
    """
    if config is None:
        config = PreprocessingConfig()
    
    # Setup date bounds
    target_date = datetime.strptime(date, "%Y-%m-%d")
    date_min = target_date.replace(hour=0, minute=0, second=0)
    date_max = target_date.replace(hour=23, minute=59, second=59)
    
    print(f"🚀 Processing {date}")
    print(f"📊 GPS bounds: Lat {config.LAT_MIN}-{config.LAT_MAX}, "
          f"Lon {config.LON_MIN}-{config.LON_MAX}")
    
    # Load and process files
    date_folder = os.path.join(data_dir, f"date={date}")
    files = sorted(glob(os.path.join(date_folder, "*.parquet")))
    
    all_trajectories = []
    traj_counter = 0
    stats = {
        'filtered_outliers': 0,
        'filtered_coordinates': 0,
        'filtered_dates': 0
    }
    
    for file_num, f in enumerate(tqdm(files, desc="Processing"), 1):
        try:
            df = pl.read_parquet(f, columns=config.ESSENTIAL_COLS)
            
            if df.height == 0:
                continue
            
            # STEP 1: Decode binary columns
            binary_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                          if dtype == pl.Binary]
            
            for col in binary_cols:
                df = df.with_columns([
                    pl.col(col).map_elements(
                        lambda x: x.decode('utf-8', errors='ignore').strip('"') 
                                 if x else None,
                        return_dtype=pl.String
                    ).alias(col)
                ])
            
            # STEP 2: Parse timestamps
            df = df.with_columns([
                pl.col("vehicle_timestamp")
                .str.strip_chars('"')
                .str.to_integer(strict=False)
                .alias("timestamp_numeric")
            ]).with_columns([
                (pl.datetime(1970, 1, 1) + 
                 pl.col("timestamp_numeric") * pl.duration(seconds=1))
                .alias("timestamp")
            ]).filter(pl.col("timestamp").is_not_null())
            
            # STEP 3: Date validation
            before_date = df.height
            df = df.filter(
                pl.col("timestamp").is_between(date_min, date_max, closed="both")
            )
            stats['filtered_dates'] += (before_date - df.height)
            
            # STEP 4: GPS coordinate validation
            before_coord = df.height
            df = df.filter(
                pl.col("latitude").is_between(config.LAT_MIN, config.LAT_MAX) &
                pl.col("longitude").is_between(config.LON_MIN, config.LON_MAX) &
                pl.col("latitude").is_not_null() &
                pl.col("longitude").is_not_null()
            )
            stats['filtered_coordinates'] += (before_coord - df.height)
            
            if df.height == 0:
                continue
            
            # STEP 5: Trajectory grouping with outlier detection
            trip_groups = df.group_by("trip_id").agg([
                pl.len().alias("point_count"),
                pl.col("vehicle_timestamp"),
                pl.col("latitude"), 
                pl.col("longitude"),
                pl.col("vehicle_id"),
                pl.col("timestamp_numeric"),
                pl.col("timestamp")
            ])
            
            # Filter outliers
            outliers = trip_groups.filter(
                pl.col("point_count") > config.MAX_POINTS
            )
            stats['filtered_outliers'] += outliers.height
            
            trip_groups = trip_groups.filter(
                (pl.col("point_count") >= config.MIN_POINTS) & 
                (pl.col("point_count") <= config.MAX_POINTS)
            )
            
            # STEP 6: Reconstruct trajectories
            for row in trip_groups.iter_rows():
                trip_id_val = row[0]
                point_count = row[1]
                
                traj_df = pl.DataFrame({
                    "trip_id": [trip_id_val] * point_count,
                    "vehicle_timestamp": row[2],
                    "latitude": row[3],
                    "longitude": row[4], 
                    "vehicle_id": row[5],
                    "timestamp_numeric": row[6],
                    "timestamp": row[7],
                    "trajectory_id": [traj_counter] * point_count
                }).sort("timestamp")
                
                all_trajectories.append(traj_df)
                traj_counter += 1
            
            del df, trip_groups
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error in file {file_num}: {e}")
            continue
    
    # STEP 7: Consolidate and add metadata
    if all_trajectories:
        print(f"\n💾 Consolidating {len(all_trajectories)} trajectories...")
        combined = pl.concat(all_trajectories)
        
        # Add temporal and spatial features
        combined = combined.with_columns([
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.minute().alias("minute"),
            pl.col("timestamp").diff().dt.total_seconds().alias("time_diff_sec"),
            # Basic speed estimation components
            (pl.col("latitude").diff() * 111).alias("lat_diff_km"),
            (pl.col("longitude").diff() * 85).alias("lon_diff_km"),
        ])
        
        # Print quality report
        print(f"\n🔍 QUALITY VALIDATION:")
        print(f"   📊 GPS points: {combined.height:,}")
        print(f"   🚍 Trajectories: {combined.select('trajectory_id').unique().height:,}")
        print(f"   🛡️  Filtered - Dates: {stats['filtered_dates']:,}, "
              f"Coords: {stats['filtered_coordinates']:,}, "
              f"Outliers: {stats['filtered_outliers']:,}")
        
        return combined
    
    else:
        print("❌ No valid trajectories created")
        return None


# ===== VALIDATION PIPELINE =====
def validate_processed_data(df: pl.DataFrame) -> dict:
    """
    Comprehensive validation of processed trajectory data.
    
    Returns:
        Dictionary with validation results and quality metrics
    """
    validation_results = {
        'total_points': df.height,
        'total_trajectories': df.select('trajectory_id').unique().height,
        'unique_trips': df.select('trip_id').unique().height,
        'unique_vehicles': df.select('vehicle_id').unique().height,
        'time_range': (df['timestamp'].min(), df['timestamp'].max()),
        'gps_coverage': {
            'lat': (df['latitude'].min(), df['latitude'].max()),
            'lon': (df['longitude'].min(), df['longitude'].max())
        },
        'trajectory_stats': {
            'mean_length': df.group_by('trajectory_id').len()['len'].mean(),
            'median_length': df.group_by('trajectory_id').len()['len'].median(),
            'min_length': df.group_by('trajectory_id').len()['len'].min(),
            'max_length': df.group_by('trajectory_id').len()['len'].max()
        }
    }
    
    return validation_results


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    # Example: Process one day of data
    config = PreprocessingConfig()
    
    df = preprocess_trajectories(
        data_dir="/path/to/data",
        date="2025-08-19",
        config=config
    )
    
    if df is not None:
        validation = validate_processed_data(df)
        print(f"\n✅ Processing complete: {validation['total_trajectories']:,} trajectories")
