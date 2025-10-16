"""
Feature Engineering for Bus ETA Prediction
Author: Pratheek Shanbhogue
Competition: IISc Bengaluru Last Mile Challenge 2025 - Task 1

Ultra-safe 42-feature extraction pipeline designed for production deployment
with zero data leakage. Combines GPS trajectory analysis with temporal and
geographic context modeling.
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime
from typing import Dict, Optional


class ETAFeatureExtractor:
    """
    Production-safe feature extractor for bus ETA prediction.
    
    Extracts 42 carefully designed features from GPS trajectories with
    explicit data leakage prevention. All features use only information
    available at prediction time.
    
    Feature Categories:
        - Temporal (19): Time-of-day, day-of-week, cyclical encodings
        - Geographic (7): Location context, city zones
        - Target (7): Destination-aware features
        - Movement (7): Speed, acceleration, smoothness
        - Progress (1): Journey advancement
        - Traffic (3): Rush hour modeling
        - Speed Context (2): Comparative speed metrics
    """
    
    def __init__(self, stop_coords: Optional[Dict] = None):
        """
        Initialize feature extractor.
        
        Args:
            stop_coords: Dictionary mapping stop_id -> {'lat': float, 'lon': float}
        """
        self.stop_coords = stop_coords or {}
        
        # Bengaluru city constants (public knowledge - no leakage)
        self.city_center = (12.9716, 77.5946)
        self.city_avg_speed = 25.0  # Historical city average km/h
        self.rush_hours = {
            'morning': (7, 10),
            'evening': (17, 20)
        }
        self.peak_hours = {
            'morning': (8, 9),
            'evening': (18, 19)
        }
    
    def extract_features(self, 
                        gps_trajectory: pd.DataFrame,
                        use_up_to_idx: Optional[int] = None,
                        target_stop_id: Optional[str] = None) -> Dict:
        """
        Extract 42 production-safe features from GPS trajectory.
        
        Args:
            gps_trajectory: DataFrame with columns:
                - timestamp: GPS reading timestamp
                - latitude: GPS latitude
                - longitude: GPS longitude
            use_up_to_idx: Only use GPS points up to this index
                          (for simulating real-time prediction)
            target_stop_id: Target destination stop ID (if known)
            
        Returns:
            Dictionary with 42 feature values
            
        Example:
            >>> extractor = ETAFeatureExtractor(stop_coords)
            >>> features = extractor.extract_features(
            ...     trajectory_df, use_up_to_idx=10, target_stop_id='STOP123'
            ... )
        """
        # Limit to available data (real-time simulation)
        if use_up_to_idx is not None:
            df = gps_trajectory.iloc[:use_up_to_idx + 1].copy()
        else:
            df = gps_trajectory.copy()
        
        # Handle edge case: insufficient data
        if len(df) < 2:
            return self._default_features()
        
        # Prepare data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Smooth coordinates (minimal backward window to reduce noise)
        if len(df) >= 3:
            window = min(3, len(df))
            df['lat_smooth'] = df['latitude'].rolling(
                window=window, min_periods=1
            ).mean()
            df['lon_smooth'] = df['longitude'].rolling(
                window=window, min_periods=1
            ).mean()
        else:
            df['lat_smooth'] = df['latitude']
            df['lon_smooth'] = df['longitude']
        
        # Extract all feature categories
        temporal_features = self._extract_temporal_features(df)
        geographic_features = self._extract_geographic_features(df)
        target_features = self._extract_target_features(df, target_stop_id)
        movement_features = self._extract_movement_features(df)
        progress_features = self._extract_progress_features(df)
        traffic_features = self._extract_traffic_features(temporal_features)
        speed_context = self._extract_speed_context(movement_features)
        
        # Combine all features
        features = {
            **temporal_features,
            **geographic_features,
            **target_features,
            **movement_features,
            **progress_features,
            **traffic_features,
            **speed_context
        }
        
        return features
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> Dict:
        """Extract 19 temporal features"""
        current_time = df['timestamp'].iloc[-1]
        
        hour = current_time.hour
        day_of_week = current_time.weekday()
        month = current_time.month
        day_of_month = current_time.day
        
        return {
            # Basic time components
            'hour': int(hour),
            'day_of_week': int(day_of_week),
            'month': int(month),
            
            # Temporal flags
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_holiday_season': 1 if month in [4, 10, 12] else 0,
            'is_morning_rush': 1 if self.rush_hours['morning'][0] <= hour <= self.rush_hours['morning'][1] else 0,
            'is_evening_rush': 1 if self.rush_hours['evening'][0] <= hour <= self.rush_hours['evening'][1] else 0,
            'is_peak_morning': 1 if self.peak_hours['morning'][0] <= hour <= self.peak_hours['morning'][1] else 0,
            'is_peak_evening': 1 if self.peak_hours['evening'][0] <= hour <= self.peak_hours['evening'][1] else 0,
            'is_business_hours': 1 if 9 <= hour <= 18 else 0,
            'is_lunch_time': 1 if 12 <= hour <= 14 else 0,
            
            # Cyclical encodings (preserve periodic patterns)
            'hour_sin': float(np.sin(2 * np.pi * hour / 24)),
            'hour_cos': float(np.cos(2 * np.pi * hour / 24)),
            'dow_sin': float(np.sin(2 * np.pi * day_of_week / 7)),
            'dow_cos': float(np.cos(2 * np.pi * day_of_week / 7)),
            'month_sin': float(np.sin(2 * np.pi * month / 12)),
            'month_cos': float(np.cos(2 * np.pi * month / 12)),
            'day_sin': float(np.sin(2 * np.pi * day_of_month / 31)),
            'day_cos': float(np.cos(2 * np.pi * day_of_month / 31)),
        }
    
    def _extract_geographic_features(self, df: pd.DataFrame) -> Dict:
        """Extract 7 geographic features"""
        current_lat = float(df['lat_smooth'].iloc[-1])
        current_lon = float(df['lon_smooth'].iloc[-1])
        
        # Distance from Bengaluru city center
        dist_from_center = self._haversine_distance(
            current_lat, current_lon,
            self.city_center[0], self.city_center[1]
        )
        
        # Urban zone classification
        is_city_center = 1 if dist_from_center < 5 else 0
        is_suburban = 1 if 5 <= dist_from_center < 15 else 0
        
        # Normalized coordinates (km from city center)
        lat_normalized = (current_lat - self.city_center[0]) * 111
        lon_normalized = (current_lon - self.city_center[1]) * 85
        
        return {
            'current_lat': float(current_lat),
            'current_lon': float(current_lon),
            'lat_normalized': float(lat_normalized),
            'lon_normalized': float(lon_normalized),
            'dist_from_center': float(dist_from_center),
            'is_city_center': is_city_center,
            'is_suburban': is_suburban,
        }
    
    def _extract_target_features(self, df: pd.DataFrame, 
                                 target_stop_id: Optional[str]) -> Dict:
        """Extract 7 target destination features"""
        current_lat = float(df['lat_smooth'].iloc[-1])
        current_lon = float(df['lon_smooth'].iloc[-1])
        
        # Default values if no target
        target_lat = target_lon = distance_to_target = bearing_to_target = 0.0
        lat_diff_to_target = lon_diff_to_target = 0.0
        is_target_central = 0
        
        # If target stop known, calculate features
        if target_stop_id and str(target_stop_id) in self.stop_coords:
            target_lat = self.stop_coords[str(target_stop_id)]['lat']
            target_lon = self.stop_coords[str(target_stop_id)]['lon']
            
            distance_to_target = self._haversine_distance(
                current_lat, current_lon, target_lat, target_lon
            )
            bearing_to_target = self._calculate_bearing(
                current_lat, current_lon, target_lat, target_lon
            )
            
            lat_diff_to_target = target_lat - current_lat
            lon_diff_to_target = target_lon - current_lon
            
            # Is target in city center?
            target_dist_from_center = self._haversine_distance(
                target_lat, target_lon,
                self.city_center[0], self.city_center[1]
            )
            is_target_central = 1 if target_dist_from_center < 5 else 0
        
        return {
            'target_lat': float(target_lat),
            'target_lon': float(target_lon),
            'distance_to_target': float(distance_to_target),
            'bearing_to_target': float(bearing_to_target),
            'lat_diff_to_target': float(lat_diff_to_target),
            'lon_diff_to_target': float(lon_diff_to_target),
            'is_target_central': is_target_central,
        }
    
    def _extract_movement_features(self, df: pd.DataFrame) -> Dict:
        """Extract 7 movement pattern features"""
        # Default values
        current_speed = avg_recent_speed = speed_volatility = 0.0
        max_recent_speed = recent_acceleration = movement_smoothness = 0.0
        time_since_slow_movement = 5.0
        
        if len(df) < 2:
            return {
                'current_speed': current_speed,
                'avg_recent_speed': avg_recent_speed,
                'speed_volatility': speed_volatility,
                'max_recent_speed': max_recent_speed,
                'recent_acceleration': recent_acceleration,
                'movement_smoothness': movement_smoothness,
                'time_since_slow_movement': time_since_slow_movement,
            }
        
        # Analyze recent GPS points (last 4 for safety)
        recent_window = min(4, len(df))
        recent_df = df.tail(recent_window)
        
        speeds = []
        accelerations = []
        bearing_changes = []
        
        # Calculate movement metrics
        for i in range(1, len(recent_df)):
            lat_diff = recent_df.iloc[i]['lat_smooth'] - recent_df.iloc[i-1]['lat_smooth']
            lon_diff = recent_df.iloc[i]['lon_smooth'] - recent_df.iloc[i-1]['lon_smooth']
            time_diff = (recent_df.iloc[i]['timestamp'] - 
                        recent_df.iloc[i-1]['timestamp']).total_seconds()
            
            if time_diff > 0:
                # Speed calculation
                distance = math.sqrt((lat_diff * 111)**2 + (lon_diff * 85)**2)
                speed = (distance / time_diff) * 3600  # km/h
                speed = max(0, min(speed, 100))  # Cap at reasonable values
                speeds.append(speed)
                
                # Acceleration
                if len(speeds) >= 2:
                    acceleration = (speeds[-1] - speeds[-2]) / max(time_diff / 3600, 0.01)
                    accelerations.append(acceleration)
                
                # Direction consistency (smoothness)
                if i >= 2:
                    prev_bearing = self._calculate_bearing(
                        recent_df.iloc[i-2]['lat_smooth'], recent_df.iloc[i-2]['lon_smooth'],
                        recent_df.iloc[i-1]['lat_smooth'], recent_df.iloc[i-1]['lon_smooth']
                    )
                    curr_bearing = self._calculate_bearing(
                        recent_df.iloc[i-1]['lat_smooth'], recent_df.iloc[i-1]['lon_smooth'],
                        recent_df.iloc[i]['lat_smooth'], recent_df.iloc[i]['lon_smooth']
                    )
                    bearing_diff = abs(curr_bearing - prev_bearing)
                    if bearing_diff > math.pi:
                        bearing_diff = 2 * math.pi - bearing_diff
                    bearing_changes.append(bearing_diff)
        
        # Aggregate movement statistics
        if speeds:
            current_speed = speeds[-1]
            avg_recent_speed = np.mean(speeds)
            speed_volatility = np.std(speeds) if len(speeds) > 1 else 0.0
            max_recent_speed = np.max(speeds)
        
        if accelerations:
            recent_acceleration = np.mean(accelerations)
        
        if bearing_changes:
            movement_smoothness = 1 / (1 + np.mean(bearing_changes))
        else:
            movement_smoothness = 1.0
        
        # Time since last bus stop (slow movement indicator)
        time_since_slow_movement = self._get_time_since_slow_movement(df)
        
        return {
            'current_speed': float(current_speed),
            'avg_recent_speed': float(avg_recent_speed),
            'speed_volatility': float(speed_volatility),
            'max_recent_speed': float(max_recent_speed),
            'recent_acceleration': float(recent_acceleration),
            'movement_smoothness': float(movement_smoothness),
            'time_since_slow_movement': float(time_since_slow_movement),
        }
    
    def _extract_progress_features(self, df: pd.DataFrame) -> Dict:
        """Extract 1 journey progress feature"""
        start_lat = df['lat_smooth'].iloc[0]
        start_lon = df['lon_smooth'].iloc[0]
        current_lat = df['lat_smooth'].iloc[-1]
        current_lon = df['lon_smooth'].iloc[-1]
        
        distance_from_start = self._haversine_distance(
            current_lat, current_lon, start_lat, start_lon
        )
        
        return {
            'distance_from_start': float(distance_from_start),
        }
    
    def _extract_traffic_features(self, temporal_features: Dict) -> Dict:
        """Extract 3 traffic context features"""
        # Rush hour intensity
        traffic_density_score = (
            temporal_features['is_morning_rush'] * 0.3 +
            temporal_features['is_evening_rush'] * 0.3 +
            temporal_features['is_peak_morning'] * 0.2 +
            temporal_features['is_peak_evening'] * 0.2
        )
        
        # Day-of-week factor
        weekend_factor = 0.7 if temporal_features['is_weekend'] else 1.0
        
        # Seasonal factor
        seasonal_factor = 1.1 if temporal_features['is_holiday_season'] else 1.0
        
        return {
            'traffic_density_score': float(traffic_density_score),
            'weekend_factor': float(weekend_factor),
            'seasonal_factor': float(seasonal_factor),
        }
    
    def _extract_speed_context(self, movement_features: Dict) -> Dict:
        """Extract 2 speed context features"""
        current_speed = movement_features['current_speed']
        
        # Relative to city average
        speed_vs_city_avg = (current_speed / self.city_avg_speed 
                            if self.city_avg_speed > 0 else 1.0)
        
        # Speed category
        speed_category = self._categorize_speed(current_speed)
        
        return {
            'speed_vs_city_avg': float(speed_vs_city_avg),
            'speed_category': int(speed_category),
        }
    
    def _get_time_since_slow_movement(self, df: pd.DataFrame) -> float:
        """Time since bus was at a stop (slow movement < 8 km/h)"""
        if len(df) < 2:
            return 5.0
        
        current_time = df['timestamp'].iloc[-1]
        
        # Look backwards for slow movement
        for i in range(len(df) - 2, -1, -1):
            if i > 0:
                lat_diff = df.iloc[i]['lat_smooth'] - df.iloc[i-1]['lat_smooth']
                lon_diff = df.iloc[i]['lon_smooth'] - df.iloc[i-1]['lon_smooth']
                time_diff = (df.iloc[i]['timestamp'] - 
                           df.iloc[i-1]['timestamp']).total_seconds()
                
                if time_diff > 0:
                    distance = math.sqrt((lat_diff * 111)**2 + (lon_diff * 85)**2)
                    speed = (distance / time_diff) * 3600
                    
                    if speed < 8:  # Likely a bus stop
                        return min(
                            (current_time - df.iloc[i]['timestamp']).total_seconds() / 60,
                            20.0
                        )
            
            # Don't look back more than 20 minutes
            if (current_time - df.iloc[i]['timestamp']).total_seconds() > 1200:
                break
        
        return 5.0
    
    def _categorize_speed(self, speed: float) -> int:
        """Categorize speed into buckets"""
        if speed < 8: return 0      # Stopped/very slow
        elif speed < 20: return 1   # City crawl
        elif speed < 35: return 2   # Normal city traffic
        elif speed < 50: return 3   # Fast city movement
        else: return 4              # Highway speed
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two GPS points (km)"""
        if lat1 == lat2 and lon1 == lon2:
            return 0.0
        
        R = 6371.0  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _calculate_bearing(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """Calculate bearing between two GPS points (radians)"""
        if lat1 == lat2 and lon1 == lon2:
            return 0.0
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        bearing = math.atan2(y, x)
        return bearing
    
    def _default_features(self) -> Dict:
        """Default feature values for edge cases"""
        return {
            # Temporal (19)
            'hour': 12, 'day_of_week': 2, 'month': 6,
            'is_weekend': 0, 'is_holiday_season': 0,
            'is_morning_rush': 0, 'is_evening_rush': 0,
            'is_peak_morning': 0, 'is_peak_evening': 0,
            'is_business_hours': 1, 'is_lunch_time': 0,
            'hour_sin': 0.0, 'hour_cos': 1.0,
            'dow_sin': 0.0, 'dow_cos': 1.0,
            'month_sin': 0.0, 'month_cos': 1.0,
            'day_sin': 0.0, 'day_cos': 1.0,
            
            # Geographic (7)
            'current_lat': 12.97, 'current_lon': 77.59,
            'lat_normalized': 0.0, 'lon_normalized': 0.0,
            'dist_from_center': 8.0,
            'is_city_center': 0, 'is_suburban': 1,
            
            # Target (7)
            'target_lat': 12.97, 'target_lon': 77.59,
            'distance_to_target': 2.5, 'bearing_to_target': 0.0,
            'lat_diff_to_target': 0.0, 'lon_diff_to_target': 0.0,
            'is_target_central': 0,
            
            # Movement (7)
            'current_speed': 25.0, 'avg_recent_speed': 25.0,
            'speed_volatility': 3.0, 'max_recent_speed': 28.0,
            'recent_acceleration': 0.0, 'movement_smoothness': 0.8,
            'time_since_slow_movement': 5.0,
            
            # Progress (1)
            'distance_from_start': 1.5,
            
            # Traffic (3)
            'traffic_density_score': 0.2,
            'weekend_factor': 1.0, 'seasonal_factor': 1.0,
            
            # Speed context (2)
            'speed_vs_city_avg': 1.0, 'speed_category': 2,
        }


def add_polynomial_stop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add polynomial features for stop sequence position.
    
    Captures non-linear patterns in ETA as bus progresses through route.
    
    Args:
        df: DataFrame with 'stop_sequence_position' column
        
    Returns:
        DataFrame with added polynomial features
    """
    if 'stop_sequence_position' not in df.columns:
        raise ValueError("DataFrame must have 'stop_sequence_position' column")
    
    df['stop_position_squared'] = df['stop_sequence_position'] ** 2
    df['stop_position_cubed'] = df['stop_sequence_position'] ** 3
    
    # Assuming route length (adjust based on your data)
    if 'total_stops' in df.columns:
        df['stops_remaining'] = df['total_stops'] - df['stop_sequence_position']
    
    return df


def add_rush_hour_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add continuous rush hour intensity feature.
    
    Models traffic density as continuous function peaking at rush hours.
    
    Args:
        df: DataFrame with 'hour' column
        
    Returns:
        DataFrame with 'rush_hour_intensity' feature
    """
    if 'hour' not in df.columns:
        raise ValueError("DataFrame must have 'hour' column")
    
    df['rush_hour_intensity'] = 0.0
    
    # Morning rush (peaks at 8am)
    morning_rush = (df['hour'] >= 7) & (df['hour'] <= 9)
    df.loc[morning_rush, 'rush_hour_intensity'] = (
        1.0 - abs(df.loc[morning_rush, 'hour'] - 8) / 2.0
    ).clip(0, 1)
    
    # Evening rush (peaks at 6:30pm)
    evening_rush = (df['hour'] >= 17) & (df['hour'] <= 20)
    df.loc[evening_rush, 'rush_hour_intensity'] = (
        1.0 - abs(df.loc[evening_rush, 'hour'] - 18.5) / 2.5
    ).clip(0, 1)
    
    return df


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    print(" Bus ETA Feature Engineering Module")
    print("\n Production-safe 42-feature extraction")
    print(" Feature Categories:")
    print("   - Temporal (19): Time patterns with cyclical encoding")
    print("   - Geographic (7): Location and city zone context")
    print("   - Target (7): Destination-aware features")
    print("   - Movement (7): Speed, acceleration, smoothness")
    print("   - Progress (1): Journey advancement")
    print("   - Traffic (3): Rush hour and seasonal modeling")
    print("   - Speed Context (2): Comparative speed metrics")
    print("\n Usage:")
    print("   from feature_engineering import ETAFeatureExtractor")
    print("   extractor = ETAFeatureExtractor(stop_coords)")
    print("   features = extractor.extract_features(trajectory_df)")
