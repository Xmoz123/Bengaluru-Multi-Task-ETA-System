"""
Production Inference Pipeline for Bus ETA Prediction
Author: Pratheek Shanbhogue
Competition: IISc Bengaluru Last Mile Challenge 2025 - Task 1

Real-time ETA prediction for sequential bus stops using trained ensemble models.
Handles multi-stop prediction with monotonicity constraints and fold averaging.
"""

import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class ETAInferencePipeline:
    """
    Production inference pipeline for bus ETA prediction.
    
    Predicts cumulative ETAs for a sequence of 5 future stops using
    a trained ensemble of LightGBM, CatBoost, and XGBoost models.
    
    Features:
        - Real-time prediction from GPS trajectory
        - Sequential multi-stop prediction (5 stops)
        - K-fold ensemble averaging
        - Monotonicity enforcement
        - Historical feature integration
    """
    
    def __init__(self, 
                 lgb_models: List,
                 cb_models: List,
                 xgb_models: List,
                 ensemble_weights: Dict[str, float],
                 feature_names: List[str],
                 historical_lookup: Dict):
        """
        Initialize inference pipeline.
        
        Args:
            lgb_models: List of trained LightGBM models (one per fold)
            cb_models: List of trained CatBoost models (one per fold)
            xgb_models: List of trained XGBoost models (one per fold)
            ensemble_weights: Dict with keys 'lgb', 'cb', 'xgb' and weight values
            feature_names: List of feature names in training order
            historical_lookup: Historical ETA statistics dictionary
        """
        self.lgb_models = lgb_models
        self.cb_models = cb_models
        self.xgb_models = xgb_models
        self.ensemble_weights = ensemble_weights
        self.feature_names = feature_names
        self.historical_lookup = historical_lookup
        
        print(f"✅ Inference pipeline initialized")
        print(f"   Models: {len(lgb_models)} LGB + {len(cb_models)} CB + {len(xgb_models)} XGB")
        print(f"   Ensemble: LGB={ensemble_weights['lgb']:.2f}, "
              f"CB={ensemble_weights['cb']:.2f}, XGB={ensemble_weights['xgb']:.2f}")
        print(f"   Features: {len(feature_names)}")
    
    def predict_sequence(self, 
                        test_trajectory: Dict,
                        future_stops: List[Dict]) -> List[float]:
        """
        Predict cumulative ETAs for sequence of 5 future stops.
        
        Args:
            test_trajectory: Dictionary with current trajectory state:
                - route_id: Route identifier
                - current_time: Current timestamp
                - current_speed: Current speed (km/h)
                - current_lat/lon: Current GPS coordinates
                - avg_recent_speed, speed_volatility, etc.
            future_stops: List of 5 stop dictionaries:
                - stop_id: Stop identifier
                - lat/lon: Stop GPS coordinates
                
        Returns:
            List of 5 cumulative ETAs (minutes from now)
            
        Example:
            >>> pipeline = ETAInferencePipeline(models, weights, features, lookup)
            >>> trajectory = {
            ...     'route_id': '335E',
            ...     'current_time': pd.Timestamp('2025-10-16 09:00:00'),
            ...     'current_speed': 28.5,
            ...     'current_lat': 12.9352,
            ...     'current_lon': 77.6245
            ... }
            >>> stops = [
            ...     {'stop_id': 'STOP1', 'lat': 12.9400, 'lon': 77.6300},
            ...     # ... 4 more stops
            ... ]
            >>> etas = pipeline.predict_sequence(trajectory, stops)
            >>> # etas = [3.2, 8.7, 14.5, 21.3, 28.9] (cumulative minutes)
        """
        if len(future_stops) != 5:
            raise ValueError(f"Expected 5 future stops, got {len(future_stops)}")
        
        cumulative_etas = []
        cumulative_time = 0.0
        
        # Predict each stop sequentially
        for stop_idx, stop in enumerate(future_stops):
            # Extract features for this specific stop
            features = self._extract_stop_features(
                trajectory=test_trajectory,
                target_stop=stop,
                stop_sequence_position=stop_idx,
                cumulative_time_so_far=cumulative_time
            )
            
            # Predict inter-stop time
            inter_stop_time = self._predict_inter_stop_time(features)
            
            # Accumulate time (minimum 0.5 min between stops)
            cumulative_time += max(0.5, inter_stop_time)
            cumulative_etas.append(cumulative_time)
        
        # Enforce monotonicity (times must increase)
        cumulative_etas = self._enforce_monotonicity(cumulative_etas)
        
        return cumulative_etas
    
    def _predict_inter_stop_time(self, features: Dict) -> float:
        """
        Predict time to reach next stop using ensemble.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Predicted inter-stop time (minutes)
        """
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure all required features exist (fill missing with 0)
        for feat in self.feature_names:
            if feat not in features_df.columns:
                features_df[feat] = 0.0
        
        # Reorder to match training
        features_df = features_df[self.feature_names]
        
        # Get numeric features for LGB/XGB
        features_numeric = features_df.select_dtypes(include=[np.number])
        
        # Predict with each fold and average
        lgb_preds = [
            model.predict(features_numeric, num_iteration=model.best_iteration)[0]
            for model in self.lgb_models
        ]
        cb_preds = [
            model.predict(features_df)[0]
            for model in self.cb_models
        ]
        xgb_preds = [
            model.predict(features_numeric)[0]
            for model in self.xgb_models
        ]
        
        # Average across folds
        lgb_pred = np.mean(lgb_preds)
        cb_pred = np.mean(cb_preds)
        xgb_pred = np.mean(xgb_preds)
        
        # Weighted ensemble
        inter_stop_time = (
            self.ensemble_weights['lgb'] * lgb_pred +
            self.ensemble_weights['cb'] * cb_pred +
            self.ensemble_weights['xgb'] * xgb_pred
        )
        
        return float(inter_stop_time)
    
    def _extract_stop_features(self,
                               trajectory: Dict,
                               target_stop: Dict,
                               stop_sequence_position: int,
                               cumulative_time_so_far: float) -> Dict:
        """
        Extract 60+ features for predicting one specific stop.
        
        Combines current trajectory state, target stop information,
        historical patterns, and temporal context.
        """
        # Get trajectory info
        route_id = trajectory.get('route_id', 0)
        current_time = trajectory.get('current_time', pd.Timestamp.now())
        hour = current_time.hour
        dow = current_time.dayofweek
        month = current_time.month
        
        current_speed = trajectory.get('current_speed', 20.0)
        current_lat = trajectory.get('current_lat', 12.95)
        current_lon = trajectory.get('current_lon', 77.58)
        
        # Get target stop info
        target_lat = target_stop.get('lat', 12.95)
        target_lon = target_stop.get('lon', 77.58)
        target_stop_id = target_stop.get('stop_id', 0)
        
        # Geographic calculations
        lat_diff = target_lat - current_lat
        lon_diff = target_lon - current_lon
        
        # Haversine distance
        straight_line_dist_km = self._haversine_distance(
            current_lat, current_lon, target_lat, target_lon
        )
        
        # Distance from Bengaluru center
        center_lat, center_lon = 12.9716, 77.5946
        dist_from_center = self._haversine_distance(
            current_lat, current_lon, center_lat, center_lon
        )
        
        # Historical lookup
        historical_features = self._get_historical_features(
            route_id, stop_sequence_position, hour, dow
        )
        
        # Speed features
        avg_recent_speed = trajectory.get('avg_recent_speed', current_speed)
        speed_volatility = trajectory.get('speed_volatility',
                                         abs(current_speed - avg_recent_speed))
        max_recent_speed = trajectory.get('max_recent_speed', current_speed * 1.2)
        recent_acceleration = trajectory.get('recent_acceleration', 0.0)
        movement_smoothness = trajectory.get('movement_smoothness',
                                            1.0 - min(speed_volatility / 20.0, 1.0))
        
        # Journey progress
        distance_from_start = trajectory.get(
            'distance_from_start',
            cumulative_time_so_far * current_speed / 60.0
        )
        
        # Rush hour intensity
        rush_hour_intensity = self._calculate_rush_hour_intensity(hour)
        
        # Compile all features
        features = {
            # === TEMPORAL (16 features) ===
            'hour': hour,
            'day_of_week': dow,
            'month': month,
            'is_weekend': 1 if dow >= 5 else 0,
            'is_friday': 1 if dow == 4 else 0,
            'is_morning_rush': 1 if 7 <= hour <= 9 else 0,
            'is_evening_rush': 1 if 17 <= hour <= 20 else 0,
            'is_peak_morning': 1 if hour == 8 else 0,
            'is_peak_evening': 1 if hour == 18 else 0,
            'is_business_hours': 1 if 9 <= hour <= 17 else 0,
            'is_lunch_time': 1 if 12 <= hour <= 14 else 0,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dow_sin': np.sin(2 * np.pi * dow / 7),
            'dow_cos': np.cos(2 * np.pi * dow / 7),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'rush_hour_intensity': rush_hour_intensity,
            
            # === CURRENT LOCATION (8 features) ===
            'current_lat': current_lat,
            'current_lon': current_lon,
            'lat_normalized': (current_lat - 12.85) / 0.3,
            'lon_normalized': (current_lon - 77.45) / 0.3,
            'dist_from_center': dist_from_center,
            'is_city_center': 1 if dist_from_center < 5 else 0,
            'is_suburban': 1 if 5 <= dist_from_center < 15 else 0,
            'is_outer_city': 1 if dist_from_center >= 15 else 0,
            
            # === SPEED (12 features) ===
            'current_speed': current_speed,
            'avg_recent_speed': avg_recent_speed,
            'speed_volatility': speed_volatility,
            'max_recent_speed': max_recent_speed,
            'recent_acceleration': recent_acceleration,
            'movement_smoothness': movement_smoothness,
            'distance_from_start': distance_from_start,
            'speed_vs_city_avg': current_speed / 25.0,
            'speed_stopped': 1 if current_speed < 5 else 0,
            'speed_slow': 1 if 5 <= current_speed < 15 else 0,
            'speed_normal': 1 if 15 <= current_speed < 35 else 0,
            'speed_fast': 1 if current_speed >= 35 else 0,
            
            # === TARGET LOCATION (12 features) ===
            'target_lat': target_lat,
            'target_lon': target_lon,
            'lat_diff': lat_diff,
            'lon_diff': lon_diff,
            'is_distant_target': 1 if straight_line_dist_km > 5 else 0,
            'is_nearby_target': 1 if straight_line_dist_km < 1 else 0,
            'is_very_close_target': 1 if straight_line_dist_km < 0.2 else 0,
            'dist_at_stop': 1 if straight_line_dist_km < 0.05 else 0,
            'dist_very_close': 1 if 0.05 <= straight_line_dist_km < 0.5 else 0,
            'dist_close': 1 if 0.5 <= straight_line_dist_km < 2 else 0,
            'dist_medium': 1 if 2 <= straight_line_dist_km < 5 else 0,
            'dist_far': 1 if straight_line_dist_km >= 5 else 0,
            
            # === IDENTIFIERS (2 features) ===
            'target_stop_id': target_stop_id,
            'route_id': route_id,
            
            # === SEQUENCE POSITION (4 features) ===
            'stop_sequence_position': stop_sequence_position,
            'stop_position_squared': stop_sequence_position ** 2,
            'stop_position_cubed': stop_sequence_position ** 3,
            'stops_remaining': 5 - stop_sequence_position,
            
            # === TRAFFIC (2 features) ===
            'traffic_density_score': trajectory.get('traffic_density_score', 0.0),
            'weekend_factor': 0.7 if dow >= 5 else 1.0,
            
            # === HISTORICAL (9 features) ===
            **historical_features
        }
        
        return features
    
    def _get_historical_features(self, route_id: Any, 
                                stop_seq: int, 
                                hour: int, 
                                dow: int) -> Dict:
        """Get historical ETA statistics from lookup table"""
        lookup_key = (route_id, stop_seq, hour, dow)
        
        if lookup_key in self.historical_lookup:
            hist_stats = self.historical_lookup[lookup_key]
            historical_eta_mean = hist_stats.get('historical_eta_mean', 15.0)
            historical_eta_median = hist_stats.get('historical_eta_median', 12.0)
            historical_eta_std = hist_stats.get('historical_eta_std', 5.0)
            historical_eta_min = hist_stats.get('historical_eta_min', 5.0)
            historical_eta_max = hist_stats.get('historical_eta_max', 30.0)
            historical_sample_count = hist_stats.get('historical_sample_count', 0)
            historical_confidence = min(historical_sample_count / 50.0, 1.0)
        else:
            # Fallback defaults
            historical_eta_mean = 15.0
            historical_eta_median = 12.0
            historical_eta_std = 5.0
            historical_eta_min = 5.0
            historical_eta_max = 30.0
            historical_sample_count = 0
            historical_confidence = 0.0
        
        historical_eta_range = historical_eta_max - historical_eta_min
        historical_eta_cv = historical_eta_std / (historical_eta_mean + 1e-6)
        
        return {
            'historical_eta_mean': historical_eta_mean,
            'historical_eta_median': historical_eta_median,
            'historical_eta_std': historical_eta_std,
            'historical_sample_count': historical_sample_count,
            'historical_eta_min': historical_eta_min,
            'historical_eta_max': historical_eta_max,
            'historical_confidence': historical_confidence,
            'historical_eta_range': historical_eta_range,
            'historical_eta_cv': historical_eta_cv,
        }
    
    @staticmethod
    def _calculate_rush_hour_intensity(hour: int) -> float:
        """Calculate continuous rush hour intensity (0-1)"""
        rush_hour_intensity = 0.0
        
        # Morning rush (peaks at 8am)
        if 7 <= hour <= 9:
            rush_hour_intensity = 1.0 - abs(hour - 8) / 2.0
        # Evening rush (peaks at 6:30pm)
        elif 17 <= hour <= 20:
            rush_hour_intensity = 1.0 - abs(hour - 18.5) / 2.5
        
        return max(0.0, min(1.0, rush_hour_intensity))
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate great-circle distance (km)"""
        R = 6371.0  # Earth radius in km
        
        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    @staticmethod
    def _enforce_monotonicity(cumulative_etas: List[float]) -> List[float]:
        """
        Enforce monotonically increasing cumulative times.
        
        If a later stop has earlier ETA than previous stop,
        adjust it to be at least 0.5 min later.
        """
        monotonic_etas = [cumulative_etas[0]]
        
        for i in range(1, len(cumulative_etas)):
            if cumulative_etas[i] <= monotonic_etas[-1]:
                # Ensure at least 0.5 min between stops
                monotonic_etas.append(monotonic_etas[-1] + 0.5)
            else:
                monotonic_etas.append(cumulative_etas[i])
        
        return monotonic_etas


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    print("🚦 Bus ETA Inference Pipeline")
    print("\n Production-ready real-time prediction")
    print(" Capabilities:")
    print("   - Sequential 5-stop ETA prediction")
    print("   - K-fold ensemble averaging")
    print("   - Monotonicity enforcement")
    print("   - Historical feature integration")
    print("   - 60+ feature extraction per stop")
    print("\nUsage:")
    print("   from inference import ETAInferencePipeline")
    print("   pipeline = ETAInferencePipeline(models, weights, features, lookup)")
    print("   etas = pipeline.predict_sequence(trajectory, future_stops)")
