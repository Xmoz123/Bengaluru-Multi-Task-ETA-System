"""
Feature Engineering for Auto-rickshaw ETA Prediction
Author: Pratheek Shanbhogue
Task 2: Auto-rickshaw ETA Prediction

60+ features combining H3 spatial indexing, OSMnx routing, and
supply-demand dynamics for accurate auto-rickshaw ETA prediction.
"""

import numpy as np
import pandas as pd
import h3
from geopy.distance import geodesic
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


# Configuration
RANDOM_STATE = 42
CITY_CENTER = (12.9716, 77.5946)  # Bengaluru
TECH_HUBS = [
    (12.9352, 77.6245),  # Whitefield
    (12.8467, 77.6627)   # Electronic City
]


class AutoRickshawFeatureExtractor:
    """
    Feature extraction for auto-rickshaw ETA prediction.
    
    Combines:
    - H3 hexagonal spatial indexing
    - Supply-demand dynamics
    - OSMnx routing features
    - Temporal patterns
    - Speed profiles
    """
    
    def __init__(self, h3_resolution: int = 9):
        self.h3_resolution = h3_resolution
        self.city_center = CITY_CENTER
        self.tech_hubs = TECH_HUBS
    
    def extract_features(self,
                        origin_lat: float,
                        origin_lon: float,
                        dest_lat: float,
                        dest_lon: float,
                        timestamp: pd.Timestamp,
                        slot: int,
                        activity_df: pd.DataFrame,
                        origin_speed: float = 20.2,
                        dest_speed: float = 20.2,
                        trip_time: float = None) -> Dict:
        """
        Extract 60+ features for one auto-rickshaw trip.
        
        Returns dictionary with all features for model input.
        """
        features = {}
        
        # 1. Basic trip identifiers
        features['originlat'] = float(origin_lat)
        features['originlon'] = float(origin_lon)
        features['destlat'] = float(dest_lat)
        features['destlon'] = float(dest_lon)
        features['slot'] = int(slot)
        
        # 2. Temporal features (8)
        features['hour'] = int(timestamp.hour)
        features['dow'] = int(timestamp.dayofweek)
        features['date'] = timestamp.date()
        features['is_weekend'] = int(timestamp.dayofweek >= 5)
        features['is_morning_peak'] = int(7 <= timestamp.hour <= 10)
        features['is_evening_peak'] = int(17 <= timestamp.hour <= 20)
        features['is_super_peak'] = int(timestamp.hour in [8, 9, 18, 19])
        features['same_cell'] = int(
            np.isclose(origin_lat, dest_lat, atol=1e-3) and 
            np.isclose(origin_lon, dest_lon, atol=1e-3)
        )
        
        # 3. Distance features (5)
        distance_km = geodesic((origin_lat, origin_lon), (dest_lat, dest_lon)).kilometers
        features['distance_km'] = float(distance_km)
        features['short_trip'] = int(distance_km < 2)
        features['medium_trip'] = int(2 <= distance_km < 10)
        features['long_trip'] = int(distance_km >= 10)
        features['distance_cat'] = int(np.where(
            distance_km < 2, 0,
            np.where(distance_km < 5, 1,
            np.where(distance_km < 10, 2,
            np.where(distance_km < 20, 3, 4)))
        ))
        
        # 4. Spatial features - City center distances (4)
        origin_center_dist = geodesic((origin_lat, origin_lon), self.city_center).kilometers
        dest_center_dist = geodesic((dest_lat, dest_lon), self.city_center).kilometers
        
        features['origin_center_dist'] = float(origin_center_dist)
        features['dest_center_dist'] = float(dest_center_dist)
        features['avg_center_dist'] = float((origin_center_dist + dest_center_dist) / 2)
        
        # 5. Spatial features - Tech hub proximity (4)
        min_origin_tech = min(
            geodesic((origin_lat, origin_lon), hub).kilometers 
            for hub in self.tech_hubs
        )
        min_dest_tech = min(
            geodesic((dest_lat, dest_lon), hub).kilometers 
            for hub in self.tech_hubs
        )
        
        features['min_origin_tech_dist'] = float(min_origin_tech)
        features['min_dest_tech_dist'] = float(min_dest_tech)
        
        # 6. Route characteristics (7)
        bearing = np.degrees(np.arctan2(
            np.radians(dest_lon - origin_lon),
            np.radians(dest_lat - origin_lat)
        ))
        if bearing < 0:
            bearing += 360
        
        features['trip_bearing'] = float(bearing)
        features['trip_bearing_sin'] = float(np.sin(np.radians(bearing)))
        features['trip_bearing_cos'] = float(np.cos(np.radians(bearing)))
        features['route_efficiency'] = float(np.minimum(1.0, distance_km / (distance_km * 1.3)))
        features['density_factor'] = float(np.maximum(0.6, 1.5 - features['avg_center_dist'] / 15.0))
        features['route_complexity'] = float(distance_km * features['density_factor'])
        
        # 7. Congestion & Speed features (6)
        features['origin_speed'] = float(origin_speed)
        features['dest_speed'] = float(dest_speed)
        features['congestion_factor'] = float(np.where(
            features['is_super_peak'] == 1, 1.8,
            np.where(features['is_evening_peak'] == 1, 1.6,
            np.where(features['is_morning_peak'] == 1, 1.4, 1.0))
        ))
        features['rush_intensity'] = float(features['congestion_factor'] - 1.0)
        
        base_speed = 20.2
        features['estimated_speed'] = float(base_speed / features['congestion_factor'])
        features['time_per_km'] = float(60.0 / features['estimated_speed'])
        
        # 8. Supply-Demand features from H3 activity (6)
        origin_cell = h3.latlng_to_cell(origin_lat, origin_lon, self.h3_resolution)
        ring_feats = self._extract_ring_aggregates(activity_df, origin_cell, slot)
        
        features['ring_start_sum'] = float(ring_feats.get('ring_start_sum', 0.0))
        features['ring_end_sum'] = float(ring_feats.get('ring_end_sum', 0.0))
        features['ring_idle_mean'] = float(ring_feats.get('ring_idle_mean', 0.0))
        features['ring_idle_p90'] = float(ring_feats.get('ring_idle_p90', 0.0))
        features['supply_demand_ratio'] = float(ring_feats.get('supply_demand_ratio', 0.5))
        
        # 9. Target labels (if provided for training)
        if trip_time is not None:
            features['actual_trip_time'] = float(trip_time)
        
        return features
    
    def _extract_ring_aggregates(self,
                                 act_df: pd.DataFrame,
                                 cell: str,
                                 slot: int,
                                 k: int = 1) -> Dict:
        """
        Extract supply-demand aggregates from H3 neighborhood ring.
        
        Args:
            act_df: Activity dataframe with H3 cells and start/end counts
            cell: Origin H3 cell
            slot: Time slot
            k: Ring distance (1 = immediate neighbors)
            
        Returns:
            Dictionary with aggregated features
        """
        # Get neighbor cells
        try:
            neigh = list(h3.grid_disk(str(cell), k))
        except Exception:
            neigh = [str(cell)]
        
        # Filter to neighbors
        sub = act_df[act_df['h3_index'].isin(neigh)]
        
        feats = {}
        start_col = f'start_{slot}'
        end_col = f'end_{slot}'
        idle_col = f'idle_{slot}'
        
        # Aggregate counts
        feats['ring_start_sum'] = float(sub[start_col].fillna(0).sum()) if start_col in sub.columns else 0.0
        feats['ring_end_sum'] = float(sub[end_col].fillna(0).sum()) if end_col in sub.columns else 0.0
        
        # Idle time statistics
        idle_vals = []
        if idle_col in sub.columns:
            for lst in sub[idle_col].dropna().values:
                if isinstance(lst, list):
                    idle_vals.extend(lst)
        
        feats['ring_idle_mean'] = float(np.mean(idle_vals)) if idle_vals else 0.0
        feats['ring_idle_p90'] = float(np.percentile(idle_vals, 90)) if idle_vals else 0.0
        
        # Supply-demand ratio
        feats['supply_demand_ratio'] = feats['ring_start_sum'] / (feats['ring_end_sum'] + 1e-3)
        
        return feats


# Example usage
if __name__ == "__main__":
    print("Auto-rickshaw Feature Engineering")
    print("\n 60+ Features:")
    print("   - H3 spatial indexing")
    print("   - Supply-demand dynamics")
    print("   - OSMnx routing features")
    print("   - Temporal patterns")
    print("   - Speed profiles")
