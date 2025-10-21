#!/usr/bin/env python3
"""
BLMC AutoRickshaw ETA - V9.10 GEOGRAPHY-AWARE + FAILURE DETECTION
Target: 70-78 TTT Score (From 83.03)
Strategy: Geo-based corrections + proper failure detection + short-ride optimization
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from geopy.distance import geodesic
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ChampionshipMAPEEnsemble - Complete Implementation (Same as V9.8)
# ============================================================================
class ChampionshipMAPEEnsemble:
    """OOF-stacked ensemble with 56+ championship features"""
    
    def __init__(self, failure_threshold=9.0, timeout_minutes=30.0, n_splits=5, eps=1e-3):
        self.failure_threshold = float(failure_threshold)
        self.timeout_minutes = float(timeout_minutes)
        self.n_splits = int(n_splits)
        self.eps = float(eps)
        
        self.models_accept = []
        self.models_pickup = []
        self.models_trip = []
        self.blender_accept = None
        self.blender_pickup = None
        self.blender_trip = None
        self.accept_gate_folds = []
        self.accept_calibrator = None
        self.accept_gate_threshold = 0.5
        self.scaler = None
    
    def _build_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering - EXACT ORDER from training"""
        feats = {}
        
        feats["origin_lat"] = df["origin_lat"].values
        feats["origin_lon"] = df["origin_lon"].values
        feats["dest_lat"] = df["dest_lat"].values
        feats["dest_lon"] = df["dest_lon"].values
        
        distances = [geodesic((olat, olon), (dlat, dlon)).kilometers 
                    for olat, olon, dlat, dlon in zip(df["origin_lat"], df["origin_lon"], 
                                                       df["dest_lat"], df["dest_lon"])]
        feats["distance_km"] = np.array(distances, dtype=float)
        
        ts = pd.to_datetime(df["timestamp"])
        feats["hour"] = ts.dt.hour.values
        feats["dow"] = ts.dt.dayofweek.values
        feats["day"] = ts.dt.day.values
        feats["month"] = ts.dt.month.values
        
        feats["hour_sin"] = np.sin(2 * np.pi * feats["hour"] / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * feats["hour"] / 24)
        feats["dow_sin"] = np.sin(2 * np.pi * feats["dow"] / 7)
        feats["dow_cos"] = np.cos(2 * np.pi * feats["dow"] / 7)
        feats["day_sin"] = np.sin(2 * np.pi * feats["day"] / 31)
        feats["day_cos"] = np.cos(2 * np.pi * feats["day"] / 31)
        
        feats["is_weekend"] = (feats["dow"] >= 5).astype(int)
        feats["is_monday"] = (feats["dow"] == 0).astype(int)
        feats["is_friday"] = (feats["dow"] == 4).astype(int)
        feats["is_super_peak"] = np.isin(feats["hour"], [8, 9, 18, 19]).astype(int)
        feats["is_morning_peak"] = ((feats["hour"] >= 7) & (feats["hour"] <= 10)).astype(int)
        feats["is_evening_peak"] = ((feats["hour"] >= 17) & (feats["hour"] <= 21)).astype(int)
        feats["is_night"] = ((feats["hour"] >= 22) | (feats["hour"] <= 5)).astype(int)
        feats["is_business_hours"] = ((feats["hour"] >= 9) & (feats["hour"] <= 17) & 
                                     (feats["is_weekend"] == 0)).astype(int)
        feats["is_lunch_hour"] = np.isin(feats["hour"], [12, 13]).astype(int)
        
        rush = np.where(feats["is_super_peak"] == 1, 1.8,
                np.where(feats["is_evening_peak"] == 1, 1.56,
                    np.where(feats["is_morning_peak"] == 1, 1.30, 1.0)))
        feats["rush_intensity"] = rush
        feats["rush_hour_intensity"] = np.maximum(
            1 - np.abs(feats["hour"] - 9) / 4,
            1 - np.abs(feats["hour"] - 18) / 4
        ).clip(0, 1)
        
        city_center = (12.9716, 77.5946)
        tech_hubs = [(12.9352, 77.6245), (12.8467, 77.6627)]
        
        origin_center_dist = []
        dest_center_dist = []
        min_origin_tech = []
        min_dest_tech = []
        
        for olat, olon, dlat, dlon in zip(df["origin_lat"], df["origin_lon"], 
                                          df["dest_lat"], df["dest_lon"]):
            oc = geodesic((olat, olon), city_center).kilometers
            dc = geodesic((dlat, dlon), city_center).kilometers
            origin_center_dist.append(oc)
            dest_center_dist.append(dc)
            
            ot = min([geodesic((olat, olon), hub).kilometers for hub in tech_hubs])
            dt = min([geodesic((dlat, dlon), hub).kilometers for hub in tech_hubs])
            min_origin_tech.append(ot)
            min_dest_tech.append(dt)
        
        feats["origin_center_dist"] = np.array(origin_center_dist)
        feats["dest_center_dist"] = np.array(dest_center_dist)
        feats["min_origin_tech_dist"] = np.array(min_origin_tech)
        feats["min_dest_tech_dist"] = np.array(min_dest_tech)
        feats["avg_center_dist"] = (feats["origin_center_dist"] + feats["dest_center_dist"]) / 2
        
        feats["demand_intensity"] = np.where(feats["hour"] == 18, 2.2,
                                     np.where(feats["hour"] == 19, 2.0,
                                         np.where(feats["is_morning_peak"] == 1, 1.6, 1.0)))
        feats["supply_scarcity"] = feats["demand_intensity"] * (1.0 + feats["origin_center_dist"] / 20)
        feats["demand_pressure"] = feats["rush_intensity"] * np.where(feats["is_weekend"] == 1, 0.8, 1.5)
        
        feats["density_factor"] = np.maximum(0.6, 1.5 - feats["avg_center_dist"] / 15.0)
        feats["route_complexity"] = feats["distance_km"] * feats["density_factor"] * feats["rush_intensity"]
        feats["congestion_factor"] = np.where(feats["is_super_peak"] == 1, 1.8,
                                      np.where(feats["is_evening_peak"] == 1, 1.6,
                                          np.where(feats["is_morning_peak"] == 1, 1.4, 1.0)))
        
        feats["distance_cat"] = np.where(feats["distance_km"] < 2, 0,
                                np.where(feats["distance_km"] < 5, 1,
                                    np.where(feats["distance_km"] < 10, 2,
                                        np.where(feats["distance_km"] < 20, 3, 4))))
        feats["short_trip"] = (feats["distance_km"] < 2).astype(int)
        feats["medium_trip"] = ((feats["distance_km"] >= 2) & (feats["distance_km"] < 10)).astype(int)
        feats["long_trip"] = (feats["distance_km"] >= 10).astype(int)
        
        bearings = []
        for olat, olon, dlat, dlon in zip(df["origin_lat"], df["origin_lon"], 
                                          df["dest_lat"], df["dest_lon"]):
            bearing = np.degrees(np.arctan2(np.radians(dlon - olon), np.radians(dlat - olat)))
            if bearing < 0:
                bearing += 360
            bearings.append(bearing)
        
        feats["trip_bearing"] = np.array(bearings)
        feats["trip_bearing_sin"] = np.sin(np.radians(feats["trip_bearing"]))
        feats["trip_bearing_cos"] = np.cos(np.radians(feats["trip_bearing"]))
        
        feats["route_efficiency"] = np.minimum(1.0, feats["distance_km"] / (feats["distance_km"] * 1.3))
        
        base_speed = 20.2
        feats["estimated_speed"] = base_speed / feats["congestion_factor"]
        feats["time_per_km"] = 60.0 / feats["estimated_speed"]
        
        feats["weekend_demand_factor"] = np.where(feats["is_weekend"] == 1, 0.8, 1.2)
        feats["weekday_rush_factor"] = np.where(feats["is_weekend"] == 0, feats["rush_intensity"], 1.0)
        
        feats["peak_distance_interaction"] = feats["rush_intensity"] * feats["distance_km"]
        feats["center_rush_interaction"] = feats["avg_center_dist"] * feats["rush_intensity"]
        
        feats["supply_demand_balance"] = feats["demand_intensity"] / (feats["supply_scarcity"] + 0.1)
        feats["congestion_distance_factor"] = feats["congestion_factor"] * feats["distance_km"]
        feats["tech_hub_accessibility"] = 1.0 / (1.0 + np.minimum(feats["min_origin_tech_dist"], 
                                                                   feats["min_dest_tech_dist"]))
        
        return pd.DataFrame(feats)
    
    def predict_championship(self, origin_lat, origin_lon, dest_lat, dest_lon, timestamp):
        """Predict using loaded models"""
        row = pd.DataFrame([{
            "origin_lat": float(origin_lat),
            "origin_lon": float(origin_lon),
            "dest_lat": float(dest_lat),
            "dest_lon": float(dest_lon),
            "timestamp": pd.to_datetime(timestamp)
        }])
        
        feat = self._build_features_df(row)
        X = self.scaler.transform(feat)
        
        def comp(models):
            lgb_log = np.mean([m[0].predict(X) for m in models], axis=0)
            cat_log = np.mean([m[1].predict(X) for m in models], axis=0)
            q50_raw = np.mean([m[2].predict(X) for m in models], axis=0)
            Xb = np.vstack([lgb_log, cat_log, np.log1p(np.maximum(q50_raw, 1e-3))]).T
            return Xb
        
        a_log = self.blender_accept.predict(comp(self.models_accept))
        p_log = self.blender_pickup.predict(comp(self.models_pickup))
        t_log = self.blender_trip.predict(comp(self.models_trip))
        
        pa = float(np.expm1(np.maximum(a_log, np.log1p(1e-3))))
        pb = float(np.expm1(np.maximum(p_log, np.log1p(1e-3))))
        pc = float(np.expm1(np.maximum(t_log, np.log1p(1e-3))))
        
        gate_p = float(np.mean([g.predict_proba(X)[:,1] for g in self.accept_gate_folds], axis=0)[0])
        gate_p = float(self.accept_calibrator.predict([gate_p])[0])
        
        # Return raw predictions (don't apply gate logic here - V9.10 will handle it)
        pa = float(np.clip(pa, 0.1, 60.0))
        pb = float(np.clip(pb, 0.5, 60.0))
        pc = float(np.clip(pc, 1.0, 300.0))
        
        return {"acceptance_time": pa, "pickup_time": pb, "trip_time": pc, "gate_probability": gate_p}


class ChampionshipEnhancedETAPipeline:
    def __init__(self):
        self.ens = ChampionshipMAPEEnsemble()
        self.is_trained = False
    
    def predict_one(self, origin_lat, origin_lon, dest_lat, dest_lon, timestamp):
        if not self.is_trained:
            distance = geodesic((origin_lat, origin_lon), (dest_lat, dest_lon)).kilometers
            hour = pd.to_datetime(timestamp).hour
            base_acceptance = 1.2 if hour in [18,19] else 0.9 if 8<=hour<=10 else 0.7
            return {
                "acceptance_time": float(np.clip(base_acceptance, 0.3, 30.0)),
                "pickup_time": float(np.clip(4.2 + 0.1*distance, 0.5, 60.0)),
                "trip_time": float(np.clip(2.0*distance + 3.0, 1.0, 300.0)),
                "gate_probability": 0.0
            }
        return self.ens.predict_championship(origin_lat, origin_lon, dest_lat, dest_lon, timestamp)


def load_artifacts(save_dir="/app/championship_models"):
    print(f"Loading artifacts from: {save_dir}")
    
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Model directory not found: {save_dir}")
    
    pl = ChampionshipEnhancedETAPipeline()
    ens = pl.ens
    
    ens.models_accept = joblib.load(os.path.join(save_dir, "models_accept.joblib"))
    ens.models_pickup = joblib.load(os.path.join(save_dir, "models_pickup.joblib"))
    ens.models_trip = joblib.load(os.path.join(save_dir, "models_trip.joblib"))
    
    ens.blender_accept = joblib.load(os.path.join(save_dir, "blender_accept.joblib"))
    ens.blender_pickup = joblib.load(os.path.join(save_dir, "blender_pickup.joblib"))
    ens.blender_trip = joblib.load(os.path.join(save_dir, "blender_trip.joblib"))
    
    ens.scaler = joblib.load(os.path.join(save_dir, "scaler.joblib"))
    ens.accept_gate_folds = joblib.load(os.path.join(save_dir, "accept_gate_folds.joblib"))
    ens.accept_calibrator = joblib.load(os.path.join(save_dir, "accept_calibrator.joblib"))
    
    with open(os.path.join(save_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    
    ens.accept_gate_threshold = float(meta["accept_gate_threshold"])
    ens.failure_threshold = float(meta["failure_threshold"])
    ens.timeout_minutes = float(meta["timeout_minutes"])
    ens.n_splits = int(meta["n_splits"])
    ens.eps = float(meta["eps"])
    
    setattr(ens, "_trained_", True)
    pl.is_trained = True
    
    print("✅ All artifacts loaded successfully")
    return pl


# ============================================================================
# V9.10 Geography-Aware System
# ============================================================================
class GeographyAwareSystem:
    """V9.10: Advanced geo-based corrections for Bengaluru"""
    
    def __init__(self):
        # Bengaluru key locations
        self.CITY_CENTER = (12.9716, 77.5946)
        self.AIRPORT = (13.1986, 77.7066)
        self.WHITEFIELD = (12.9698, 77.7500)
        self.KORAMANGALA = (12.9352, 77.6245)
        self.ELECTRONIC_CITY = (12.8467, 77.6627)
        self.INDIRANAGAR = (12.9784, 77.6408)
        self.MARATHAHALLI = (12.9591, 77.7010)
        self.JAYANAGAR = (12.9250, 77.5838)
        self.SILK_BOARD = (12.9175, 77.6223)
        self.OUTER_RING_ROAD = (12.9352, 77.6245)
        
        # High-density zones (fast acceptance, quick pickup)
        self.high_density_zones = [
            self.CITY_CENTER,
            self.KORAMANGALA,
            self.INDIRANAGAR,
            self.JAYANAGAR
        ]
        
        # Tech hubs (peak hour = high demand)
        self.tech_hubs = [
            self.WHITEFIELD,
            self.ELECTRONIC_CITY,
            self.KORAMANGALA
        ]
        
        # Railway stations
        self.railway_stations = [
            (12.9766, 77.5993),  # Bangalore City Junction
            (13.0743, 77.5858),  # Yeshwantpur
            (12.9180, 77.5836),  # Cantonment
        ]
    
    def get_zone_type(self, lat, lon):
        """Identify zone type based on proximity"""
        location = (lat, lon)
        
        # Check if in high-density zone (within 2km)
        for zone in self.high_density_zones:
            if geodesic(location, zone).kilometers < 2.0:
                return 'high_density'
        
        # Check if in tech hub (within 3km)
        for hub in self.tech_hubs:
            if geodesic(location, hub).kilometers < 3.0:
                return 'tech_hub'
        
        # Check distance from city center
        center_dist = geodesic(location, self.CITY_CENTER).kilometers
        
        if center_dist < 5:
            return 'central'
        elif center_dist < 10:
            return 'mid_city'
        else:
            return 'suburban'
    
    def estimate_driver_density(self, lat, lon, hour):
        """Estimate driver availability based on location"""
        
        # Distance to nearest high-activity zone
        distances_to_zones = [
            geodesic((lat, lon), zone).kilometers
            for zone in self.high_density_zones
        ]
        
        nearest_zone_dist = min(distances_to_zones)
        
        # Driver density score (1.0 = high, 0.0 = low)
        if nearest_zone_dist < 1:
            density_score = 1.0
        elif nearest_zone_dist < 3:
            density_score = 0.8
        elif nearest_zone_dist < 5:
            density_score = 0.6
        elif nearest_zone_dist < 10:
            density_score = 0.4
        else:
            density_score = 0.2
        
        # Adjust for time of day
        if hour in [8, 9, 18, 19]:  # Peak demand
            density_score *= 0.7  # Fewer available drivers
        elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Late night
            density_score *= 0.5  # Very few drivers
        
        return density_score
    
    def get_route_characteristics(self, origin_lat, origin_lon, dest_lat, dest_lon):
        """Analyze route direction and corridors"""
        
        # Calculate bearing (direction)
        bearing = np.degrees(np.arctan2(
            np.radians(dest_lon - origin_lon),
            np.radians(dest_lat - origin_lat)
        ))
        if bearing < 0:
            bearing += 360
        
        # Classify direction
        if 45 <= bearing < 135:
            direction = 'eastward'
        elif 135 <= bearing < 225:
            direction = 'southward'
        elif 225 <= bearing < 315:
            direction = 'westward'
        else:
            direction = 'northward'
        
        # Check if route crosses major corridors
        route_midpoint = (
            (origin_lat + dest_lat) / 2,
            (origin_lon + dest_lon) / 2
        )
        
        # Known congestion corridors in Bengaluru
        silk_board = geodesic(route_midpoint, self.SILK_BOARD).kilometers < 2
        orr = geodesic(route_midpoint, self.OUTER_RING_ROAD).kilometers < 3
        
        return {
            'direction': direction,
            'crosses_silk_board': silk_board,
            'crosses_orr': orr,
            'bearing': bearing
        }
    
    def check_transport_hub(self, lat, lon):
        """Check if location is near major transport hub"""
        
        airport_dist = geodesic((lat, lon), self.AIRPORT).kilometers
        
        station_dists = [
            geodesic((lat, lon), station).kilometers
            for station in self.railway_stations
        ]
        nearest_station_dist = min(station_dists)
        
        if airport_dist < 2:
            return 'airport'
        elif nearest_station_dist < 1:
            return 'railway'
        else:
            return None


# ============================================================================
# V9.10 Heuristics (Same as V9.8)
# ============================================================================
class V9EnhancedHeuristics:
    """V9.10 heuristics"""
    
    def __init__(self):
        self.SILK_BOARD = (12.9175, 77.6223)
        self.OUTER_RING_ROAD = (12.9352, 77.6245)
        self.WHITEFIELD = (12.9698, 77.7500)
        self.CITY_CENTER = (12.9716, 77.5946)
    
    def calculate_realistic_trip_time(self, distance, hour, is_weekend):
        is_peak = hour in [8, 9, 18, 19]
        
        if is_peak and not is_weekend:
            if distance < 3:
                speed = 12
            elif distance < 7:
                speed = 18
            else:
                speed = 25
        else:
            if distance < 3:
                speed = 20
            elif distance < 7:
                speed = 28
            else:
                speed = 35
        
        if is_weekend and not is_peak:
            speed *= 1.15
        
        time_minutes = (distance / speed) * 60
        return time_minutes
    
    def _route_near_point(self, olat, olon, dlat, dlon, point, threshold_km):
        mid_lat = (olat + dlat) / 2
        mid_lon = (olon + dlon) / 2
        dist = geodesic((mid_lat, mid_lon), point).kilometers
        return dist < threshold_km
    
    def apply_corridor_logic(self, trip_pred, origin_lat, origin_lon, dest_lat, dest_lon, hour):
        crosses_silk_board = self._route_near_point(origin_lat, origin_lon, dest_lat, dest_lon, 
                                                     self.SILK_BOARD, 2.0)
        crosses_orr = self._route_near_point(origin_lat, origin_lon, dest_lat, dest_lon, 
                                             self.OUTER_RING_ROAD, 3.0)
        
        penalty = 1.0
        if crosses_silk_board and hour in [8, 9, 18, 19]:
            penalty *= 1.20
        if crosses_orr and hour in [8, 9, 18, 19]:
            penalty *= 1.12
        
        return trip_pred * penalty
    
    def apply_minimum_time_floor(self, trip_pred, distance):
        absolute_minimum = (distance / 15.0) * 60
        
        if distance < 2:
            floor = 2.5
        elif distance < 5:
            floor = 4.0
        elif distance < 10:
            floor = 7.0
        else:
            floor = max(absolute_minimum * 0.8, 10.0)
        
        return max(trip_pred, floor)
    
    def smart_acceptance_clipping(self, acceptance_pred, hour, distance):
        if hour in [18, 19]:
            if distance > 15:
                return min(acceptance_pred, 2.5)
            elif distance > 10:
                return min(acceptance_pred, 1.9)
            else:
                return min(acceptance_pred, 1.6)
        elif hour in [8, 9]:
            if distance > 12:
                return min(acceptance_pred, 2.1)
            else:
                return min(acceptance_pred, 1.6)
        else:
            return min(acceptance_pred, 1.3)
    
    def adjust_pickup_by_distance(self, pickup_pred, distance):
        if distance > 15:
            return pickup_pred * 1.12
        elif distance > 10:
            return pickup_pred * 1.06
        elif distance < 3:
            return pickup_pred * 0.93
        return pickup_pred
    
    def apply_weekday_pattern(self, trip_pred, is_weekend, hour):
        if is_weekend:
            if hour in [8, 9, 10]:
                return trip_pred * 0.85
            elif hour in [20, 21, 22]:
                return trip_pred * 1.08
            else:
                return trip_pred * 0.92
        return trip_pred
    
    def apply_variance_penalty(self, trip_pred, distance, hour):
        if hour in [18, 19] and distance > 10:
            return trip_pred * 0.88
        elif hour in [8, 9, 18, 19] and distance > 7:
            return trip_pred * 0.93
        return trip_pred


# ============================================================================
# V9.10 Route Calibrator (Same as V9.8)
# ============================================================================
class V9RouteCalibrator:
    """V9.10 route intelligence"""
    
    def __init__(self):
        self.heuristics = V9EnhancedHeuristics()
        
        self.calibration_factors = {
            'acceptance': 0.80,
            'pickup': 0.86,
            'trip': 0.75
        }
        
        self.route_scaling = {
            'easy_routes': {'acceptance': 0.88, 'pickup': 0.88, 'trip': 0.78},
            'medium_routes': {'acceptance': 0.84, 'pickup': 0.84, 'trip': 0.74},
            'hard_routes': {'acceptance': 0.78, 'pickup': 0.81, 'trip': 0.68},
            'failure_routes': {'acceptance': 0.82, 'pickup': 0.84, 'trip': 0.70}
        }
        
        self.variance_bounds = {
            'acceptance': {'min': 0.3, 'max': 30.0},  # V9.10: Allow up to 30!
            'pickup': {'min': 0.5, 'max': 5.0},
            'trip': {'min': 1.0, 'max': 18.0}
        }
    
    def apply_distance_penalty(self, trip_pred, distance):
        if distance > 15:
            return trip_pred * 0.80
        elif distance > 10:
            return trip_pred * 0.85
        elif distance > 7:
            return trip_pred * 0.90
        elif distance > 5:
            return trip_pred * 0.93
        return trip_pred
    
    def classify_route(self, distance, hour, center_dist):
        if distance < 3:
            base = 'easy'
        elif distance > 15:
            base = 'hard'
        else:
            base = 'medium'
        
        if hour in [18, 19]:
            if base == 'easy':
                return 'medium_routes'
            elif base == 'medium':
                return 'hard_routes'
            else:
                return 'failure_routes'
        elif 8 <= hour <= 10:
            if base == 'hard':
                return 'failure_routes'
            return base + '_routes'
        return base + '_routes'
    
    def apply_route_calibration(self, ensemble_pred, distance, hour, center_dist, 
                                origin_lat, origin_lon, dest_lat, dest_lon, is_weekend):
        route_class = self.classify_route(distance, hour, center_dist)
        scaling = self.route_scaling[route_class]
        
        calibrated_acceptance = (0.40 * ensemble_pred['acceptance_time'] + 
                                0.60 * ensemble_pred['acceptance_time'] * scaling['acceptance'])
        calibrated_pickup = (0.40 * ensemble_pred['pickup_time'] + 
                            0.60 * ensemble_pred['pickup_time'] * scaling['pickup'])
        calibrated_trip = (0.35 * ensemble_pred['trip_time'] + 
                          0.65 * ensemble_pred['trip_time'] * scaling['trip'])
        
        calibrated_trip = self.apply_distance_penalty(calibrated_trip, distance)
        
        physics_time = self.heuristics.calculate_realistic_trip_time(distance, hour, is_weekend)
        calibrated_trip = 0.70 * calibrated_trip + 0.30 * physics_time
        
        calibrated_trip = self.heuristics.apply_corridor_logic(
            calibrated_trip, origin_lat, origin_lon, dest_lat, dest_lon, hour
        )
        
        calibrated_trip = self.heuristics.apply_variance_penalty(calibrated_trip, distance, hour)
        calibrated_trip = self.heuristics.apply_weekday_pattern(calibrated_trip, is_weekend, hour)
        calibrated_trip = self.heuristics.apply_minimum_time_floor(calibrated_trip, distance)
        
        calibrated_acceptance = self.heuristics.smart_acceptance_clipping(calibrated_acceptance, hour, distance)
        calibrated_pickup = self.heuristics.adjust_pickup_by_distance(calibrated_pickup, distance)
        
        calibrated_acceptance = np.clip(calibrated_acceptance,
                                       self.variance_bounds['acceptance']['min'],
                                       self.variance_bounds['acceptance']['max'])
        calibrated_pickup = np.clip(calibrated_pickup,
                                   self.variance_bounds['pickup']['min'],
                                   self.variance_bounds['pickup']['max'])
        calibrated_trip = np.clip(calibrated_trip,
                                 self.variance_bounds['trip']['min'],
                                 self.variance_bounds['trip']['max'])
        
        return {
            'acceptance_time': calibrated_acceptance,
            'pickup_time': calibrated_pickup,
            'trip_time': calibrated_trip
        }


# ============================================================================
# V9.10 Hybrid Predictor - GEOGRAPHY-AWARE + FAILURE DETECTION
# ============================================================================
class V9HybridPredictor:
    """V9.10: Geography-aware + proper failure detection + debug prints"""
    
    def __init__(self):
        print("Loading championship ensemble artifacts...")
        self.championship_pipeline = load_artifacts("/app/championship_models")
        print("✅ Championship ensemble loaded")
        
        self.v9_calibrator = V9RouteCalibrator()
        self.geo_system = GeographyAwareSystem()
        
        # V9.10: Base scaling (from V9.8)
        self.competitive_scaling = {
            'acceptance': 0.81,
            'pickup': 0.78,
            'trip': 0.64
        }
        
        # V9.10: Failure detection thresholds
        self.failure_gate_threshold = 0.55  # Gate probability threshold
        self.failure_ensemble_threshold = 7.5  # Raw ensemble prediction threshold
        
        # V9.10: Short-ride correction factors
        self.short_ride_factors = {
            'ultra_short': {'acceptance': 0.50, 'pickup': 0.55, 'trip': 0.45},  # <1 km
            'very_short': {'acceptance': 0.60, 'pickup': 0.65, 'trip': 0.55},   # 1-2 km
            'short': {'acceptance': 0.72, 'pickup': 0.76, 'trip': 0.68},        # 2-4 km
            'medium': {'acceptance': 0.82, 'pickup': 0.82, 'trip': 0.75}        # 4-8 km
        }
        
        # Debug tracking
        self.failure_count = 0
        self.failure_details = []
        self.geo_correction_stats = []
        
        print(f"\n🌍 V9.10 GEOGRAPHY-AWARE + FAILURE DETECTION")
        print(f"   Base scaling: {self.competitive_scaling}")
        print(f"   Failure thresholds: gate={self.failure_gate_threshold}, ensemble={self.failure_ensemble_threshold}")
    
    def _classify_ride_length(self, distance):
        """Classify ride by distance"""
        if distance < 1:
            return 'ultra_short'
        elif distance < 2:
            return 'very_short'
        elif distance < 4:
            return 'short'
        elif distance < 8:
            return 'medium'
        else:
            return 'long'
    
    def _detect_failure(self, ensemble_pred, gate_p, distance, hour, is_weekend):
        """V9.10: Detect if ride likely to fail acceptance"""
        
        # Multiple signals for failure detection
        high_gate = gate_p >= self.failure_gate_threshold
        high_ensemble = ensemble_pred['acceptance_time'] >= self.failure_ensemble_threshold
        
        # Very long rides during peak hours
        extreme_distance_peak = (distance > 20) and (hour in [18, 19]) and (not is_weekend)
        
        # Combined failure detection
        is_failure = high_gate or high_ensemble or extreme_distance_peak
        
        return is_failure, {
            'high_gate': high_gate,
            'high_ensemble': high_ensemble,
            'extreme_distance_peak': extreme_distance_peak,
            'gate_p': round(gate_p, 3),
            'ensemble_pa': round(ensemble_pred['acceptance_time'], 2)
        }
    
    def predict_v9_hybrid(self, origin_lat, origin_lon, dest_lat, dest_lon, timestamp, rid=None):
        """V9.10: Full prediction with geo-awareness + failure detection"""
        
        # Get ensemble prediction
        ensemble_pred = self.championship_pipeline.predict_one(
            origin_lat, origin_lon, dest_lat, dest_lon, timestamp
        )
        
        gate_p = ensemble_pred.get('gate_probability', 0.0)
        
        # Calculate features
        distance = geodesic((origin_lat, origin_lon), (dest_lat, dest_lon)).kilometers
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        city_center = (12.9716, 77.5946)
        center_dist = geodesic((origin_lat, origin_lon), city_center).kilometers
        
        # === V9.10 NEW: Failure Detection ===
        is_failure, failure_info = self._detect_failure(ensemble_pred, gate_p, distance, hour, is_weekend)
        
        if is_failure:
            self.failure_count += 1
            debug_info = {
                'rid': rid,
                'distance': round(distance, 2),
                'hour': hour,
                **failure_info
            }
            self.failure_details.append(debug_info)
            
            print(f"   🚨 FAILURE DETECTED [RID {rid}]: dist={debug_info['distance']}km, " +
                  f"hour={hour}, gate_p={debug_info['gate_p']}, " +
                  f"ensemble_pa={debug_info['ensemble_pa']}, " +
                  f"triggers={[k for k,v in failure_info.items() if k.startswith(('high_', 'extreme_')) and v]}")
            
            # Return spec-compliant failure pattern
            return {
                'acceptance_time': 30.0,
                'pickup_time': 1.0,
                'trip_time': 1.0
            }
        
        # === Normal Prediction Flow ===
        
        # Apply base calibration
        calibrated_pred = self.v9_calibrator.apply_route_calibration(
            ensemble_pred, distance, hour, center_dist,
            origin_lat, origin_lon, dest_lat, dest_lon, is_weekend
        )
        
        # Apply base scaling
        pred = {
            'acceptance_time': calibrated_pred['acceptance_time'] * self.competitive_scaling['acceptance'],
            'pickup_time': calibrated_pred['pickup_time'] * self.competitive_scaling['pickup'],
            'trip_time': calibrated_pred['trip_time'] * self.competitive_scaling['trip']
        }
        
        # === V9.10 GEO-BASED ENHANCEMENTS ===
        
        # 1. Zone identification
        origin_zone = self.geo_system.get_zone_type(origin_lat, origin_lon)
        dest_zone = self.geo_system.get_zone_type(dest_lat, dest_lon)
        
        # 2. Driver density
        density = self.geo_system.estimate_driver_density(origin_lat, origin_lon, hour)
        
        # 3. Route characteristics
        route_info = self.geo_system.get_route_characteristics(
            origin_lat, origin_lon, dest_lat, dest_lon
        )
        
        # 4. Transport hubs
        origin_hub = self.geo_system.check_transport_hub(origin_lat, origin_lon)
        dest_hub = self.geo_system.check_transport_hub(dest_lat, dest_lon)
        
        # Track corrections for debug
        corrections_applied = []
        original_pred = pred.copy()
        
        # Apply zone-based corrections
        if origin_zone == 'high_density':
            pred['acceptance_time'] *= 0.75
            pred['pickup_time'] *= 0.85
            corrections_applied.append(f"high_density_origin")
        elif origin_zone == 'suburban':
            pred['acceptance_time'] *= 1.20
            pred['pickup_time'] *= 1.15
            corrections_applied.append(f"suburban_origin")
        
        # Same zone short trips = very fast
        if origin_zone == dest_zone and origin_zone == 'high_density' and distance < 5:
            pred['acceptance_time'] *= 0.70
            pred['trip_time'] *= 0.75
            corrections_applied.append("intra_zone_short")
        
        # Driver density corrections
        if density > 0.8:
            pred['acceptance_time'] *= 0.70
            pred['pickup_time'] *= 0.85
            corrections_applied.append("high_driver_density")
        elif density < 0.3:
            pred['acceptance_time'] *= 1.30
            pred['pickup_time'] *= 1.20
            corrections_applied.append("low_driver_density")
        
        # Route directionality corrections
        if route_info['direction'] == 'eastward' and hour in [8, 9]:
            pred['trip_time'] *= 1.15
            corrections_applied.append("eastward_morning_peak")
        elif route_info['direction'] == 'westward' and hour in [18, 19]:
            pred['trip_time'] *= 1.20
            corrections_applied.append("westward_evening_peak")
        
        if route_info['crosses_silk_board'] and hour in [8, 9, 18, 19]:
            pred['trip_time'] *= 1.25
            corrections_applied.append("silk_board_congestion")
        
        # Transport hub logic
        if origin_hub == 'railway':
            pred['acceptance_time'] *= 0.70
            pred['pickup_time'] *= 0.80
            corrections_applied.append("railway_origin")
        elif origin_hub == 'airport' or dest_hub == 'airport':
            pred['acceptance_time'] *= 0.85
            pred['trip_time'] *= 1.10
            corrections_applied.append("airport_trip")
        
        # Short-ride specific corrections
        ride_type = self._classify_ride_length(distance)
        if ride_type in self.short_ride_factors:
            factors = self.short_ride_factors[ride_type]
            pred['acceptance_time'] *= factors['acceptance']
            pred['pickup_time'] *= factors['pickup']
            pred['trip_time'] *= factors['trip']
            corrections_applied.append(f"short_ride_{ride_type}")
            
            # Extra off-peak reduction
            if not (hour in [8, 9, 18, 19]) and distance < 5:
                pred['acceptance_time'] *= 0.88
                pred['trip_time'] *= 0.90
                corrections_applied.append("short_offpeak")
            
            # Weekend reduction
            if is_weekend and distance < 5:
                pred['acceptance_time'] *= 0.92
                pred['trip_time'] *= 0.93
                corrections_applied.append("short_weekend")
        
        # Final bounds
        pred['acceptance_time'] = max(0.3, min(pred['acceptance_time'], 9.0))
        pred['pickup_time'] = max(0.5, min(pred['pickup_time'], 60.0))
        pred['trip_time'] = max(1.0, min(pred['trip_time'], 300.0))
        
        # Track geo corrections
        if corrections_applied:
            self.geo_correction_stats.append({
                'rid': rid,
                'distance': round(distance, 2),
                'origin_zone': origin_zone,
                'dest_zone': dest_zone,
                'density': round(density, 2),
                'corrections': corrections_applied,
                'before': {k: round(v, 2) for k, v in original_pred.items()},
                'after': {k: round(v, 2) for k, v in pred.items()}
            })
        
        return pred


def parse_coordinates(coord_str):
    clean_str = coord_str.strip('()').strip('"').replace(' ', '')
    lat_str, lon_str = clean_str.split(',')
    return float(lat_str), float(lon_str)


def load_input_data(input_path):
    print(f"Loading input from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} requests")
    
    processed = []
    for _, row in df.iterrows():
        start_lat, start_lon = parse_coordinates(row['ride_start_location'])
        end_lat, end_lon = parse_coordinates(row['ride_end_location'])
        time_str = row['ride_request_time']
        timestamp = pd.Timestamp(f'2025-08-15 {time_str}')
        
        processed.append({
            'rid': row['rid'],
            'timestamp': timestamp,
            'origin_lat': start_lat,
            'origin_lon': start_lon,
            'dest_lat': end_lat,
            'dest_lon': end_lon
        })
    
    return pd.DataFrame(processed)


def main():
    parser = argparse.ArgumentParser(description='BLMC V9.10 Geography-Aware + Failure Detection')
    parser.add_argument('--input-json', default='/app/data/input.csv')
    parser.add_argument('--output-json', default='/app/out/output.json')
    args = parser.parse_args()
    
    print("🏆 BLMC AutoRickshaw ETA - V9.10 GEOGRAPHY-AWARE + FAILURE DETECTION")
    print("🌍 Geo-based corrections + Zone intelligence + Driver density")
    print("🚨 Proper failure detection (outputs 30, 1, 1 pattern)")
    print("🎯 Target: 70-78 TTT Score")
    print("=" * 75)
    
    possible_input_paths = ['/app/data/input.csv', '/app/data.csv', args.input_json]
    
    if os.path.exists('/app/data'):
        try:
            for f in os.listdir('/app/data'):
                if f.endswith('.csv'):
                    possible_input_paths.insert(0, f'/app/data/{f}')
        except:
            pass
    
    input_path = None
    for path in possible_input_paths:
        if os.path.exists(path):
            input_path = path
            break
    
    if not input_path:
        print(f"\n❌ ERROR: Could not find input CSV file!")
        sys.exit(1)
    
    print(f"\n✅ Using input file: {input_path}")
    
    args.output_json = '/app/out/output.json'
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Output directory ready: {output_dir}")
    
    try:
        test_data = load_input_data(input_path)
        predictor = V9HybridPredictor()
        
        print(f"\n🔍 Generating V9.10 predictions for {len(test_data)} requests...")
        print("=" * 75)
        
        results = {}
        for _, row in test_data.iterrows():
            try:
                pred = predictor.predict_v9_hybrid(
                    row['origin_lat'], row['origin_lon'],
                    row['dest_lat'], row['dest_lon'],
                    row['timestamp'],
                    rid=int(row['rid'])
                )
                
                results[int(row['rid'])] = {
                    "pa": round(pred['acceptance_time'], 3),
                    "pb": round(pred['pickup_time'], 3),
                    "pc": round(pred['trip_time'], 3)
                }
                
            except Exception as e:
                print(f"⚠️ Error on request {row['rid']}: {e}")
                results[int(row['rid'])] = {"pa": 1.0, "pb": 2.0, "pc": 6.0}
        
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("=" * 75)
        print(f"✅ Saved {len(results)} predictions to {args.output_json}")
        
        # === V9.10 COMPREHENSIVE DIAGNOSTICS ===
        print(f"\n📊 V9.10 GEOGRAPHY-AWARE + FAILURE DETECTION - DIAGNOSTICS:")
        print(f"=" * 75)
        
        # Failure detection summary
        print(f"\n🚨 FAILURE DETECTION:")
        print(f"   Detected: {predictor.failure_count} / {len(test_data)} " +
              f"({100*predictor.failure_count/len(test_data):.1f}%)")
        
        if predictor.failure_count > 0:
            print(f"\n   Details:")
            for detail in predictor.failure_details:
                print(f"      {detail}")
        
        # Geo correction summary
        print(f"\n🌍 GEO-BASED CORRECTIONS APPLIED:")
        print(f"   Rides with corrections: {len(predictor.geo_correction_stats)} / {len(test_data)}")
        
        if len(predictor.geo_correction_stats) > 0:
            print(f"\n   Sample corrections:")
            for i, stat in enumerate(predictor.geo_correction_stats[:5], 1):
                print(f"      #{i} [RID {stat['rid']}]: {stat['distance']}km, " +
                      f"zone={stat['origin_zone']}→{stat['dest_zone']}, " +
                      f"density={stat['density']}")
                print(f"         Applied: {', '.join(stat['corrections'])}")
                print(f"         Before: {stat['before']}")
                print(f"         After: {stat['after']}")
        
        # Overall prediction summary
        preds_df = pd.DataFrame.from_dict(results, orient='index')
        normal_rides = preds_df[preds_df['pa'] < 30.0]
        failure_rides = preds_df[preds_df['pa'] == 30.0]
        
        print(f"\n📊 PREDICTION SUMMARY:")
        if len(normal_rides) > 0:
            print(f"   Normal Rides ({len(normal_rides)}):")
            print(f"      Acceptance: {normal_rides['pa'].mean():.2f} ± {normal_rides['pa'].std():.2f} min")
            print(f"      Pickup: {normal_rides['pb'].mean():.2f} ± {normal_rides['pb'].std():.2f} min")
            print(f"      Trip: {normal_rides['pc'].mean():.2f} ± {normal_rides['pc'].std():.2f} min")
        
        if len(failure_rides) > 0:
            print(f"\n   Failure Rides ({len(failure_rides)}):")
            print(f"      All set to: pa=30.0, pb=1.0, pc=1.0 (spec-compliant)")
        
        print(f"\n📈 COMPARISON TO V9.8:")
        print(f"   V9.8 (83.03): No failure detection, no geo-awareness")
        print(f"   V9.10: + Failure detection + Geo corrections")
        print(f"   Expected: 70-78 score (5-13 point improvement)")
        
        print("\n🚀 V9.10 geography-aware submission ready!")
        print("=" * 75)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

class AutoETAPredictor:
    """
    Simplified interface for Task 3 multi-modal integration.
    Wraps V9HybridPredictor (V9.10) with clean API.
    """
    
    def __init__(self):
        """Initialize the V9.10 Geography-Aware predictor"""
        print("🚗 Loading AutoETAPredictor (V9.10 Geography-Aware)...")
        self.predictor = V9HybridPredictor()
        print("✅ AutoETAPredictor ready for Task 3")
    
    def predict_auto_ride(self, origin_lat, origin_lon, dest_lat, dest_lon, timestamp):
        """
        Predict autorickshaw ride components for Task 3
        
        Args:
            origin_lat (float): Origin latitude
            origin_lon (float): Origin longitude
            dest_lat (float): Destination latitude
            dest_lon (float): Destination longitude
            timestamp (datetime): Request time in IST timezone
            
        Returns:
            dict: {
                'acceptance_time': float,  # Minutes
                'pickup_time': float,      # Minutes
                'trip_time': float         # Minutes
            }
        """
        # Use V9.10 predictor with all geo-awareness and failure detection
        result = self.predictor.predict_v9_hybrid(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            timestamp=timestamp,
            rid=None
        )
        
        # Extract only the 3 components needed for Task 3
        return {
            'acceptance_time': float(result['acceptance_time']),
            'pickup_time': float(result['pickup_time']),
            'trip_time': float(result['trip_time'])
        }