#!/usr/bin/env python3
"""
Bengaluru Bus ETA Prediction - Docker Submission
v27 FINAL: Scale 0.15 + Distance/Rush Hour Heuristics
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings('ignore')

from embedded_data import ROUTE_TO_STOPS, STOPS_DATA

print("✅ Loaded embedded route and stop data")

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points in kilometers"""
    R = 6371.0
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

class SubmissionValidator:
    @staticmethod
    def enforce_monotonic_cumulative_times(cumulative_etas):
        """Ensure cumulative ETAs are strictly increasing"""
        corrected = [cumulative_etas[0]]
        for i in range(1, len(cumulative_etas)):
            min_required = corrected[-1] + 0.5
            corrected.append(max(cumulative_etas[i], min_required))
        return corrected

validator = SubmissionValidator()

print("📂 Loading trained models...")

MODEL_DIR = './models/'

lgb_models = []
for i in range(5):
    model = lgb.Booster(model_file=f'{MODEL_DIR}lgb_fold{i}.txt')
    lgb_models.append(model)

cb_models = []
for i in range(5):
    model = cb.CatBoostRegressor()
    model.load_model(f'{MODEL_DIR}cb_fold{i}.cbm')
    cb_models.append(model)

xgb_models = []
for i in range(5):
    model = xgb.XGBRegressor()
    model.load_model(f'{MODEL_DIR}xgb_fold{i}.json')
    xgb_models.append(model)

ensemble_weights = joblib.load(f'{MODEL_DIR}ensemble_weights.pkl')
feature_names = joblib.load(f'{MODEL_DIR}feature_names.pkl')
historical_lookup = joblib.load(f'{MODEL_DIR}historical_lookup.pkl')

print(f"✅ Loaded models: {len(lgb_models)} LGB, {len(cb_models)} CB, {len(xgb_models)} XGB")
print(f"✅ Historical patterns: {len(historical_lookup):,}")
print(f"✅ Routes in embedded data: {len(ROUTE_TO_STOPS)}")
print(f"✅ Stops in embedded data: {len(STOPS_DATA)}")

try:
    cb_feature_names = cb_models[0].feature_names_
    print(f"✅ CatBoost feature order: {len(cb_feature_names)} features")
except:
    cb_feature_names = feature_names
    print("⚠️ Using default feature order for CatBoost")

# ⚙️ v27 FINAL TUNING
SCALE_FACTOR = 0.15  # Base scale factor
print(f"⚙️ v27 FINAL Scale: {SCALE_FACTOR} + Heuristics")

def extract_features_for_stop(trajectory, target_stop, stop_sequence_position, 
                               cumulative_time_so_far, historical_lookup, total_future_stops):
    """Extract ALL features for one stop prediction"""
    
    route_id = trajectory.get('route_id', 0)
    current_time = trajectory.get('current_time', pd.Timestamp.now('UTC'))
    hour = current_time.hour
    dow = current_time.dayofweek
    month = current_time.month
    
    current_speed = trajectory.get('current_speed', 20.0)
    current_lat = trajectory.get('current_lat', 12.95)
    current_lon = trajectory.get('current_lon', 77.58)
    
    target_lat = target_stop.get('lat', 12.95)
    target_lon = target_stop.get('lon', 77.58)
    target_stop_id = target_stop.get('stop_id', 0)
    
    lat_diff = target_lat - current_lat
    lon_diff = target_lon - current_lon
    
    straight_line_dist_km = haversine_distance(current_lat, current_lon, target_lat, target_lon)
    
    center_lat, center_lon = 12.9716, 77.5946
    dist_from_center = sqrt((current_lat - center_lat)**2 + (current_lon - center_lon)**2) * 111
    
    lookup_key = (route_id, stop_sequence_position, hour, dow)
    if lookup_key in historical_lookup:
        hist = historical_lookup[lookup_key]
        historical_eta_mean = hist.get('historical_eta_mean', 15.0)
        historical_eta_median = hist.get('historical_eta_median', 12.0)
        historical_eta_std = hist.get('historical_eta_std', 5.0)
        historical_eta_min = hist.get('historical_eta_min', 5.0)
        historical_eta_max = hist.get('historical_eta_max', 30.0)
        historical_sample_count = hist.get('historical_sample_count', 0)
        historical_confidence = min(historical_sample_count / 50.0, 1.0)
    else:
        historical_eta_mean = 15.0
        historical_eta_median = 12.0
        historical_eta_std = 5.0
        historical_eta_min = 5.0
        historical_eta_max = 30.0
        historical_sample_count = 0
        historical_confidence = 0.0
    
    historical_eta_range = historical_eta_max - historical_eta_min
    historical_eta_cv = historical_eta_std / (historical_eta_mean + 1e-6)
    
    rush_hour_intensity = 0.0
    if 7 <= hour <= 9:
        rush_hour_intensity = 1.0 - abs(hour - 8) / 2.0
    elif 17 <= hour <= 20:
        rush_hour_intensity = 1.0 - abs(hour - 18.5) / 2.5
    rush_hour_intensity = max(0.0, min(1.0, rush_hour_intensity))
    
    avg_recent_speed = trajectory.get('avg_recent_speed', current_speed)
    speed_volatility = trajectory.get('speed_volatility', abs(current_speed - avg_recent_speed))
    max_recent_speed = trajectory.get('max_recent_speed', current_speed * 1.2)
    recent_acceleration = trajectory.get('recent_acceleration', 0.0)
    movement_smoothness = trajectory.get('movement_smoothness', 1.0 - min(speed_volatility / 20.0, 1.0))
    distance_from_start = trajectory.get('distance_from_start', cumulative_time_so_far * current_speed / 60.0)
    
    speed_vs_city_avg = current_speed / 25.0
    speed_stopped = 1 if current_speed < 5 else 0
    speed_slow = 1 if 5 <= current_speed < 15 else 0
    speed_normal = 1 if 15 <= current_speed < 35 else 0
    speed_fast = 1 if current_speed >= 35 else 0
    
    lat_normalized = (current_lat - 12.85) / 0.3
    lon_normalized = (current_lon - 77.45) / 0.3
    is_city_center = 1 if dist_from_center < 5 else 0
    is_suburban = 1 if 5 <= dist_from_center < 15 else 0
    is_outer_city = 1 if dist_from_center >= 15 else 0
    
    is_distant_target = 1 if straight_line_dist_km > 5 else 0
    is_nearby_target = 1 if straight_line_dist_km < 1 else 0
    is_very_close_target = 1 if straight_line_dist_km < 0.2 else 0
    dist_at_stop = 1 if straight_line_dist_km < 0.05 else 0
    dist_very_close = 1 if 0.05 <= straight_line_dist_km < 0.5 else 0
    dist_close = 1 if 0.5 <= straight_line_dist_km < 2 else 0
    dist_medium = 1 if 2 <= straight_line_dist_km < 5 else 0
    dist_far = 1 if straight_line_dist_km >= 5 else 0
    
    traffic_density_score = trajectory.get('traffic_density_score', 0.0)
    weekend_factor = 1.0
    stops_remaining = max(0, total_future_stops - stop_sequence_position - 1)
    
    features = {
        'hour': hour, 'day_of_week': dow, 'month': month,
        'is_weekend': 1 if dow >= 5 else 0,
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
        'current_lat': current_lat, 'current_lon': current_lon,
        'lat_normalized': lat_normalized, 'lon_normalized': lon_normalized,
        'dist_from_center': dist_from_center,
        'is_city_center': is_city_center, 'is_suburban': is_suburban, 'is_outer_city': is_outer_city,
        'current_speed': current_speed, 'avg_recent_speed': avg_recent_speed,
        'speed_volatility': speed_volatility, 'max_recent_speed': max_recent_speed,
        'recent_acceleration': recent_acceleration, 'movement_smoothness': movement_smoothness,
        'distance_from_start': distance_from_start, 'speed_vs_city_avg': speed_vs_city_avg,
        'speed_stopped': speed_stopped, 'speed_slow': speed_slow,
        'speed_normal': speed_normal, 'speed_fast': speed_fast,
        'target_lat': target_lat, 'target_lon': target_lon,
        'lat_diff': lat_diff, 'lon_diff': lon_diff,
        'is_distant_target': is_distant_target, 'is_nearby_target': is_nearby_target,
        'is_very_close_target': is_very_close_target,
        'dist_at_stop': dist_at_stop, 'dist_very_close': dist_very_close,
        'dist_close': dist_close, 'dist_medium': dist_medium, 'dist_far': dist_far,
        'traffic_density_score': traffic_density_score, 'weekend_factor': weekend_factor,
        'target_stop_id': target_stop_id, 'route_id': route_id,
        'stop_sequence_position': stop_sequence_position,
        'stop_position_squared': stop_sequence_position ** 2,
        'stop_position_cubed': stop_sequence_position ** 3,
        'stops_remaining': stops_remaining,
        'is_friday': 1 if dow == 4 else 0,
        'rush_hour_intensity': rush_hour_intensity,
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
    
    return features

def predict_sequence(trajectory, future_stops):
    """Predict ETAs with model + historical + heuristics"""
    
    cumulative_time = 0.0
    eta_predictions = []
    total_future_stops = len(future_stops)
    
    for stop_idx, stop in enumerate(future_stops):
        features = extract_features_for_stop(
            trajectory, stop, stop_idx, cumulative_time, historical_lookup,
            total_future_stops
        )
        
        features_df = pd.DataFrame([features])
        
        for feat in feature_names:
            if feat not in features_df.columns:
                features_df[feat] = 0.0
        
        features_df_ordered = features_df[feature_names]
        features_numeric = features_df_ordered.select_dtypes(include=[np.number])
        
        for feat in cb_feature_names:
            if feat not in features_df.columns:
                features_df[feat] = 0.0
        features_df_cb = features_df[cb_feature_names]
        
        lgb_preds = [m.predict(features_numeric, num_iteration=m.best_iteration)[0] for m in lgb_models]
        cb_preds = [m.predict(features_df_cb)[0] for m in cb_models]
        xgb_preds = [m.predict(features_numeric)[0] for m in xgb_models]
        
        lgb_pred = np.mean(lgb_preds)
        cb_pred = np.mean(cb_preds)
        xgb_pred = np.mean(xgb_preds)
        
        inter_stop_time = (ensemble_weights['lgb'] * lgb_pred + 
                          ensemble_weights['cb'] * cb_pred + 
                          ensemble_weights['xgb'] * xgb_pred) * SCALE_FACTOR
        
        # Historical blending
        hist_lookup_key = (trajectory.get('route_id'), stop_idx, 
                          trajectory.get('current_time').hour, 
                          trajectory.get('current_time').dayofweek)
        if hist_lookup_key in historical_lookup:
            hist_data = historical_lookup[hist_lookup_key]
            hist_eta = hist_data.get('historical_eta_mean', inter_stop_time)
            confidence = hist_data.get('historical_confidence', 0.0)
            if confidence > 0.3:
                inter_stop_time = 0.7 * inter_stop_time + 0.3 * hist_eta * SCALE_FACTOR
        
        # 🔧 HEURISTIC 1: Distance-based adjustment
        target_distance = haversine_distance(
            trajectory.get('current_lat'), 
            trajectory.get('current_lon'),
            stop['lat'], 
            stop['lon']
        )
        
        # Models tend to underestimate long distances
        if target_distance > 2.0:  # > 2km
            inter_stop_time += 0.25  # +15 seconds
        elif target_distance > 1.0:  # > 1km
            inter_stop_time += 0.15  # +9 seconds
        
        # 🔧 HEURISTIC 2: Rush hour penalty
        hour = trajectory.get('current_time').hour
        if 8 <= hour <= 10:  # Morning rush
            inter_stop_time *= 1.08  # +8%
        elif 17 <= hour <= 19:  # Evening rush
            inter_stop_time *= 1.10  # +10%
        
        # 🔧 HEURISTIC 3: Sequential slowdown (later stops take longer)
        if stop_idx >= 3:  # After 3rd stop, traffic compounds
            inter_stop_time *= 1.05  # +5%
        
        cumulative_time += max(0.5, inter_stop_time)
        eta_predictions.append(cumulative_time)
    
    return eta_predictions

def find_current_stop_via_gps(last_gps_lat, last_gps_lon, route_id):
    """Find current stop using GPS coordinates"""
    route_key = str(route_id)
    
    if route_key not in ROUTE_TO_STOPS:
        print(f"   ❌ Route {route_id} not found")
        return None, None
    
    stop_ids = ROUTE_TO_STOPS[route_key]
    min_distance = float('inf')
    closest_stop_id = None
    closest_stop_position = -1
    
    print(f"   🔍 GPS: ({last_gps_lat:.6f}, {last_gps_lon:.6f})")
    
    for idx, stop_id in enumerate(stop_ids):
        stop_id_str = str(stop_id)
        
        if stop_id_str in STOPS_DATA:
            stop_info = STOPS_DATA[stop_id_str]
        elif stop_id_str.isdigit() and int(stop_id_str) in STOPS_DATA:
            stop_info = STOPS_DATA[int(stop_id_str)]
        else:
            continue
        
        distance = haversine_distance(last_gps_lat, last_gps_lon, 
                                     stop_info['lat'], stop_info['lon'])
        
        if distance < min_distance:
            min_distance = distance
            closest_stop_id = stop_id
            closest_stop_position = idx
    
    print(f"   📍 Stop: {closest_stop_id}, Dist: {min_distance*1000:.0f}m, Pos: {closest_stop_position+1}/{len(stop_ids)}")
    
    return closest_stop_id, closest_stop_position

def get_future_stops_for_route_gps(route_id, current_stop_position, predict_count=6):
    """Get future stops from current position"""
    route_key = str(route_id)
    
    if route_key in ROUTE_TO_STOPS:
        stop_ids = ROUTE_TO_STOPS[route_key]
    else:
        return []
    
    start_idx = current_stop_position + 1
    future_stop_ids = stop_ids[start_idx : start_idx + predict_count]
    
    future_stops = []
    for stop_id in future_stop_ids:
        stop_id_str = str(stop_id)
        
        if stop_id_str in STOPS_DATA:
            stop_info = STOPS_DATA[stop_id_str]
        elif stop_id_str.isdigit() and int(stop_id_str) in STOPS_DATA:
            stop_info = STOPS_DATA[int(stop_id_str)]
        else:
            continue
        
        future_stops.append({
            'stop_id': int(stop_id_str) if stop_id_str.isdigit() else stop_id_str,
            'lat': stop_info['lat'],
            'lon': stop_info['lon']
        })
    
    return future_stops

def process_input_json(input_src):
    """Process input JSON (file path or dict) and generate predictions."""
    if isinstance(input_src, dict):
        data = input_src
    else:
        with open(input_src,'r',encoding='utf-8-sig') as f:
            data = json.load(f)
    print(f"📥 Processing {len(data)} sequences...\n")
    out = {}
    inp_dir = os.path.dirname(input_src) if isinstance(input_src,str) else ""
    for seq_id, parquet_path in data.items():
        try:
            paths = ([parquet_path] if os.path.isabs(parquet_path) else
                     [os.path.join(inp_dir,parquet_path),
                      os.path.join("/app/data",parquet_path),
                      os.path.join("/app",parquet_path),
                      parquet_path])
            pfile = next((p for p in paths if os.path.exists(p)),None)
            if pfile is None:
                print(f"⚠️ Seq {seq_id}: File not found")
                continue

            df = pd.read_parquet(pfile)
            print(f"🔍 Seq {seq_id}: {len(df)} rows")
            raw = df.iloc[0].get('route_id')
            rid = str(raw).strip().strip('"').strip("'")
            if rid.replace('-','').replace('.','').isdigit():
                try:
                    rid=str(int(float(rid)))
                except:
                    pass
            print(f"   Route: {rid}")

            last = df.iloc[-1]
            lat = last.get('latitude', last.get('lat',12.95))
            lon = last.get('longitude',last.get('lon',77.58))
            csid, cpos = find_current_stop_via_gps(lat,lon,rid)
            if csid is None:
                print("   ❌ No current stop")
                continue

            # 1) Compute current_time UTC
            rawt = last.get('system_time', last.get('vehicle_timestamp', datetime.now()))
            try:
                t0 = pd.to_datetime(rawt)
            except:
                t0 = pd.Timestamp.now()
            current_time = t0.tz_localize('UTC') if t0.tzinfo is None else t0.tz_convert('UTC')

            # 2) Future stops
            fsts = get_future_stops_for_route_gps(rid, cpos, 6)
            if not fsts:
                print("   ⚠️ No future stops – attempting backward interpolation")
                
                # Get historical timing for this route segment
                route_key = str(rid)
                if route_key in ROUTE_TO_STOPS:
                    all_stops = ROUTE_TO_STOPS[route_key]
                    
                    # Estimate time remaining based on historical averages
                    # Find how many stops back we can look
                    lookback_stops = min(3, cpos)  # Look at last 3 stops
                    
                    if lookback_stops > 0 and cpos < len(all_stops):
                        # Use historical average speed for this route/time
                        hour = current_time.hour
                        avg_speed_kmh = 20 if 8 <= hour <= 19 else 25  # Peak vs off-peak
                        
                        # Estimate remaining distance to end of route
                        remaining_stops = len(all_stops) - cpos - 1
                        avg_dist_per_stop = 0.5  # km (rough estimate)
                        remaining_dist = remaining_stops * avg_dist_per_stop
                        
                        # Time to reach end of route
                        time_to_end = (remaining_dist / avg_speed_kmh) * 60  # minutes
                        
                        # Output the last stop on route with estimated time
                        last_stop_id = all_stops[-1]
                        arrival_time = current_time + pd.Timedelta(minutes=time_to_end)
                        out[rid] = { 
                            str(csid): current_time.strftime('%Y-%m-%d %H:%M:%S'),
                            str(last_stop_id): arrival_time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        print(f"   ✅ Interpolated: {remaining_stops} stops remaining, ETA: {time_to_end:.1f}min")
                        continue
                
                # Complete fallback if interpolation fails
                out[rid] = { str(csid): current_time.strftime('%Y-%m-%d %H:%M:%S') }
                continue

            print(f"   🎯 {len(fsts)} stops: {fsts[0]['stop_id']} → {fsts[-1]['stop_id']}")

            traj={
                'route_id':int(rid) if rid.isdigit() else 0,
                'current_lat':lat,'current_lon':lon,
                'current_speed':last.get('speed',20.0),
                'avg_recent_speed':df['speed'].tail(10).mean() if 'speed' in df else 20.0,
                'speed_volatility':df['speed'].tail(10).std() if 'speed' in df else 5.0,
                'current_time':current_time
            }
            etas = predict_sequence(traj,fsts)
            preds={}
            for i,st in enumerate(fsts):
                at = current_time+pd.Timedelta(minutes=etas[i])
                preds[str(st['stop_id'])]=at.strftime('%Y-%m-%d %H:%M:%S')
            print(f"   ⏱️ ETAs: {etas[0]:.1f}min → {etas[-1]:.1f}min\n")
            out[rid]=preds

        except Exception as e:
            print(f"❌ Seq {seq_id}: {e}\n")
            continue

    return out

if __name__=="__main__":
    possible=[sys.argv[1] if len(sys.argv)>1 else None,
              "/app/data/input.json","input.json","./input.json",
              "/app/input.json","/data/input.json","/input.json"]
    inp=next((p for p in possible if p and os.path.exists(p)),None)
    if not inp:
        print("❌ ERROR: Cannot find input.json")
        sys.exit(1)
    print("="*70)
    print("🚀 BENGALURU BUS ETA - v27 FINAL")
    print("="*70)
    print("🎯 Scale 0.15 + Distance/Rush/Sequential Heuristics")
    print("="*70)
    out = process_input_json(inp)
    if not out:
        print("⚠️ No predictions!")
        sys.exit(1)
    print(f"✅ {len(out)} routes predicted")
    for path in ["/app/out/output.json","/app/data/output.json","/app/output.json","output.json"]:
        try:
            os.makedirs(os.path.dirname(path),exist_ok=True)
            with open(path,'w') as f:
                json.dump(out,f,indent=2)
            print(f"   ✅ {path}")
        except:
            print(f"   ⚠️ {path}: failed")

class BusETAPredictor:
    """Wrapper for Task 3 multi-modal integration."""

    def __init__(self):
        print("🚌 Initializing BusETAPredictor (v27)...")
        print("✅ BusETAPredictor ready for Task 3")

    def get_stop_coordinates(self, bus_stop_id):
        sid = str(bus_stop_id)
        if sid in STOPS_DATA:
            s = STOPS_DATA[sid]
            return s['lat'], s['lon']
        return None

    def predict_bus_waiting_time(self, bus_stop_id, passenger_arrival_time_utc, bus_parquet_path):
        input_json={str(bus_stop_id):bus_parquet_path}
        all_eta=process_input_json(input_json)
        for etas in all_eta.values():
            if str(bus_stop_id) in etas:
                arrive_utc=datetime.fromisoformat(etas[str(bus_stop_id)])
                return max(0.0,(arrive_utc-passenger_arrival_time_utc).total_seconds()/60.0)
        return 2.0

    def predict_bus_ride_duration(self, start_stop_id, end_stop_id, boarding_time_utc, bus_parquet_path):
        input_json = {str(start_stop_id): bus_parquet_path}
        all_eta = process_input_json(input_json)

        # No ETAs at all
        if not all_eta:
            return 0.0

        # Only boarding stop returned (check inside the route-level dict)
        for route_id, stops_dict in all_eta.items():
            if len(stops_dict) == 1:
                # If the single stop is the start stop and start==end, return 0
                if str(start_stop_id) in stops_dict and str(start_stop_id) == str(end_stop_id):
                    return 0.0
                # If start != end but only start is in dict, also return 0 (no future stops to reach end)
                if str(start_stop_id) in stops_dict:
                    return 0.0

        # Both stops present → use ETA difference
        for etas in all_eta.values():
            s, e = str(start_stop_id), str(end_stop_id)
            if s in etas and e in etas:
                t_s = datetime.fromisoformat(etas[s])
                t_e = datetime.fromisoformat(etas[e])
                return max(0.0, (t_e - t_s).total_seconds() / 60.0)

        # Same stop IDs → zero travel
        if str(start_stop_id) == str(end_stop_id):
            return 0.0

        # Fallback: haversine
        coords1 = self.get_stop_coordinates(start_stop_id)
        coords2 = self.get_stop_coordinates(end_stop_id)
        if coords1 is None or coords2 is None:
            return 2.5
        lat1, lon1 = coords1
        lat2, lon2 = coords2
        dist_km = haversine_distance(lat1, lon1, lat2, lon2)
        avg_speed = 22 if 8 <= boarding_time_utc.hour <= 19 else 28  # hour-aware speed
        return max(2.0, dist_km / avg_speed * 60.0)
