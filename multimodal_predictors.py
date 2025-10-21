#!/usr/bin/env python3
"""
BLMC Task 3: Multi-Modal Journey Prediction
Combines Auto (Task 2) + Bus (Task 1) predictions
WITH PROPER FAILURE CASCADE LOGIC & JOINT CALIBRATION
"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import warnings
from geopy.distance import geodesic
warnings.filterwarnings('ignore')


from auto_predictor import AutoETAPredictor
from bus_predictor import BusETAPredictor


class MultiModalPredictor:
    """
    Predicts all 8 components of a multi-modal journey:
    AUTO 1 → BUS → AUTO 2
    
    WITH FAILURE CASCADE HANDLING PER SPEC:
    - If a1 >= 9 min: Return [30, 1, 1, 1, 1, 1, 1, 1]
    - If a6 >= 9 min: Return [a1, a2, a3, a4, a5, 30, 1, 1]
    """
    
    def __init__(self):
        print("🚀 Initializing Multi-Modal Predictor...")
        
        # Load Task 2 (Auto) predictor
        self.auto_predictor = AutoETAPredictor()
        print("✅ Auto predictor loaded")
        
        # Load Task 1 (Bus) predictor
        self.bus_predictor = BusETAPredictor()
        print("✅ Bus predictor loaded")
        
        # Timezone setup
        self.ist = pytz.timezone('Asia/Kolkata')
        self.utc = pytz.UTC
        
        # SPEC-DEFINED CONSTANTS
        self.ACCEPTANCE_FAILURE_THRESHOLD = 9.0  # Minutes
        self.FAILURE_ACCEPTANCE_VALUE = 30.0     # Minutes
        self.FAILURE_OTHER_VALUE = 1.0           # Minutes
        
        # JOINT CALIBRATION PARAMETERS
        self.CALIBRATION_ALPHA = 0.75  # Scale factor
        self.CALIBRATION_BETA = -3.5  # Offset in minutes
        self.CALIBRATION_MIN_TOTAL = 9.0 # Prevent under-prediction on very short journeys
        
        print(f"⚠️  Failure threshold: {self.ACCEPTANCE_FAILURE_THRESHOLD} min")
        print(f"⚠️  Failure values: acceptance={self.FAILURE_ACCEPTANCE_VALUE}, others={self.FAILURE_OTHER_VALUE}")
        print(f"🎯 Joint calibration: α={self.CALIBRATION_ALPHA}, β={self.CALIBRATION_BETA}")
    
    def parse_location(self, location_str):
        """Parse 'lat,lon' or '(lat, lon)' string to floats"""
        try:
            # Remove parentheses and extra spaces
            cleaned = location_str.strip().replace('(', '').replace(')', '').replace(' ', '')
            parts = cleaned.split(',')
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
        except Exception as e:
            print(f"⚠️ Error parsing location '{location_str}': {e}")
            return 12.9716, 77.5946  # Default to Bengaluru center
    
    def parse_time_ist(self, time_str):
        """Parse HH:MM:SS IST string to pandas.Timestamp (with dayofweek)."""
        try:
            today = datetime.now(self.ist).date()
            t = datetime.strptime(time_str.strip(), "%H:%M:%S").time()
            dt = datetime.combine(today, t)
            dt_aware = self.ist.localize(dt)
            # Wrap in pandas Timestamp so .dayofweek exists
            return pd.Timestamp(dt_aware)
        except Exception as e:
            print(f"⚠️ Error parsing time '{time_str}': {e}")
            # Fallback to now as pandas Timestamp
            return pd.Timestamp.now(self.ist)
    
    def ist_to_utc(self, ist_dt):
        """Convert IST datetime to UTC"""
        return ist_dt.astimezone(self.utc)
    
    def utc_to_ist(self, utc_dt):
        """Convert UTC datetime to IST"""
        return utc_dt.astimezone(self.ist)
    
    def predict_journey(self, journey_data):
        """
        Predict all 8 components of a multi-modal journey
        WITH PROPER FAILURE CASCADE HANDLING
        
        Args:
            journey_data: dict with keys:
                - jid: Journey ID
                - auto_1_start_lat, auto_1_start_lon
                - auto_1_request_time_ist: datetime
                - bus_stop_A_id: int
                - bus_parquet_path: str
                - bus_stop_B_id: int
                - auto_2_end_lat, auto_2_end_lon
        
        Returns:
            dict: {a1, a2, a3, a4, a5, a6, a7, a8} in minutes
        """
        
        jid = journey_data['jid']
        print(f"\n{'='*60}")
        print(f"🚗 Processing Journey {jid}")
        print(f"{'='*60}")
        
        try:
            # ========================================
            # AUTO RIDE 1: Home → Bus Stop A
            # ========================================
            print("🚗 AUTO 1: Predicting acceptance, pickup, ride...")
            
            auto1_start_lat = journey_data['auto_1_start_lat']
            auto1_start_lon = journey_data['auto_1_start_lon']
            auto1_request_time = journey_data['auto_1_request_time_ist']
            
            # Get bus stop A coordinates
            bus_stop_A_coords = self.bus_predictor.get_stop_coordinates(
                journey_data['bus_stop_A_id']
            )
            
            if bus_stop_A_coords is None:
                print(f"⚠️ Bus stop A not found, using defaults")
                bus_stop_A_coords = (12.9716, 77.5946)
            
            # Predict Auto 1
            auto1_result = self.auto_predictor.predict_auto_ride(
                origin_lat=auto1_start_lat,
                origin_lon=auto1_start_lon,
                dest_lat=bus_stop_A_coords[0],
                dest_lon=bus_stop_A_coords[1],
                timestamp=auto1_request_time
            )
            
            a1 = auto1_result['acceptance_time']
            a2 = auto1_result['pickup_time']
            a3 = auto1_result['trip_time']
            
            print(f"   ✅ a1 (acceptance): {a1:.2f} min")
            print(f"   ✅ a2 (pickup): {a2:.2f} min")
            print(f"   ✅ a3 (ride): {a3:.2f} min")
            
            # ========================================
            # CRITICAL: CHECK FIRST AUTO FAILURE
            # ========================================
            if a1 >= self.ACCEPTANCE_FAILURE_THRESHOLD:
                print(f"\n🚨 FAILURE: First auto acceptance ({a1:.2f}) >= {self.ACCEPTANCE_FAILURE_THRESHOLD} min")
                print(f"   Returning failure cascade: [30, 1, 1, 1, 1, 1, 1, 1]")
                print(f"{'='*60}\n")
                
                return {
                    'a1': self.FAILURE_ACCEPTANCE_VALUE,
                    'a2': self.FAILURE_OTHER_VALUE,
                    'a3': self.FAILURE_OTHER_VALUE,
                    'a4': self.FAILURE_OTHER_VALUE,
                    'a5': self.FAILURE_OTHER_VALUE,
                    'a6': self.FAILURE_OTHER_VALUE,
                    'a7': self.FAILURE_OTHER_VALUE,
                    'a8': self.FAILURE_OTHER_VALUE
                }
            
            # Calculate when passenger arrives at bus stop A
            passenger_arrival_at_A = auto1_request_time + timedelta(minutes=a1+a2+a3)
            print(f"   📍 Passenger arrives at Stop A: {passenger_arrival_at_A.strftime('%H:%M:%S')} IST")
            
            # ========================================
            # BUS WAITING TIME: At Stop A
            # ========================================
            print("🚌 BUS: Calculating waiting time...")
            
            # CRITICAL: Convert IST to UTC for bus data
            passenger_arrival_at_A_utc = self.ist_to_utc(passenger_arrival_at_A)
            
            a4 = self.bus_predictor.predict_bus_waiting_time(
                bus_stop_id=journey_data['bus_stop_A_id'],
                passenger_arrival_time_utc=passenger_arrival_at_A_utc,
                bus_parquet_path=journey_data['bus_parquet_path']
            )
            
            print(f"   ✅ a4 (waiting): {a4:.2f} min")
            
            # Calculate when passenger boards bus
            bus_boarding_time = passenger_arrival_at_A + timedelta(minutes=a4)
            print(f"   🚌 Bus boarding: {bus_boarding_time.strftime('%H:%M:%S')} IST")
            
            # ========================================
            # BUS RIDE: Stop A → Stop B
            # ========================================
            print("🚌 BUS: Predicting ride duration...")
            
            # CRITICAL: Convert to UTC for bus data
            bus_boarding_time_utc = self.ist_to_utc(bus_boarding_time)
            
            a5 = self.bus_predictor.predict_bus_ride_duration(
                start_stop_id=journey_data['bus_stop_A_id'],
                end_stop_id=journey_data['bus_stop_B_id'],
                boarding_time_utc=bus_boarding_time_utc,
                bus_parquet_path=journey_data['bus_parquet_path']
            )
            
            print(f"   ✅ a5 (bus ride): {a5:.2f} min")
            
            # Calculate when passenger gets off at stop B
            passenger_arrival_at_B = bus_boarding_time + timedelta(minutes=a5)
            print(f"   📍 Passenger arrives at Stop B: {passenger_arrival_at_B.strftime('%H:%M:%S')} IST")
            
            # ========================================
            # AUTO RIDE 2: Bus Stop B → Destination
            # ========================================
            print("🚗 AUTO 2: Predicting acceptance, pickup, ride...")
            
            # Get bus stop B coordinates
            bus_stop_B_coords = self.bus_predictor.get_stop_coordinates(
                journey_data['bus_stop_B_id']
            )
            
            if bus_stop_B_coords is None:
                print(f"⚠️ Bus stop B not found, using defaults")
                bus_stop_B_coords = (12.9716, 77.5946)
            
            auto2_end_lat = journey_data['auto_2_end_lat']
            auto2_end_lon = journey_data['auto_2_end_lon']
            
            # Predict Auto 2
            auto2_result = self.auto_predictor.predict_auto_ride(
                origin_lat=bus_stop_B_coords[0],
                origin_lon=bus_stop_B_coords[1],
                dest_lat=auto2_end_lat,
                dest_lon=auto2_end_lon,
                timestamp=passenger_arrival_at_B
            )
            
            a6 = auto2_result['acceptance_time']
            a7 = auto2_result['pickup_time']
            a8 = auto2_result['trip_time']
            
            print(f"   ✅ a6 (acceptance): {a6:.2f} min")
            print(f"   ✅ a7 (pickup): {a7:.2f} min")
            print(f"   ✅ a8 (ride): {a8:.2f} min")
            
            # ========================================
            # CRITICAL: CHECK SECOND AUTO FAILURE
            # ========================================
            if a6 >= self.ACCEPTANCE_FAILURE_THRESHOLD:
                print(f"\n🚨 FAILURE: Second auto acceptance ({a6:.2f}) >= {self.ACCEPTANCE_FAILURE_THRESHOLD} min")
                print(f"   Returning partial failure: [a1, a2, a3, a4, a5, 30, 1, 1]")
                print(f"{'='*60}\n")
                
                return {
                    'a1': round(a1, 2),
                    'a2': round(a2, 2),
                    'a3': round(a3, 2),
                    'a4': round(a4, 2),
                    'a5': round(a5, 2),
                    'a6': self.FAILURE_ACCEPTANCE_VALUE,
                    'a7': self.FAILURE_OTHER_VALUE,
                    'a8': self.FAILURE_OTHER_VALUE
                }
            
            # ========================================
            # JOINT CALIBRATION FOR TASK 3
            # ========================================
            # Calculate total journey distance
            dist_auto1 = geodesic((auto1_start_lat, auto1_start_lon), bus_stop_A_coords).kilometers
            dist_auto2 = geodesic(bus_stop_B_coords, (auto2_end_lat, auto2_end_lon)).kilometers

# Approximate bus distance (assume straight-line * 1.3 for route complexity)
            dist_bus = geodesic(bus_stop_A_coords, bus_stop_B_coords).kilometers * 1.3
            total_distance = dist_auto1 + dist_bus + dist_auto2
            
            total_raw = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
            print(f"\n🔧 Applying piecewise calibration (distance: {total_distance:.1f} km)...")
            print(f"   Raw total: {total_raw:.2f} min")
            
            # Distance-based calibration parameters
            if total_distance < 5:
               alpha, beta, min_floor = 0.82, -1.5, 8.0   # Light reduction for short journeys
            elif total_distance < 10:
               alpha, beta, min_floor = 0.76, -3.0, 9.0   # Medium reduction
            elif total_distance < 20:
               alpha, beta, min_floor = 0.70, -4.0, 10.0  # Heavy reduction
            else:
               alpha, beta, min_floor = 0.65, -5.0, 12.0  # Extreme reduction for long journeys

            total_calibrated = max(min_floor, alpha * total_raw + beta)
            print(f"   Calibrated total: {total_calibrated:.2f} min (α={alpha}, β={beta})")

            # Proportionally scale all components
            scale_factor = total_calibrated / total_raw if total_raw > 0 else 1.0
            print(f"   Scale factor: {scale_factor:.4f}")
            
            a1 = round(a1 * scale_factor, 2)
            a2 = round(a2 * scale_factor, 2)
            a3 = round(a3 * scale_factor, 2)
            a4 = round(a4 * scale_factor, 2)
            a5 = round(a5 * scale_factor, 2)
            a6 = round(a6 * scale_factor, 2)
            a7 = round(a7 * scale_factor, 2)
            a8 = round(a8 * scale_factor, 2)
            
            # Calculate final arrival time
            final_arrival = passenger_arrival_at_B + timedelta(minutes=a6+a7+a8)
            total_journey_time = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
            
            print(f"\n{'='*60}")
            print(f"✅ Journey {jid} Complete!")
            print(f"   Total time: {total_journey_time:.2f} min")
            print(f"   Final arrival: {final_arrival.strftime('%H:%M:%S')} IST")
            print(f"{'='*60}\n")
            
            return {
                'a1': a1,
                'a2': a2,
                'a3': a3,
                'a4': a4,
                'a5': a5,
                'a6': a6,
                'a7': a7,
                'a8': a8
            }
            
        except Exception as e:
            print(f"❌ Error in journey {jid}: {e}")
            import traceback
            traceback.print_exc()
            
            # COMPLETE FAILURE: Both autos failed pattern
            print(f"   Returning complete failure pattern: [30, 1, 1, 1, 1, 30, 1, 1]")
            return {
                'a1': self.FAILURE_ACCEPTANCE_VALUE,
                'a2': self.FAILURE_OTHER_VALUE,
                'a3': self.FAILURE_OTHER_VALUE,
                'a4': self.FAILURE_OTHER_VALUE,
                'a5': self.FAILURE_OTHER_VALUE,
                'a6': self.FAILURE_ACCEPTANCE_VALUE,
                'a7': self.FAILURE_OTHER_VALUE,
                'a8': self.FAILURE_OTHER_VALUE
            }
    
    def predict_from_csv(self, input_csv_path):
        """
        Process entire input CSV and return predictions
        
        Args:
            input_csv_path: Path to input.csv
            
        Returns:
            dict: {jid: {a1, ..., a8}}
        """
        print(f"\n{'='*70}")
        print(f"📊 MULTI-MODAL JOURNEY PREDICTION - Task 3")
        print(f"{'='*70}\n")
        
        # Read input CSV
        df = pd.read_csv(input_csv_path)
        print(f"📥 Loaded {len(df)} journeys from {input_csv_path}")
        
        results = {}
        
        for idx, row in df.iterrows():
            try:
                # Parse journey data
                jid = int(row['jid'])
                
                auto1_start_lat, auto1_start_lon = self.parse_location(
                    row['auto_1_ride_start_location']
                )
                
                auto1_request_time = self.parse_time_ist(
                    row['auto_1_ride_request_time']
                )
                
                auto2_end_lat, auto2_end_lon = self.parse_location(
                    row['auto_2_ride_end_location']
                )
                
                journey_data = {
                    'jid': jid,
                    'auto_1_start_lat': auto1_start_lat,
                    'auto_1_start_lon': auto1_start_lon,
                    'auto_1_request_time_ist': auto1_request_time,
                    'bus_stop_A_id': int(row['auto_1_ride_end_bus_stop_ID']),
                    'bus_parquet_path': row['path_to_the_parquet_file'],
                    'bus_stop_B_id': int(row['bus_trip_end_bus_stop_ID']),
                    'auto_2_end_lat': auto2_end_lat,
                    'auto_2_end_lon': auto2_end_lon
                }
                
                # Predict journey
                prediction = self.predict_journey(journey_data)
                results[jid] = prediction
                
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                import traceback
                traceback.print_exc()
                
                # COMPLETE FAILURE FALLBACK
                jid = int(row['jid'])
                results[jid] = {
                    'a1': 30.0, 'a2': 1.0, 'a3': 1.0,
                    'a4': 1.0, 'a5': 1.0,
                    'a6': 30.0, 'a7': 1.0, 'a8': 1.0
                }
        
        print(f"\n✅ Processed {len(results)}/{len(df)} journeys")
        return results



if __name__ == "__main__":
    # Test locally
    predictor = MultiModalPredictor()
    results = predictor.predict_from_csv('/app/data/input.csv')
    
    import json
    with open('/app/out/output.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Predictions saved to output.json")
