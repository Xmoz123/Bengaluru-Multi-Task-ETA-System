# Task 2: Auto-rickshaw ETA Prediction

Auto-rickshaw ETA prediction combining OSMnx road network routing, H3 spatial features, and supply-demand dynamics for Bengaluru urban mobility.

**Problem:** Predict auto-rickshaw travel time using:
- Origin and destination coordinates
- Time slot (15-minute intervals, 96 slots/day)
- Road network topology via OpenStreetMap
- Supply-demand patterns from H3 hexagonal spatial indexing
- Slot-specific speed profiles

---

**Solution Pipeline:**

**1. OSMnx Road Network** (`osmnx_routing.py`)

Downloads and builds Bengaluru road network for time-aware routing:
- Downloads OpenStreetMap drive network for Bengaluru (~15k nodes, ~35k edges)
- Builds time-aware graphs with slot-specific speeds (96 time slots/day)
- Integrates observed speed data from auto-rickshaw GPS traces
- Provides realistic routing time estimates using Dijkstra's algorithm

**Key Features:**
- Road class-based speed priors: motorway (55 km/h), trunk (45 km/h), primary (35 km/h), residential (18 km/h)
- Time-slot-aware routing adjusts for traffic patterns (rush hour vs off-peak)
- Robust fallback (15 min default) for routing failures
- In-memory graph caching for fast repeated queries

**2. Feature Engineering** (`feature_engineering.py`)

60+ features combining spatial indexing, routing, and movement patterns:

**Feature Categories:**
- Temporal (8): Hour, day of week, weekend flag, morning/evening/super peak indicators
- Distance (5): Geodesic distance, categorization (short <2km, medium 2-10km, long >10km)
- Spatial - City Center (4): Distance from Bengaluru center, tech hub proximity
- Route Characteristics (7): Bearing, bearing sin/cos, route efficiency, density factor, complexity
- Speed & Congestion (6): Origin/destination speeds, congestion multiplier (1.0-1.8x), rush intensity, estimated speed
- H3 Supply-Demand (6): Start/end counts in neighborhood, idle time mean/P90, supply-demand ratio
- Tech Hub Proximity (2): Minimum distance to Whitefield and Electronic City tech hubs

**Innovation:**
- H3 hexagonal spatial indexing (resolution 9 = ~174m edge) for efficient neighborhood aggregation
- Supply-demand dynamics: Trip start/end counts, idle time statistics by hexagon
- Time-aware congestion: Super peak (8-9am, 6-7pm) gets 1.8x multiplier
- OSMnx routing integration provides ground-truth baseline for trip times

**3. Ensemble Training** (`train_model.py`)

LightGBM + CatBoost weighted ensemble:

**Two Models:**
- LightGBM: Fast gradient boosting, 63 leaves, depth 8, 0.05 learning rate
- CatBoost: Handles categorical features natively (hour, dow, distance_cat), depth 7

**Training Strategy:**
1. 80-20 train-validation split with shuffling
2. Individual model training with early stopping (100 rounds patience)
3. Grid search over weight combinations: [0.3-0.7] for LGB, [0.7-0.3] for CB
4. Weighted averaging for final predictions
5. Best weights selected by validation MAE

**4. Validation & Analysis** (`validate_model.py`)

Comprehensive performance evaluation:
- **Core metrics:** MAE, RMSE, R²
- **Error statistics:** Mean, median, standard deviation of residuals
- **Accuracy bands:** Percentage within 1/2/5 minutes
- **Error percentiles:** P50, P90, P95
- **Performance by distance:** Breakdown across <2km, 2-5km, 5-10km, 10-20km, >20km bins
- **Performance by time slot:** Hourly MAE analysis (0-23 hours)
- **Diagnostic plots:** Predicted vs Actual, Residual distribution, Residuals vs Predicted

---

**Data Analysis**

**Target Distribution:**

![Target Distribution Analysis](images/target_distribution_analysis.png)

**Key Observations:**
- Right-skewed distribution with median around 9-12 minutes
- 80% of trips complete within 20 minutes
- 95% of trips complete within 45 minutes
- Mode in 5-15 minute range (~85,000 trips)
- Long tail extends to 120+ minutes for distant destinations
- Box plot shows outliers beyond 60 minutes (heavy traffic or long-distance rides)

**Feature Correlations:**

![Feature Correlations](images/feature_correlations.png)

**Strongest Predictors:**
- **Distance features** show highest correlation (0.635-0.655): `straight_line_distance`, `estimated_route_distance`
- **Positive correlation** with distance categories: `dist_far` (0.422), `dist_medium` (0.383), `is_distant_target` (0.526)
- **Negative correlation** with proximity indicators: `is_nearby_target` (-0.445), `is_very_close_target` (-0.333), `dist_at_stop` (-0.264)
- **Movement features** show weak correlation: `movement_smoothness` (0.096), `speed_volatility` (-0.098), `speed_slow` (-0.098)
- **Spatial insights:** Distance dominates prediction; movement dynamics add marginal value

---

**Technical Highlights**

**Geospatial Innovation:**
- OSMnx road network integration with 15-minute time slots (96/day)
- H3 hexagonal spatial indexing (resolution 9) for scalable neighborhood queries
- Tech hub proximity features capture Whitefield and Electronic City demand patterns
- Route efficiency = actual distance / (detour-adjusted distance)

**Feature Engineering:**
- 60+ features from 4 data sources: coordinates, time, road network, activity data
- Supply-demand dynamics via H3 cell aggregation (k=1 ring neighbors)
- Time-aware congestion modeling: Super peak 1.8x, evening rush 1.6x, morning rush 1.4x
- Categorical feature handling leverages CatBoost strength

**Model Engineering:**
- LightGBM + CatBoost ensemble for complementary strengths
- LightGBM: Fast inference, numeric features only
- CatBoost: Handles categorical natively, better for hour/dow interactions
- Weighted averaging optimized via grid search (typical: 60% LGB, 40% CB)
- Early stopping prevents overfitting (100-round patience)

---

**Usage Example**
Step 1: Build road network (one-time setup)
from osmnx_routing import build_base_road_network, build_time_aware_graph

G_base = build_base_road_network("Bengaluru, India")
G_time = build_time_aware_graph(
'bengaluru_graph.pkl',
speed_df,
date='2023-06-15',
slot=32 # 8:00-8:15 AM
)

Step 2: Extract features for a trip
from feature_engineering import AutoRickshawFeatureExtractor

extractor = AutoRickshawFeatureExtractor(h3_resolution=9)
features = extractor.extract_features(
origin_lat=12.9716, origin_lon=77.5946, # Bengaluru city center
dest_lat=12.9352, dest_lon=77.6245, # Whitefield
timestamp=pd.Timestamp('2023-06-15 08:00:00'),
slot=32,
activity_df=activity_df,
origin_speed=22.5,
dest_speed=18.3
)

Step 3: Train ensemble
from train_model import AutoRickshawEnsembleTrainer

trainer = AutoRickshawEnsembleTrainer(random_state=42)
X_train, X_val, y_train, y_val = trainer.prepare_training_data(features_df)
lgb_model, cb_model = trainer.train_ensemble(X_train, y_train, X_val, y_val)

Step 4: Make predictions
y_pred_ensemble = trainer.predict(X_val)

Step 5: Validate performance
from validate_model import ModelValidator

validator = ModelValidator()
metrics = validator.evaluate(y_val, y_pred_ensemble, "Validation")
validator.print_metrics(metrics)
validator.plot_diagnostics(y_val, y_pred_ensemble, save_path='diagnostics.png')


---

**Performance Metrics**

**Model Performance:**
- LightGBM: ~3.2 MAE minutes (fast inference, 0.8-1.5ms per prediction)
- CatBoost: ~3.4 MAE minutes (better categorical handling)
- **Ensemble: ~3.1 MAE minutes** (0.1-0.3 min improvement via weighted averaging)

**Accuracy Bands (Validation Set):**
- Within 1 minute: ~35% of predictions
- Within 2 minutes: ~58% of predictions
- Within 5 minutes: ~85% of predictions

**Performance by Distance:**
- Short trips (<2km): 2.1 MAE (best accuracy, less variability)
- Medium trips (2-5km): 2.8 MAE
- Long trips (5-10km): 3.5 MAE
- Very long trips (>10km): 5.2 MAE (higher uncertainty due to route choices)

**Performance by Time:**
- Off-peak (10am-4pm, 9pm-6am): 2.9 MAE
- Peak hours (8-9am, 6-7pm): 3.6 MAE (13% higher error due to traffic variability)
- Tech hub destinations: More predictable patterns (consistent commute routes)

---


---

**Key Innovations**

**1. H3 Spatial Indexing**
- Hexagonal grid (resolution 9 = ~174m edge length) provides uniform spatial partitioning
- Efficient neighborhood queries via `grid_disk(k=1)` for 7-cell ring
- Supply-demand aggregation: Trip starts, ends, idle times at cell level
- Scalable to city-wide patterns without coordinate precision issues

**2. Time-Aware Routing**
- 96 time slots per day (15-minute intervals) capture intraday speed variation
- Slot-specific speed profiles from observed GPS data
- Road class priors for missing observations (e.g., motorway 55 km/h, residential 18 km/h)
- Realistic travel time estimates via Dijkstra shortest path weighted by `travel_time_min`

**3. Supply-Demand Dynamics**
- Auto-rickshaw start/end counts per H3 cell indicate local availability
- Idle time statistics (mean, P90) proxy driver wait times
- Supply-demand ratio = starts / (ends + ε) captures imbalance
- Features capture temporal-spatial patterns in auto availability

**4. Ensemble Strategy**
- LightGBM for speed and numerical feature efficiency
- CatBoost for categorical feature handling (hour, dow, distance bins)
- Weighted averaging (60-40 split typical) optimizes complementary strengths
- Grid search over [0.3-0.7] weight range finds best validation MAE

---

**Future Improvements**

1. **Real-time traffic integration:** Live congestion data from Google/TomTom APIs
2. **Weather impact:** Rainfall, fog, temperature effects on travel time and driver supply
3. **Driver behavior modeling:** Aggressive vs conservative routing preferences
4. **Multi-modal integration:** Transfer points with bus/metro for combined journeys
5. **Surge pricing correlation:** High demand periods → lower supply → longer wait times

---
