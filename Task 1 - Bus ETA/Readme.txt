# Task 1: Bus ETA Prediction

## Overview
Production-ready GPS trajectory-based ETA prediction system for Bengaluru bus network. This task implements a complete ML pipeline from 60GB raw GPS data to real-time inference with ensemble modeling.

## Problem Statement
Predict cumulative estimated time of arrival (ETA) for a sequence of 5 future bus stops using:
- Real-time GPS trajectories (latitude, longitude, timestamp)
- Historical route performance data
- Temporal and geographic context
- Traffic patterns and rush hour modeling

## Solution Architecture

### 1. Data Preprocessing (data_preprocessing.py)
Robust GPS trajectory cleaning and quality assurance pipeline:
- **Input:** 60GB raw GPS trajectories with 47 columns
- **Outlier Detection:** IQR-based speed filtering, coordinate validation
- **Quality Checks:** Duplicate removal, temporal ordering, missing data handling
- **Output:** Clean, validated trajectories ready for feature extraction

**Key Features:**
- Processes millions of GPS points with parallel processing
- Production-safe error handling and logging
- Configurable quality thresholds

### 2. Feature Engineering (feature_engineering.py)
Production-safe extraction of 42 features with zero data leakage:

**Feature Categories:**
- **Temporal (19):** Hour, day of week, cyclical encodings, rush hour indicators
- **Geographic (7):** Current location, city center distance, urban zone classification
- **Target (7):** Destination coordinates, bearing, distance features
- **Movement (7):** Speed, acceleration, smoothness, time since last stop
- **Progress (1):** Distance traveled from route start
- **Traffic (3):** Rush hour intensity, weekend factors, seasonal patterns
- **Speed Context (2):** Relative speed metrics, speed categorization

**Design Principles:**
- **Leakage Prevention:** Only uses information available at prediction time
- **Real-time Safe:** Minimal backward-looking window (3 GPS points)
- **Production Ready:** Handles edge cases and missing data gracefully

### 3. Ensemble Training (train_ensemble.py)
K-fold cross-validation with multi-algorithm ensemble:

**Models:**
- **LightGBM:** Fast inference, 2.520 MAE
- **CatBoost:** Handles categorical features, 2.850 MAE  
- **XGBoost:** Strong baseline, 2.577 MAE

**Training Strategy:**
- 5-fold cross-validation for robustness
- Out-of-fold (OOF) predictions for unbiased evaluation
- Grid search for optimal ensemble weights
- Historical ETA lookup table integration

**Final Ensemble:** 2.566 MAE with optimized weights (LGB=0.50, CB=0.25, XGB=0.25)

### 4. Production Inference (inference.py)
Real-time prediction pipeline for sequential multi-stop ETA:

**Features:**
- Sequential 5-stop prediction with cumulative time tracking
- K-fold model averaging for stable predictions
- Monotonicity enforcement (ETAs must increase)
- Historical feature integration at inference time
- Minimum inter-stop time constraints (0.5 min)

## Model Performance

![Model Diagnostics](model_diagnostics.png)

### Key Results
- **Ensemble MAE:** 2.566 minutes (per inter-stop segment)
- **OOF RMSE:** 4.334 minutes
- **Cross-validation Stability:** σ < 0.05 across 5 folds
- **Prediction Quality:** Strong linear correlation (R² > 0.95)
- **Bias:** Near-zero residual mean (0.012 minutes)

### Feature Importance (Top 5)
1. **historical_eta_mean** (6.7) - Route-specific historical patterns
2. **lat_diff** (2.1) - Relative latitude to target stop
3. **lon_diff** (2.0) - Relative longitude to target stop
4. **historical_eta_median** (0.9) - Robust central tendency
5. **historical_eta_max** (0.8) - Upper bound pattern recognition

### Performance Insights
- Error decreases with route progress (Stop 4: 2.41 MAE vs Stop 0: 3.08 MAE)
- LightGBM shows best inference speed/accuracy trade-off
- Historical features dominate importance rankings
- Ensemble provides 0.054 min improvement over best single model

## Technical Highlights

**Data Engineering:**
- Processed 60GB GPS trajectories with quality filtering
- 195,724 training samples after cleaning
- 56,386 unique historical route-time patterns

**Feature Engineering Innovation:**
- Production-safe leakage prevention design
- Cyclical encoding for temporal periodicity
- Rush hour intensity modeling (continuous 0-1 scale)
- Historical lookup with fallback defaults

**Model Engineering:**
- K-fold averaging reduces prediction variance
- Ensemble weight optimization via grid search
- Monotonicity constraints for logical consistency

## Usage
## Usage
```python
# Import feature extractor and inference pipeline
from feature_engineering import ETAFeatureExtractor
from inference import ETAInferencePipeline

# 1. Initialize the feature extractor with stop coordinates
extractor = ETAFeatureExtractor(stop_coords=stop_coordinates)

# 2. Extract features from a GPS trajectory for a specific target stop
features = extractor.extract_features(
    gps_trajectory=trajectory_df,
    target_stop_id='STOP_123'
)

# 3. Load trained models into the inference pipeline
pipeline = ETAInferencePipeline(
    lgb_models=lgb_models,
    cb_models=cb_models,
    xgb_models=xgb_models,
    ensemble_weights={'lgb': 0.50, 'cb': 0.25, 'xgb': 0.25},
    feature_names=feature_names,
    historical_lookup=historical_lookup
)

# 4. Predict sequential ETAs for the next 5 stops
etas = pipeline.predict_sequence(
    test_trajectory=current_trajectory,
    future_stops=next_5_stops
)

# 5. Output cumulative ETAs in minutes
print(etas)  # Example output: [3.2, 8.7, 14.5, 21.3, 28.9]
``` 


## Performance Metrics

**Training:**
- Dataset: 195,724 samples, 60 features
- Training time: ~45 minutes (5-fold CV, 3 models)
- Memory: ~8GB peak usage

**Inference:**
- Prediction time: <100ms per 5-stop sequence
- Model size: ~50MB (all 15 models)
- Production ready: Thread-safe, error-handled

## Future Improvements

1. **Real-time traffic integration:** External traffic API features
2. **Route deviation handling:** Off-route detection and re-routing
3. **Weather impact modeling:** Precipitation and visibility features
4. **Passenger load estimation:** Dwell time prediction enhancement
5. **Dynamic model updating:** Online learning for concept drift

## Competition Context

**IISc Bengaluru Last Mile Challenge 2025 - Task 1**
- Ranking: Competitive performance (Top 10)
- Dataset: 60GB Bengaluru bus GPS trajectories
- Evaluation: Mean Absolute Error (MAE) on held-out test set
