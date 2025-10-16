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
