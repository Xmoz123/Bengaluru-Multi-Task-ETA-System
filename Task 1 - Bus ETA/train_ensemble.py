"""
Ensemble Training Pipeline for Bus ETA Prediction
Author: Pratheek Shanbhogue
Competition: IISc Bengaluru Last Mile Challenge 2025 - Task 1

K-Fold cross-validation with LightGBM, CatBoost, and XGBoost ensemble.
Includes historical ETA lookup and production-ready validation.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import joblib
import os
import gc
import warnings

warnings.filterwarnings('ignore')


# ===== CONFIGURATION =====
class TrainingConfig:
    """Training pipeline configuration"""
    N_FOLDS = 5
    RANDOM_SEED = 42
    
    # Model parameters
    LGB_PARAMS = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 7,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbosity': -1,
        'n_jobs': -1
    }
    
    CB_PARAMS = {
        'iterations': 1500,
        'depth': 7,
        'learning_rate': 0.05,
        'loss_function': 'RMSE',
        'eval_metric': 'MAE',
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 100
    }
    
    XGB_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 1500,
        'early_stopping_rounds': 100,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }


# ===== HISTORICAL FEATURE ENGINEERING =====
def create_historical_lookup(df: pd.DataFrame, target_col: str = 'target') -> dict:
    """
    Create historical ETA lookup table from training data.
    
    Groups by route, stop sequence, hour, and day to compute
    historical statistics for each segment.
    
    Args:
        df: Training dataframe with route_id, stop_sequence_position,
            hour, day_of_week, and target columns
        target_col: Name of target column
        
    Returns:
        Dictionary mapping (route, stop, hour, dow) -> statistics
    """
    print("🔧 Creating historical ETA lookup table...")
    
    historical_etas = df.groupby(
        ['route_id', 'stop_sequence_position', 'hour', 'day_of_week']
    ).agg({
        target_col: ['mean', 'median', 'std', 'count', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    historical_etas.columns = [
        'route_id', 'stop_sequence_position', 'hour', 'day_of_week',
        'historical_eta_mean', 'historical_eta_median', 
        'historical_eta_std', 'historical_sample_count',
        'historical_eta_min', 'historical_eta_max'
    ]
    
    print(f"   ✅ Created {len(historical_etas):,} unique patterns")
    print(f"   📊 Coverage: {historical_etas['route_id'].nunique()} routes, "
          f"{historical_etas['stop_sequence_position'].nunique()} stops")
    
    # Convert to lookup dictionary
    historical_lookup = historical_etas.set_index(
        ['route_id', 'stop_sequence_position', 'hour', 'day_of_week']
    ).to_dict('index')
    
    return historical_lookup


def add_historical_features(df: pd.DataFrame, historical_lookup: dict,
                           overall_mean: float, overall_std: float) -> pd.DataFrame:
    """
    Add historical ETA features to dataframe.
    
    Args:
        df: Dataframe with route_id, stop_sequence_position, hour, day_of_week
        historical_lookup: Historical statistics lookup
        overall_mean: Fallback mean for missing patterns
        overall_std: Fallback std for missing patterns
        
    Returns:
        Dataframe with added historical features
    """
    # Create lookup keys
    df['_lookup_key'] = list(zip(
        df['route_id'], df['stop_sequence_position'],
        df['hour'], df['day_of_week']
    ))
    
    # Initialize columns
    hist_cols = [
        'historical_eta_mean', 'historical_eta_median', 
        'historical_eta_std', 'historical_sample_count',
        'historical_eta_min', 'historical_eta_max'
    ]
    
    for col in hist_cols:
        df[col] = np.nan
    
    # Fill from lookup
    for idx, row in df.iterrows():
        key = row['_lookup_key']
        if key in historical_lookup:
            stats = historical_lookup[key]
            for col in hist_cols:
                df.at[idx, col] = stats.get(col, np.nan)
    
    # Fill missing values
    df['historical_eta_mean'] = df['historical_eta_mean'].fillna(overall_mean)
    df['historical_eta_median'] = df['historical_eta_median'].fillna(overall_mean)
    df['historical_eta_std'] = df['historical_eta_std'].fillna(overall_std)
    df['historical_sample_count'] = df['historical_sample_count'].fillna(0)
    df['historical_eta_min'] = df['historical_eta_min'].fillna(overall_mean * 0.5)
    df['historical_eta_max'] = df['historical_eta_max'].fillna(overall_mean * 2.0)
    
    # Derived features
    df['historical_confidence'] = np.clip(
        df['historical_sample_count'] / 50.0, 0.0, 1.0
    )
    df['historical_eta_range'] = (
        df['historical_eta_max'] - df['historical_eta_min']
    )
    df['historical_eta_cv'] = (
        df['historical_eta_std'] / (df['historical_eta_mean'] + 1e-6)
    )
    
    df = df.drop(columns=['_lookup_key'])
    
    return df


# ===== K-FOLD ENSEMBLE TRAINING =====
def train_kfold_ensemble(X: pd.DataFrame, y: np.ndarray, 
                        config: TrainingConfig = None) -> dict:
    """
    Train ensemble using K-Fold cross-validation.
    
    Args:
        X: Feature dataframe
        y: Target values
        config: Training configuration
        
    Returns:
        Dictionary with trained models and results
    """
    if config is None:
        config = TrainingConfig()
    
    print(f"\n🏆 TRAINING ENSEMBLE - {config.N_FOLDS}-FOLD CV")
    print("=" * 70)
    
    # Storage
    lgb_models = []
    cb_models = []
    xgb_models = []
    fold_scores = {'lgb': [], 'cb': [], 'xgb': []}
    oof_predictions = {
        'lgb': np.zeros(len(X)),
        'cb': np.zeros(len(X)),
        'xgb': np.zeros(len(X))
    }
    
    # Identify categorical features
    cat_features = [col for col in X.columns 
                   if 'route_id' in col or 'stop_id' in col]
    
    # Create numeric version
    X_numeric = X.select_dtypes(include=[np.number])
    
    # K-Fold setup
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, 
               random_state=config.RANDOM_SEED)
    
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n📊 FOLD {fold + 1}/{config.N_FOLDS}")
        print("-" * 50)
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        X_train_numeric = X_train_fold.select_dtypes(include=[np.number])
        X_val_numeric = X_val_fold.select_dtypes(include=[np.number])
        
        # === LightGBM ===
        print("   🔵 Training LightGBM...")
        lgb_train = lgb.Dataset(X_train_numeric, y_train_fold)
        lgb_val = lgb.Dataset(X_val_numeric, y_val_fold, reference=lgb_train)
        
        lgb_model = lgb.train(
            config.LGB_PARAMS,
            lgb_train,
            num_boost_round=1500,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(0)
            ]
        )
        
        lgb_pred = lgb_model.predict(X_val_numeric, 
                                     num_iteration=lgb_model.best_iteration)
        oof_predictions['lgb'][val_idx] = lgb_pred
        lgb_mae = mean_absolute_error(y_val_fold, lgb_pred)
        fold_scores['lgb'].append(lgb_mae)
        lgb_models.append(lgb_model)
        
        print(f"      MAE: {lgb_mae:.3f} min")
        
        del lgb_train, lgb_val
        gc.collect()
        
        # === CatBoost ===
        print("   🟠 Training CatBoost...")
        cb_model = cb.CatBoostRegressor(**config.CB_PARAMS)
        
        if cat_features:
            cb_model.set_params(cat_features=cat_features)
        
        cb_model.fit(
            X_train_fold, y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            use_best_model=True
        )
        
        cb_pred = cb_model.predict(X_val_fold)
        oof_predictions['cb'][val_idx] = cb_pred
        cb_mae = mean_absolute_error(y_val_fold, cb_pred)
        fold_scores['cb'].append(cb_mae)
        cb_models.append(cb_model)
        
        print(f"      MAE: {cb_mae:.3f} min")
        gc.collect()
        
        # === XGBoost ===
        print("   🟢 Training XGBoost...")
        xgb_model = xgb.XGBRegressor(**config.XGB_PARAMS)
        
        xgb_model.fit(
            X_train_numeric, y_train_fold,
            eval_set=[(X_val_numeric, y_val_fold)],
            verbose=False
        )
        
        xgb_pred = xgb_model.predict(X_val_numeric)
        oof_predictions['xgb'][val_idx] = xgb_pred
        xgb_mae = mean_absolute_error(y_val_fold, xgb_pred)
        fold_scores['xgb'].append(xgb_mae)
        xgb_models.append(xgb_model)
        
        print(f"      MAE: {xgb_mae:.3f} min")
        gc.collect()
    
    # Optimize ensemble weights
    print("\n🎯 Optimizing ensemble weights...")
    best_weights, best_mae = optimize_ensemble_weights(
        oof_predictions, y
    )
    
    # Calculate OOF scores
    results = {
        'lgb_models': lgb_models,
        'cb_models': cb_models,
        'xgb_models': xgb_models,
        'ensemble_weights': best_weights,
        'fold_scores': fold_scores,
        'oof_predictions': oof_predictions,
        'best_mae': best_mae,
        'feature_names': X_numeric.columns.tolist()
    }
    
    # Print summary
    print_training_summary(results, y)
    
    return results


def optimize_ensemble_weights(oof_predictions: dict, y: np.ndarray) -> tuple:
    """
    Find optimal ensemble weights through grid search.
    
    Args:
        oof_predictions: Dictionary with OOF predictions for each model
        y: True target values
        
    Returns:
        Tuple of (best_weights, best_mae)
    """
    weight_combinations = [
        {'lgb': 0.40, 'cb': 0.35, 'xgb': 0.25},
        {'lgb': 0.45, 'cb': 0.30, 'xgb': 0.25},
        {'lgb': 0.35, 'cb': 0.40, 'xgb': 0.25},
        {'lgb': 0.33, 'cb': 0.34, 'xgb': 0.33},
        {'lgb': 0.50, 'cb': 0.25, 'xgb': 0.25},
    ]
    
    best_mae = float('inf')
    best_weights = None
    
    for weights in weight_combinations:
        ensemble_pred = (weights['lgb'] * oof_predictions['lgb'] + 
                        weights['cb'] * oof_predictions['cb'] + 
                        weights['xgb'] * oof_predictions['xgb'])
        mae = mean_absolute_error(y, ensemble_pred)
        
        print(f"   LGB={weights['lgb']:.2f}, CB={weights['cb']:.2f}, "
              f"XGB={weights['xgb']:.2f} → MAE={mae:.3f}")
        
        if mae < best_mae:
            best_mae = mae
            best_weights = weights
    
    print(f"\n🏆 BEST ENSEMBLE: LGB={best_weights['lgb']:.2f}, "
          f"CB={best_weights['cb']:.2f}, XGB={best_weights['xgb']:.2f}")
    print(f"   OOF MAE: {best_mae:.3f} minutes")
    
    return best_weights, best_mae


def print_training_summary(results: dict, y: np.ndarray):
    """Print comprehensive training summary"""
    print("\n" + "=" * 70)
    print("🎯 TRAINING SUMMARY")
    print("=" * 70)
    
    # Individual model scores
    lgb_mae = mean_absolute_error(y, results['oof_predictions']['lgb'])
    cb_mae = mean_absolute_error(y, results['oof_predictions']['cb'])
    xgb_mae = mean_absolute_error(y, results['oof_predictions']['xgb'])
    
    print(f"\n🔵 LightGBM:")
    print(f"   Fold MAEs: {[f'{s:.3f}' for s in results['fold_scores']['lgb']]}")
    print(f"   Mean: {np.mean(results['fold_scores']['lgb']):.3f} ± "
          f"{np.std(results['fold_scores']['lgb']):.3f}")
    print(f"   OOF MAE: {lgb_mae:.3f}")
    
    print(f"\n🟠 CatBoost:")
    print(f"   Fold MAEs: {[f'{s:.3f}' for s in results['fold_scores']['cb']]}")
    print(f"   Mean: {np.mean(results['fold_scores']['cb']):.3f} ± "
          f"{np.std(results['fold_scores']['cb']):.3f}")
    print(f"   OOF MAE: {cb_mae:.3f}")
    
    print(f"\n🟢 XGBoost:")
    print(f"   Fold MAEs: {[f'{s:.3f}' for s in results['fold_scores']['xgb']]}")
    print(f"   Mean: {np.mean(results['fold_scores']['xgb']):.3f} ± "
          f"{np.std(results['fold_scores']['xgb']):.3f}")
    print(f"   OOF MAE: {xgb_mae:.3f}")
    
    print(f"\n🏆 ENSEMBLE:")
    print(f"   Weights: {results['ensemble_weights']}")
    print(f"   OOF MAE: {results['best_mae']:.3f} minutes")
    print(f"   Improvement: {min(lgb_mae, cb_mae, xgb_mae) - results['best_mae']:.3f} min")
    
    print("=" * 70)


# ===== MODEL PERSISTENCE =====
def save_models(results: dict, historical_lookup: dict, output_dir: str = 'models'):
    """
    Save trained models and artifacts.
    
    Args:
        results: Training results dictionary
        historical_lookup: Historical ETA lookup
        output_dir: Output directory path
    """
    print(f"\n💾 Saving models to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LightGBM models
    for i, model in enumerate(results['lgb_models']):
        model.save_model(f'{output_dir}/lgb_fold{i}.txt')
    
    # Save CatBoost models
    for i, model in enumerate(results['cb_models']):
        model.save_model(f'{output_dir}/cb_fold{i}.cbm')
    
    # Save XGBoost models
    for i, model in enumerate(results['xgb_models']):
        model.save_model(f'{output_dir}/xgb_fold{i}.json')
    
    # Save ensemble weights
    joblib.dump(results['ensemble_weights'], f'{output_dir}/ensemble_weights.pkl')
    
    # Save feature names
    joblib.dump(results['feature_names'], f'{output_dir}/feature_names.pkl')
    
    # Save historical lookup
    joblib.dump(historical_lookup, f'{output_dir}/historical_lookup.pkl')
    
    print(f"✅ Saved {len(results['lgb_models'])} models per type + artifacts")


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    # Load your processed data
    df = pd.read_parquet("processed_data.parquet")
    targets = df['target'].values
    X = df.drop(columns=['target'])
    
    # Create historical lookup
    historical_lookup = create_historical_lookup(df)
    
    # Add historical features
    X = add_historical_features(
        X, historical_lookup,
        overall_mean=targets.mean(),
        overall_std=targets.std()
    )
    
    # Train ensemble
    results = train_kfold_ensemble(X, targets)
    
    # Save models
    save_models(results, historical_lookup)
    
    print("\nTraining complete!")
