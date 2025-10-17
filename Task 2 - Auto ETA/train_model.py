"""
Training Pipeline for Auto-rickshaw ETA Prediction
Author: Pratheek Shanbhogue
Task 2: Auto-rickshaw ETA Prediction

LightGBM + CatBoost ensemble with weighted averaging.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings

warnings.filterwarnings('ignore')


class AutoRickshawEnsembleTrainer:
    """
    Ensemble training pipeline: LightGBM + CatBoost
    
    Features:
    - Dual model training (LGB + CatBoost)
    - Weighted ensemble averaging
    - Train/validation split
    - Model persistence
    - Performance comparison
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.lgb_model = None
        self.cb_model = None
        self.ensemble_weights = {'lgb': 0.6, 'cb': 0.4}  # Default weights
        self.feature_names = None
        self.training_stats = {}
    
    def prepare_training_data(self,
                             df: pd.DataFrame,
                             target_col: str = 'actual_trip_time',
                             test_size: float = 0.2) -> tuple:
        """
        Prepare features and labels for training.
        
        Args:
            df: DataFrame with extracted features
            target_col: Name of target column
            test_size: Validation set fraction
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        # Identify feature columns (exclude target and identifiers)
        exclude_cols = [
            target_col, 'date', 'originlat', 'originlon', 
            'destlat', 'destlon', 'slot'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical features
        categorical_features = ['hour', 'dow', 'distance_cat']
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        X = df[feature_cols].copy()
        y = df[target_col].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            shuffle=True
        )
        
        self.feature_names = feature_cols
        self.categorical_features = [col for col in categorical_features if col in feature_cols]
        
        print(f" Training data prepared:")
        print(f"   Train: {len(X_train):,} samples")
        print(f"   Validation: {len(X_val):,} samples")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Categorical: {len(self.categorical_features)}")
        
        return X_train, X_val, y_train, y_val
    
    def train_ensemble(self,
                      X_train: pd.DataFrame,
                      y_train: np.ndarray,
                      X_val: pd.DataFrame,
                      y_val: np.ndarray) -> tuple:
        """
        Train LightGBM + CatBoost ensemble.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Tuple of (lgb_model, cb_model)
        """
        print(f"\n Training LightGBM + CatBoost Ensemble...")
        
        # ===== Train LightGBM =====
        print(f"\n Training LightGBM...")
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 8,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'verbosity': -1,
            'n_jobs': -1
        }
        
        # Create numeric dataset for LightGBM
        X_train_numeric = X_train.select_dtypes(include=[np.number])
        X_val_numeric = X_val.select_dtypes(include=[np.number])
        
        lgb_train = lgb.Dataset(X_train_numeric, y_train)
        lgb_val = lgb.Dataset(X_val_numeric, y_val, reference=lgb_train)
        
        self.lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        lgb_train_pred = self.lgb_model.predict(X_train_numeric, num_iteration=self.lgb_model.best_iteration)
        lgb_val_pred = self.lgb_model.predict(X_val_numeric, num_iteration=self.lgb_model.best_iteration)
        
        lgb_train_mae = mean_absolute_error(y_train, lgb_train_pred)
        lgb_val_mae = mean_absolute_error(y_val, lgb_val_pred)
        
        print(f"   ✅ LightGBM complete:")
        print(f"      Best iteration: {self.lgb_model.best_iteration}")
        print(f"      Train MAE: {lgb_train_mae:.3f} min")
        print(f"      Val MAE: {lgb_val_mae:.3f} min")
        
        # ===== Train CatBoost =====
        print(f"\n Training CatBoost...")
        
        cb_params = {
            'iterations': 2000,
            'depth': 7,
            'learning_rate': 0.05,
            'loss_function': 'RMSE',
            'eval_metric': 'MAE',
            'random_seed': self.random_state,
            'verbose': 100,
            'early_stopping_rounds': 100
        }
        
        self.cb_model = cb.CatBoostRegressor(**cb_params)
        
        # CatBoost can handle categorical features directly
        if self.categorical_features:
            cat_feature_indices = [X_train.columns.get_loc(col) for col in self.categorical_features]
            self.cb_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_feature_indices,
                use_best_model=True
            )
        else:
            self.cb_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True
            )
        
        cb_train_pred = self.cb_model.predict(X_train)
        cb_val_pred = self.cb_model.predict(X_val)
        
        cb_train_mae = mean_absolute_error(y_train, cb_train_pred)
        cb_val_mae = mean_absolute_error(y_val, cb_val_pred)
        
        print(f"      CatBoost complete:")
        print(f"      Best iteration: {self.cb_model.best_iteration_}")
        print(f"      Train MAE: {cb_train_mae:.3f} min")
        print(f"      Val MAE: {cb_val_mae:.3f} min")
        
        # ===== Optimize Ensemble Weights =====
        print(f"\n Optimizing ensemble weights...")
        best_weights, ensemble_mae = self._optimize_weights(
            lgb_val_pred, cb_val_pred, y_val
        )
        self.ensemble_weights = best_weights
        
        # Store stats
        self.training_stats = {
            'lgb_train_mae': lgb_train_mae,
            'lgb_val_mae': lgb_val_mae,
            'lgb_best_iter': self.lgb_model.best_iteration,
            'cb_train_mae': cb_train_mae,
            'cb_val_mae': cb_val_mae,
            'cb_best_iter': self.cb_model.best_iteration_,
            'ensemble_val_mae': ensemble_mae,
            'ensemble_weights': self.ensemble_weights,
            'num_features': len(self.feature_names)
        }
        
        print(f"\n Ensemble training complete!")
        print(f"   Best weights: LGB={best_weights['lgb']:.2f}, CB={best_weights['cb']:.2f}")
        print(f"   Ensemble Val MAE: {ensemble_mae:.3f} min")
        print(f"   Improvement: {min(lgb_val_mae, cb_val_mae) - ensemble_mae:.3f} min")
        
        return self.lgb_model, self.cb_model
    
    def _optimize_weights(self, lgb_pred, cb_pred, y_true):
        """Find optimal ensemble weights via grid search"""
        best_mae = float('inf')
        best_weights = None
        
        weight_combinations = [
            {'lgb': 0.7, 'cb': 0.3},
            {'lgb': 0.6, 'cb': 0.4},
            {'lgb': 0.5, 'cb': 0.5},
            {'lgb': 0.4, 'cb': 0.6},
            {'lgb': 0.3, 'cb': 0.7},
        ]
        
        for weights in weight_combinations:
            ensemble_pred = weights['lgb'] * lgb_pred + weights['cb'] * cb_pred
            mae = mean_absolute_error(y_true, ensemble_pred)
            
            print(f"   LGB={weights['lgb']:.1f}, CB={weights['cb']:.1f} → MAE={mae:.3f}")
            
            if mae < best_mae:
                best_mae = mae
                best_weights = weights
        
        return best_weights, best_mae
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Weighted ensemble predictions
        """
        if self.lgb_model is None or self.cb_model is None:
            raise ValueError("Models not trained yet")
        
        # LightGBM prediction (numeric features only)
        X_numeric = X.select_dtypes(include=[np.number])
        lgb_pred = self.lgb_model.predict(X_numeric, num_iteration=self.lgb_model.best_iteration)
        
        # CatBoost prediction (all features)
        cb_pred = self.cb_model.predict(X)
        
        # Weighted ensemble
        ensemble_pred = (self.ensemble_weights['lgb'] * lgb_pred + 
                        self.ensemble_weights['cb'] * cb_pred)
        
        return ensemble_pred
    
    def get_feature_importance(self, model_type: str = 'lgb', top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            model_type: 'lgb' or 'cb'
            top_n: Number of top features
            
        Returns:
            DataFrame with feature importance
        """
        if model_type == 'lgb':
            if self.lgb_model is None:
                raise ValueError("LightGBM model not trained")
            importance = self.lgb_model.feature_importance(importance_type='gain')
            feature_names = self.lgb_model.feature_name()
        elif model_type == 'cb':
            if self.cb_model is None:
                raise ValueError("CatBoost model not trained")
            importance = self.cb_model.get_feature_importance()
            feature_names = self.cb_model.feature_names_
        else:
            raise ValueError("model_type must be 'lgb' or 'cb'")
        
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feat_imp.head(top_n)
    
    def save_models(self,
                   lgb_path: str = 'autorickshaw_lgb.txt',
                   cb_path: str = 'autorickshaw_cb.cbm',
                   metadata_path: str = 'ensemble_metadata.pkl'):
        """Save both models and metadata"""
        if self.lgb_model is None or self.cb_model is None:
            raise ValueError("Models not trained yet")
        
        # Save models
        self.lgb_model.save_model(lgb_path)
        self.cb_model.save_model(cb_path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'ensemble_weights': self.ensemble_weights,
            'training_stats': self.training_stats,
            'random_state': self.random_state
        }
        joblib.dump(metadata, metadata_path)
        
        print(f"\n Models saved:")
        print(f"   LightGBM: {lgb_path}")
        print(f"   CatBoost: {cb_path}")
        print(f"   Metadata: {metadata_path}")
    
    def load_models(self,
                   lgb_path: str = 'autorickshaw_lgb.txt',
                   cb_path: str = 'autorickshaw_cb.cbm',
                   metadata_path: str = 'ensemble_metadata.pkl'):
        """Load both models and metadata"""
        # Load models
        self.lgb_model = lgb.Booster(model_file=lgb_path)
        self.cb_model = cb.CatBoostRegressor()
        self.cb_model.load_model(cb_path)
        
        # Load metadata
        metadata = joblib.load(metadata_path)
        self.feature_names = metadata['feature_names']
        self.categorical_features = metadata['categorical_features']
        self.ensemble_weights = metadata['ensemble_weights']
        self.training_stats = metadata['training_stats']
        self.random_state = metadata['random_state']
        
        print(f"   Ensemble loaded:")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Weights: LGB={self.ensemble_weights['lgb']:.2f}, CB={self.ensemble_weights['cb']:.2f}")
        print(f"   Val MAE: {self.training_stats['ensemble_val_mae']:.3f} min")


# Example usage
if __name__ == "__main__":
    print("Auto-rickshaw Ensemble Training Pipeline")
    print("\n Capabilities:")
    print("   - LightGBM + CatBoost dual training")
    print("   - Weighted ensemble optimization")
    print("   - Feature importance analysis")
    print("   - Model persistence")
    print("\n Usage:")
    print("   trainer = AutoRickshawEnsembleTrainer()")
    print("   X_train, X_val, y_train, y_val = trainer.prepare_training_data(df)")
    print("   lgb_model, cb_model = trainer.train_ensemble(X_train, y_train, X_val, y_val)")
    print("   trainer.save_models()")
