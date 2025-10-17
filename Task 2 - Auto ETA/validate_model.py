"""
Model Validation and Performance Analysis
Author: Pratheek Shanbhogue
Task 2: Auto-rickshaw ETA Prediction

Comprehensive validation metrics, error analysis, and performance visualization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class ModelValidator:
    """
    Validation and performance analysis for auto-rickshaw ETA models.
    
    Features:
    - Multiple evaluation metrics
    - Error distribution analysis
    - Performance by distance category
    - Performance by time slot
    - Prediction quality diagnostics
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                dataset_name: str = "Validation") -> dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary with metrics
        """
        # Core metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Error statistics
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        
        # Accuracy bands
        within_1min = np.mean(np.abs(errors) <= 1.0) * 100
        within_2min = np.mean(np.abs(errors) <= 2.0) * 100
        within_5min = np.mean(np.abs(errors) <= 5.0) * 100
        
        # Percentile errors
        p50 = np.percentile(np.abs(errors), 50)
        p90 = np.percentile(np.abs(errors), 90)
        p95 = np.percentile(np.abs(errors), 95)
        
        metrics = {
            'dataset': dataset_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_error': mean_error,
            'median_error': median_error,
            'std_error': std_error,
            'within_1min': within_1min,
            'within_2min': within_2min,
            'within_5min': within_5min,
            'p50_error': p50,
            'p90_error': p90,
            'p95_error': p95,
            'n_samples': len(y_true)
        }
        
        self.results[dataset_name] = metrics
        
        return metrics
    
    def print_metrics(self, metrics: dict):
        """Print formatted metrics report"""
        print(f"\n  {metrics['dataset']} Set Performance")
        print("=" * 50)
        print(f"Samples: {metrics['n_samples']:,}")
        print(f"\n Core Metrics:")
        print(f"   MAE:  {metrics['mae']:.3f} minutes")
        print(f"   RMSE: {metrics['rmse']:.3f} minutes")
        print(f"   R²:   {metrics['r2']:.4f}")
        print(f"\n Error Statistics:")
        print(f"   Mean:   {metrics['mean_error']:+.3f} minutes")
        print(f"   Median: {metrics['median_error']:+.3f} minutes")
        print(f"   Std:    {metrics['std_error']:.3f} minutes")
        print(f"\n Accuracy Bands:")
        print(f"   Within 1 min:  {metrics['within_1min']:.1f}%")
        print(f"   Within 2 min:  {metrics['within_2min']:.1f}%")
        print(f"   Within 5 min:  {metrics['within_5min']:.1f}%")
        print(f"\n Error Percentiles:")
        print(f"   P50: {metrics['p50_error']:.3f} minutes")
        print(f"   P90: {metrics['p90_error']:.3f} minutes")
        print(f"   P95: {metrics['p95_error']:.3f} minutes")
    
    def analyze_by_distance(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           distances: np.ndarray) -> pd.DataFrame:
        """
        Analyze performance by distance category.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            distances: Trip distances in km
            
        Returns:
            DataFrame with metrics per distance category
        """
        # Categorize distances
        dist_cats = pd.cut(
            distances,
            bins=[0, 2, 5, 10, 20, 100],
            labels=['<2km', '2-5km', '5-10km', '10-20km', '>20km']
        )
        
        results = []
        for cat in dist_cats.cat.categories:
            mask = dist_cats == cat
            if mask.sum() == 0:
                continue
            
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            count = mask.sum()
            
            results.append({
                'distance_category': cat,
                'count': count,
                'mae': mae,
                'rmse': rmse
            })
        
        df = pd.DataFrame(results)
        
        print("\n Performance by Distance Category")
        print(df.to_string(index=False))
        
        return df
    
    def analyze_by_timeslot(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           slots: np.ndarray) -> pd.DataFrame:
        """
        Analyze performance by time slot.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            slots: Time slots (0-95)
            
        Returns:
            DataFrame with metrics per hour
        """
        # Convert slots to hours
        hours = slots // 4
        
        results = []
        for hour in range(24):
            mask = hours == hour
            if mask.sum() == 0:
                continue
            
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            count = mask.sum()
            
            results.append({
                'hour': hour,
                'count': count,
                'mae': mae
            })
        
        df = pd.DataFrame(results)
        
        print("\n⏰ Performance by Hour of Day")
        print(df.to_string(index=False))
        
        return df
    
    def plot_diagnostics(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        save_path: str = None):
        """
        Create diagnostic plots.
        
        Creates:
        1. Predicted vs Actual scatter plot
        2. Residual distribution
        3. Residuals vs Predicted
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.3, s=1)
        axes[0].plot([0, y_true.max()], [0, y_true.max()], 'r--', lw=2, label='Perfect prediction')
        axes[0].set_xlabel('Actual ETA (min)')
        axes[0].set_ylabel('Predicted ETA (min)')
        axes[0].set_title('Predicted vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Residual distribution
        residuals = y_pred - y_true
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residual (Predicted - Actual)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residual Distribution (Mean: {np.mean(residuals):.3f})')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Residuals vs Predicted
        axes[2].scatter(y_pred, residuals, alpha=0.3, s=1)
        axes[2].axhline(0, color='r', linestyle='--', lw=2)
        axes[2].set_xlabel('Predicted ETA (min)')
        axes[2].set_ylabel('Residual')
        axes[2].set_title('Residuals vs Predicted')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n Diagnostic plots saved to {save_path}")
        
        plt.show()
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare multiple models if multiple evaluations done.
        
        Returns:
            DataFrame comparing all evaluated models
        """
        if not self.results:
            print("No evaluation results available")
            return pd.DataFrame()
        
        comparison = pd.DataFrame(self.results).T
        comparison = comparison[[
            'n_samples', 'mae', 'rmse', 'r2', 
            'within_1min', 'within_2min', 'within_5min'
        ]]
        
        print("\n Model Comparison")
        print(comparison.to_string())
        
        return comparison


# Example usage
if __name__ == "__main__":
    print(" Auto-rickshaw Model Validation")
    print("\n Capabilities:")
    print("   - Comprehensive evaluation metrics")
    print("   - Error distribution analysis")
    print("   - Performance by distance/time")
    print("   - Diagnostic visualizations")
    print("\n Usage:")
    print("   validator = ModelValidator()")
    print("   metrics = validator.evaluate(y_true, y_pred)")
    print("   validator.print_metrics(metrics)")
    print("   validator.plot_diagnostics(y_true, y_pred)")
