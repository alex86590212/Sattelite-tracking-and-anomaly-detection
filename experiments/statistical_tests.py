"""Statistical tests for model comparison (1-2 days evaluation).

Implements:
- Paired Wilcoxon signed-rank test (Transformer vs LSTM, Transformer vs Bayesian)
- Cliff's delta effect size
- Horizon robustness (Wilcoxon per horizon + Holm-Bonferroni correction)
- Bayesian calibration (empirical coverage, NLL comparison)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from typing import Dict, Tuple, List, Optional
from datetime import timedelta, timezone
from scipy import stats
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from data.tle_loader import TLELoader
from data.telemetry_processor import TelemetryProcessor
from physics_model.sgp4_propagator import SGP4Propagator
from ml_models.lstm_model import OrbitLSTM
from ml_models.transformer_model import OrbitTransformer
from ml_models.bayesian_model import BayesianOrbitPredictor


def load_trained_model(model_type: str, model_path: Path, device: str = 'cpu'):
    """Load a trained model from disk."""
    if model_type == 'lstm':
        model = OrbitLSTM(input_size=6, hidden_size=128, num_layers=2, dropout=0.01, output_size=6)
    elif model_type == 'transformer':
        model = OrbitTransformer(
            input_size=6, d_model=128, nhead=8, num_layers=2,
            dim_feedforward=256, dropout=0.01, output_size=6
        )
    elif model_type == 'bayesian':
        model = BayesianOrbitPredictor(input_size=6, hidden_size=128, num_layers=2, output_size=6)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def compute_position_errors(predictions: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Compute position errors in km."""
    return np.linalg.norm(predictions[:, :3] - observed[:, :3], axis=1)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size.
    
    Returns:
        Effect size: -1 (all x < y) to +1 (all x > y)
        |d| < 0.147: negligible
        |d| < 0.33: small
        |d| < 0.474: medium
        |d| >= 0.474: large
    """
    n_x, n_y = len(x), len(y)
    dominance = 0
    
    for xi in x:
        for yj in y:
            if xi > yj:
                dominance += 1
            elif xi < yj:
                dominance -= 1
    
    delta = dominance / (n_x * n_y)
    return delta


def holm_bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    Returns:
        (corrected_p_values, rejected)
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    corrected = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)
    
    for i, idx in enumerate(sorted_indices):
        corrected[idx] = p_values[idx] * (n - i)
        if i > 0:
            corrected[idx] = max(corrected[idx], corrected[sorted_indices[i-1]])
    
    rejected = corrected <= alpha
    return corrected, rejected


def empirical_coverage(predictions: np.ndarray, std: np.ndarray, observed: np.ndarray, sigma: float = 1.0) -> float:
    """
    Compute empirical coverage for Bayesian predictions.
    
    Args:
        predictions: Mean predictions (n_samples, 6)
        std: Standard deviations (n_samples, 6)
        observed: True values (n_samples, 6)
        sigma: Number of standard deviations (1.0 for 1σ, 2.0 for 2σ)
    
    Returns:
        Coverage percentage
    """
    errors = np.abs(predictions - observed)
    thresholds = sigma * std
    within_bounds = np.all(errors <= thresholds, axis=1)
    return np.mean(within_bounds) * 100


def negative_log_likelihood(predictions: np.ndarray, std: np.ndarray, observed: np.ndarray) -> float:
    """
    Compute negative log-likelihood for Bayesian predictions.
    
    Assumes Gaussian distribution.
    """
    errors = predictions - observed
    nll = 0.5 * np.sum((errors / (std + 1e-8))**2) + np.sum(np.log(std + 1e-8))
    nll += 0.5 * len(predictions) * np.log(2 * np.pi)
    return nll / len(predictions)


def main():
    """Run statistical tests on models."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Statistical tests for model comparison')
    parser.add_argument('--satellite_id', type=int, default=25544)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--sequence_length', type=int, default=60)
    parser.add_argument('--test_duration_hours', type=float, default=48.0)  # 2 days
    parser.add_argument('--num_horizons', type=int, default=10)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Statistical Tests: Model Comparison (1-2 days)")
    print("=" * 80)
    
    # Load TLE and generate test data
    print("\n[1/5] Loading TLE data and generating test telemetry...")
    loader = TLELoader()
    
    # Use hardcoded TLE data to avoid API issues
    # ISS (ZARYA) TLE data
    tle_lines = [
        "ISS (ZARYA)",
        "1 25544U 98067A   26003.40374414  .00014825  00000+0  27593-3 0  9996",
        "2 25544  51.6328  35.2048 0007560 335.4964  24.5664 15.49064236546216"
    ]
    
    try:
        # Parse the TLE data
        tle_data = loader.parse_tle(tle_lines)
        sat = loader.create_satrec(tle_data, tle_lines)
        print(f"Loaded TLE for: {tle_data['name']}")
    except Exception as e:
        print(f"Error: Failed to parse hardcoded TLE: {e}")
        raise Exception(f"Cannot proceed without TLE data. Error: {e}")
    
    processor = TelemetryProcessor(time_step_minutes=1.0)
    # Generate test data from AFTER training period
    start_time = tle_data['epoch'] + timedelta(hours=25)
    test_telemetry = processor.generate_telemetry(
        sat, start_time, duration_hours=args.test_duration_hours, add_noise=True
    )
    
    if len(test_telemetry) == 0:
        raise ValueError("No test telemetry generated!")
    
    print(f"Generated {len(test_telemetry)} test telemetry points")
    print(f"Test period: {test_telemetry['timestamp'].min()} to {test_telemetry['timestamp'].max()}")
    
    # Load normalization statistics (from training period)
    print("\n[2/5] Loading normalization statistics...")
    sample_telemetry = processor.generate_telemetry(
        sat, tle_data['epoch'], duration_hours=24.0, add_noise=True
    )
    _, _, train_mean, train_std = processor.prepare_sequences(
        sample_telemetry,
        sequence_length=args.sequence_length,
        prediction_horizon=1,
        normalize=True
    )
    
    # Prepare test sequences
    print("Preparing test sequences...")
    X_test, y_test, _, _ = processor.prepare_sequences(
        test_telemetry,
        sequence_length=args.sequence_length,
        prediction_horizon=1,
        normalize=True,
        mean=train_mean,
        std=train_std
    )
    
    if len(X_test) == 0:
        raise ValueError("No test sequences generated!")
    
    print(f"Prepared {len(X_test)} test sequences")
    
    # Load models and generate predictions
    print("\n[3/5] Loading models and generating predictions...")
    model_dir = Path(__file__).parent.parent / 'ml_models'
    models = {}
    predictions = {}
    uncertainties = {}  # For Bayesian model
    
    for model_type in ['lstm', 'transformer', 'bayesian']:
        model_path = model_dir / f'{model_type}_trained.pt'
        if not model_path.exists():
            print(f"  WARNING: Model {model_type} not found, skipping...")
            continue
        
        print(f"  Loading {model_type.upper()}...")
        model = load_trained_model(model_type, model_path, device=args.device)
        models[model_type] = model
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(args.device)
            output = model(X_tensor)
            
            if model_type == 'bayesian':
                pred_mean, pred_std, _ = output
                predictions[model_type] = pred_mean.cpu().numpy()
                uncertainties[model_type] = pred_std.cpu().numpy()
            else:
                predictions[model_type] = output.cpu().numpy()
    
    # Compute errors for all models
    print("\n[4/5] Computing errors...")
    errors = {}
    for model_type, pred in predictions.items():
        errors[model_type] = compute_position_errors(pred, y_test)
        print(f"  {model_type.upper()}: Mean error = {np.mean(errors[model_type]):.6f} km, "
              f"Std = {np.std(errors[model_type]):.6f} km")
    
    # A. Accuracy tests (Paired Wilcoxon + Cliff's delta)
    print("\n[5/5] Running statistical tests...")
    print("\n" + "=" * 80)
    print("A. ACCURACY TESTS (Core Claim)")
    print("=" * 80)
    
    results_table = []
    
    # Transformer vs LSTM
    if 'transformer' in errors and 'lstm' in errors:
        trans_errors = errors['transformer']
        lstm_errors = errors['lstm']
        
        # Paired Wilcoxon signed-rank test
        stat, p_value = wilcoxon(trans_errors, lstm_errors, alternative='two-sided')
        
        # Cliff's delta
        delta = cliffs_delta(trans_errors, lstm_errors)
        
        # Mean and std
        trans_mean = np.mean(trans_errors)
        trans_std = np.std(trans_errors)
        lstm_mean = np.mean(lstm_errors)
        lstm_std = np.std(lstm_errors)
        
        results_table.append({
            'Comparison': 'Transformer vs LSTM',
            'Model1': f'{trans_mean:.6f} ± {trans_std:.6f}',
            'Model2': f'{lstm_mean:.6f} ± {lstm_std:.6f}',
            'p-value': f'{p_value:.6f}',
            "Cliff's δ": f'{delta:.4f}',
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
        
        print(f"\nTransformer vs LSTM:")
        print(f"  Transformer: {trans_mean:.6f} ± {trans_std:.6f} km")
        print(f"  LSTM:        {lstm_mean:.6f} ± {lstm_std:.6f} km")
        print(f"  Wilcoxon p-value: {p_value:.6f}")
        print(f"  Cliff's delta: {delta:.4f}")
        if abs(delta) < 0.147:
            effect = "negligible"
        elif abs(delta) < 0.33:
            effect = "small"
        elif abs(delta) < 0.474:
            effect = "medium"
        else:
            effect = "large"
        print(f"  Effect size: {effect}")
    
    # Transformer vs Bayesian
    if 'transformer' in errors and 'bayesian' in errors:
        trans_errors = errors['transformer']
        bay_errors = errors['bayesian']
        
        # Paired Wilcoxon signed-rank test
        stat, p_value = wilcoxon(trans_errors, bay_errors, alternative='two-sided')
        
        # Cliff's delta
        delta = cliffs_delta(trans_errors, bay_errors)
        
        # Mean and std
        trans_mean = np.mean(trans_errors)
        trans_std = np.std(trans_errors)
        bay_mean = np.mean(bay_errors)
        bay_std = np.std(bay_errors)
        
        results_table.append({
            'Comparison': 'Transformer vs Bayesian',
            'Model1': f'{trans_mean:.6f} ± {trans_std:.6f}',
            'Model2': f'{bay_mean:.6f} ± {bay_std:.6f}',
            'p-value': f'{p_value:.6f}',
            "Cliff's δ": f'{delta:.4f}',
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
        
        print(f"\nTransformer vs Bayesian:")
        print(f"  Transformer: {trans_mean:.6f} ± {trans_std:.6f} km")
        print(f"  Bayesian:    {bay_mean:.6f} ± {bay_std:.6f} km")
        print(f"  Wilcoxon p-value: {p_value:.6f}")
        print(f"  Cliff's delta: {delta:.4f}")
        if abs(delta) < 0.147:
            effect = "negligible"
        elif abs(delta) < 0.33:
            effect = "small"
        elif abs(delta) < 0.474:
            effect = "medium"
        else:
            effect = "large"
        print(f"  Effect size: {effect}")
    
    # B. Horizon robustness
    print("\n" + "=" * 80)
    print("B. HORIZON ROBUSTNESS")
    print("=" * 80)
    
    # Test different prediction horizons
    horizons = np.linspace(1, min(args.num_horizons, len(X_test) // 10), args.num_horizons, dtype=int)
    horizon_p_values = {}
    
    for model_type in ['transformer', 'lstm', 'bayesian']:
        if model_type not in errors:
            continue
        
        print(f"\n{model_type.upper()} - Wilcoxon tests per horizon:")
        horizon_results = []
        
        for horizon in horizons:
            # Sample errors at this horizon
            horizon_errors = errors[model_type][::horizon]
            baseline_errors = errors[model_type]  # Compare against full dataset
            
            # Wilcoxon test: horizon errors vs baseline
            if len(horizon_errors) > 10:  # Need sufficient samples
                # Check if samples are identical (all differences are zero)
                baseline_subset = baseline_errors[:len(horizon_errors)]
                differences = np.array(horizon_errors) - np.array(baseline_subset)
                
                if np.allclose(differences, 0.0):
                    # Samples are identical, no statistical test needed
                    p_val = 1.0  # Perfect match, p-value = 1.0 (no difference)
                    stat = 0.0
                    print(f"  Horizon {horizon}: p = {p_val:.6f} (samples identical)")
                else:
                    # Use 'zsplit' zero_method which handles zeros better
                    try:
                        stat, p_val = wilcoxon(horizon_errors, baseline_subset, alternative='two-sided', zero_method='zsplit')
                        horizon_results.append((horizon, p_val))
                        print(f"  Horizon {horizon}: p = {p_val:.6f}")
                    except ValueError as e:
                        # If still fails, skip this horizon
                        print(f"  Horizon {horizon}: Cannot compute (all differences zero or insufficient variation)")
                        continue
        
        if horizon_results:
            p_vals = np.array([p for _, p in horizon_results])
            corrected_p, rejected = holm_bonferroni_correction(p_vals)
            
            horizon_p_values[model_type] = {
                'horizons': [h for h, _ in horizon_results],
                'p_values': p_vals,
                'corrected_p': corrected_p,
                'rejected': rejected
            }
            
            print(f"  Holm-Bonferroni corrected:")
            for i, (horizon, orig_p, corr_p, rej) in enumerate(zip(
                [h for h, _ in horizon_results],
                p_vals,
                corrected_p,
                rejected
            )):
                print(f"    Horizon {horizon}: p = {corr_p:.6f} {'*' if rej else ''}")
    
    # C. Bayesian calibration
    print("\n" + "=" * 80)
    print("C. BAYESIAN CALIBRATION")
    print("=" * 80)
    
    if 'bayesian' in predictions and 'bayesian' in uncertainties:
        bay_pred = predictions['bayesian']
        bay_std = uncertainties['bayesian']
        
        # Empirical coverage
        coverage_1sigma = empirical_coverage(bay_pred, bay_std, y_test, sigma=1.0)
        coverage_2sigma = empirical_coverage(bay_pred, bay_std, y_test, sigma=2.0)
        
        print(f"\nEmpirical Coverage:")
        print(f"  1σ coverage: {coverage_1sigma:.2f}% (expected: 68.27%)")
        print(f"  2σ coverage: {coverage_2sigma:.2f}% (expected: 95.45%)")
        
        # NLL comparison
        bay_nll = negative_log_likelihood(bay_pred, bay_std, y_test)
        
        # Compare with deterministic models (use constant uncertainty estimate)
        det_nlls = {}
        for model_type in ['transformer', 'lstm']:
            if model_type in predictions:
                # Estimate uncertainty as std of errors
                model_errors = errors[model_type]
                estimated_std = np.std(model_errors)
                # Create constant std array
                const_std = np.ones_like(predictions[model_type]) * estimated_std
                det_nlls[model_type] = negative_log_likelihood(
                    predictions[model_type], const_std, y_test
                )
        
        print(f"\nNegative Log-Likelihood (NLL):")
        print(f"  Bayesian: {bay_nll:.6f}")
        for model_type, nll in det_nlls.items():
            print(f"  {model_type.upper()}: {nll:.6f}")
            improvement = ((nll - bay_nll) / nll) * 100
            print(f"    Improvement: {improvement:+.2f}%")
    
    # Output summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    if results_table:
        df = pd.DataFrame(results_table)
        print("\nAccuracy Tests:")
        print(df.to_string(index=False))
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    if results_table:
        df.to_csv(output_dir / 'statistical_tests_accuracy.csv', index=False)
        print(f"\nResults saved to {output_dir / 'statistical_tests_accuracy.csv'}")
    
    print("\n" + "=" * 80)
    print("Statistical tests complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

