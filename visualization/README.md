# Visualization Module

This module provides plotting utilities for satellite tracking, predictions, and anomaly detection.

## Available Plotting Functions

### 1. Trajectory Visualization (`plot_trajectory.py`)

**`plot_3d_trajectory()`**
- **What**: 3D orbit visualization with Earth sphere
- **When**: After generating telemetry or predictions
- **Where**: Use in `main.py` or evaluation scripts
- **Example**:
```python
from visualization.plot_trajectory import plot_3d_trajectory
plot_3d_trajectory(telemetry, title="ISS Trajectory", save_path="orbit_3d.png")
```

**`plot_2d_projections()`**
- **What**: XY, XZ, YZ plane projections
- **When**: For detailed orbit analysis
- **Where**: Analysis notebooks or evaluation

**`plot_position_over_time()`**
- **What**: Position components (x, y, z) vs time
- **When**: To see orbital period and position evolution
- **Where**: Training analysis, debugging

**`plot_velocity_over_time()`**
- **What**: Velocity components and magnitude vs time
- **When**: To analyze orbital speed changes
- **Where**: Physics model validation

### 2. Prediction Visualization (`plot_predictions.py`)

**`plot_predictions_vs_observed()`**
- **What**: Compare predictions vs actual telemetry
- **When**: After model training/evaluation
- **Where**: In `experiments/evaluate.py` or notebooks
- **Example**:
```python
from visualization.plot_predictions import plot_predictions_vs_observed
plot_predictions_vs_observed(observed_df, predicted_df, 
                            title="LSTM Predictions vs Observed")
```

**`plot_residuals()`**
- **What**: Residual analysis (errors, statistics, histograms)
- **When**: To understand model errors
- **Where**: Model evaluation and debugging

**`plot_prediction_error_over_horizon()`**
- **What**: Error vs prediction horizon
- **When**: To assess prediction stability
- **Where**: Model comparison and evaluation

### 3. Anomaly Visualization (`plot_anomalies.py`)

**`plot_anomalies_over_time()`**
- **What**: Anomalies highlighted on trajectory
- **When**: After anomaly detection
- **Where**: In `main.py` demo or evaluation scripts
- **Example**:
```python
from visualization.plot_anomalies import plot_anomalies_over_time
plot_anomalies_over_time(telemetry, anomaly_flags, anomaly_scores,
                        title="Detected Anomalies")
```

**`plot_anomaly_types()`**
- **What**: Different anomaly types (thruster, drag, sensor)
- **When**: When using `ResidualAnomalyDetector`
- **Where**: Anomaly analysis

### 4. Training Visualization (`plot_training.py`)

**`plot_training_curves()`**
- **What**: Training/validation loss over epochs
- **When**: During/after model training
- **Where**: In `experiments/train_model.py`
- **Example**:
```python
from visualization.plot_training import plot_training_curves
plot_training_curves(train_losses, val_losses, 
                    save_path="training_curves.png")
```

**`plot_uncertainty_bounds()`**
- **What**: Predictions with confidence intervals
- **When**: Using Bayesian models
- **Where**: Uncertainty analysis, model evaluation

## Usage Examples

### In `main.py` (Demo)
```python
from visualization.plot_trajectory import plot_3d_trajectory
from visualization.plot_anomalies import plot_anomalies_over_time

# After generating telemetry
plot_3d_trajectory(telemetry, title="ISS Orbit", save_path="demo_orbit.png")

# After anomaly detection
plot_anomalies_over_time(telemetry_with_anomalies, detected_anomalies, 
                        anomaly_scores, save_path="demo_anomalies.png")
```

### In `experiments/evaluate.py`
```python
from visualization.plot_predictions import plot_predictions_vs_observed, plot_residuals

# After evaluation
plot_predictions_vs_observed(observed, predictions, 
                            save_path="evaluation_predictions.png")
plot_residuals(residuals, save_path="evaluation_residuals.png")
```

### In `experiments/train_model.py`
```python
from visualization.plot_training import plot_training_curves

# After training
plot_training_curves(train_losses, val_losses, 
                    save_path="training_curves.png")
```

## When to Plot

1. **During Development**: Use plots to debug data generation and preprocessing
2. **After Training**: Visualize model performance and errors
3. **During Evaluation**: Compare different models and methods
4. **For Reports**: Generate publication-quality figures
5. **In Notebooks**: Interactive exploration and analysis

## Output

- **Display**: Call functions without `save_path` to display plots
- **Save**: Provide `save_path` to save as PNG (300 DPI)
- **Format**: All plots saved as high-resolution PNG files

## Dependencies

- `matplotlib`: For all plotting
- `numpy`: For calculations
- `pandas`: For DataFrame handling

All dependencies are in `requirements.txt`.

