# Satellite Tracking and Anomaly Detection

A machine learning system for predicting satellite positions and detecting anomalies in orbit behavior using real or simulated telemetry data.

## Project Goals

- **Orbit Prediction**: Use ML to predict future satellite positions with uncertainty estimates
- **Anomaly Detection**: Detect thruster faults, drag spikes, sensor glitches, and other anomalies
- **Hybrid Approach**: Combine physics-based models (SGP4) with ML corrections for best accuracy

## Features

### Inputs
- **Orbital Elements**: TLE (Two-Line Element) data from public catalogs
- **Time-Series Telemetry**: Position, velocity, and error residuals
- **Environmental Disturbances**: Drag, solar activity (optional)

### AI Tasks
- **Orbit Prediction**: Forecast future satellite positions
- **Anomaly Detection**: Identify unusual orbit behavior
- **Uncertainty Estimation**: Provide confidence intervals for predictions

### Outputs
- **Future Satellite Position**: Predicted positions with timestamps
- **Confidence Intervals**: Uncertainty bounds for predictions
- **Anomaly Alerts**: Flags for thruster faults, drag spikes, sensor glitches

## Project Structure

```
├── data/                      # Data loading and preprocessing
│   ├── tle_loader.py          # TLE data fetching and parsing
│   └── telemetry_processor.py # Telemetry generation and processing
│
├── physics_model/             # Physics-based baseline
│   └── sgp4_propagator.py    # SGP4 orbital propagation
│
├── ml_models/                 # Machine learning models
│   ├── lstm_model.py         # LSTM/GRU for orbit prediction
│   ├── transformer_model.py  # Transformer for long-horizon prediction
│   └── bayesian_model.py     # Bayesian NN for uncertainty
│
├── anomaly_detection/         # Anomaly detection methods
│   ├── autoencoder.py        # Autoencoder-based detection
│   ├── prediction_error.py   # Error threshold detection
│   └── bayesian_surprise.py  # Bayesian surprise scores
│
├── uncertainty/               # Uncertainty quantification
│   └── estimation.py         # Uncertainty estimation methods
│
├── experiments/               # Training and evaluation
│   ├── train_model.py        # Model training script
│   └── evaluate.py           # Evaluation metrics
│
└── requirements.txt          # Python dependencies
```

## Quick Start

### Installation

1. Clone the repository:
```bash
cd /Users/baris/projects/Sattelite-tracking-and-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### 1. Load TLE Data

```python
from data.tle_loader import TLELoader

loader = TLELoader()
tle_lines = loader.fetch_tle_from_celestrak(catalog_number=25544)  # ISS
tle_data = loader.parse_tle(tle_lines[:3])
sat = loader.create_satrec(tle_data)
```

#### 2. Generate Telemetry

```python
from data.telemetry_processor import TelemetryProcessor
from datetime import datetime

processor = TelemetryProcessor(time_step_minutes=1.0)
telemetry = processor.generate_telemetry(
    sat, datetime.now(), duration_hours=24.0, add_noise=True
)
```

#### 3. Physics-Based Prediction

```python
from physics_model.sgp4_propagator import SGP4Propagator

propagator = SGP4Propagator(sat)
trajectory = propagator.propagate_trajectory(
    start_time=datetime.now(),
    end_time=datetime.now() + timedelta(hours=1),
    time_step_minutes=1.0
)
```

#### 4. Train ML Model

```bash
python experiments/train_model.py --model lstm --epochs 50 --satellite_id 25544
```

#### 5. Anomaly Detection

```python
from anomaly_detection.prediction_error import PredictionErrorDetector

detector = PredictionErrorDetector(threshold_percentile=95.0)
detector.fit(training_errors)
anomalies, scores = detector.detect(observed, predicted)
```

## Technical Details

### Physics Model (Baseline)

The project uses **SGP4** (Simplified General Perturbations 4) as the physics-based baseline:
- Handles atmospheric drag (via B* parameter)
- Includes J2 oblateness effects
- Provides fast, accurate short-term predictions
- ML models learn to correct residuals from this baseline

### ML Models

#### Option A: LSTM/GRU
- **Use Case**: Predict orbit residuals over time
- **Strengths**: Learns non-linear drift, handles sequential data well
- **Best For**: Short to medium-term predictions

#### Option B: Transformer
- **Use Case**: Long-horizon orbit prediction
- **Strengths**: Handles missing telemetry, attention mechanism
- **Best For**: Long-term forecasting, irregular sampling

#### Option C: Bayesian Neural Network
- **Use Case**: Uncertainty-aware prediction
- **Strengths**: Provides uncertainty estimates, not just predictions
- **Best For**: Mission-critical applications requiring confidence bounds

### Anomaly Detection Methods

1. **Autoencoders**: Learn normal patterns, flag deviations
2. **Prediction Error Thresholds**: Flag when ML predictions exceed thresholds
3. **Bayesian Surprise**: Measure how "surprising" observations are

### Evaluation Metrics

- **Position Error (km)**: Mean position prediction error
- **Prediction Horizon Stability**: Error over different time horizons
- **False Alarm Rate**: Rate of false anomaly detections
- **Detection Latency**: Time to detect anomalies after they occur

## Example Workflow

```python
# 1. Load satellite data
loader = TLELoader()
tle_data = loader.parse_tle(tle_lines)
sat = loader.create_satrec(tle_data)

# 2. Generate training data
processor = TelemetryProcessor()
telemetry = processor.generate_telemetry(sat, start_time, duration_hours=24.0)

# 3. Prepare sequences
X, y = processor.prepare_sequences(telemetry, sequence_length=60, prediction_horizon=1)

# 4. Train model
model = OrbitLSTM(input_size=6, hidden_size=128, num_layers=2)
# ... training code ...

# 5. Predict with uncertainty
mean, std, kl = model.forward(X_test, num_samples=10)

# 6. Detect anomalies
detector = PredictionErrorDetector()
anomalies, scores = detector.detect(observed, predicted)
```

## Data Sources

- **TLE Data**: [Celestrak](https://celestrak.org/) - Public satellite catalog
- **Historical Tracks**: Can be extended with historical orbit data
- **Simulated Data**: Generate realistic telemetry with configurable noise

## Research Applications

This hybrid physics-ML approach is valuable for:
- **Space Situational Awareness**: Track satellites and debris
- **Mission Planning**: Predict satellite positions for operations
- **Anomaly Response**: Early detection of satellite issues
- **Collision Avoidance**: Predict close approaches

## Notes

- The physics model (SGP4) provides a strong baseline
- ML models learn to correct residuals, not replace physics
- This hybrid approach is preferred in aerospace applications
- Uncertainty quantification is critical for mission-critical use

## Contributing

Contributions welcome! Areas for improvement:
- Additional perturbation models
- More sophisticated drag modeling
- Real-time telemetry integration
- Visualization tools
- Extended evaluation metrics

## License

MIT License - see LICENSE file for details

## Acknowledgments

- SGP4 library: [pypi/sgp4](https://pypi.org/project/sgp4/)
- TLE data: [Celestrak](https://celestrak.org/)
- Inspired by hybrid physics-ML approaches in aerospace
