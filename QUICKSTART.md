# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run Demo

```bash
# Run the main demonstration
python main.py
```

This will:
1. Load TLE data for the International Space Station (ISS)
2. Generate simulated telemetry
3. Run physics-based predictions
4. Set up ML models
5. Demonstrate anomaly detection

## Train a Model

```bash
# Train LSTM model
python experiments/train_model.py --model lstm --epochs 50 --satellite_id 25544

# Train Transformer model
python experiments/train_model.py --model transformer --epochs 50

# Train Bayesian model
python experiments/train_model.py --model bayesian --epochs 50
```

## Use Different Satellites

Find satellite NORAD IDs from [Celestrak](https://celestrak.org/):
- ISS: 25544
- Hubble: 20580
- GPS satellites: Various (use group='gps-ops')

```bash
python experiments/train_model.py --satellite_id 20580  # Hubble
```

## Next Steps

1. **Customize Models**: Edit model architectures in `ml_models/`
2. **Add Features**: Extend `data/telemetry_processor.py` with additional features
3. **Real Data**: Integrate real-time telemetry streams
4. **Visualization**: Add plotting utilities for trajectories and anomalies

## Troubleshooting

**Import errors**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**TLE fetch fails**: Check internet connection. TLE data is fetched from Celestrak.

**CUDA errors**: Models default to CPU. For GPU:
```bash
python experiments/train_model.py --device cuda
```

