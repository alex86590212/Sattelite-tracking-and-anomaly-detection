# Data Sources

This document explains where the satellite tracking system gets its data from.

## Primary Data Source: Celestrak

### TLE (Two-Line Element) Data

**Source**: [Celestrak](https://celestrak.org/) - Public satellite catalog maintained by Dr. T.S. Kelso

**URL**: `https://celestrak.org/NORAD/elements/gp.php`

**What it provides**:
- Real-time and historical orbital elements for thousands of satellites
- Updated regularly (multiple times per day for active satellites)
- Free and publicly available
- Standard format used by space agencies worldwide

**How we use it**:
```python
from data.tle_loader import TLELoader

loader = TLELoader()
# Fetch by NORAD catalog number
tle_lines = loader.fetch_tle_from_celestrak(catalog_number=25544)  # ISS

# Or fetch by group
tle_lines = loader.fetch_tle_from_celestrak(group='stations')  # Space stations
```

**Popular satellites**:
- **ISS (International Space Station)**: NORAD ID `25544`
- **Hubble Space Telescope**: NORAD ID `20580`
- **GPS satellites**: Use group `'gps-ops'`
- **Starlink satellites**: Use group `'starlink'`
- **Weather satellites**: Use group `'weather'`

**TLE Format**:
Each TLE contains 3 lines:
1. Satellite name
2. Line 1: Orbital parameters (inclination, eccentricity, mean motion, etc.)
3. Line 2: Additional parameters (RAAN, argument of perigee, mean anomaly, etc.)

## Telemetry Data Generation

### Current Implementation: Simulated Telemetry

**Source**: Generated from TLE data using SGP4 propagation

**How it works**:
1. Load TLE data from Celestrak
2. Use SGP4 propagator to compute position/velocity at each time step
3. Add realistic noise to simulate sensor measurements
4. Generate time-series telemetry (position, velocity, timestamps)

**Code**:
```python
from data.telemetry_processor import TelemetryProcessor
from datetime import datetime

processor = TelemetryProcessor(time_step_minutes=1.0)
telemetry = processor.generate_telemetry(
    sat=satellite_object,
    start_time=datetime.now(),
    duration_hours=24.0,
    add_noise=True,        # Add realistic sensor noise
    noise_std=0.001        # Noise standard deviation in km
)
```

**Output columns**:
- `timestamp`: Time of measurement
- `x, y, z`: Position in ECI coordinates (km)
- `vx, vy, vz`: Velocity in ECI coordinates (km/s)
- `position_error`: Estimated position uncertainty (km)

## Alternative Data Sources

### Real Telemetry Data

For production use, you can integrate real telemetry from:

1. **Space-Track.org** (requires free account)
   - More comprehensive than Celestrak
   - Historical data available
   - API access available

2. **NORAD (North American Aerospace Defense Command)**
   - Official source of TLE data
   - Space-Track is the public interface

3. **Satellite Operators**
   - Direct telemetry feeds from satellite operators
   - Higher precision than TLE data
   - May require agreements/access

4. **Ground Station Networks**
   - Real-time tracking data
   - Higher update rates than TLE
   - Examples: NORAD tracking network, commercial services

### Historical Data

For training on historical patterns:

1. **Space-Track Historical Archive**
   - Years of historical TLE data
   - Useful for training on long-term patterns

2. **NASA Horizons System**
   - Precise ephemeris data
   - For validation and comparison

3. **ESA Space Debris Office**
   - European satellite tracking data
   - Additional coverage

## Data Flow

```
┌─────────────────┐
│  Celestrak API  │  ← Real TLE data (free, public)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   TLE Loader    │  ← Parse TLE format
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SGP4 Propagator│  ← Physics-based orbit prediction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Telemetry Gen.   │  ← Add noise, create time-series
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Training    │  ← Train models on telemetry
└─────────────────┘
```

## Customizing Data Sources

### Using Different TLE Sources

Modify `data/tle_loader.py` to add new sources:

```python
def fetch_tle_from_spacetrack(self, catalog_number: int, api_key: str):
    """Fetch from Space-Track (requires account)."""
    # Implementation here
    pass
```

### Using Real Telemetry

Replace `generate_telemetry()` with real data loading:

```python
def load_real_telemetry(self, file_path: str) -> pd.DataFrame:
    """Load real telemetry from file/API."""
    # Load from CSV, database, API, etc.
    return telemetry_df
```

## Notes

- **TLE data is free and publicly available** - no API keys needed for Celestrak
- **TLE update frequency**: Active satellites updated multiple times daily
- **TLE accuracy**: Good for short-term predictions (hours to days)
- **For high precision**: Use real telemetry or precise ephemeris data
- **Simulated telemetry**: Good for development and testing; use real data for production

## Useful Links

- [Celestrak](https://celestrak.org/) - Primary TLE source
- [Space-Track](https://www.space-track.org/) - Comprehensive catalog (requires account)
- [NORAD](https://www.norad.mil/) - Official source
- [NASA Horizons](https://ssd.jpl.nasa.gov/horizons/) - Precise ephemeris
- [SGP4 Documentation](https://pypi.org/project/sgp4/) - Python SGP4 library

