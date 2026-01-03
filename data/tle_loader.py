"""Utilities for loading and parsing Two-Line Element (TLE) data."""

import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sgp4.api import Satrec, jday, WGS72
from sgp4.conveniences import jday_datetime
try:
    # Try different import methods for twoline2rv
    try:
        from sgp4.io import twoline2rv
        HAS_TWOLINE2RV = True
    except ImportError:
        # Some versions have it in api module
        from sgp4.api import twoline2rv
        HAS_TWOLINE2RV = True
except ImportError:
    # Fallback if twoline2rv is not available
    HAS_TWOLINE2RV = False


class TLELoader:
    """Load and parse TLE data from various sources."""
    
    def __init__(self):
        self.catalog_url = "https://celestrak.org/NORAD/elements/gp.php"
    
    def fetch_tle_from_celestrak(
        self, 
        catalog_number: Optional[int] = None,
        group: Optional[str] = None
    ) -> List[str]:
        """
        Fetch TLE data from Celestrak.
        
        Args:
            catalog_number: NORAD catalog number (e.g., 25544 for ISS)
            group: Group name (e.g., 'stations', 'weather', 'noaa')
        
        Returns:
            List of TLE lines (3 lines per satellite: name, line1, line2)
        """
        params = {}
        if catalog_number:
            params['CATNR'] = catalog_number
        elif group:
            params['GROUP'] = group
        
        try:
            response = requests.get(self.catalog_url, params=params, timeout=10)
            response.raise_for_status()
            lines = response.text.strip().split('\n')
            return lines
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch TLE data: {e}")
    
    def parse_tle(self, tle_lines: List[str]) -> Dict:
        """
        Parse TLE lines into structured data.
        
        Args:
            tle_lines: List of 3 lines (name, line1, line2)
        
        Returns:
            Dictionary with parsed TLE data
        """
        if len(tle_lines) < 3:
            raise ValueError("TLE must have at least 3 lines")
        
        name = tle_lines[0].strip()
        line1 = tle_lines[1].strip()
        line2 = tle_lines[2].strip()
        
        # Parse line 1
        catalog_number = int(line1[2:7])
        classification = line1[7]
        int_designator = line1[9:17]
        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32].strip())
        mean_motion_dot = float(line1[33:43].strip())
        mean_motion_ddot_str = line1[44:50].strip().replace(' ', '')
        mean_motion_ddot_exp = line1[50:52].strip().replace(' ', '')
        if mean_motion_ddot_str and mean_motion_ddot_exp:
            mean_motion_ddot = float('0.' + mean_motion_ddot_str) * 10 ** float(mean_motion_ddot_exp)
        else:
            mean_motion_ddot = 0.0
        bstar_str = line1[53:59].strip().replace(' ', '')
        bstar_exp = line1[59:61].strip().replace(' ', '')
        if bstar_str and bstar_exp:
            bstar = float('0.' + bstar_str) * 10 ** float(bstar_exp)
        else:
            bstar = 0.0
        element_set_number = int(line1[64:68])
        
        # Parse line 2
        inclination = float(line2[8:16].strip())
        raan = float(line2[17:25].strip())  # Right Ascension of Ascending Node
        eccentricity_str = line2[26:33].strip().replace(' ', '')
        if eccentricity_str:
            eccentricity = float('0.' + eccentricity_str)
        else:
            eccentricity = 0.0
        argument_of_perigee = float(line2[34:42].strip())
        mean_anomaly = float(line2[43:51].strip())
        mean_motion = float(line2[52:63].strip())
        revolution_number = int(line2[63:68])
        
        # Calculate epoch datetime
        if epoch_year < 57:
            epoch_year += 2000
        else:
            epoch_year += 1900
        
        epoch_datetime = datetime(epoch_year, 1, 1) + timedelta(days=epoch_day - 1)
        
        return {
            'name': name,
            'catalog_number': catalog_number,
            'classification': classification,
            'international_designator': int_designator,
            'epoch': epoch_datetime,
            'epoch_year': epoch_year,
            'epoch_day': epoch_day,
            'inclination': inclination,
            'raan': raan,
            'eccentricity': eccentricity,
            'argument_of_perigee': argument_of_perigee,
            'mean_anomaly': mean_anomaly,
            'mean_motion': mean_motion,
            'mean_motion_dot': mean_motion_dot,
            'mean_motion_ddot': mean_motion_ddot,
            'bstar': bstar,
            'element_set_number': element_set_number,
            'revolution_number': revolution_number,
            'line1': line1,
            'line2': line2
        }
    
    def create_satrec(self, tle_data: Dict, tle_lines: Optional[List[str]] = None) -> Satrec:
        """
        Create a Satrec object from parsed TLE data.
        
        Args:
            tle_data: Parsed TLE dictionary
            tle_lines: Optional original TLE lines (name, line1, line2)
                      If provided, uses built-in parser for better accuracy
        
        Returns:
            Satrec object for SGP4 propagation
        """
        # Use built-in TLE parser if lines are available
        if HAS_TWOLINE2RV and tle_lines is not None and len(tle_lines) >= 3:
            try:
                sat = Satrec.twoline2rv(tle_lines[1], tle_lines[2], WGS72)
                return sat
            except (AttributeError, TypeError):
                # Fallback to function call
                result = twoline2rv(tle_lines[1], tle_lines[2], WGS72)
                if isinstance(result, Satrec):
                    return result
                elif isinstance(result, tuple) and len(result) == 2:
                    sat, error = result
                    if error == 0:
                        return sat
            except Exception:
                pass  # Fall through to manual initialization
        
        # Fallback to manual initialization
        # sgp4init requires positional arguments, not keyword arguments
        # Order: whichconst, opsmode, satnum, epoch, xbstar, xndot, xnddot,
        #        xecco, xargpo, xinclo, xmo, xno_units, xnodeo
        sat = Satrec()
        
        # Calculate epoch in the format expected by sgp4init
        # Format: YYDDD.dddddddd where YY is 2-digit year, DDD is day of year
        epoch_year_2digit = tle_data['epoch_year'] % 100
        epoch_julian = epoch_year_2digit * 1000 + tle_data['epoch_day']
        
        try:
            sat.sgp4init(
                0,               # whichconst (0 = WGS72)
                'i',             # opsmode ('a' = AFSPC, 'i' = improved)
                tle_data['catalog_number'],
                epoch_julian,
                tle_data['bstar'],
                tle_data['mean_motion_dot'],
                tle_data['mean_motion_ddot'],
                tle_data['eccentricity'],
                tle_data['argument_of_perigee'],
                tle_data['inclination'],
                tle_data['mean_anomaly'],
                tle_data['mean_motion'],
                tle_data['raan']
            )
            
            return sat
        except Exception as e:
            raise ValueError(f"Failed to initialize satellite: {e}. Check TLE data validity.")
    
    def load_multiple_tles(self, tle_text: str) -> List[Dict]:
        """
        Load multiple TLEs from a text block.
        
        Args:
            tle_text: Multi-line string with TLE data
        
        Returns:
            List of parsed TLE dictionaries
        """
        lines = tle_text.strip().split('\n')
        tles = []
        
        i = 0
        while i < len(lines):
            if lines[i].strip() and not lines[i].startswith('1 ') and not lines[i].startswith('2 '):
                # This is a name line
                if i + 2 < len(lines):
                    tle_lines = [lines[i], lines[i+1], lines[i+2]]
                    try:
                        tle_data = self.parse_tle(tle_lines)
                        tles.append(tle_data)
                    except Exception as e:
                        print(f"Error parsing TLE at line {i}: {e}")
                    i += 3
                else:
                    break
            else:
                i += 1
        
        return tles
    
    def load_satellite(self, catalog_number: int = 25544) -> Tuple[Satrec, Dict]:
        """
        Helper function to load satellite TLE and create Satrec object.
        This is the recommended way to load satellite data.
        
        Args:
            catalog_number: NORAD catalog number (default: 25544 for ISS)
        
        Returns:
            Tuple of (Satrec object, TLE data dictionary)
        
        Raises:
            Exception: If TLE loading or parsing fails
        """
        tle_lines = self.fetch_tle_from_celestrak(catalog_number=catalog_number)
        tle_data = self.parse_tle(tle_lines[:3])
        sat = self.create_satrec(tle_data, tle_lines[:3])
        return sat, tle_data

