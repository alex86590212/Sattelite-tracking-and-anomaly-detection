"""Utilities for loading and parsing Two-Line Element (TLE) data."""

import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sgp4.api import Satrec, jday
from sgp4.conveniences import jday_datetime


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
        epoch_day = float(line1[20:32])
        mean_motion_dot = float(line1[33:43])
        mean_motion_ddot = float('0.' + line1[44:50]) * 10 ** float(line1[50:52])
        bstar = float('0.' + line1[53:59]) * 10 ** float(line1[59:61])
        element_set_number = int(line1[64:68])
        
        # Parse line 2
        inclination = float(line2[8:16])
        raan = float(line2[17:25])  # Right Ascension of Ascending Node
        eccentricity = float('0.' + line2[26:33])
        argument_of_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])
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
    
    def create_satrec(self, tle_data: Dict) -> Satrec:
        """
        Create a Satrec object from parsed TLE data.
        
        Args:
            tle_data: Parsed TLE dictionary
        
        Returns:
            Satrec object for SGP4 propagation
        """
        sat = Satrec()
        sat.sgp4init(
            whichconst=0,  # WGS72
            opsmode='i',    # 'a' = AFSPC, 'i' = improved
            satnum=tle_data['catalog_number'],
            epoch=(tle_data['epoch_year'] % 100) * 1000 + tle_data['epoch_day'],
            xbstar=tle_data['bstar'],
            xndot=tle_data['mean_motion_dot'],
            xnddot=tle_data['mean_motion_ddot'],
            xecco=tle_data['eccentricity'],
            xargpo=tle_data['argument_of_perigee'],
            xinclo=tle_data['inclination'],
            xmo=tle_data['mean_anomaly'],
            xno_units=tle_data['mean_motion'],
            xnodeo=tle_data['raan']
        )
        return sat
    
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

