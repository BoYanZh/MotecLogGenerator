"""High-precision WGS84 Geodesic Ellipsoid Coordinate Utilities."""

import numpy as np

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0                # Semi-major axis in meters
WGS84_E2 = 0.00669437999014        # First eccentricity squared


def get_wgs84_geodesic_factors(latitude_deg: float):
    """
    Computes exact WGS84 Geodesic meters-per-degree for Latitude and Longitude
    at a given reference latitude (phi0) using WGS84 ellipsoid radii of curvature:
      - M(phi): Meridian radius of curvature (Latitude)
      - N(phi): Prime vertical radius of curvature (Longitude)
    """
    phi0 = np.radians(latitude_deg)
    sin_phi = np.sin(phi0)
    denom = np.sqrt(1.0 - WGS84_E2 * sin_phi**2)

    n_phi = WGS84_A / denom
    m_phi = WGS84_A * (1.0 - WGS84_E2) / (denom**3)

    m_per_deg_lat = m_phi * (np.pi / 180.0)
    m_per_deg_lon = n_phi * np.cos(phi0) * (np.pi / 180.0)

    return m_per_deg_lat, m_per_deg_lon


def enu_to_wgs84(easting_m, northing_m, lat0_deg: float, lon0_deg: float):
    """ Converts ENU (East-North-Up) offsets in meters to WGS84 Latitude & Longitude degrees. """
    m_per_deg_lat, m_per_deg_lon = get_wgs84_geodesic_factors(lat0_deg)

    lat = lat0_deg + (northing_m / m_per_deg_lat)
    lon = lon0_deg + (easting_m / m_per_deg_lon)

    return lat, lon


def wgs84_to_enu(lat_deg, lon_deg, lat0_deg: float, lon0_deg: float):
    """ Converts WGS84 Latitude & Longitude degrees to ENU (East-North-Up) offsets in meters. """
    m_per_deg_lat, m_per_deg_lon = get_wgs84_geodesic_factors(lat0_deg)

    easting = (lon_deg - lon0_deg) * m_per_deg_lon
    northing = (lat_deg - lat0_deg) * m_per_deg_lat

    return easting, northing
