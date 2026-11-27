"""Vectorized WGS84 inverse distances, with Karney fallback for difficult pairs.

Vincenty (1975), equations 3, 4, 6 and 13-19:
https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf
Only static data features use this CPU transform, never learnable coordinates.
"""
import numpy as np
from geopy.distance import geodesic


def distances_from(origin,points):
    """Return kilometres from one [longitude, latitude] point to N such points."""
    origin,points = np.asarray(origin,dtype=np.float64),np.asarray(points,dtype=np.float64)
    if (origin.shape!=(2,) or points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(origin).all()
            or not np.isfinite(points).all() or abs(origin[0])>180 or abs(origin[1])>90
            or (np.abs(points[:,0])>180).any() or (np.abs(points[:,1])>90).any()):
        raise ValueError('WGS84 distances require finite longitude/latitude point coordinates')
    major,flattening = 6378.137,1/298.257223563
    minor = major*(1-flattening)
    longitude = np.deg2rad((points[:,0]-origin[0]+180)%360-180)
    latitude = np.deg2rad(points[:,1])
    reduced_origin = np.arctan((1-flattening)*np.tan(np.deg2rad(origin[1])))
    reduced = np.arctan((1-flattening)*np.tan(latitude))
    sin1,cos1 = np.sin(reduced_origin),np.cos(reduced_origin)
    sin2,cos2 = np.sin(reduced),np.cos(reduced)
    lam = longitude.copy()
    coincident = (longitude==0)&(points[:,1]==origin[1])
    # Geodesic longitude is undefined at the poles; let Karney handle these pairs.
    fallback = (abs(origin[1])==90)|(np.abs(points[:,1])==90)
    converged = coincident.copy()
    for _ in range(100):
        sin_lam,cos_lam = np.sin(lam),np.cos(lam)
        sin_sigma = np.hypot(cos2*sin_lam,cos1*sin2-sin1*cos2*cos_lam)
        cos_sigma = sin1*sin2+cos1*cos2*cos_lam
        sigma = np.arctan2(sin_sigma,cos_sigma)
        sin_alpha = np.divide(cos1*cos2*sin_lam,sin_sigma,out=np.zeros_like(lam),where=sin_sigma>1e-30)
        cos_alpha2 = np.maximum(0.,1-sin_alpha**2)
        middle = cos_sigma-np.divide(2*sin1*sin2,cos_alpha2,out=np.zeros_like(lam),where=cos_alpha2>1e-30)
        coefficient = flattening/16*cos_alpha2*(4+flattening*(4-3*cos_alpha2))
        updated = longitude+(1-coefficient)*flattening*sin_alpha*(sigma+coefficient*sin_sigma*(
            middle+coefficient*cos_sigma*(-1+2*middle**2)))
        converged = (np.abs(updated-lam)<1e-13)|coincident
        if np.all(converged|fallback):
            break
        lam = updated
    factor = cos_alpha2*(major**2-minor**2)/minor**2
    a = 1+factor/16384*(4096+factor*(-768+factor*(320-175*factor)))
    b = factor/1024*(256+factor*(-128+factor*(74-47*factor)))
    correction = b*sin_sigma*(middle+b/4*(cos_sigma*(-1+2*middle**2)
        -b/6*middle*(-3+4*sin_sigma**2)*(-3+4*middle**2)))
    result = minor*a*(sigma-correction)
    result[coincident] = 0.
    fallback = fallback|~converged|~np.isfinite(result)
    for index in np.flatnonzero(fallback):
        result[index] = geodesic(origin[::-1],points[index,::-1]).km
    return result
