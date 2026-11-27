"""Validate static POI point geometry without evaluating coordinate strings."""
import json

import numpy as np
import pandas as pd


def spherical_distance(left,right):
    """Broadcast WGS84 point coordinates on the original 6371 km sphere."""
    left,right = np.deg2rad(left),np.deg2rad(right)
    difference = left-right
    a = np.sin(difference[...,1]/2)**2+np.cos(left[...,1])*np.cos(right[...,1])*np.sin(difference[...,0]/2)**2
    a = np.clip(a,0,1)
    return 2*6371*np.arctan2(np.sqrt(a),np.sqrt(1-a))


def read_points(folder,metadata,dataset,num_nodes,model_name):
    geo = pd.read_csv(folder/(metadata.get('geo_file',dataset)+'.geo'))
    try:
        coordinates = np.asarray([json.loads(value) for value in geo['coordinates']],dtype=np.float64)
    except (ValueError,TypeError,KeyError) as error:
        raise ValueError(f'{model_name} requires JSON point coordinates in .geo') from error
    if (coordinates.shape!=(num_nodes,2) or not np.isfinite(coordinates).all()
            or (np.abs(coordinates[:,0])>180).any() or (np.abs(coordinates[:,1])>90).any()):
        raise ValueError(f'{model_name} requires finite longitude/latitude point coordinates')
    return geo,coordinates
