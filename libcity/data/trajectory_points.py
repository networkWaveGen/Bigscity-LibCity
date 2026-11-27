"""Validate static POI point geometry without evaluating coordinate strings."""
import json

import numpy as np
import pandas as pd


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
