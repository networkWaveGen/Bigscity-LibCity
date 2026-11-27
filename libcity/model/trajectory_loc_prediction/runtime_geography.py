"""Spherical distances from centered longitude/latitude buffers, in kilometres."""
import torch


def spherical_distance(left,right,origin,*,radius):
    with torch.autocast(left.device.type,enabled=False):
        difference = torch.deg2rad(left-right)
        lat1 = torch.deg2rad(left[...,1]+origin[1])
        lat2 = torch.deg2rad(right[...,1]+origin[1])
        a = (difference[...,1]/2).sin().square()+lat1.cos()*lat2.cos()*(difference[...,0]/2).sin().square()
        a = a.clamp(0,1)
        return 2*radius*torch.atan2(a.sqrt(),(1-a).sqrt())
