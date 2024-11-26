from glob import glob
import numpy as np

def load_data(type="asip"):
    if type=="asip":
        path = glob('/dmidata/users/maxb/ASIP_OSISAF_dataset/ASIP_L3/*nc')
    else:
        path = glob('/dmidata/users/maxb/ASIP_OSISAF_dataset/OSISAF_NRT/*/*/*nh*amsr2_????????1200.nc')
    return path
