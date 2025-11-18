#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 20:05:45 2025

@author: sebinjohn
"""

import matplotlib.pyplot as plt
from obspy import read_events
from eqcorrscan.utils.catalog_utils import filter_picks
from obspy import Catalog
import pandas as pd
import numpy as np

# --- Load catalog ---
local_catalog_file = "/Users/sebinjohn/Tracy_arm/data/catalog_with_picks.xml"
catalog = read_events(local_catalog_file)

catalog = Catalog([ev for ev in catalog if len(ev.picks) > 0])

catalog.write("/Users/sebinjohn/Tracy_arm/data/reloc/evs.pha", format="HYPODDPHA")


stas=sorted({pick.waveform_id.station_code for ev in catalog for pick in ev.picks})

stat_loc=pd.read_csv("/Users/sebinjohn/Tracy_arm/data/reloc/Alaska_network_station_location.csv")
lon_sta=np.array([])
lat_sta=np.array([])
for ele in stas:
    lon_sta=np.append(lon_sta,(stat_loc[stat_loc["Station Code"]==ele]["Longitude"].iloc[0]))
    lat_sta=np.append(lat_sta,(stat_loc[stat_loc["Station Code"]==ele]["Latitude"].iloc[0]))

with open("/Users/sebinjohn/Tracy_arm/data/reloc/allsta.dat", "w") as file:
    for i in range(len(stas)):
        file.write(str(stas[i])+" "+str(lat_sta[i])+" "+str(lon_sta[i])+"\n")
 