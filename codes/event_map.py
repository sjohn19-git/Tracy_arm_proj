#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 19:08:43 2025

@author: sebinjohn
"""

import pygmt
import pandas as pd
from obspy import read_events
from eqcorrscan.utils.catalog_utils import filter_picks

# --- Load catalog ---
local_catalog_file = "/Users/sebinjohn/Tracy_arm/data/catalog_with_picks.xml"
catalog1 = read_events(local_catalog_file)
catalog2 = filter_picks(catalog=catalog1, top_n_picks=10)


# --- Extract event information ---
data = []
for event in catalog2:
    ori = event.preferred_origin() or event.origins[0]
    lat = ori.latitude
    lon = ori.longitude
    depth = ori.depth / 1000 if ori.depth else 0  # convert to km
    mag = event.preferred_magnitude().mag if event.preferred_magnitude() else None
    data.append([lat, lon, depth, mag])

df = pd.DataFrame(data, columns=["Latitude", "Longitude", "Depth_km", "Magnitude"])

# --- Define map region (buffer around events) ---
region = [
    df["Longitude"].min() - 0.2,
    df["Longitude"].max() + 0.2,
    df["Latitude"].min() - 0.1,
    df["Latitude"].max() + 0.1,
]


# --- Load global DEM (SRTM15+, 15 arc-second) ---
grid = pygmt.datasets.load_earth_relief(resolution="01s", region=region)

# --- Create shaded relief effect ---
# 'intensity' controls shading (illumination from NW)
pygmt.grdgradient(grid=grid, azimuth=315, normalize="t1", outgrid="intensity.nc")


proj="M6i"
# --- Create figure ---
fig = pygmt.Figure()

with pygmt.config(MAP_FRAME_TYPE="plain", MAP_FRAME_PEN="2p,black"):
    fig.basemap(region=region, projection="M6i", frame=["af"])

# --- Plot shaded DEM: land gray, ocean blue ---
fig.grdimage(grid=grid, cmap="grayC", shading="intensity.nc")



cmap1 = pygmt.makecpt(
    cmap="viridis",
    series=[df["Magnitude"].min(), df["Magnitude"].max(),0.1],
    continuous=True
)
# --- Plot events (linked to colorbar through cmap) ---
fig.plot(
    x=df["Longitude"],
    y=df["Latitude"],
    style="c0.25c",
    fill=df["Magnitude"],
    cmap=True,      # uses the active CPT created above
    pen="black",
)

# --- Add colorbar that reflects magnitude ---
fig.colorbar(
    frame=['xaf+lMagnitude'],
    position="JMR+w5c/0.4c+o0.5c/0c"
)

# --- Add scale bar ---
fig.basemap(map_scale="jBL+w5+o0.5c/0.5c+f")

fig.show()

