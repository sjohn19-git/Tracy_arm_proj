#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 07:42:39 2025

@author: sebinjohn
"""

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from obspy.core.event import Pick
from obspy.core.event import WaveformStreamID
from obspy.core.event import Pick
from obspy.core.event import ResourceIdentifier
from obspy.core.event import Arrival
from tqdm import tqdm
from obspy import read_events

client = Client("IRIS")
t1 = UTCDateTime("2025-07-25T00:00:00")
t2 = UTCDateTime("2025-08-10T14:00:00")
catalog = client.get_events(
    starttime=t1, endtime=t2, minmagnitude=1, minlatitude=57.39, maxlatitude=58.28,
    minlongitude=-134.80, maxlongitude=-132.77)

####
df = pd.read_csv("/Users/sebinjohn/Tracy_arm/data/TracyArm_Precursors.csv")

df["event_id"] = None

counts = df.groupby("origin.orid").size()

         
used_catalog_idx = set()  # to enforce one catalog event per ORID
# Loop over ORIDs
for orid, group in df.groupby("origin.orid"):
    csv_otime = UTCDateTime(float(group.iloc[0]["origin.time"]))
    csv_lat   = float(group.iloc[0]["origin.lat"])
    csv_lon   = float(group.iloc[0]["origin.lon"])
    csv_mag   = float(group.iloc[0]["netmag.magnitude"])

    
    
    # Compute ΔTime for all catalog events not yet used
    delta_time = np.array([
        abs(ev.origins[0].time - csv_otime) if i not in used_catalog_idx else np.inf
        for i, ev in enumerate(catalog)
    ])

    # Best matching catalog event
    min_idx = np.argmin(delta_time)
    if delta_time[min_idx] > 0.001:  # optional sanity check
        print(f"Problem ORID {orid}: ΔTime={delta_time[min_idx]:.6f}s")
        continue
    else:
        best_event = catalog[min_idx]
        event_id = best_event.resource_id.id
    
        # Assign event_id to all rows in df for this ORID
        df.loc[df["origin.orid"] == orid, "event_id"] = event_id
    
        # Mark this catalog event as used
        used_catalog_idx.add(min_idx)
        
df_unassigned = df[df["event_id"].isna()]

print(f"Number of unassigned rows: {len(df_unassigned)}")
print(df_unassigned.head())


############updated qml

# Create a mapping from event_id to ObsPy Event for faster lookup
event_map = {ev.resource_id.id: ev for ev in catalog}

# Loop over assigned rows in df
for idx, row in tqdm(df[df["event_id"].notna()].iterrows()):
    event_id = row["event_id"]
    if event_id not in event_map:
        continue  # safety check

    ev = event_map[event_id]
    origin = ev.origins[0]

    
    network = row["snetsta.snet"]
    station = row["arrival.sta"]
    phase = row["arrival.iphase"] if "arrival.iphase" in row else None
    
    chosen_channel = row.get("arrival.chan", None)
    chosen_location = row.get("arrival.loc", "")

    pick = Pick()
    pick.time = UTCDateTime(float(row["arrival.time"]))  
    pick.waveform_id = WaveformStreamID(
    network_code=network,
    station_code=station,
    channel_code=chosen_channel,
    location_code=chosen_location)
    
    pick.phase_hint = phase
    pick.resource_id = ResourceIdentifier()

    ev.picks.append(pick)

    arrival = Arrival()
    arrival.pick_id = pick.resource_id
    arrival.phase = pick.phase_hint
    origin.arrivals.append(arrival)


catalog.write("/Users/sebinjohn/Tracy_arm/data/catalog_with_picks.xml", format="QUAKEML")

#####SANITY CHECKS
local_catalog_file = "/Users/sebinjohn/Tracy_arm/data/catalog_with_picks.xml"
catalog = read_events(local_catalog_file)


events_no_picks = [ev for ev in catalog if not hasattr(ev, "picks") or len(ev.picks) == 0]

print(f"Number of events with no picks: {len(events_no_picks)}")

# Optionally, print their resource IDs
for ev in events_no_picks:
    print(ev.resource_id.id)

import random

# Select a random event from the catalog
random_event = random.choice(catalog)

ev_arrivals=random_event.origins[0].arrivals
ev_picks=random_event.picks

event_id = random_event.resource_id.id

# Find corresponding rows in the dataframe
matching_rows = df[df["event_id"] == event_id]
    
ortim_df=UTCDateTime(float(matching_rows.iloc[0]["origin.time"]))
ortim_ev=random_event.origins[0].time

print(f"Randomly selected event: {event_id}")
print(f"Number of matching rows in df: {len(matching_rows)}")


# Print number of arrivals and picks
print(f"Number of arrivals in origin: {len(ev_arrivals)}")
print(f"Number of picks in event: {len(ev_picks)}")

# Print origin times
print(f"Origin time (CSV): {ortim_df}")
print(f"Origin time (Catalog): {ortim_ev}")

print("Catalog Picks vs Corresponding CSV Times:")

for i, pick in enumerate(ev_picks, 1):
    # Find the matching row(s) in the dataframe
    match_df_rows = matching_rows[
        (matching_rows["arrival.sta"] == pick.waveform_id.station_code) &
        (matching_rows["arrival.iphase"] == pick.phase_hint)
    ]
    
    csv_times = [UTCDateTime(float(t)) for t in match_df_rows["arrival.time"]]
    
    print(f"{i}. Station: {pick.waveform_id.station_code}, Phase: {pick.phase_hint}")
    print(f"   Catalog pick time: {pick.time}")
    print(f"   CSV arrival times: {csv_times}")
    print("-----")



from collections import Counter

nslc_counter = Counter()

for ev in catalog:
    for pick in ev.picks:
        wsid = pick.waveform_id
        n = wsid.network_code or ""
        s = wsid.station_code or ""
        l = wsid.location_code or ""
        c = wsid.channel_code or ""
        nslc_counter[(n, s, l, c)] += 1

# Sort by NSLC
sorted_counts = sorted(nslc_counter.items())

print(f"Number of unique NSLCs: {len(sorted_counts)}")
for (n, s, l, c), count in sorted_counts:
    print(f"Network: {n}, Station: {s}, Location: {l}, Channel: {c} -> {count} picks")
    
# Find the event with the maximum magnitude
max_event = max(catalog, key=lambda ev: ev.magnitudes[0].mag if ev.magnitudes else -999)

# Extract magnitude
max_mag = max_event.magnitudes[0].mag if max_event.magnitudes else None

# Extract origin info
origin = max_event.preferred_origin() or max_event.origins[0]
lat, lon, depth_km = origin.latitude, origin.longitude, origin.depth / 1000.0  # depth in km

print(f"Largest event magnitude: {max_mag}")
print(f"Location: Latitude={lat:.3f}, Longitude={lon:.3f}, Depth={depth_km:.2f} km")
print(f"Time: {origin.time}")