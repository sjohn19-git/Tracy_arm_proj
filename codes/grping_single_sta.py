#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 15:05:47 2025

@author: sebinjohn
"""

from obspy import Stream
from eqcorrscan.core.match_filter import Party
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm
from obspy import read
from obspy import UTCDateTime
from collections import Counter
from obsplus import WaveBank
from pathlib import Path
from obspy import UTCDateTime
from eqcorrscan.utils.clustering import cluster
from obspy import read_events
from eqcorrscan.utils.catalog_utils import filter_picks
from obspy.clients.fdsn import Client
import pandas as pd
import numpy as np

client = Client("IRIS")

local_catalog_file = "/Users/sebinjohn/Tracy_arm/data/catalog_with_picks.xml"
catalog = read_events(local_catalog_file)
templates = filter_picks(catalog=catalog, top_n_picks=10)



station = "S32K"

time_diffs = []   # store in seconds
event_times = []  # origin times for plotting on x-axis

for ev in catalog:
    origin = ev.origins[0]
    ot = origin.time

    # Get P arrival pick for station S32K
    p_pick = None
    for pk in ev.picks:
        if pk.phase_hint == "P" and pk.waveform_id.station_code == station:
            p_pick = pk.time
            break

    if p_pick is not None:
        diff = p_pick - ot     # seconds
        time_diffs.append(diff)
        event_times.append(ot.datetime)

# ---- Plot ----
plt.figure(figsize=(10,5))
plt.scatter(event_times, time_diffs, alpha=0.7, edgecolor='k')

plt.axhline(0, color='r', linestyle='--', label="Origin Time")

plt.xlabel("Event Origin Time", fontsize=12)
plt.ylabel("P_pick – Origin Time (s)", fontsize=12)
plt.title(f"Time Difference Between S32K P-pick and Origin Time", fontsize=14)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.legend()
plt.show()


outdir = Path("/Users/sebinjohn/Tracy_arm/data/seismic")
bank = WaveBank(outdir) 
bank.update_index()
avail=bank.get_availability_df() 

party = Party()
party.read('/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.tgz', read_detection_catalog=False)
print(party)
len(party)

party.decluster(0.25)
print(len(party))

all_st = read("/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.mseed")


def get_station_coords(station):
    """
    Fetch station coordinates and network code from IRIS.

    Searches multiple networks in order: AK, AT, AV, CN
    """
    networks = ["AK", "AT", "AV", "CN"]
    for net in networks:
        try:
            inv = client.get_stations(network=net, station=station, level="station")
            sta = inv[0][0]
            return sta.latitude, sta.longitude, net  # lat, lon, station_code, network
        except Exception:
            continue
    return None, None, None


stations = ["R32K", "SIT", "U33K", "EDCR","S31K","PLBC","S32K"]  # put all your stations here
data = []

for sta in stations:
    lat, lon, net = get_station_coords(sta)
    data.append({"station": sta, "lat": lat, "lon": lon, "network": net})

# Create DataFrame
station_df = pd.DataFrame(data)
station_df.set_index("station", inplace=True)


preferred_station = "S32K"
pick_times = []
sta_sts=[]
c=0
pred=False
for fam in party:
    c+=1
    print(f"{c}/101")
    for det in fam:
        det_ev = det.event
        p_time =det_ev.origins[0].time+np.mean(time_diffs)
        stt=p_time-5.5
        ett=p_time+9.5
        sti=bank.get_waveforms(network="*",station="S32K",channel="BHZ",starttime=stt,endtime=ett)
        sti.filter("bandpass", freqmin=2.0, freqmax=8.0, corners=4, zerophase=True)
        for trii in sti:
            trii.trim(starttime=trii.stats.starttime + 5, endtime=trii.stats.endtime - 5)
        sta_sts.append((sti,det.id))
        

output_file="/Users/sebinjohn/Tracy_arm/data/all_streams/streams_eqcorr_S32K.pkl"

#Save template_list to a file
with open(output_file, "wb") as f:
    pickle.dump(sta_sts, f)

with open(output_file, "rb") as f:
    sta_sts = pickle.load(f)

stream_list=[(st, det_id) for (st, det_id) in sta_sts
            if len(st[0].data) == 251]

# Extract all detection IDs from the template_list
all_ids_temp = [tid for _, tid in stream_list]

# Count occurrences
counts = Counter(all_ids_temp)

# Find duplicates
duplicates_temp = {tid: count for tid, count in counts.items() if count > 1}
print("there are ",len(duplicates_temp),"detection duplicates in all families")
num_unique_tids = len(set(all_ids_temp))
print(f"Number of unique detection IDs: {num_unique_tids}")


all_ids = [tid for _, tid in stream_list]

import random
import matplotlib.pyplot as plt

# --- Select up to 100 random streams ---
N = 100
sel = random.sample(stream_list, min(N, len(stream_list)))

plt.figure(figsize=(10, 14))

offset = 0
spacing = 0.8  # vertical separation

for (st, detid) in sel:
    # Each st is a 1-trace Stream
    tr = st[0]

    # Normalize amplitude for clearer overlay
    data = tr.data / max(abs(tr.data))

    # Shift vertically
    plt.plot(tr.times(), data + offset, lw=0.7, color="black")

    #plt.text(tr.times()[0]-0.1, offset, str(detid), fontsize=7)
    offset += spacing

plt.title("100 Random S32K BHZ Streams (Filtered 2–8 Hz)")
plt.xlabel("Time (s) relative to trimmed window")
plt.ylabel("Stream Index")
plt.tight_layout()
plt.show()





thresh=0.75

start = time.time()

groups = cluster(
    template_list=stream_list,
    show=False,
    corr_thresh=thresh,
    cores=4,replace_nan_distances_with=1)


end = time.time()
elapsed = end - start
print(f"Time elapsed: {elapsed:.2f} seconds")




output_file = f"/Users/sebinjohn/Tracy_arm/data/groups/grps_eqcor_S32K_{thresh}.pkl" 

# with open(output_file, "wb") as f:    
#     pickle.dump(groups, f)

with open(output_file, "rb") as f:
    groups = pickle.load(f)

min_size=20
groupsf = [cl for cl in groups if len(cl) >= min_size]
c=0  
for ele in groupsf:
    c+=len(ele)
    
