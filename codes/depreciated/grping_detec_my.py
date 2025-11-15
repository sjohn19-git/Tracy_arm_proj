#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:48:55 2025

@author: sebinjohn
"""

from eqcorrscan.core.match_filter import Party
from obspy import read
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from obspy.signal.filter import envelope
from obspy import UTCDateTime
from obspy.core.event import Catalog, Event
from eqcorrscan.utils.clustering import cluster
import json
import pickle
import matplotlib.dates as mdates

party = Party()

# load the QuakeML detections
party.read('/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.tgz',read_detection_catalog=False)
print(party)

fig = party.plot(plot_grouped=True)

# Load the stream file you saved earlier
st = read("/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.mseed")
avg_peak_dict = {}



for i in tqdm(range(0,101)):
    fam=party[i]
    #print(fam)
    streams = fam.extract_streams(stream=st, length=15, prepick=5.5)
    peak_values = [] 
    for det in tqdm(fam):
        #print(det)
        ide=det.id
        detst=streams[ide]
        detst_filt = detst.copy()
        detst_filt.filter("bandpass", freqmin=2.0, freqmax=8.0, corners=4, zerophase=True)
        for tr in detst_filt:
            tr.trim(starttime=tr.stats.starttime + 5,endtime=tr.stats.endtime - 5)
        detst_env = detst_filt.copy()
        for tr in detst_filt:
            env = envelope(tr.data)
            peak_amp = np.max(env)
            peak_values.append(peak_amp)
        avg_peak = np.mean(peak_values) if peak_values else 0
        # Use family template name + detection id as key
        key = f"{ide}"
        avg_peak_dict[key] = avg_peak


# Save
with open("/Users/sebinjohn/Tracy_arm/data/party/non_clustered/avg_peak_dict.json", "w") as f:
    json.dump(avg_peak_dict, f)

# Load later
with open("/Users/sebinjohn/Tracy_arm/data/party/non_clustered/avg_peak_dict.json", "r") as f:
    avg_peak_dict = json.load(f)

cat = party.get_catalog()

det_times = {}
for ev in cat:
    det_times[ev.resource_id.id]=ev.origins[0].time            


from datetime import datetime

def normalize_id(s):
    """Convert nformatdets-style ID into det.id-style ID."""
    if "T" in s and "." in s:  # looks like old format
        front, back = s.rsplit("_", 1)
        dt = datetime.strptime(back, "%Y%m%dT%H%M%S.%f")
        reformatted_back = dt.strftime("%Y%m%d_%H%M%S%f")
        return f"{front}_{reformatted_back}"
    return s  # already in det.id format


sorted_dets = sorted(det_times.items(), key=lambda x: x[1])
selected_ids = []
i = 0
while i < len(sorted_dets):
    window_start = sorted_dets[i][1]
    window_end = window_start + 0.25
    # Get detections in this window
    window_dets = [det for det in sorted_dets if window_start <= det[1] < window_end]
    if not window_dets:
        i += 1
        continue
    # Pick detection with max avg_peak
    nformatdets=[d[0] for d in window_dets]
    formatdetsid=[normalize_id(d) for d in nformatdets]
    values=[]
    for idef in formatdetsid:
        value=avg_peak_dict.get(idef,np.nan)
        values.append(value)
    values=np.array(values)
    if np.isnan(values).any():
        print("issue with code")
        break
    best_evid=np.argmax(values)
    selected_ids.append((nformatdets[best_evid],formatdetsid[best_evid]))
    # Skip to next detection after this window
    i += len(window_dets)
    print(i)



# Get normalized IDs
normalized_ids = [t[1] for t in selected_ids]

# Check for duplicates
duplicates = [det_id for det_id, count in Counter(normalized_ids).items() if count > 1]

print("Duplicates:" if duplicates else "No duplicates found")
for det_id in duplicates:
    print(det_id)

new_cat = Catalog()
seen_ids = set()


selected_first_ids = set([t[0] for t in selected_ids])

for ev in cat:
    rid = ev.resource_id.id
    if rid in selected_first_ids and rid not in seen_ids:
        new_cat.append(ev)
        seen_ids.add(rid)

print(f"Original catalog: {len(cat)} events")
print(f"Filtered catalog: {len(new_cat)} events")

output_file = "/Users/sebinjohn/Tracy_arm/data/filtered_catalog/filtered_catalog_2.5s.xml"
new_cat.write(output_file, format="QUAKEML")
print(f"Filtered catalog saved to: {output_file}")

from obspy import read_events

# Read the QuakeML catalog
new_cat = read_events(output_file)


# Collect all event IDs
event_ids = [ev.resource_id.id for ev in new_cat]

# Count duplicates
dup_ids = [eid for eid, count in Counter(event_ids).items() if count > 1]

# Extract origin times
times = [ev.origins[0].time.datetime for ev in new_cat]
times.sort()

# Build cumulative counts
cumulative_counts = np.arange(1, len(times) + 1)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, cumulative_counts, drawstyle="steps-post", color="darkred", lw=1.8)

# Formatting
ax.set_xlabel("Time")
ax.set_ylabel("Cumulative Number of Events")
ax.set_title("Cumulative Event Count from Filtered Catalog")

ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))  # tick every day
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d")) 
plt.xticks(rotation=45)

ax.grid(True, ls="--", alpha=0.6)

plt.tight_layout()
plt.show()
fig.savefig("/Users/sebinjohn/Downloads/cumulative.png",dpi=600)

import pandas as pd 
times_series = pd.Series(times)

# Bin events by day
counts_per_day = times_series.dt.floor("D").value_counts().sort_index()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(counts_per_day.index, counts_per_day.values, 
       width=0.8, color="darkred", alpha=0.7)

# Log scale for y-axis
ax.set_yscale("log")

# Formatting
ax.set_xlabel("Date")
ax.set_ylabel("Number of Events (log scale)")
ax.set_title("Daily Event Counts (Log Scale) from Filtered Catalog")

# X-axis: show ticks every day, format as "Mon-DD"
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
plt.xticks(rotation=45)

ax.grid(True, ls="--", alpha=0.6)

plt.tight_layout()
plt.show()

# Save figure
fig.savefig("/Users/sebinjohn/Downloads/daily_event_counts_log.png", dpi=600)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(counts_per_day.index, counts_per_day.values, 
       width=0.8, color="darkgrey", alpha=0.7)

# Formatting
ax.set_xlabel("Date")
ax.set_ylabel("Number of Events")
ax.set_title("Daily Event Counts from Filtered Catalog")

# X-axis: show ticks every day, format as "Mon-DD"
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
plt.xticks(rotation=45)

ax.grid(True, ls="--", alpha=0.6)
ax.set_ylim([0,100])
plt.tight_layout()
plt.show()

# Save figure
fig.savefig("/Users/sebinjohn/Downloads/daily_event_counts.png", dpi=600)




############

# Make a set of normalized IDs for fast lookup
selected_normalized_ids = set([t[1] for t in selected_ids])


template_list = []
for i in tqdm(range(0, 101)):
    fam = party[i]

    # Only keep detections whose normalized ID is in selected_normalized_ids
    filtered_dets = [det for det in fam if det.id in selected_normalized_ids]

    if not filtered_dets:
        print(f"all events closeby for,{fam},{i}")
        continue

    for det in filtered_dets:
        sti = det.extract_stream(stream=st, length=15, prepick=5.5)
        sti.filter("bandpass", freqmin=2.0, freqmax=8.0, corners=4, zerophase=True)
        for trii in sti:
            trii.trim(starttime=trii.stats.starttime + 5,endtime=trii.stats.endtime - 5)
        template_list.append((sti, det.id))



# Save template_list to a file
with open("/Users/sebinjohn/Tracy_arm/data/clustering_det/templates_id.pkl", "wb") as f:
    pickle.dump(template_list, f)

print("template_list saved to template_list.pkl")

with open("/Users/sebinjohn/Tracy_arm/data/clustering_det/templates_id.pkl", "rb") as f:
    template_list = pickle.load(f)
    
# Extract all detection IDs from the template_list
all_ids = [tid for _, tid in template_list]

# Count occurrences
counts = Counter(all_ids)

# Find duplicates
duplicates = {tid: count for tid, count in counts.items() if count > 1}


from obspy import Stream
from eqcorrscan.core.match_filter import Party
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from eqcorrscan.utils.clustering import cluster
import numpy as np
import time
from tqdm import tqdm
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import timedelta
from matplotlib.colors import LogNorm
from obspy import read
from collections import defaultdict
from obspy import Trace
import random
from obspy import UTCDateTime
from collections import Counter
import math
import itertools
import matplotlib.cm as cm
import matplotlib.ticker as ticker
# -------------------------------
# Load Party / Detections
# -------------------------------
party = Party()
party.read('/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.tgz', read_detection_catalog=False)
print(party)

# Build detection times dictionary: det.id -> UTCDateTime
cat = party.get_catalog()
det_times = {}
for ev in cat:
    det_times[ev.resource_id.id] = ev.origins[0].time


# Helper: normalize ID formats
def normalize_id(s):
    if "T" in s and "." in s:  # old nformatdets-style
        front, back = s.rsplit("_", 1)
        dt = datetime.strptime(back, "%Y%m%dT%H%M%S.%f")
        reformatted_back = dt.strftime("%Y%m%d_%H%M%S%f")
        return f"{front}_{reformatted_back}"
    return s

det_times = {
    normalize_id(k): v for k, v in det_times.items()
}

# -------------------------------
# Load templates
# -------------------------------
with open("/Users/sebinjohn/Tracy_arm/data/clustering_det/templates_id.pkl", "rb") as f:
    template_list = pickle.load(f)

# Extract all detection IDs from the template_list
all_ids = [tid for _, tid in template_list]

# Count occurrences
counts = Counter(all_ids)

# Find duplicates
duplicates = {tid: count for tid, count in counts.items() if count > 1}
print("there are ",len(duplicates),"detection duplicates in all families")

fil_ch_temp = []

# Loop over each template
for stream, tid in template_list:
    for tr in stream:
        # Check if trace is vertical component
        if tr.stats.channel[-1].upper() == "Z":
            # Create a Stream with just this trace
            st = Stream(traces=[tr.copy()])  # copy to avoid modifying original trace
            if st[0].data.shape[0]==251:
            # Add to filtered list
                fil_ch_temp.append((st, tid))

all_ids = [tid for _, tid in fil_ch_temp]

# Count occurrences
counts = Counter(all_ids)

# Find duplicates
duplicates = {tid: count for tid, count in counts.items() if count > 1}
print("there are ",len(duplicates),
      "channels across detections which are common to a detection.bacuase of many more\nchannels for each detection")


# Get unique IDs
unique_ids = set(all_ids)

# Number of unique IDs
num_unique_ids = len(unique_ids)

print(f"Number of unique detections: {num_unique_ids}")



thresh=0.75

start = time.time()

groups = cluster(
    template_list=fil_ch_temp,
    show=False,
    corr_thresh=thresh,
    cores=6,replace_nan_distances_with=1)


end = time.time()
elapsed = end - start
print(f"Time elapsed: {elapsed:.2f} seconds")



output_file = f"/Users/sebinjohn/Tracy_arm/data/clustering_det/eqcorr_cluster_{thresh}.pkl" 

# with open(output_file, "wb") as f:    
#     pickle.dump(groups, f)

with open(output_file, "rb") as f:
    groups = pickle.load(f)

