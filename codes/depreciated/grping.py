#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 13:19:52 2025

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

# Load the stream file you saved earlier
all_st = read("/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.mseed")
stream_list = []


for i in tqdm(range(0, 100)):
    fam = party[i]
    for det in fam:
        print(det)
        sti = det.extract_stream(stream=all_st, length=15, prepick=5.5)
        sti.filter("bandpass", freqmin=2.0, freqmax=8.0, corners=4, zerophase=True)
        for trii in sti:
            trii.trim(starttime=trii.stats.starttime + 5,endtime=trii.stats.endtime - 5)
        stream_list.append((sti, det.id))

output_file="/Users/sebinjohn/Tracy_arm/data/all_streams/streams_eqcorr.pkl"

# #Save template_list to a file
# with open(output_file, "wb") as f:
#     pickle.dump(stream_list, f)

with open(output_file, "rb") as f:
    stream_list = pickle.load(f)





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

plt.title("100 Random Streams (Filtered 2–8 Hz)")
plt.xlabel("Time (s) relative to trimmed window")
plt.ylabel("Stream Index")
plt.tight_layout()
plt.show()




# Extract all detection IDs from the template_list
all_ids_temp = [tid for _, tid in stream_list]

# Count occurrences
counts = Counter(all_ids_temp)

# Find duplicates
duplicates_temp = {tid: count for tid, count in counts.items() if count > 1}
print("there are ",len(duplicates_temp),"detection duplicates in all families")
num_unique_tids = len(set(all_ids_temp))
print(f"Number of unique detection IDs: {num_unique_tids}")



fil_ch_temp = []

# Loop over each template
for stream, tid in stream_list:
    for tr in stream:
        # Check if trace is vertical component
        if tr.stats.channel[-1].upper() == "Z":
            # Create a Stream with just this trace
            st = Stream(traces=[tr.copy()])  # copy to avoid modifying original trace
            if st[0].data.shape[0]==251:
            # Add to filtered list
                fil_ch_temp.append((st, tid))

all_ids = [tid for _, tid in fil_ch_temp]

import random
from obspy.signal.cross_correlation import correlate, xcorr_max
import numpy as np
# ---------------------------------------------------------------------
# STEP 2 — Separate SIT and S32K traces
# ---------------------------------------------------------------------

sit_traces  = [(st, tid) for (st, tid) in fil_ch_temp if st[0].stats.station == "SIT"]
s32k_traces = [(st, tid) for (st, tid) in fil_ch_temp if st[0].stats.station == "S32K"]

print("SIT BHZ traces:", len(sit_traces))
print("S32K BHZ traces:", len(s32k_traces))


# ---------------------------------------------------------------------
# STEP 3 — Select N random traces from each group
# ---------------------------------------------------------------------

N = 500   # <<< CHANGE THIS ANYTIME

sel_sit  = random.sample(sit_traces,  min(N, len(sit_traces)))
sel_s32k = random.sample(s32k_traces, min(N, len(s32k_traces)))

# For S32K-vs-S32K comparisons, another independent sample:
sel_s32k_2 = random.sample(s32k_traces, min(N, len(s32k_traces)))


# ---------------------------------------------------------------------
# STEP 4 — Cross-correlation function
# ---------------------------------------------------------------------

def corr_val(traceA, traceB, max_lag=100):
    """Return correlation coefficient at best lag."""
    n = min(len(traceA.data), len(traceB.data))
    a = traceA.data[:n]
    b = traceB.data[:n]

    c = correlate(a, b, max_lag)
    val = max(c)
    
    return val


# ---------------------------------------------------------------------
# STEP 5 — Compute correlation lists
# ---------------------------------------------------------------------

corr_sit_vs_s32k = []
for (stA, _), (stB, _) in zip(sel_sit, sel_s32k):
    corr_sit_vs_s32k.append(corr_val(stA[0], stB[0]))


corr_s32k_vs_s32k = []
for (stA, _), (stB, _) in zip(sel_s32k, sel_s32k_2):
    corr_s32k_vs_s32k.append(corr_val(stA[0], stB[0]))


# ---------------------------------------------------------------------
# STEP 6 — Plot
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.scatter(np.arange(0,N),corr_sit_vs_s32k, label="SIT vs S32K",color="orange")
plt.scatter(np.arange(0,N),corr_s32k_vs_s32k, color="blue", label="S32K vs S32K")

plt.axhline(0, color="k", lw=0.5)
plt.ylabel("Correlation Coefficient")
plt.xlabel("Pair Index")
plt.title(f"Cross-Correlation Comparison (N = {N})")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()



thresh=0.75

start = time.time()

groups = cluster(
    template_list=fil_ch_temp,
    show=False,
    corr_thresh=thresh,
    cores=4,replace_nan_distances_with=1)


end = time.time()
elapsed = end - start
print(f"Time elapsed: {elapsed:.2f} seconds")




output_file = f"/Users/sebinjohn/Tracy_arm/data/groups/grps_eqcor_{thresh}.pkl" 

# with open(output_file, "wb") as f:    
#     pickle.dump(groups, f)

with open(output_file, "rb") as f:
    groups = pickle.load(f)
