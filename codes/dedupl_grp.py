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
