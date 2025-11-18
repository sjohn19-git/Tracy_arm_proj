#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 14:00:53 2025

@author: sebinjohn
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from eqcorrscan.core.match_filter import Party
from obsplus import WaveBank
import numpy as np
from obspy import Stream
import random
from tqdm import tqdm

outdir = Path("/Users/sebinjohn/Tracy_arm/data/seismic")
bank = WaveBank(outdir)
bank.update_index()
avail = bank.get_availability_df()

party = Party()
party.read('/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.tgz',
           read_detection_catalog=False)
print(party)
len(party)

party.decluster(0.25)
print(len(party))

THRESH=0.75

OUTPUT_FILE = f"/Users/sebinjohn/Tracy_arm/data/groups/grps_eqcor_S32K_{THRESH}.pkl"

# with open(output_file, "wb") as f:
#     pickle.dump(groups, f)

with open(OUTPUT_FILE, "rb") as f:
    groups = pickle.load(f)
    

min_size=20
groupsf = [cl for cl in groups if len(cl) <= min_size]


outliers = []

for cluster in groupsf:
    for item in cluster:
        outliers.append(item)



def plot_trs(trs, plot_n=50, spacing=0.9, scale=1.0, wiggle=True, stack_tr=None,grp_id=None,Ns=None,title=None,rand=False):
    """
    Plot raw traces for a group with station names, using wiggle-style plotting.
    Optionally overlay a stacked trace in red.

    Parameters
    ----------
    trs : list of Trace
        Traces to plot
    grp_id : str or int
        Group identifier for labeling the figure
    plot_n : int
        Maximum number of traces to plot
    spacing : float
        Vertical spacing between traces
    scale : float
        Amplitude scaling factor
    wiggle : bool
        If True, use wiggle plot style (fill positive lobes)
    stack_tr : Trace or None
        Optional stacked trace to plot in red on top
    """
    if isinstance(trs, Stream):
        trs=[tr for tr in trs]
    # Select traces to plot
    plot_n = min(plot_n, len(trs))
    if rand:
        selected_trs = random.sample(trs, plot_n)
    else:
        selected_trs = trs[:plot_n]
        
    selected_trs.sort(key=lambda tr: (tr.stats.station, tr.stats.channel))
    # --- Figure ---
    fig_height = max(6, plot_n * spacing*0.4+5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    ylabels, ytick_positions = [], []
    max_time = 0
    for i, tr in enumerate(selected_trs):
        # Normalize and scale
        tr_data = tr.data - np.mean(tr.data)                # remove DC offset
        tr_data = tr_data / np.max(np.abs(tr_data)) * scale # normalize to scale
        timei = np.arange(len(tr_data)) / tr.stats.sampling_rate
        max_time = max(max_time, timei[-1])
        # Offset each trace vertically
        offset = i * spacing
        if wiggle:
            ax.fill_between(timei, offset, tr_data + offset, where=(tr_data > 0),
                            color="k", alpha=0.8, lw=0)
            ax.plot(timei, tr_data + offset, color="k", lw=0.5)
        else:
            ax.plot(timei, tr_data + offset, color="k", lw=0.5)
        if isinstance(Ns,list):
            ylabels.append(f"{tr.stats.station}.{tr.stats.channel}.grp {i}.(N={Ns[i]})")
        else:
            ylabels.append(f"{tr.stats.station}.{tr.stats.channel}")
        ytick_positions.append(offset)

    # --- Overlay stacked trace if provided ---
    
    if stack_tr is not None:
        stack_offset = len(selected_trs) * spacing + spacing  # place above all traces
        stack_data = stack_tr.data / np.max(np.abs(stack_tr.data)) * scale
        stack_time = np.arange(len(stack_data)) / stack_tr.stats.sampling_rate
        ax.plot(stack_time, stack_data + stack_offset, color="r", lw=2, label="Stacked trace")

    # --- Formatting ---
    ax.set_xlim(0, max_time)  # show full trace length
    if stack_tr is not None:
        ax.set_ylim(-spacing, stack_offset+2*spacing)
    else:
        ax.set_ylim(-spacing, len(selected_trs) * spacing)
        
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trace offset")
    if not title:
        ax.set_title(f"Waveforms for randomly selected {plot_n} detections group {grp_id}", fontsize=14, fontweight="bold")
    else:
        ax.set_title(title, fontsize=14, fontweight="bold")
        

    if stack_tr is not None:
        ax.legend()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig(f"/Users/sebinjohn/Downloads/wiggle.pdf",dpi=300)


dets = [det[1] for det in outliers]
contri_fams = []
filtered_dets = []
for i in tqdm(range(0, 100), desc="Filtering detections"):
    fam = party[i]
    fam_filtered = [det for det in fam if det.id in dets]
    filtered_dets.extend(fam_filtered)
    if fam_filtered:
        contri_fams.append([fam,len(fam_filtered)])
print(f"Found {len(contri_fams)} families in this group")

total_dets = 0
for fam, ndets in contri_fams:
    print(f"Template {fam.template.name} | {ndets} detections")
    total_dets += ndets

print(f"Total detections across families: {total_dets}")
print(f"Detections in group: {len(dets)}")

search_len=50

n_samples = min(search_len, len(filtered_dets))
selected_dets = random.sample(filtered_dets, n_samples)

t1=20.5
t2=40   
traces = []
detect_times=[]
prf_stas=["S32K"]   # dynamically chosen
prf_cha="BHZ"
fmin=2
fmax=8
filt=True

for detection in tqdm(selected_dets, desc="Processing detections"):
    det_ev = detection.event
    p_time =det_ev.origins[0].time+16.65
    stt=p_time-t1
    ett=p_time+t2
    detect_times.append(p_time)
    sti=bank.get_waveforms(network="*",station=prf_stas,starttime=stt,endtime=ett)  
    sti = sti.select(channel=prf_cha)
    if filt:
        sti.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    for trii in sti:
        trii.trim(starttime=trii.stats.starttime + 5, endtime=trii.stats.endtime - 5)
        traces.append(trii)
spacing=0.9
plot_trs(traces, plot_n=50, spacing=spacing, scale=1,
         wiggle=False,rand=True, stack_tr=None, grp_id="Orphans")