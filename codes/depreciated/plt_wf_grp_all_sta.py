#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 06:48:50 2025

@author: sebinjohn
"""

from obspy import Stream
from eqcorrscan.core.match_filter import Party
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time
from tqdm import tqdm
import matplotlib.cm as cm
from obspy import read
from collections import defaultdict
from obspy import Trace
import random
from obspy import UTCDateTime
from collections import Counter
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from eqcorrscan.utils.stacking import align_traces
from obsplus import WaveBank
from pathlib import Path
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from seisbench.models import PhaseNet

party = Party()
party.read('/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.tgz', read_detection_catalog=False)
print(party)



outdir = Path("/Users/sebinjohn/Tracy_arm/data/seismic")
bank = WaveBank(outdir) 
bank.update_index()
avail=bank.get_availability_df() 

model = PhaseNet.from_pretrained("stead")

def align_stream(stacked_stream,pad_value=0.0):
    """
    Align waveforms in traces using cross-correlation on a specific window,
    and plot both unaligned and aligned traces.
    Returns: list of aligned Trace objects
    """
    stacked_stream.plot()
    preds = model.classify(stacked_stream)
    picks=preds.picks
    
    aligned_stream = Stream()

    # Build dictionary of pick times per station
    pick_dict = {}
    for pick in picks:
        net, sta = pick.trace_id.split(".")[:2]
        try:
            peak_time=pick_dict[(net, sta)]
            if pick.peak_time<peak_time:
                pick_dict[(net, sta)] = pick.peak_time
            else:
                continue
        except:
            pick_dict[(net, sta)] = pick.peak_time
    traces_with_picks = [tr for tr in stacked_stream if (tr.stats.network, tr.stats.station) in pick_dict]
    if not traces_with_picks:
        print("No traces with the chosen phase found.")
        

      
    # compute common length (shortest or longest, your choice)
    nmax = max(len(tr.data) for tr in traces_with_picks)
      
    # reference pick time (for alignment)
    ref_picks = [pick_dict[(tr.stats.network, tr.stats.station)] for tr in traces_with_picks]
    ref_pick=ref_picks[0]
    # --- shift and zero-pad traces ---
    for tr in traces_with_picks:
        pick_time = pick_dict[(tr.stats.network, tr.stats.station)]
        shift_sec = (ref_pick - pick_time)
        print(tr,shift_sec)
        shift_samples = int(round(shift_sec * traces_with_picks[0].stats.sampling_rate))
      
        data = tr.data.astype(float)
        # create zero-padded array
        aligned = np.ones(nmax) * pad_value
      
        if shift_samples >= 0:
            aligned[shift_samples:shift_samples+len(data)] = data[:nmax-shift_samples]
        else:
            aligned[:len(data)+shift_samples] = data[-shift_samples:len(data)]
      
        tr_aligned = Trace(data=aligned, header=tr.stats)
        tr_aligned.stats.processing = tr.stats.get('processing', [])
        aligned_stream.append(tr_aligned)
    
    aligned_stream.plot() 
    print(f"Aligned {len(aligned_stream)} traces")     
    return aligned_stream


def plot_trs(trs, grp_id, plot_n=50, spacing=0.5, scale=1.0, wiggle=True, stack_tr=None):
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
    selected_trs = random.sample(trs, plot_n)

    # --- Figure ---
    fig_height = int(plot_n * spacing)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    ylabels, ytick_positions = [], []
    max_time = 0
    for i, tr in enumerate(selected_trs):
        # Normalize and scale
        tr_data = tr.data / np.max(np.abs(tr.data)) * scale
        timei = np.arange(len(tr_data)) / tr.stats.sampling_rate
        max_time = max(max_time, timei[-1])
        # Offset each trace vertically
        offset = i * spacing
        if wiggle:
            ax.fill_between(timei, offset, tr_data + offset, where=(tr_data > 0),
                            color="k", alpha=0.8, lw=0)
            ax.plot(timei, tr_data + offset, color="k", lw=0.5)
        else:
            ax.plot(timei, tr_data + offset, color="k", lw=0.6)

        # Station labels
        ylabels.append(f"{tr.stats.station}.{tr.stats.channel}")
        ytick_positions.append(offset)

    # --- Overlay stacked trace if provided ---
    if stack_tr is not None:
        stack_data = stack_tr.data / np.max(np.abs(stack_tr.data)) * scale
        stack_time = np.arange(len(stack_data)) / stack_tr.stats.sampling_rate
        stack_offset = plot_n * spacing + spacing  # place above all traces
        ax.plot(stack_time, stack_data + stack_offset, color="r", lw=2, label="Stacked trace")

    # --- Formatting ---
    ax.set_xlim(0, max_time)  # show full trace length
    ax.set_ylim(-spacing, (plot_n - 1) * spacing + (3 * spacing)) 
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trace offset")
    ax.set_title(f"Waveforms for group {grp_id}", fontsize=14, fontweight="bold")

    if stack_tr is not None:
        ax.legend()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig("/Users/sebinjohn/Downloads/wiggle.pdf",dpi=300)

 
def stack_waveforms(groupsf, grp_id=0,prf_stas=["S32K"], prf_cha="*H*",search_len=100,plot_n=50,
                    wiggle=True,align=False,fmin=2,fmax=8):
    """
    Stack waveforms from a group of detections and return with cluster ID.
    """
    grp = groupsf[grp_id]
    
    print(f"\nGroup {grp_id} | {len(grp)} detections")
    stations = set()
    for st, tid in grp:
        sta,ch= st[0].stats.station, st[0].stats.channel  
        stations.add(sta+"_"+ch)
    print(f"  Stations: {', '.join(sorted(stations))}")
    print("extraction detections from Party.....")
    
    dets = [det[1] for det in grp]
    
    contri_fams = []
    filtered_dets = []
    for i in tqdm(range(0, 101), desc="Filtering detections"):
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
            
    if len(filtered_dets) == 0:
        print(f"No detections found in family {grp_id}.")
        return grp_id, Stream()
    n_samples = min(search_len, len(filtered_dets))
    selected_dets = random.sample(filtered_dets, n_samples)
    
    print("processing and extracting streams from all stations....")
    
    traces = []

    for detection in tqdm(selected_dets, desc="Processing detections"):
        stt=detection.detect_time-70
        ett=detection.detect_time+110
        sti=bank.get_waveforms(network="*",station=prf_stas,starttime=stt,endtime=ett)  
        sti = sti.select(channel=prf_cha)
        sti.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

        for trii in sti:
            trii.trim(starttime=trii.stats.starttime + 5, endtime=trii.stats.endtime - 5)
            traces.append(trii)
    
    print(f"\nFound {len(traces)} traces in total after processing.")
    
    
    grouped = defaultdict(list)
    for tr in traces:
        key = (tr.stats.station, tr.stats.channel)
        grouped[key].append(tr)
    for (sta, cha), trs in grouped.items():
        print(f"Found {len(trs)} traces for station {sta}, channel {cha}")
    
    stacked_stream = Stream()
    for (sta, cha), trs in grouped.items():
        number_of_stacks=len(trs)
        print(f"Stacking {len(trs)} traces for station {sta}, channel {cha}")
        if align:
            trs=align_stream(stacked_stream,pad_value=0.0)
    
        # Collect data arrays (make sure they are the same length)
        min_len = min(len(tr.data) for tr in trs)
        print(f"minimum length of taces for {sta} is {min_len}")
        all_data = np.array([tr.data[:min_len] for tr in trs])
    
        # Stack by mean
        data_stack = np.mean(all_data, axis=0)
    
        # Copy header from first trace, adjust start time if needed
        header = trs[0].stats.copy()
        header.starttime = UTCDateTime(2025, 8, 10)  # dummy reference time
    
        # Create stacked trace
        new_tr = Trace(data=data_stack.astype(np.float32), header=header)
        stacked_stream.append(new_tr)
        plot_trs(trs, grp_id, plot_n=50, spacing=0.8, scale=1.0, wiggle=wiggle, stack_tr=new_tr)
    return grp_id, stacked_stream



thresh=0.75
output_file = f"/Users/sebinjohn/Tracy_arm/data/groups/grps_eqcor_{thresh}.pkl" 


with open(output_file, "rb") as f:
    groups = pickle.load(f)

min_size=20
groupsf = [cl for cl in groups if len(cl) >= min_size]


grp_id, stacked_stream = stack_waveforms(
    groupsf, 
    grp_id=1,     # <- fixed
    prf_stas=["S32K","SIT","R32K","EDCR","S31K"], 
    prf_cha="*H*",
    search_len=100,
    plot_n=40,
    wiggle=True,
    align=False,
    fmin=2,
    fmax=8
)

fig_height = int(len(stacked_stream)* 0.9)
fig, ax = plt.subplots(figsize=(12, fig_height))
fig=stacked_stream.plot(equal_scale=False,fig=fig)

#aligned_stream=align_stream(stacked_stream,pad_value=0.0)

plot_trs(stacked_stream, grp_id, plot_n=50, spacing=0.9, scale=1.0, wiggle=False, stack_tr=None)

