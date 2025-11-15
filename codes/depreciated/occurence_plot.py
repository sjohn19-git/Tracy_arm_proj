#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 21:18:45 2025

@author: sebinjohn
"""


from obspy import Stream
from eqcorrscan.core.match_filter import Party
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
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
import matplotlib.ticker as ticker
from eqcorrscan.utils.stacking import align_traces


thresh=0.75
output_file = f"/Users/sebinjohn/Tracy_arm/data/clustering_det/eqcorr_cluster_{thresh}.pkl" 


with open(output_file, "rb") as f:
    groups = pickle.load(f)

def align_trs(trs, grp_id, t1=1, t2=15, plot_n=50, shift_len=3):
    """
    Align waveforms in traces using cross-correlation on a specific window,
    and plot both unaligned and aligned traces.
    Returns: list of aligned Trace objects
    """

    # Trim for alignment window
    p_trs = []
    for tr in trs:
        trim_tr = tr.copy()
        trim_tr = trim_tr.trim(starttime=tr.stats.starttime + t1,
                               endtime=tr.stats.starttime + t2)
        p_trs.append(trim_tr)

    # cross-correlate
    shift = int(shift_len * trs[0].stats.sampling_rate)
    shifts, corre = align_traces(p_trs, shift, master=p_trs[0],
                                 positive=True, plot=False)

    aligned_traces = []
    shift_info = {}

    for cid, tr in enumerate(trs):
        data = tr.data
        n = len(data)
        shift_sec = shifts[cid]
        shift_samples = int(shift_sec * tr.stats.sampling_rate)

        aligned_data = np.zeros_like(data)
        if shift_samples > 0:
            if shift_samples < n:
                aligned_data[shift_samples:] = data[:n - shift_samples]
        elif shift_samples < 0:
            shift_samples = abs(shift_samples)
            if shift_samples < n:
                aligned_data[:n - shift_samples] = data[shift_samples:]
        else:
            aligned_data = data.copy()

        aligned_tr = tr.copy()
        aligned_tr.data = aligned_data
        aligned_traces.append(aligned_tr)
        shift_info[cid] = shift_sec

    # --- Plotting (optional, keep as before) ---
    # select subset for plotting
    plot_n = min(plot_n, len(trs))
    selected_idx = random.sample(range(len(trs)), plot_n)

    fig_height = max(10, plot_n * 0.5)
    fig, axes = plt.subplots(2, 1, figsize=(12, fig_height), sharex=True)

    # Unaligned
    cluster_offset = 0
    ylabels, ytick_positions = [], []
    for cid in selected_idx:
        tr = trs[cid]
        trace_norm = tr.data / np.max(np.abs(tr.data))
        times_sec = np.arange(len(trace_norm)) / tr.stats.sampling_rate
        axes[0].plot(times_sec, trace_norm + cluster_offset, color="r", lw=0.8)
        ylabels.append(f"{tr.stats.station}.{tr.stats.channel}")
        ytick_positions.append(cluster_offset)
        cluster_offset += 1.5
    axes[0].set_yticks(ytick_positions)
    axes[0].set_yticklabels(ylabels, fontsize=8)
    axes[0].set_title(f"Unaligned traces for group {grp_id}")

    # Aligned
    cluster_offset = 0
    ylabels, ytick_positions = [], []
    for cid in selected_idx:
        tr = aligned_traces[cid]
        trace_norm = tr.data / np.max(np.abs(tr.data))
        times_sec = np.arange(len(trace_norm)) / tr.stats.sampling_rate
        axes[1].plot(times_sec, trace_norm + cluster_offset, color="k", lw=0.8)
        ylabels.append(f"{tr.stats.station}.{tr.stats.channel} (sh {shift_info[cid]:.2f}s)")
        ytick_positions.append(cluster_offset)
        cluster_offset += 1.5
    axes[1].set_yticks(ytick_positions)
    axes[1].set_yticklabels(ylabels, fontsize=8)
    axes[1].set_title(f"Aligned traces for group {grp_id}")
    axes[1].set_xlabel("Seconds")

    fig.suptitle(f"Waveform alignment for group {grp_id} (Shift window: {shift_len}s)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return aligned_traces


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

    # Select traces to plot
    plot_n = min(plot_n, len(trs))
    selected_trs = random.sample(trs, plot_n)

    # --- Figure ---
    fig_height = max(8, plot_n * spacing * 0.4 + 4)
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
    fig.savefig(f"/Users/sebinjohn/Downloads/wiggle.pdf",dpi=300)

 
def stack_waveforms(groupsf, all_st, grp_id=0,prf_sta="S32K", search_len=100,plot_n=50,
                    t1=1,t2=10,wiggle=True,align=False,shift_len=1):
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
        sti = detection.extract_stream(stream=all_st, length=40, prepick=10)
        sti = sti.select(channel="*Z")
        sti.filter("bandpass", freqmin=2.0, freqmax=8.0, corners=4, zerophase=True)

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
    
    if prf_sta is None:
        # Select the station with the most traces
        (sel_sta, sel_cha), trs = max(grouped.items(), key=lambda kv: len(kv[1]))
        print(f"\nSelected station with most traces: {sel_sta}_{sel_cha} ({len(trs)} traces)")
        selected_groups = { (sel_sta, sel_cha): trs }
    else:
        # Use only preferred station
        selected_groups = {k: v for k, v in grouped.items() if k[0] == prf_sta}
        if not selected_groups:
            print(f"\n⚠️ Preferred station {prf_sta} not found in group {grp_id}, skipping...")
            return grp_id, Stream()
        else:
            print(f"\nUsing preferred station: {prf_sta}")
    stacked_stream = Stream()
    for (sta, cha), trs in selected_groups.items():
        number_of_stacks=len(trs)
        print(f"Stacking {len(trs)} traces for station {sta}, channel {cha}")
        if align:
            trs=align_trs(trs, grp_id, t1=t1, t2=t2, plot_n=50, shift_len=shift_len)
    
        # Collect data arrays (make sure they are the same length)
        min_len = min(len(tr.data) for tr in trs)
        all_data = np.array([tr.data[:min_len] for tr in trs])
    
        # Stack by mean
        data_stack = np.mean(all_data, axis=0)
    
        # Copy header from first trace, adjust start time if needed
        header = trs[0].stats.copy()
        header.starttime = UTCDateTime(2025, 1, 1)  # dummy reference time
    
        # Create stacked trace
        new_tr = Trace(data=data_stack.astype(np.float32), header=header)
        stacked_stream.append(new_tr)
        plot_trs(trs, grp_id, plot_n=50, spacing=0.8, scale=1.0, wiggle=wiggle, stack_tr=new_tr)
    return grp_id, stacked_stream,number_of_stacks






all_st = read("/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.mseed") 

min_size=20
groupsf = [cl for cl in groups if len(cl) >= min_size]

grp_id, stacked_stream,n_stacks = stack_waveforms(
    groupsf, 
    all_st, 
    grp_id=5,     # <- fixed
    prf_sta="S32K", 
    search_len=100,
    plot_n=40,
    wiggle=True,
    align=False,
    t1=4,
    t2=8,
    shift_len=1
)


##########



def alighn_stcked(aligned_stackwf_dict,shift_len=3):
    """
    Align all traces in aligned_stackwf_dict using cross-correlation
    and return dictionary in the same format {cid: (Stream, nstacks)}.
    """
    # Collect all traces
    p_trs = [tr for st, nstacks in aligned_stackwf_dict.values() for tr in st]

    # Convert shift length to samples
    shift = int(shift_len * p_trs[0].stats.sampling_rate)

    # Align traces relative to one another
    shifts, corre = align_traces(p_trs, shift, positive=True, plot=False)

    aligned_dict = {}
    shift_info = {}

    # Rebuild dictionary with shifted traces
    i = 0
    for cid, (st, nstacks) in aligned_stackwf_dict.items():
        aligned_trs = []
        for tr in st:
            data = tr.data
            n = len(data)

            # shift for this trace
            shift_sec = shifts[i]
            shift_samples = int(shift_sec * tr.stats.sampling_rate)

            aligned_data = np.zeros_like(data)
            if shift_samples > 0:
                if shift_samples < n:
                    aligned_data[shift_samples:] = data[:n - shift_samples]
            elif shift_samples < 0:
                s = abs(shift_samples)
                if s < n:
                    aligned_data[:n - s] = data[s:]
            else:
                aligned_data = data.copy()

            aligned_tr = tr.copy()
            aligned_tr.data = aligned_data
            aligned_trs.append(aligned_tr)

            shift_info[(cid, tr.id)] = shift_sec
            i += 1

        aligned_dict[cid] = (Stream(aligned_trs), nstacks)

    return aligned_dict
            
    
def plot_aligned_stackwf(aligned_stackwf):
    """
    Plot aligned stack waveforms from aligned_stackwf dictionary.

    Parameters
    ----------
    aligned_stackwf : dict
        {cluster_id: (Stream, n_stacks)} dictionary of aligned traces
    """

    fig_height = max(8, len(aligned_stackwf) * 0.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    cluster_offset = 0
    ylabels = []
    ytick_positions = []

    for cid, (st, n_stacks) in aligned_stackwf.items():
        trace = st[0]
        trace_norm = trace.data / np.max(np.abs(trace.data))
        times_sec = np.arange(len(trace_norm)) / trace.stats.sampling_rate
        ax.plot(times_sec, trace_norm + cluster_offset, color="k", lw=0.8)

        # include n_stacks in label
        ylabels.append(f"{trace.stats.station}.{trace.stats.channel} | Group {cid} (N={n_stacks})")
        ytick_positions.append(cluster_offset)
        cluster_offset += 1.5

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Seconds")
    ax.set_title("Aligned Stack Waveforms per Group")
    plt.tight_layout()
    plt.show()
    fig.savefig(f"/Users/sebinjohn/Downloads/Single_Trace_Clusters_{trace.stats.station}.pdf", dpi=300)


prf_sta="S32K"
stackwf_dict={}
for grp_id in range(len(groupsf)):
    # use the loop's grp_id, not a hard-coded value
    if grp_id==17:
        align=True
    else:
        align=False
    grp_id, stacked_stream,n_stacks = stack_waveforms(
        groupsf, 
        all_st, 
        grp_id=grp_id,     # <- fixed
        prf_sta=prf_sta, 
        search_len=100,
        plot_n=40,
        wiggle=True,
        align=align,
        t1=3,
        t2=8,
        shift_len=1
    )

    if stacked_stream is not None and len(stacked_stream) > 0:
        stackwf_dict[grp_id] = (stacked_stream,n_stacks)



output_file = f"/Users/sebinjohn/Tracy_arm/data/clustering_det/eqcorr_stack_{thresh}_{prf_sta}.pkl" 


with open(output_file, "wb") as f:    
    pickle.dump(stackwf_dict, f)
    
    
with open(output_file, "rb") as f:
    stackwf_dict = pickle.load(f)

plot_aligned_stackwf(stackwf_dict)
corrected_stack=alighn_stcked(stackwf_dict,shift_len=3)

plot_aligned_stackwf(corrected_stack)
   
    



def plot_occurrence_timeline(groups, det_times,xmin=None,xmax=None, min_size=20, bin_hours=1):
    
    groupsf = [cl for cl in groups if len(cl) >= min_size]

    # Collect all event times
    all_times = []
    for cl in groupsf:
        for _, tid in cl:
            if tid in det_times:
                all_times.append(det_times[tid].datetime)

    if not all_times:
        print("No events to plot.")
        return

    # Define histogram bins (6 hours = 0.25 days in matplotlib units)
    min_time = min(all_times)
    max_time = max(all_times)
    bin_edges = mdates.drange(min_time, max_time, timedelta(hours=bin_hours))

    # Histogram counts per bin
    counts, _ = np.histogram(mdates.date2num(all_times), bins=bin_edges)

    norm = LogNorm(vmin=max(1, counts.min()), vmax=counts.max())
    cmap = plt.cm.turbo
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, cl in enumerate(groupsf):
        times = []
        for _, tid in cl:
            if tid in det_times:
                times.append(det_times[tid].datetime)
        if not times:
            continue

        # Convert to matplotlib dates
        times_mpl = mdates.date2num(times)
        times_mpl.sort()
        
        bin_indices = np.digitize(times_mpl, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(counts) - 1)  # prevent out-of-range
        colors = [cmap(norm(counts[i])) for i in bin_indices]
        # Scatter with per-event color
        ax.scatter(times_mpl, np.full(len(times_mpl), idx),
                   c=colors, s=8, alpha=0.9)

        # Lifespan line (neutral gray)
        ax.hlines(idx, times_mpl[0], times_mpl[-1], colors="lightgray", lw=0.8)
        if np.isnan(xmin):
            ax.text(times_mpl[-1] + 0.25, idx, f"{len(times)}", va="center", fontsize=8)  
        else:
            ax.text(xmax - 0.25, idx, f"{len(times)}", va="center", fontsize=8)
             
        # Label with cluster size
        
    if np.isnan(xmin):
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.set_xlim([xmin,xmax])
        
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    plt.xticks(rotation=45)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_ylabel(f"Group (≥{min_size} members)")
    ax.set_xlabel("Time")
    ax.set_title(f"Occurrence Timeline (colored by {bin_hours}-hour event density)")

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"Events per {bin_hours}-hour window")

    plt.tight_layout()
    plt.show()
    fig.savefig("/Users/sebinjohn/Downloads/Occurences_eqcorr.pdf", dpi=300)


xmin = np.nan
xmax = np.nan
xmin=mdates.date2num(datetime(2025, 8, 10, 0))
xmax=mdates.date2num(datetime(2025, 8, 11, 0))


plot_occurrence_timeline(groups, det_times,xmin,xmax,min_size=20, bin_hours=1)





########Plot record section
min_size=20
groupsf = [cl for cl in groups if len(cl) >= min_size]


def plot_recordsection(groupsf, stackwf, st, party, grp_id=17, trace_no=40, 
                       align_stat="S32K", spacing=0.5, scale=1.0, wiggle=True, 
                       outpath="/Users/sebinjohn/Downloads/record_section_wiggle.pdf"):
    """
    Plot a record section with detections and a stack.

    Parameters
    ----------
    groupsf : dict
        Grouped detections (families).
    stackwf : dict
        Stacked waveforms for each group.
    st : Stream
        Continuous waveform stream.
    party : list
        List of detection families.
    grp_id : int
        Group ID to plot.
    trace_no : int
        Number of detections to plot.
    align_stat : str
        Station to align on.
    spacing : float
        Vertical spacing between traces.
    scale : float
        Amplitude scaling factor for traces.
    wiggle : bool
        If True, use wiggle plot style (filled positive lobes).
    outpath : str
        Path to save the figure.
    """

    # --- Group + stack ---
    grp = groupsf[grp_id]
    stack = stackwf[grp_id].select(station=align_stat)
    dets = [det[1] for det in grp]

    # --- Filter detections across families ---
    filtered_dets = []
    for i in tqdm(range(101), desc="Filtering detections"):
        fam = party[i]
        fam_filtered = [det for det in fam if det.id in dets]
        if fam_filtered:
            filtered_dets.extend(fam_filtered)

    # --- Select subset of detections ---
    n_samples = min(trace_no, len(filtered_dets))
    selected_dets = random.sample(filtered_dets, n_samples)

    # --- Normalize stack ---
    stack_trace = stack[0].copy()
    stack_trace.data /= np.max(np.abs(stack_trace.data))
    stack_time = np.arange(len(stack_trace.data)) / stack_trace.stats.sampling_rate

    # --- Figure ---
    fig_height = max(8, n_samples * spacing * 0.4 + 4)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # --- Plot detections ---
    for i, det in tqdm(enumerate(selected_dets), desc="Plotting detections"):
        detst = det.extract_stream(stream=st, length=40, prepick=10)
        detst.filter("bandpass", freqmin=2.0, freqmax=8.0, corners=4, zerophase=True)

        # Trim edges
        for trii in detst:
            trii.trim(starttime=trii.stats.starttime + 5, endtime=trii.stats.endtime - 5)
        try:
            tr = detst.select(station=align_stat).select(channel="*Z")[0]
        except:
            continue
        tr_data = tr.data / np.max(np.abs(tr.data)) * scale
        time = np.arange(len(tr_data)) / tr.stats.sampling_rate

        offset = i * spacing
        if wiggle:
            ax.fill_between(time, offset, tr_data + offset, where=(tr_data > 0),
                            color="k", alpha=0.8, lw=0)
            ax.plot(time, tr_data + offset, color="k", lw=0.5)
        else:
            ax.plot(time, tr_data + offset, color="k", lw=0.6)

    # --- Plot stack on top ---
    stack_offset = n_samples * spacing + spacing
    ax.plot(stack_time, stack_trace.data * scale + stack_offset, 
            color="r", lw=2, label="Stack", zorder=10)

    # --- Formatting ---
    ax.set_xlim(0, 25)   # adjust to your signal length
    ax.set_ylim(-spacing, stack_offset + spacing*2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trace offset")
    ax.set_title(f"Record Section | Group {grp_id} | Station {align_stat}", 
                 fontsize=14, fontweight="bold")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Save figure
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    return grp


for cid in range(len(groupsf)):
    stations_in_grps = set()
    for stream, _ in groupsf[cid]:   # unpack tuple
        for tr in stream:           # stream can have multiple traces
            stations_in_grps.add(f"{tr.stats.network}.{tr.stats.station}")
    
    print(f"Stations in groupsf[{cid}]:", sorted(stations_in_grps))


# --- Example call ---
grpi=plot_recordsection(
    groupsf=groupsf, 
    stackwf=stackwf, 
    st=all_st, 
    party=party, 
    grp_id=17,          # which group to plot
    trace_no=20,        # number of detections
    align_stat="S32K",  # station to align on
    spacing=0.65,        # vertical spacing between traces
    scale=0.5,          # amplitude scaling
    wiggle=False,        # wiggle style on
    outpath="/Users/sebinjohn/Downloads/record_section_wiggle.pdf"
)


import random
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.cross_correlation import correlate

def cross_correlate_with_random(groupsfi, max_lag=50):
    """
    Pick a random reference trace from groupsfi and cross-correlate with all others.
    Returns maximum positive correlation and corresponding lag.

    Parameters
    ----------
    groupsfi : list
        List of (Stream, id) tuples.
    max_lag : int
        Maximum lag (in samples) for cross-correlation.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ref_trace : Trace
        The chosen reference trace.
    corr_results : list of (station, corr, lag)
        List of correlation results against reference.
    """
    ref_stream, _ = random.choice(groupsfi)
    ref_trace = ref_stream[0]

    corr_results = []

    for stream, _ in groupsfi:
        tr = stream[0]

        # Ensure same length
        min_len = min(len(ref_trace.data), len(tr.data))
        data_ref = ref_trace.data[:min_len]
        data_tr = tr.data[:min_len]

        # Cross-correlation
        cc = correlate(data_tr, data_ref, max_lag)
        lags = np.arange(-max_lag, max_lag + 1)

        # Only positive correlations
        pos_mask = cc > 0
        if np.any(pos_mask):
            best_idx = np.argmax(cc[pos_mask])
            valid_idxs = np.where(pos_mask)[0]
            best_idx = valid_idxs[best_idx]
            value = cc[best_idx]
            shift = lags[best_idx]
        else:
            value, shift = 0.0, None

        corr_results.append((tr.stats.station, value, shift))

    return ref_trace, corr_results


def plot_corr_results(ref_trace, corr_results):
    stations = [r[0] for r in corr_results]
    corrs = [r[1] for r in corr_results]

    plt.figure(figsize=(10, 5))
    plt.plot(corrs, color="steelblue")
    plt.axhline(0.75, color="k", linewidth=0.8)
    plt.title(f"Cross-correlation vs reference {ref_trace.stats.station}")
    plt.ylabel("Correlation coefficient")
    plt.xlabel("Station")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Example usage
ref_trace, results = cross_correlate_with_random(groupsf[11], max_lag=0)
plot_corr_results(ref_trace, results)

for sta, corr, lag in results:
    print(f"{sta}: corr={corr:.3f}, lag={lag}")




