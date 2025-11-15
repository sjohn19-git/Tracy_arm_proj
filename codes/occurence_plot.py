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
from tqdm import tqdm
import matplotlib.dates as mdates
import matplotlib.cm as cm
from datetime import timedelta
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

thresh=0.75
output_file = f"/Users/sebinjohn/Tracy_arm/data/groups/grps_eqcor_{thresh}.pkl" 

with open(output_file, "rb") as f:
    groups = pickle.load(f)
    
    

party = Party()
# load the QuakeML detections
party.read('/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.tgz',read_detection_catalog=False)
print(party)

cat = party.get_catalog()

def normalize_id(s):
    """Convert nformatdets-style ID into det.id-style ID."""
    if "T" in s and "." in s:  # looks like old format
        front, back = s.rsplit("_", 1)
        dt = datetime.strptime(back, "%Y%m%dT%H%M%S.%f")
        reformatted_back = dt.strftime("%Y%m%d_%H%M%S%f")
        return f"{front}_{reformatted_back}"
    return s  # already in det.id format


det_times = {}
for ev in cat:
    det_times[normalize_id(ev.resource_id.id)]=ev.origins[0].time            


def plot_occurrence_timeline(groups, det_times,xmin=None,xmax=None, min_size=20, bin_hours=1):
    
    groupsf = [cl for cl in groups if len(cl) >= min_size]

    # Collect all event times
    all_times = []
    for cl in groupsf:
        for _, tid in cl:
            if tid in det_times.keys():
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
xmin=mdates.date2num(datetime(2025, 8, 10, 1))
xmax=mdates.date2num(datetime(2025, 8, 10, 15))


plot_occurrence_timeline(groups, det_times,xmin,xmax,min_size=20, bin_hours=1)

n=0
groupsf = [cl for cl in groups if len(cl) >= 20]
for cl in groupsf:
    n+=len(cl)

########Plot record section

def plot_occurrence_timeline(groups, det_times, xmin=None, xmax=None,
                             min_size=20, bin_hours=1,
                             figsize=(14, 7), outfile=None):

    # Filter groups
    groupsf = [cl for cl in groups if len(cl) >= min_size]

    # Collect all detection times
    all_times = []
    for cl in groupsf:
        for _, tid in cl:
            if tid in det_times:
                all_times.append(det_times[tid].datetime)

    if not all_times:
        print("No events to plot.")
        return
    
    min_time = min(all_times)
    max_time = max(all_times)

    # Histogram binning
    bin_edges = mdates.drange(min_time, max_time, timedelta(hours=bin_hours))
    counts, _ = np.histogram(mdates.date2num(all_times), bins=bin_edges)

    # Log-normalized color scale
    vmin = max(1, counts.min())
    norm = LogNorm(vmin=vmin, vmax=counts.max())
    cmap = plt.cm.turbo

    fig, ax = plt.subplots(figsize=figsize)

    # Margin for count labels (in days)
    label_margin = 0.02 * (max_time - min_time).days if xmin is None else 0.02

    # Plot each group
    for idx, cl in enumerate(groupsf):

        # Extract times
        times = [det_times[tid].datetime for _, tid in cl if tid in det_times]
        if not times:
            continue

        times_mpl = mdates.date2num(times)
        times_mpl.sort()

        # Map each event to its histogram bin color
        bin_indices = np.digitize(times_mpl, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(counts) - 1)
        colors = [cmap(norm(counts[i])) for i in bin_indices]

        # Scatter events
        ax.scatter(times_mpl, np.full(len(times_mpl), idx),
                   c=colors, s=14, alpha=0.95, edgecolor="none")

        # Lifespan line
        ax.hlines(idx, times_mpl[0], times_mpl[-1],
                  colors="lightgray", lw=1.0)
        # ==================================================
        # Add vertical line marking landslide event
        landslide_time = datetime(2025, 8, 10, 13, 26)  # UTC
        landslide_mpl = mdates.date2num(landslide_time)
        ax.axvline(landslide_mpl, color="k", lw=1, linestyle="--")
        
        ax.text(landslide_mpl + 0.005, len(groupsf)-4.5,
                "Landslide\nAug 10 13:26",
                color="k", fontsize=9,rotation=90,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        # ==================================================
        # ----------------------
        # Improved label placement
        # ----------------------
        if xmin is None:  # No x-limits → place label at end of group line
            label_x = times_mpl[-1] + label_margin
        else:             # User x-limits → place label just inside right boundary
            label_x = xmax - label_margin

        ax.text(label_x, idx, f"{len(times)}",
                va="center", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
        # ---------------------------------------------------------------------

    # X-axis formatting
    if xmin is not None and xmax is not None:
        ax.set_xlim([xmin, xmax])
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d\n%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    
    #plt.xticks(rotation=45)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_ylabel(f"Group (≥{min_size} members)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_title(f"Occurrence Timeline (colored by {bin_hours}-hour event density)",
                 fontsize=14, weight="bold")

    # ----------------------
    # Better colorbar
    # ----------------------
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(f"Events per {bin_hours}-hour window", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    # Save if requested
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.show()


plot_occurrence_timeline(groups, det_times,min_size=20,xmin=xmin,xmax=xmax, bin_hours=1,outfile="/Users/sebinjohn/Downloads/Occurence.pdf")
