#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 09:43:53 2025

@author: sebinjohn
"""

from obspy import Stream
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from obspy import Trace
import random
from obspy import UTCDateTime
from obsplus import WaveBank
from pathlib import Path
from eqcorrscan.core.match_filter import Party
from collections import Counter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from obspy.clients.fdsn import Client
import pandas as pd
from obspy import read_events
import matplotlib.patheffects as pe
from cartopy.io import shapereader
from obspy import read
from eqcorrscan.utils.stacking import align_traces


thresh=0.75
output_file = f"/Users/sebinjohn/Tracy_arm/data/groups/grps_eqcor_S32K_{thresh}.pkl" 

with open(output_file, "rb") as f:
    groups = pickle.load(f)
    
party = Party()
party.read('/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.tgz', read_detection_catalog=False)
print(party)
len(party)

party.decluster(0.25)


outdir = Path("/Users/sebinjohn/Tracy_arm/data/seismic")
bank = WaveBank(outdir) 
bank.update_index()
avail=bank.get_availability_df() 


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

 
def stack_waveforms(groupsf, grp_id=0,prf_stas=["S32K"],prf_cha="*H*", search_len=100,plot_n=50,
                    fmin=2,fmax=8,wiggle=True,t1=15,t2=25,filt=True,spacing=0.9):
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
            
    if len(filtered_dets) == 0:
        print(f"No detections found in family {grp_id}.")
        return grp_id, Stream()
    n_samples = min(search_len, len(filtered_dets))
    selected_dets = random.sample(filtered_dets, n_samples)
    
    print("processing and extracting streams from all stations....")
    
    traces = []
    detect_times=[]

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


    print(f"\nFound {len(traces)} traces in total after processing.")
    
    
    grouped = defaultdict(list)
    for tr in traces:
        key = (tr.stats.station, tr.stats.channel)
        grouped[key].append(tr)
    for (sta, cha), trs in grouped.items():
        print(f"Found {len(trs)} traces for station {sta}, channel {cha}")

    stacked_stream = Stream()
    total_stacked_traces = 0

    for (sta, cha), trs in grouped.items():
        print(f"Stacking {len(trs)} traces for station {sta}, channel {cha}")
        st_tmp = Stream(trs)
        stacked = st_tmp.stack(npts_tol=2)

        # Adjust start time for stacked trace(s)
        for tr in stacked:
            tr.stats.starttime = UTCDateTime(2025, 1, 1)
        
        stacked_stream += stacked
        total_stacked_traces += len(trs)

        # Plot stacked vs raw
        plot_trs(trs, plot_n=plot_n, spacing=spacing, scale=1,
                 wiggle=wiggle,rand=True, stack_tr=stacked[0], grp_id=grp_id)
    return grp_id, stacked_stream,len(trs)

min_size=20
groupsf = [cl for cl in groups if len(cl) >= min_size]

Ns=[]
all_stack = Stream()
for i, group in enumerate(groupsf):
    # Count how many traces per station in this group
    station_counts = Counter(st[0].stats.station for st,ide in group)
    
    # Pick the most frequent station
    prf_sta = station_counts.most_common(1)[0][0]
    print(f"Group {i}: {len(group)} traces, preferred station = {prf_sta}, len of grp is {len(group)}")
  
    # Stack using that station as reference
    grp_id, stacked_stream,n = stack_waveforms(
        groupsf, 
        grp_id=i,
        prf_stas=["S32K"],   # dynamically chosen
        prf_cha="BHZ",
        search_len=100,
        plot_n=40,
        fmin=2,
        fmax=8,
        wiggle=False,
        t1=20.5,
        t2=40
    )
    Ns.append(n)
    # Add to global stack
    all_stack += stacked_stream
    
    
plot_trs(all_stack, plot_n=50, spacing=0.9, scale=1.0, wiggle=False, stack_tr=None,rand=False,Ns=Ns,title="Stack of detections from different groups")



def align_stream(stream, t1, t2, shift_len=200, positive=True, fill_value=0.0,master=0):
    st = stream.copy()
    master = st[master]
    
    # Convert window to absolute times if needed
    if isinstance(t1, (int, float)):
        t1 = master.stats.starttime + t1
    if isinstance(t2, (int, float)):
        t2 = master.stats.starttime + t2

    # Extract alignment window traces
    master_win = master.slice(t1, t2).data

    shifts = []
    corrs = []
    
    aligned = Stream()

    for tr in st:
        # Extract window for this trace
        data_win = tr.slice(t1, t2).data
        
        # Compute cross-correlation
        corr = np.correlate(master_win, data_win, mode="full")
        
        # Allowed shift range
        mid = len(corr) // 2
        corr_window = corr[mid - shift_len : mid + shift_len + 1]
        lags = np.arange(-shift_len, shift_len + 1)
        # NORMALIZED correlation
        corr_norm = corr_window / (np.linalg.norm(master_win) * np.linalg.norm(data_win))
        
        # best shift
        best_idx = np.argmax(corr_norm)
        shift_samples = int(lags[best_idx])
        # Save shift and correlation
        shifts.append(shift_samples)
        corrs.append(np.max(corr_norm))
        
        # --- Apply shift to full trace ---
        data = tr.data
        if shift_samples > 0:  # shift RIGHT
            new_data = np.concatenate([
                np.full(shift_samples, fill_value),
                data[:-shift_samples]
            ])
        elif shift_samples < 0:  # shift LEFT
            s = abs(shift_samples)
            new_data = np.concatenate([
                data[s:],
                np.full(s, fill_value)
            ])
        else:
            new_data = data.copy()
        
        # Add shifted trace
        new_tr = tr.copy()
        new_tr.data = new_data.astype(tr.data.dtype)
        aligned.append(new_tr)

    return aligned, shifts, corrs



aligned, shifts, corrs = align_stream(
    all_stack, 
    t1=4, t2=25,
    shift_len=900,
    fill_value=0,
    master=10
)

plot_trs(aligned, plot_n=50, spacing=0.9, scale=1.0, wiggle=False, stack_tr=None,rand=False,Ns=Ns,title="Stack of detections from different groups")



# aligned_stream=align_stream(all_stack,model,pad_value=0.0)
# plot_trs(aligned_stream, plot_n=50, spacing=0.9, scale=1.0, wiggle=False, stack_tr=None)

prf_stas=["S32K","SIT","R32K","EDCR","S31K","U33K","PLBC"]

grp_id, stacked_stream,n = stack_waveforms(
    groupsf, 
    grp_id=9,     # <- fixed
    prf_stas=prf_stas,
    prf_cha="*H*",
    search_len=300,
    plot_n=30,
    fmin=2,
    fmax=8,
    wiggle=False,
    t1=50,
    t2=130,
    filt=False,
    spacing=0.9
)

stacked_stream.write("/Users/sebinjohn/Downloads/composite.mseed", format="MSEED")

stacked_stream = read("/Users/sebinjohn/Downloads/composite.mseed")

plot_trs(stacked_stream, plot_n=50, spacing=2, scale=1.0, wiggle=False,rand=False, stack_tr=None,title="Stack of detections from group 9 on different channels")




#stacked_stream.plot()

#################
# ---- Collect unique station names from stacks ----
used_stations = list({tr.stats.station for tr in stacked_stream})
print(f"\nTotal unique stations used: {len(used_stations)}")
print("Stations:", ", ".join(sorted(used_stations)))
# ---- Load local event catalog ----
local_catalog_file = "/Users/sebinjohn/Tracy_arm/data/catalog_with_picks.xml"
catalog = read_events(local_catalog_file)
print(f"Loaded {len(catalog)} events from catalog.")

# ---- Fetch station metadata from IRIS ----
client = Client("IRIS")

networks = ["AK", "AV","AT", "CN"]  # can expand this if needed


inv = client.get_stations(
    network=",".join(networks),
    station=",".join(used_stations),
    level="station",
    starttime="2025-08-01",
    endtime="2025-08-31",
)
    
records = []
for network in inv:
    for station in network:
        records.append({
            "Network": network.code,
            "Station": station.code,
            "Latitude": station.latitude,
            "Longitude": station.longitude,
            "Elevation_m": station.elevation,
        })


# ---- Convert to DataFrame ----
df = pd.DataFrame(records)
print(f"Found {len(df)} station metadata entries from IRIS.")

# ---- Extract event coordinates from catalog ----
event_lats, event_lons, event_depths = [], [], []
for event in catalog:
    try:
        origin = event.preferred_origin() or event.origins[0]
        event_lats.append(origin.latitude)
        event_lons.append(origin.longitude)
        event_depths.append(origin.depth / 1000 if origin.depth else None)  # km
    except Exception as e:
        print(f"⚠️ Could not extract origin for event: {e}")

print(f"Extracted {len(event_lats)} event locations.")




# ---- MAP CONFIGURATION ----
proj = ccrs.LambertConformal(central_longitude=-135, central_latitude=57)
extent = [-138, -131, 55.5, 60]  # Southeast Alaska near Juneau & Tracy Arm

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=proj)
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
ax.add_feature(cfeature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS, alpha=0.3)

shpfilename = shapereader.natural_earth(resolution='10m',
                                        category='cultural',name='populated_places')
reader = shapereader.Reader(shpfilename)
places = list(reader.records())
for place in places:
    lon, lat = place.geometry.x, place.geometry.y
    name = place.attributes['NAME']
    # Only label if inside the map extent
    if extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]:
        ax.text(lon, lat, name, fontsize=8, color="darkgreen",
                transform=ccrs.PlateCarree(),
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])


# Plot stations
if not df.empty:
    ax.scatter(df["Longitude"], df["Latitude"],
               color="red", s=50, edgecolor="k",
               transform=ccrs.PlateCarree(), zorder=3)

    # Label stations
    for _, row in df.iterrows():
        ax.text(row["Longitude"] + 0.05, row["Latitude"] + 0.05,
                row["Station"], fontsize=8, transform=ccrs.PlateCarree(),path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Plot events from catalog
event_lats, event_lons = [], []
for event in catalog:
    if event.origins:
        ori = event.origins[0]
        event_lats.append(ori.latitude)
        event_lons.append(ori.longitude)

if event_lats:
    ax.scatter(event_lons, event_lats, color="blue", marker="*", s=80,
               edgecolor="white", linewidth=0.8, transform=ccrs.PlateCarree(),
               label="Events")


plt.title("Stations and Events at Tracy Arm", fontsize=14, weight="bold")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
fig.savefig("/Users/sebinjohn/map.png",dpi=200)

#############################






