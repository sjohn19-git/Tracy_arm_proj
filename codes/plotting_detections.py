#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot template + detections aligned by detection time,
using the station-channel pair that has picks in catalog.
Mark P and S picks on template and detection traces.
"""
from obspy import Stream, read_events
import matplotlib.pyplot as plt
import numpy as np
import random
from eqcorrscan.core.match_filter import Party
from obsplus import WaveBank
from pathlib import Path

######################################
# USER SETTINGS
######################################
party_file = "/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.tgz"
wavebank_dir = "/Users/sebinjohn/Tracy_arm/data/seismic"
catalog_file = "/Users/sebinjohn/Tracy_arm/data/catalog_with_picks.xml"

prepick = 5      # seconds before pick
postpick = 25    # seconds after pick
n_detections = 30
fmin, fmax = 2, 8

######################################
# LOAD DATA
######################################
party = Party().read(party_file, read_detection_catalog=False)
catalog = read_events(catalog_file)
bank = WaveBank(Path(wavebank_dir))
bank.update_index()

print(f"Loaded {len(party)} detection groups")
print(f"Loaded {len(catalog)} cataloged events")

######################################
def get_station_channel_with_picks(event):
    """Return first (station, channel) pair that has a P pick."""
    for pick in event.picks:
        wid = pick.waveform_id
        if wid.station_code and wid.channel_code:
            return wid.station_code, wid.channel_code
    return None, None


######################################
def plot_template_and_detections(event, detections, sta, cha):
    """
    Plot template and multiple detections aligned to the PICK TIME,
    mark P-picks in RED, S-picks in BLUE.
    """

    # ---- 1. Select P/S pick for this sta.cha ----
    pick = None
    for p in event.picks:
        wid = p.waveform_id
        if wid.station_code == sta and wid.channel_code == cha:
            pick = p
            break

    if pick is None:
        print(f"⚠ No pick found for {sta}.{cha} in catalog event, skipping.")
        return

    pick_time = pick.time  # <-- this is t = 0 reference

    # ---- 2. Extract TEMPLATE waveform around pick ----
    st_tpl = bank.get_waveforms(
        station=sta, channel=cha,
        starttime=pick_time - prepick-5,
        endtime=pick_time + postpick+5,
    )

    if len(st_tpl) == 0:
        print(f"⚠ No waveform in WaveBank for template at {sta}.{cha}")
        return

    st_tpl.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    for trii in st_tpl:
        trii.trim(starttime=trii.stats.starttime + 5, endtime=trii.stats.endtime - 5)
    tpl = st_tpl[0]

    # ---- 3. Prepare plot ----
    fig, ax = plt.subplots(figsize=(12, 10))
    spacing = 1.3

    # ---- Plot template ----
    y_tpl = tpl.data / np.max(np.abs(tpl.data))
    t_tpl = tpl.times(reftime=pick_time)  # pick is time zero
    ax.plot(t_tpl, y_tpl, color="red", lw=2, label="Template")

    # ---- Plot P/S pick markers for template ----
    for p in event.picks:
        wid = p.waveform_id
        if wid.station_code == sta and wid.channel_code == cha:
            rel_t = p.time - pick_time
            col = "red" if p.phase_hint.startswith("P") else "blue"
            ax.axvline(rel_t, color=col, linestyle="--", lw=1.5)

    # ---- 4. Plot DETECTIONS ----
    for i, det in enumerate(detections, start=1):

        st_det = bank.get_waveforms(
            station=sta, channel=cha,
            starttime=pick_time - prepick-5,
            endtime=pick_time + postpick+5,
        )

        if len(st_det) == 0:
            continue

        st_det.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
        for trii in st_det:
            trii.trim(starttime=trii.stats.starttime + 5, endtime=trii.stats.endtime - 5)

        tr = st_det[0]
        y = tr.data / np.max(np.abs(tr.data))
        t = tr.times(reftime=pick_time)       # relative to PICK time
        ax.plot(t, y + spacing * i, color="black", lw=0.5)

        # Draw catalog picks relative to pick_time
        for p in event.picks:
            wid = p.waveform_id
            if wid.station_code == sta:
                rel_t = p.time - pick_time
                col = "red" if p.phase_hint.startswith("P") else "blue"
                ax.axvline(rel_t, color=col, linestyle="--", lw=1)

    # ---- Final formatting ----
    ax.set_title(f"{sta}.{cha} | {len(detections)} detections (P=red / S=blue)")
    ax.set_xlabel("Time (s) relative to PICK time")
    ax.set_yticks([])
    ax.legend()
    plt.show()


######################################
# MAIN LOOP
######################################
for party_id, fam in enumerate(party):
    print(f"\n=== Party {party_id} ===")

    event = catalog[party_id]         # one-to-one pairing
    detections = random.sample(fam.detections, min(n_detections, len(fam.detections)))
    sta="S32K"
    cha="BHZ"
    plot_template_and_detections(event, detections, sta, cha)
