from eqcorrscan.utils.catalog_utils import filter_picks
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import logging
from eqcorrscan import Tribe
from obspy import read_events
from obsplus import WaveBank
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

client = Client("IRIS")
t1 = UTCDateTime("2025-07-01T00:00:00")
t2 = UTCDateTime("2025-08-15T00:00:00")


local_catalog_file = "/Users/sebinjohn/Tracy_arm/data/catalog_with_picks.xml"
catalog = read_events(local_catalog_file)


catalog = filter_picks(catalog=catalog, top_n_picks=10)

from collections import Counter

# Count how many picks each station–channel pair has
pair_counts = Counter()

for event in catalog:
    for pick in event.picks:
        wid = pick.waveform_id
        if wid and wid.station_code and wid.channel_code:
            pair = f"{wid.station_code}.{wid.channel_code}"
            pair_counts[pair] += 1

# Display sorted by station-channel name
print("Station–Channel pairs and pick counts:")
for sc in sorted(pair_counts):
    print(f"{sc:15s}  {pair_counts[sc]:5d}")

outdir = Path("/Users/sebinjohn/Tracy_arm/data/seismic")
bank = WaveBank(outdir) 
bank.update_index()
avail=bank.get_availability_df()   



tribe = Tribe().construct(
    method="from_client",client_id=bank, lowcut=2.0, highcut=8.0, samp_rate=50.0, length=5.0,
    filt_order=4, prepick=0.5, catalog=catalog, data_pad=20.,
    process_len=3600 , min_snr=3.0, parallel=True)

print(tribe)

#tribe.templates = [t for t in tribe if len({tr.stats.station for tr in t.st}) >= 4]

for trb in tribe:
    fig = trb.st.plot(equal_scale=False, size=(800, 600))
    
print(tribe)
nslcs = {tr.id for template in tribe for tr in template.st}
print(sorted(nslcs))

    
# party, st = tribe.client_detect(
#     client=client, starttime=t1, endtime=t2, threshold=9.,
#     threshold_type="MAD", trig_int=2.0, plot=False, return_stream=True)


party,st=tribe.client_detect(
    client=bank, starttime=t1, endtime=t2, threshold=9.,
    threshold_type="MAD", trig_int=2.0, plot=False, return_stream=True,ignore_length=True)


prty_p='/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party'
party.write(prty_p, format='tar')
for tr in st:
    if isinstance(tr.data, np.ma.MaskedArray):
        tr.data = tr.data.filled(0) 
st.write("/Users/sebinjohn/Tracy_arm/data/party/non_clustered/party.mseed",format="MSEED")


fig = party.plot(plot_grouped=True)


family = sorted(party.families, key=lambda f: len(f))[-1]
print(family)

fig = family.template.st.plot(equal_scale=False, size=(800, 600))
streams = family.extract_streams(stream=st, length=10, prepick=2.5)
print(family.detections[0])
fig = streams[family.detections[0].id].plot(equal_scale=False, size=(800, 600))



