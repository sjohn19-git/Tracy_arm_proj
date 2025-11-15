############

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read
import os

client = Client("IRIS")
t1 = UTCDateTime("2025-07-01T00:00:00")
t2 = UTCDateTime("2025-08-15T00:00:00")

outdir = "/Users/sebinjohn/Tracy_arm/data/seismic"
os.makedirs(outdir, exist_ok=True)

stations_info = [
    ("AK", "BCP", "", "BHZ"),
    ("AK", "PIN", "", "BHZ"),
    ("AK", "PNL", "", "BHZ"),
    ("AK", "R32K", "", "BHE"),
    ("AK", "R32K", "", "BHN"),
    ("AK", "R32K", "", "BHZ"),
    ("AK", "S31K", "", "BHE"),
    ("AK", "S31K", "", "BHN"),
    ("AK", "S31K", "", "BHZ"),
    ("AK", "S32K", "", "BHE"),
    ("AK", "S32K", "", "BHN"),
    ("AK", "S32K", "", "BHZ"),
    ("AK", "U33K", "", "BHE"),
    ("AK", "U33K", "", "BHN"),
    ("AK", "U33K", "", "BHZ"),
    ("AK", "V35K", "", "BHZ"),
    ("AT", "CRAG", "", "BHN"),
    ("AT", "SIT", "", "BHE"),
    ("AT", "SIT", "", "BHN"),
    ("AT", "SIT", "", "BHZ"),
    ("AV", "EDCR", "", "BHE"),
    ("AV", "EDCR", "", "BHN"),
    ("AV", "EDCR", "", "BHZ"),
    ("CN", "DIB", "", "HHZ"),
    ("CN", "DLBC", "", "HHN"),
    ("CN", "DLBC", "", "HHZ"),
    ("CN", "HYT", "", "HHZ"),
    ("CN", "PLBC", "", "HHN"),
    ("CN", "PLBC", "", "HHZ"),
    ("CN", "WHY", "", "HHZ"),
    ("CN", "YUK7", "", "HHZ"),
]

# Loop over stations
for net, sta, loc, chan in stations_info:
    current_day = t1
    while current_day < t2:
        next_day = current_day + 86400  # 1 day
        day_end = min(next_day, t2)

        filename = f"{net}.{sta}.{loc}.{chan}.{current_day.date}.mseed"
        outpath = os.path.join(outdir, filename)

        # Expected number of samples = duration * sampling_rate
        expected_nsamples = None

        if os.path.exists(outpath):
            try:
                st = read(outpath)
                tr = st[0]
                sr = tr.stats.sampling_rate
                expected_nsamples = int((day_end - current_day) * sr)
                actual_nsamples = tr.stats.npts

                if abs(actual_nsamples - expected_nsamples) < sr:  # within 1s worth of samples
                    print(f"Skipping {outpath} (already exists, nsamples OK).")
                    current_day = next_day
                    continue
                else:
                    print(f"Redownloading {outpath} (nsamples mismatch: {actual_nsamples} vs {expected_nsamples}).")
                    os.remove(outpath)
            except Exception as e:
                print(f"Problem reading {outpath}, redownloading: {e}")
                os.remove(outpath)

        # Download if missing or invalid
        try:
            print(f"Downloading {net}.{sta}.{loc}.{chan} from {current_day} to {day_end}...")
            st = client.get_waveforms(
                network=net,
                station=sta,
                location=loc,
                channel=chan,
                starttime=current_day,
                endtime=day_end,
                attach_response=True
            )
            st.write(outpath, format="MSEED")
            print(f"Saved to {outpath}")
        except Exception as e:
            print(f"Failed to download {net}.{sta}.{loc}.{chan} for {current_day.date}: {e}")

        current_day = next_day
