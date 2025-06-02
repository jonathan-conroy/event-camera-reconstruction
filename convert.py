import h5py
import pandas as pd
import sys
import noise2image.h5 as h5
import numpy as np

fpath = sys.argv[1]
output_type = sys.argv[2]

f = h5py.File(fpath, "r")
events = pd.DataFrame(f["CD"]["events"][:])
print(events)

if output_type == "txt":
    events['t'] = events['t']/1000000
    events = events[['t','x','y','p']]
    fpath_out = fpath[:-4] + "txt"
    events.to_csv(fpath_out, sep=" ", index=False, header = ["1280", "720", None, None])

elif output_type == "feather":
    events['t'] = events['t']/1000000
    events = events[['t','x','y','p']]
    fpath_out = fpath[:-4] + "feather"
    events = events.astype({'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int16})
    events = events.rename(columns={"p":"pol"})
    events.to_feather(fpath_out)


elif output_type == "h5":
    events = events[['x','y','p', 't']]
    dtype = np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16})
    rec = events.to_records(index=False).astype(dtype)
    fpath_out = fpath[:-5] + "_noise.h5"
    writer = h5.H5EventsWriter(fpath_out, 1280, 720)
    writer.write(rec)
    writer.close()

else:
    events['t'] = events['t']/1000000
    events = events[['t','x','y','p']]
    fpath_out = fpath[:-4] + output_type
    events.to_csv(fpath_out, sep=" ", index=False, header = ["1280", "720", None, None],
                  compression=output_type)
