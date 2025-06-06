import numpy as np
import pandas as pd
import h5py
import cv2

f_input = "reconstruction/falling.hdf5"

fname = f_input
f = h5py.File(fname, "r")
events = pd.DataFrame(f["CD"]["events"][:])
print("Loaded events...")
events = events[['t','x','y','p']]

increment = 10000 # = 0.01 second
THRESH = 2
events["timeWindow"] = (events["t"]/increment).astype(int)
events["isNoise"] = np.zeros(len(events),)
h = 720
w = 1280

def eventCounts(df):
    values, _, _ = np.histogram2d(df["x"], df["y"], bins=(w,h), range=[(0,w), (0,h)])
    return values.T

dfs_by_time = events.groupby("timeWindow")

output_dfs = []
for _, currDf in dfs_by_time:
    time = currDf["timeWindow"].iloc[0] * increment
    print(time)

    counts = eventCounts(currDf)
    density = cv2.blur(counts, (5,5)) * (5*5)
    notFlicker = cv2.filter2D(counts, -1, np.array([[1,1,1],
                                                    [1,0,1],
                                                    [1,1,1]]))
    isNoise = np.where((counts != 0) & (density < THRESH) & (notFlicker == 0))
    # isNoise = np.where((counts != 0) & (density < THRESH))
    dfCoords = pd.Series([tuple(coord) for coord in zip(currDf.y, currDf.x)])
    noiseCoords = [(int(y), int(x)) for y,x in zip(*isNoise)]
    currDf.loc[dfCoords.isin(noiseCoords).to_numpy(), "isNoise"] = 1
    output_dfs += [currDf]

print("Done filtering noise")
output_df = pd.concat(output_dfs)
output_df['t'] = output_df['t']/1000000

for isNoise, df in output_df.groupby("isNoise"):
    name_suffix = "denoise" if isNoise == 0 else "noise"
    print("-----" + name_suffix + "-----")
    df = df.reset_index()
    print(df)
    file_out = f_input[:-5] + "-" + name_suffix + ".feather"
    df = df[["t", "x", "y", "p"]]
    df.to_feather(file_out)