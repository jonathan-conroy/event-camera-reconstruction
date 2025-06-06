import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import noise2image.h5 as h5
import random
import h5py
import cv2

f_input = "reconstruction/falling-denoise.hdf5"
f_static = "reconstruction/falling-static.hdf5"
f_out = "reconstruction/correction-falling.feather"
staticTimesFile = "reconstruction/times-falling-static.npy"
activeTimesFile = "reconstruction/times-falling-active.npy"
activeMaskFile = "reconstruction/activeMask-falling.npy"

f_input_feather = "reconstruction/falling-denoise.feather"

increment = 10000 # = 0.005 second
show_static_img = False

#########################
### Load static image ###
#########################

if f_input_feather is None:
    fname = f_input
    f = h5py.File(fname, "r")
    events = pd.DataFrame(f["CD"]["events"][:])
    print("Loaded events...")
    events = events[['t','x','y','p']]
else:
    events = pd.read_feather(f_input_feather)
    events["t"] *= 1000000

start_time = events['t'].iloc[0]
end_time = events['t'].iloc[-1]

df = events[events["t"] < start_time + 1000000] # Assume 1 second of static scene at start
# ...unless there is an "static" file explicitly provided
if f_static is not None:
    fname = f_static
    f = h5py.File(fname, "r")
    df = pd.DataFrame(f["CD"]["events"][:])
    df = df[['t','x','y','p']]


h = 720
w = 1280

values_T, _, _ = np.histogram2d(df["x"], df["y"], 
                              bins=(int(w), int(h)), 
                              range=[(0,w),(0,h)])
values = values_T.T
values = cv2.blur(values, (10,10))

plt.imshow(values, cmap="gray"); plt.show()

# Manually correct for background noise (TEMP only)
correction = np.zeros(values.shape)
valh, valw = correction.shape
for x in range(valw):
    for y in range(valh):
        correction[y,x] = np.sqrt((x-(valw/2 - 10))**2 + (y-(valh/2))**2)/2
values = values - correction/500

values = cv2.bilateralFilter(values.astype(np.float32),5, 200, 200)


black_ind = np.where(values == 0)
black_y, black_x = np.where(values == 0)
num_black = len(black_x)
static_img = values

if not show_static_img:
    static_img = np.zeros((h,w))

###################
### Integration ###
###################

print("-------- Integration -----")
events["timeWindow"] = (events["t"]/increment).astype(int)

def deltas(df):
    df_plus = df[df["p"] == 1]
    df_minus = df[df["p"] == 0]
    values_plus, _, _ = np.histogram2d(df_plus["x"], df_plus["y"], bins=(w,h), range=[(0,w-1), (0,h-1)])
    values_plus = values_plus.T
    values_minus, _, _ = np.histogram2d(df_minus["x"], df_minus["y"], bins=(w,h), range=[(0,w-1), (0,h-1)])
    values_minus = values_minus.T
    values = values_plus - values_minus
    return values

print("Grouping by time...")
dfs_by_time = events.groupby("timeWindow")
activeByTime = np.zeros((len(dfs_by_time),h,w))

## Motion mask, based on time frame
runningImg = static_img

for i, (_, currDf) in enumerate(dfs_by_time):
    time = currDf["timeWindow"].iloc[0] * increment
    print(time)
    delta = deltas(currDf)
    smooth_delta = cv2.blur(delta.astype(np.float32), (20,20))# * (20*20)
    runningImg += delta
    if time < 5900000:
        continue
    plt.imshow(cv2.blur(runningImg, (20,20)), cmap="gray");plt.show()


