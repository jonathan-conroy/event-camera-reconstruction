import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import noise2image.h5 as h5
import random
import h5py
import cv2

#######################
### Find the square ###
#######################

fname = "reconstruction\eraser-over-square.hdf5"
f = h5py.File(fname, "r")
events = pd.DataFrame(f["CD"]["events"][:])
print("Loaded events...")
events['t'] = events['t']/1000000
events = events[['t','x','y','p']]

start_time = events['t'].iloc[0]
end_time = events['t'].iloc[-1]

df = events[events["t"] < start_time + 1000000/1000000] #1 second of static at start
h = 720
w = 1280

values_T, xedges, yedges = np.histogram2d(df["x"], df["y"], 
                              bins=(int(w/20), int(h/20)), 
                              range=[(0,w-1),(0,h-1)])
values = values_T.T

plt.imshow(values, cmap="gray"); plt.show()

values = cv2.bilateralFilter(values.astype(np.float32),5, 200, 200)

plt.imshow(values, cmap="gray"); plt.show()


print("Computed histogram...")
img = np.ones((h,w), dtype=np.float64)
THRESHOLD = np.average(values) * 1.5
for ind_x, _ in enumerate(xedges[:-1]):
    for ind_y, _ in enumerate(yedges[:-1]):
        if values[ind_y, ind_x] > THRESHOLD:
            xstart = int(xedges[ind_x])
            xend = int(xedges[ind_x + 1])
            ystart = int(yedges[ind_y])
            yend = int(yedges[ind_y + 1])
            if ((xstart > 200) and (xend < w - 200) and (ystart > 100) and (yend < h-100)):
                img[ystart:yend, xstart:xend] = 0


plt.imshow(img, cmap="gray");plt.show()


black_y, black_x = np.where(img == 0)
num_black = len(black_x)
static_img = img


###################
### Motion mask ###
###################
increment = 50000 # = 0.01 second

print("-------- Motion mask -----")
events["timeWindow"] = (events["t"]/increment * 1000000).astype(int)

def deltas(df):
    print(df["timeWindow"].iloc[0])
    df_plus = df[df["p"] == 1]
    df_minus = df[df["p"] == 0]
    values_plus, _, _ = np.histogram2d(df_plus["x"], df_plus["y"], bins=(w,h), range=[(0,w-1), (0,h-1)])
    values_plus = values_plus.T
    values_minus, _, _ = np.histogram2d(df_minus["x"], df_minus["y"], bins=(w,h), range=[(0,w-1), (0,h-1)])
    values_minus = values_minus.T
    return values_plus - values_minus

print("Grouping by time...")
delta_per_time = events.groupby("timeWindow").apply(deltas)


toAddEvents = []

print("Computing estimates...")
estimates_per_time = np.cumsum([x for x in delta_per_time], axis = 0)
estimates_per_time = estimates_per_time/np.max(estimates_per_time)
for i, estimate in enumerate(estimates_per_time):
    if i % 10 == 0:
        print(i)
        plt.imshow(estimate, cmap="gray")
        plt.show()
    

"""
def sufficient_plus(t, events_df):
    df = events_df[(events_df["t"] >= t) & (events_df["t"] < t + increment/1000000)]
    df_plus = df[df["p"] == 1]

    values_plus_T, xedges, yedges = np.histogram2d(df_plus["x"], df_plus["y"], 
                                bins=(int(w/10), int(h/10)), 
                                range=[(0,w-1),(0,h-1)])
    values = values_plus_T.T

    img = np.zeros((h,w), dtype=np.float64)
    THRESHOLD = 50
    for ind_x, _ in enumerate(xedges[:-1]):
        for ind_y, _ in enumerate(yedges[:-1]):
            if values[ind_y, ind_x] > THRESHOLD:
                xstart = int(xedges[ind_x])
                xend = int(xedges[ind_x + 1])
                ystart = int(yedges[ind_y])
                yend = int(yedges[ind_y + 1])
                img[ystart:yend, xstart:xend] = 1
    return img
"""

ts = np.arange(start_time, end_time, increment)
eventsToAdd = []
old_img = np.zeros((h,w))
for t in ts:
    print(t)
    img = sufficiedent_plus(t/1000000, events)
    to_add_y, to_add_x = np.where((static_img == 0) & (img == 1) & (old_img == 0))
    img = np.zeros((h,w), dtype=np.float64)
    img[to_add_y, to_add_x] = 1.0
    # plt.imshow(img, cmap="gray"); plt.show()
    to_add = zip(to_add_x, to_add_y)
    for (x,y) in to_add:
        actuallyAdd = np.random.randint(0,2)
        if True:
            times = np.random.randint(t - increment, t-increment + 1000, (10,))
            for tAdd in times:
                eventsToAdd += [(tAdd / 1000000, x, y, 0)]
    old_img = img

    # plt.imshow(img, cmap="gray"); plt.show()

print("Sorting...")
toAdd = pd.DataFrame(eventsToAdd, columns=events.columns)
print(toAdd)
print("Done w/ toAdd")
newEvents = pd.concat([toAdd, events], ignore_index=True)
newEvents.sort_values(by="t", inplace=True)
print(newEvents)
newEvents.to_feather("reconstruction\handcraft-eraser.feather")

####################################################
# The original handcrafting, to show the square... #
####################################################
exit()

tOffset = 2200000
for t in range(tOffset, tOffset + 2000):
    # In the handcrafted example, this was instead a grid of indices sparsified to 50x50?
    black_ind = np.random.randint(0, num_black, (int(num_black/50), ))
    for i in black_ind:
        x = black_x[i]
        y = black_y[i]
        events += [(t/1000000, x, y, 0)]

"""
for t in range(0,5000000, 100):
    x = random.randint(0,1280-1)
    y = random.randint(0,720-1)
    p = random.randint(0,1)
    events += [(t/1000000,x,y,p)]
"""

fname = "reconstruction\monitor-eraser-short.txt"
df_eraser = pd.read_csv(fname, skiprows=1, sep=" ", header=None, names = ["t", "x", "y", "p"])
print(df_eraser)

df_static = pd.DataFrame(events, columns=["t","x","y","p"])
print(df_static)

df = pd.concat([df_static, df_eraser])
df.sort_values(by = "t", inplace=True)
print(df)

fpath_out = "reconstruction\eraser-prefix.txt"
df.to_csv(fpath_out, sep=" ", index=False, header = ["1280", "720", None, None])
