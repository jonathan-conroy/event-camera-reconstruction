import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import noise2image.h5 as h5
import h5py

fname = "reconstruction\monitor-eraser-short.hdf5"
f = h5py.File(fname, "r")
events = pd.DataFrame(f["CD"]["events"][:])
events['t'] = events['t']/1000000
events = events[['t','x','y','p']]


# df = pd.read_csv(fname, skiprows=1, sep=" ", header=None, names = ["t", "x", "y", "p"])
h = 720
w = 1280

increment = 10000


def sufficient_plus(t, events_df):
    df = events_df[(events_df["t"] >= t) & (events_df["t"] < t + increment/1000000)]
    df_plus = df[df["p"] == 1]

    values_plus_T, xedges, yedges = np.histogram2d(df_plus["x"], df_plus["y"], 
                                bins=(int(w/10), int(h/10)), 
                                range=[(0,w-1),(0,h-1)])
    values = values_plus_T.T

    img = np.zeros((h,w), dtype=np.float64)
    THRESHOLD = 75
    for ind_x, _ in enumerate(xedges[:-1]):
        for ind_y, _ in enumerate(yedges[:-1]):
            if values[ind_y, ind_x] > THRESHOLD:
                xstart = int(xedges[ind_x])
                xend = int(xedges[ind_x + 1])
                ystart = int(yedges[ind_y])
                yend = int(yedges[ind_y + 1])
                img[ystart:yend, xstart:xend] = 1
    return img
    
ts = np.arange(2118000, 2400000, increment)
for t in ts:
    print(t)
    img = sufficient_plus(t/1000000, events)
    plt.imshow(img, cmap="gray"); plt.show()

exit()
# for _, row in events.iterrows():
#     if row["p"] == 1:
#         plus_counts[row["y"], row["x"]] += [row["t"]]
# print(plus_counts)
# exit()


def get_counts(time, events_df):
    df = events_df[(events_df["t"] >= time) & (events_df["t"] < time + increment/1000000)]
    df_plus = df[df["p"] == 1]
    df_minus = df[df["p"] == 0]

    values_plus_T, _, _ = np.histogram2d(df_plus["x"], df_plus["y"], 
                                bins=(int(w/10), int(h/10)), 
                                range=[(0,w-1),(0,h-1)])
    values_plus = values_plus_T.T

    values_minus_T, _, _ = np.histogram2d(df_minus["x"], df_minus["y"], 
                                bins=(int(w/10), int(h/10)), 
                                range=[(0,w-1),(0,h-1)])
    values_minus = values_minus_T.T
    values = values_plus - values_minus

    # img = np.ones((h,w), dtype=np.float64)
    # THRESHOLD = np.average(values) * 2.5
    # for ind_x, _ in enumerate(xedges[:-1]):
    #     for ind_y, _ in enumerate(yedges[:-1]):
    #         if values[ind_y, ind_x] > THRESHOLD:
    #             xstart = int(xedges[ind_x])
    #             xend = int(xedges[ind_x + 1])
    #             ystart = int(yedges[ind_y])
    #             yend = int(yedges[ind_y + 1])
    #             img[ystart:yend, xstart:xend] = 0

    # black_patch = ((760, 780), (330, 350))
    # white_patch = ((810, 830), (330, 350))
    # return np.avg(img[330:350, 760:780]), np.avg(img[330:350, 810:830])

    # plt.imshow(values); plt.show()
    return values

black_arr = []
white_arr = []
ts = np.arange(2118000, 2400000, increment)
for t in ts:
    print(t)
    values = get_counts(t/1000000, events)
    plt.imshow(values, cmap="gray"); plt.show()
    # black, white = get_counts(t/1000000, events)
    # black_arr += [black]
    # white_arr += [white]

plt.plot(ts, np.cumsum(white_arr))
plt.plot(ts, np.cumsum(black_arr))
plt.show()