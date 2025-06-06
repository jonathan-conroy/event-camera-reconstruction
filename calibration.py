import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import noise2image.h5 as h5
import h5py
import cv2

def load_events(fname):
    f = h5py.File(fname, "r")
    events = pd.DataFrame(f["CD"]["events"][:])
    events['t'] = events['t']/1000000
    events = events[['t','x','y','p']]
    return events

fname = "reconstruction/final-white-3.hdf5"
events = load_events(fname)
white_time = events["t"].iloc[-1] - events["t"].iloc[0]

grayscale_calib = load_events("reconstruction/final-grayscale-calib.hdf5")
gray_time = grayscale_calib["t"].iloc[-1] - grayscale_calib["t"].iloc[0]
print("GRAY TIME:", gray_time)

h = 720
w = 1280

values_T, _, _ = np.histogram2d(events["x"], events["y"], 
                                bins=(int(w), int(h)), 
                                range=[(0,w),(0,h)])
values = values_T.T
smooth_val = cv2.GaussianBlur(values.astype(np.float32), (0,0), 50)
# smooth_val = smooth_val/np.mean(smooth_val)

plt.imshow(smooth_val, cmap="gray"); plt.axis("off"); plt.show()
print("Scaling down by ", white_time)
smooth_val = smooth_val/white_time
np.save("final-white-calibration.npy", smooth_val)


gray_T, _, _ = np.histogram2d(grayscale_calib["x"], grayscale_calib["y"], 
                                bins=(int(w), int(h)), 
                                range=[(0,w),(0,h)])
gray = gray_T.T
gray = cv2.GaussianBlur(gray.astype(np.float32), (0,0), 7)
# print(np.max(gray))
# normal = np.max(white2)
plt.imshow(gray, cmap="gray"); plt.show()
plt.imshow(gray- ((gray_time)*smooth_val), cmap="gray"); plt.show()
gray = gray- ((gray_time)*smooth_val)


monitor_buckets = [[225, 265, 90, 120],
                    [380, 420, 90, 120],
                    [560, 610, 90, 120],
                    [750, 800, 95, 130],
                    [220, 260, 290, 330],
                    [380, 420, 290, 330],
                    [570, 620, 290, 330],
                    [760, 800, 295, 335],
                    [220, 260, 500, 540],
                    [390, 440, 500, 540],
                    [570, 630, 500, 540],
                    [750, 800, 500, 540]]
means = [np.mean(gray[ymin:ymax, xmin:xmax]) for (xmin, xmax, ymin, ymax) in monitor_buckets]
plt.plot([x for x in range(len(means))], means)
plt.show()

def deltas(df):
    df_plus = df[df["p"] == 1]
    df_minus = df[df["p"] == 0]
    values_plus, _, _ = np.histogram2d(df_plus["x"], df_plus["y"], bins=(w,h), range=[(0,w), (0,h)])
    values_plus = values_plus.T
    values_minus, _, _ = np.histogram2d(df_minus["x"], df_minus["y"], bins=(w,h), range=[(0,w), (0,h)])
    values_minus = values_minus.T
    values = values_plus - values_minus
    # plt.imshow(values/np.max(values), cmap="gray"); plt.show()
    return values

events = load_events("reconstruction/final-grayscale-motion.hdf5")
start_time = events["t"].iloc[0]
end_time = events["t"].iloc[-1]

moving_buckets = \
    [[210, 260, 155, 175],
     [380, 430, 155, 175],
     [575, 620, 155, 175],
     [750, 800, 160, 185],
     [200, 250, 355, 380],
     [380, 430, 365, 385],
     [570, 620, 370, 390],
     [760, 810, 375, 395],
     [200, 260, 580, 605],
     [390, 440, 580, 605],
     [560, 630, 580, 605],
     [760, 810, 570, 595]]

total_delta = deltas(events)
event_changes = [-1 * np.mean(total_delta[ymin:ymax, xmin:xmax]) for (xmin, xmax, ymin, ymax) in moving_buckets]
# plt.plot(event_changes, means); plt.show()

slope, intercept = np.polyfit(means[2:], event_changes[2:], 1)
print("slope", slope)
print("intercept", intercept)

plt.plot(means, event_changes)


plt.ylabel("Change in brightness (from white background)")
plt.xlabel("# noise events per pixel")
plt.show()

