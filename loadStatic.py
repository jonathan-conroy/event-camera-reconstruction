import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import noise2image.h5 as h5
import random
import h5py
import cv2

f_input = "reconstruction/falling-short.hdf5"#"reconstruction/final-falling.hdf5"
f_static ="reconstruction/falling-static.hdf5" # "reconstruction/final-falling-static.hdf5"
staticFileOutput = None #"reconstruction/final-handcraft-static-falling.feather"

increment = 10000 # = 0.005 second
STATIC_WAIT_TIME = 1000000


### Load events
fname = f_input
f = h5py.File(fname, "r")
events = pd.DataFrame(f["CD"]["events"][:])
print("Loaded events...")
events = events[['t','x','y','p']]

start_time = events['t'].iloc[0]
end_time = events['t'].iloc[-1]

#########################
### Load static scene ###
#########################

fname = f_static
f = h5py.File(fname, "r")
df = pd.DataFrame(f["CD"]["events"][:])
df = df[['t','x','y','p']]


h = 720
w = 1280

print(df)
img_T, xedges, yedges = np.histogram2d(df["x"], df["y"], 
                              bins=(int(w), int(h)), 
                              range=[(0,w),(0,h)])
img = img_T.T

img = cv2.GaussianBlur(img.astype(np.float32), (0,0), 7)
plt.imshow(img, cmap="gray"); plt.show()

whiteCalib = np.load("final-white-calibration.npy")

static_time = (df["t"].iloc[-1] - df["t"].iloc[0])/1000000
print("Static time", static_time)

img = img - static_time * whiteCalib


def noise2events(noise):
    if noise < 1:
        return 0
    else:
        return int(0.6006955733814385 * noise + 0.6357338809236908)

black_ind = np.where(img == 0)
black_y, black_x = np.where(img == 0)
num_black = len(black_x)
static_img = 0.6006955733814385 * img + 0.6357338809236908
plt.imshow(-1 * static_img, cmap="gray"); plt.axis("off"); plt.show()

if __name__ == "__main__":
    # Save a set of events, which (in principle) can be fed to E2VID to reconstruct the static scene
    staticEvents = []
    for x in range(w):
        for y in range(h):
            timesToAdd = np.random.randint(0, 1000, (noise2events(img[y,x]),))
            for t in timesToAdd:
                staticEvents += [(t/1000000, x, y, 0)]
    staticEvents = pd.DataFrame(staticEvents, columns=["t", "x", "y", "p"])
    staticEvents.sort_values(by="t", inplace=True)
    staticEvents.to_feather(staticFileOutput)