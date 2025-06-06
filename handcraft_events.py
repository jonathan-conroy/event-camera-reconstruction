import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import noise2image.h5 as h5
import random
import h5py
import cv2

f_input = "reconstruction/final-falling.hdf5"
f_static = "reconstruction/final-falling-static.hdf5"
f_out = "reconstruction/final-correction-falling.feather"
# staticTimesFile = "reconstruction/times-falling-static.npy"
# activeTimesFile = "reconstruction/times-falling-active.npy"
activeMaskFile = "reconstruction/activeMask-final-falling.npy"
# timestampsFile = "reconstruction/timestamps-falling.npy"

increment = 10000 # = 0.005 second
STATIC_WAIT_TIME = 1000000


#############################
### Load the static scene ###
############################
import loadStatic
static_img = loadStatic.static_img

fname = f_input
f = h5py.File(fname, "r")
events = pd.DataFrame(f["CD"]["events"][:])
print("Loaded events...")
events = events[['t','x','y','p']]

start_time = events['t'].iloc[0]
end_time = events['t'].iloc[-1]

h = 720
w = 1280

###################
### Motion mask ###
###################

print("-------- Motion mask -----")
events["timeWindow"] = (events["t"]/increment).astype(int)

def deltas(df):
    df_plus = df[df["p"] == 1]
    df_minus = df[df["p"] == 0]
    values_plus, _, _ = np.histogram2d(df_plus["x"], df_plus["y"], bins=(w,h), range=[(0,w), (0,h)])
    values_plus = values_plus.T
    values_minus, _, _ = np.histogram2d(df_minus["x"], df_minus["y"], bins=(w,h), range=[(0,w), (0,h)])
    values_minus = values_minus.T
    values = values_plus + values_minus
    return values, (values_plus - values_minus)

print("Grouping by time...")
dfs_by_time = events.groupby("timeWindow")
activeByTime = np.zeros((len(dfs_by_time),h,w))

## Motion mask, based on time frame
eventDfsToAdd = []
eventsToAdd = []
times = []
currActive = np.zeros((h,w))
timeSinceLastDelta = np.zeros((h,w))
correctionPolarity = np.zeros((h,w))
runningImg = np.zeros((h, w))
NUM_RUN_DELTA = 1
runningDeltas = np.zeros((NUM_RUN_DELTA, h, w))

MORPH_SIZE = 50
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH_SIZE,MORPH_SIZE))
for i, (_, currDf) in enumerate(dfs_by_time):
    time = currDf["timeWindow"].iloc[0] * increment
    print(time)
    times += [time]
    timeSinceLastDelta += 1
    deltaEvents, delta = deltas(currDf)
    smooth_delta = np.zeros((h + MORPH_SIZE + 1, w + MORPH_SIZE + 1), dtype=np.float32)
    smooth_delta[0:h, 0:w] = deltaEvents
    smooth_delta = cv2.blur(deltaEvents.astype(np.float32), (20,20)) * (20*20)
    runningDeltas[i%NUM_RUN_DELTA] = smooth_delta

    thresh_naive = np.zeros((h,w))
    thresh_naive[smooth_delta > 800] = 1

    if i == 0:
        continue
    activeByTime[i] = cv2.morphologyEx(smooth_delta, cv2.MORPH_CLOSE, kernel)[0:h, 0:w]

    thresholdEventCount = np.zeros((h,w), dtype=np.uint8)
    thresholdEventCount[activeByTime[i] > 800] = 1
    _, thresholdEventCount, _, _ = cv2.floodFill(thresholdEventCount,None, (0,0), 1)
    thresholdEventCount = np.invert(thresholdEventCount.astype(np.bool))
    activeByTime[i][thresholdEventCount == True] = 800+1

    activeByTime[i] = cv2.GaussianBlur(activeByTime[i], (0, 0), 50)


    CHANG_THRESH = 800
    runningImg += delta
    currentlyActive = np.zeros((h,w))
    currentlyActive[activeByTime[i] > 400] = 1
    activeByTime[i] = currentlyActive

    if i < 1:
        continue
    ind = np.where((activeByTime[i] == 1) & (activeByTime[i-1] == 0))
    currActive[ind] = 1
    timeSinceLastDelta[ind] = 0

    """Inject events..."""
    for (y, x) in zip(*ind):
        num_entries = int(static_img[y,x])
        toAddTimes = np.random.randint(time - increment/2, time-increment/2 + 1000, (num_entries * 4,) )
        for t in toAddTimes:
            eventsToAdd += [(t, x, y, 0)]
    
    continue
    """ Experimentation: inject events to ``reset'' after the object passes"""
    # if i < 5:
    #     continue
    ind = np.where((np.sum(activeByTime[i-4:i,:,:], axis=0) == 0) & (activeByTime[i-5] > 800))
    # ind = np.where((timeSinceLastDelta * increment > STATIC_WAIT_TIME) &
    #                (currActive == 1))
    currActive[ind] = 0

    num_entries = len(ind[0]) * num_rep
    if num_entries > 0:
        # print(ind)
        toAddTimes = np.random.randint(time - STATIC_WAIT_TIME + 2*increment,
                                        time - STATIC_WAIT_TIME + 2*increment + 1000, 
                                        (num_entries,))
        ys = np.repeat(ind[0], num_rep)
        xs = np.repeat(ind[1], num_rep)
        ps = np.ones((num_entries,)) #np.repeat((correctionPolarity[ind] == 0).astype(int), num_rep)
        arrToAdd = np.array([toAddTimes, xs, ys, ps])
        dfToAdd = pd.DataFrame(arrToAdd.T, columns = ["t", "x", "y", "p"])
        eventDfsToAdd += [dfToAdd]

"""Experimentation: inject events to ``reset'' the scene at the end"""
# At the end, assume everything is static...
# ind = np.where(currActive == 1)
# currActive[ind] = 0

# num_rep = 100
# num_entries = len(ind[0]) * num_rep
# if num_entries > 0:
#     print("STATIC", ind)
#     randomOffsets = np.random.randint(2*increment, 2*increment + 1000, (num_entries,))
#     startingTimes = np.repeat(time - (timeSinceLastDelta[ind] * increment), num_rep)
#     print(startingTimes)
#     toAddTimes = startingTimes + randomOffsets
#     ys = np.repeat(ind[0], num_rep)
#     xs = np.repeat(ind[1], num_rep)
#     ps = np.ones((num_entries,)) #np.repeat((correctionPolarity[ind] == 0).astype(int), num_rep)
#     arrToAdd = np.array([toAddTimes, xs, ys, ps])
#     dfToAdd = pd.DataFrame(arrToAdd.T, columns = ["t", "x", "y", "p"])
#     eventDfsToAdd += [dfToAdd]


    # plt.imshow(currActive, cmap="gray"); plt.show()
events = events[["t", "x", "y", "p"]]
print("Sorting...")
newEvents = pd.DataFrame(eventsToAdd, columns=["t", "x", "y", "p"])#pd.concat([events, *eventDfsToAdd], ignore_index=True)
newEvents = pd.concat([events, newEvents], ignore_index=True)
newEvents.sort_values(by="t", inplace=True)
newEvents["t"] = newEvents["t"] / 1000000
print(newEvents)
newEvents.to_feather(f_out)
np.save(activeMaskFile, activeByTime)

"""
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
"""