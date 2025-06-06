import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import imageio.v3 as iio
import loadStatic
import glob
import cv2
import math
import scipy

activeMaskFile = "reconstruction/activeMask-final-falling.npy"
activeFolder = "reconstruction/final-falling/"
staticFile = "reconstruction/handcraft-static-falling/frame_0004476000.png"
outputFolder = "reconstruction/merged-final-falling/"

events = loadStatic.events
static_img = iio.imread(staticFile)#loadStatic.static_img
static_img = loadStatic.static_img
static_img = -1 * static_img
static_img = static_img - np.min(static_img)

masks = np.load(activeMaskFile)
print(masks)

increment = 5000
MORPH_SIZE = 100
THRESH_EVENT_COUNT = 5


plt.imshow(static_img); plt.show()

# Copied from E2VID resampling
def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array)) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
        return (idx - 1), array[idx - 1]
    else:
        return idx, array[idx]

images_paths = [f for f in glob.glob(activeFolder + "*.png", recursive=False)]
images_paths = sorted(images_paths)
stamps = np.loadtxt(activeFolder + "timestamps.txt")
stamps = np.sort(stamps)
###

images = np.array([iio.imread(p) for p in images_paths])
white_calibration_img = np.mean(images[0][0:200, 0:200])
print(white_calibration_img)
white_calibration_static = np.mean(static_img[0:200, 0:200])
# breakpoint()
static_img = (static_img * float(white_calibration_img/white_calibration_static)).astype(np.uint8)

start_time = 150130 # events["t"].iloc[0]
end_time = events["t"].iloc[-1]
h,w = images[0].shape

print("START", start_time)
print("EMD", end_time)
print("INCREMENT", increment)
print("Enumerated over", len(range(start_time, end_time, increment)))
print("masks:", masks.shape)

for i, t in enumerate(range(start_time, end_time, increment)):
    print("Currently at time", t/1000000)
    img_index, img_time = find_nearest(stamps, t/1000000)
    path_to_img = images_paths[img_index]

    currEvents = events[(events["t"] < img_time * 1000000) & (events["t"] > img_time * 1000000 - 2 * increment)]

    values_T, _, _ = np.histogram2d(currEvents["x"], currEvents["y"], 
                                bins=(int(w), int(h)), 
                                range=[(0,w),(0,h)])
    values = values_T.T
    

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH_SIZE,MORPH_SIZE))
    eventCountSmooth = values
    eventCountSmooth = cv2.morphologyEx(values, cv2.MORPH_CLOSE, kernel)[0:h, 0:w]
    # plt.show(eventCountSmooth, cmap="gray"); plt.axis("off"); plt.show()

    thresholdEventCount = np.zeros((h,w), dtype=np.uint8)
    thresholdEventCount[eventCountSmooth > THRESH_EVENT_COUNT] = 1
    _, thresholdEventCount, _, _ = cv2.floodFill(thresholdEventCount,None, (0,0), 1)
    thresholdEventCount = np.invert(thresholdEventCount.astype(np.bool))

    eventCountSmooth[thresholdEventCount == True] = THRESH_EVENT_COUNT * 2
    # plt.show(eventCountSmooth, cmap="gray"); plt.axis("off"); plt.show()


    eventCountSmooth = cv2.GaussianBlur(eventCountSmooth, (0, 0), 50)
    # plt.imshow((eventCountSmooth/2) ** 2, cmap="gray"); plt.axis("off"); plt.show()

    print("t = ", img_time, " at index ", img_index)

    mask = masks[i]
    # plt.imshow(mask, cmap="gray"); plt.show()
    activeImage = images[img_index]

    weightStatic = np.ones(activeImage.shape)
    weightActive = (eventCountSmooth/2) ** 2 #mask/500
    # plt.imshow(weightActive, cmap="gray"); plt.show()
    # print(weightActive)
    image = (activeImage * weightActive + static_img * weightStatic)/(weightActive + weightStatic)
    
    # image[ind] = static_img[ind]

    tmask = np.zeros(masks[i].shape)
    tmask[masks[i] < 1000] = 1
    # plt.imshow(weightActive, cmap="gray"); plt.show()
    # plt.imshow(image, cmap="gray"); plt.axis("off"); plt.show()
    fout = outputFolder + 'frame_{:010d}.png'.format(i)
    # breakpoint()
    iio.imwrite(fout, image.astype(np.uint8))

print(start_time)