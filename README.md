# Scene reconstruction from a stationary event camera  
Computational Photography COSC 273   
Jonathan Conroy   
Spring 2025  

### Code structure
Data from the event camera should be collected using Metavision software and saved as an .hdf5 file in the directory /reconstruction.  
After obtaining this event data, run `convert.py` to convert into a .txt or .feather file,
then run `reconstruct_video.ps1` to reconstruct the dynamic components of the scene using E2VID.
The script `handcraft_events.py` could be used to inject events to the event stream to minimize ghosting (before E2VID reconstruction).

Run `python merge.py` after running E2VID, to estimate the static background and merge it with the E2VID reconstruction.

The dependencies (including those for E2VID and Noise2Image) are listed in `requirements.txt`

**List of scripts:**
- `convert.py:`
   Converts .hdf5 file provided by Metavision software to either .txt
   or .feather files.
   (E2VID was modified to accept .feather files, which are faster than .txt)
   Can be run with `python convert.py <input file> <txt/feather>
- `denoise.py:`
    Implements the denoising algorithm of Feng et. al.
    Modify the variables at the start of the file to select input/output filenames
    Run with `python denoise.py`
- `calibration.py:`
    Computes and stores the background noise on a white monitor,
    and displays the relation between noise events and brightness on a color checker.
    Run with `python calibration.py'
- `loadStatic.py:`
    Call `import loadStatic.py` to gain access to `loadStatic.static_img`,
    representing the relative brightness of the static scene.
    Modify the variables at the start of the file to select input file.
- `naiveIntegration.py:`
    Can be run as `python naiveIntegration.py`, after modifying variables at start of file
    to select the input. Displays the result of summing over all events, after denoising.
- `merge.py:`
    Run as `python merge.py` after modifying variables at start of file
    to select the input and output.
    Merges the static background and dynamic foreground, and
    outputs a set of images based on a weighted average according to a motion mask.
- `handcraft_events.py:`
    Outputs a file containing the event stream, plus some inejcted events to minimize
    ghosting effects from E2VID.
    Injected events are computed from a motion mask + the static scene reconstruction.
    Run with `python handcraft_events.py`
- `reconstruct_video.ps1:`
    A script which runs E2VID
    Run as `reconstruct_video.ps1 <dataset-name>`, assuming that the .feather file of events
    is stored at `\reconstruction\<dataset-name>.feather`
