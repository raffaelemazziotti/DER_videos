# Direct Emotional Response (DER) Video Dataset

A dataset of synchronized widefield behavioral recordings collected from head-fixed mice during the Direct Emotional Response (DER) paradigm.  
The dataset includes video previews, movement matrices, trigger visualizations, and metadata for every recording.

## Overview

This dataset consists of widefield behavioral videos acquired while mice were exposed to controlled emotional stimuli. Each video contains:

- A 30+ minute session per subject
- A side-view camera capturing the mouse’s behavioral responses
- Six shock intensities, each repeated six times
- 30 trials per subject
- A trigger signal encoded directly in the video pixel data
- Preprocessed materials: GIF previews, movement matrices, trigger visualization plots, and structured metadata

All files are accompanied by a `video_info.json` metadata file containing frame count, FPS, duration, resolution, and number of detected trials.

## Subjects and Recording Structure

- Total subjects: 6
- Trials per subject: 30
- Trial design:
  - 5 stimulus intensities
  - 6 repetitions per intensity
- Recording duration: ~30 minutes per subject
- Spatial alignment:
  All videos are aligned so that the mouse’s eye appears in the same image location across recordings.

## Trigger Encoding

Each video encodes the stimulation trigger in the top-left pixel of every frame.

- Pixel value > 200 → trigger ON
- Pixel value ≤ 200 → trigger OFF

These values form the binary trigger vector used to detect trial onset timing.  
A global trigger visualization plot is available:

During preprocessing, the first 10×10 pixels were removed from the analysis region to eliminate the trigger encoding.


## Metadata (`video_info.json`)

For each recording, the following information is stored:

- fps
- total_frames
- resolution
- duration_sec
- duration_min
- num_trials
- first_trigger_frame
- last_trigger_frame
- intensity_counts (dict)
- valid_trials
- invalid_trials


Example:

```json
{
  "2024-02-23_08-51-34-ECO_3_000_M-der": {
    "fps": 60,
    "total_frames": 112000,
    "resolution": [1280, 1024],
    "duration_min": 31.1,
    "num_trials": 30,
    "first_trigger_frame": 5304,
    "last_trigger_frame": 44537,
    "intensity_counts": {
            "0": 6,
            "1": 6,
            "2": 6,
            "3": 6,
            "4": 6
        },
    "valid_trials": 13,
    "invalid_trials": 17
  }
}
```

## Dataset Structure

```
res/
│
├── upright/
│     ├── subject1_video.avi
│     ├── subject2_video.avi
│     └── ...
│
└── scrambled/
      ├── subject1_video_scrambled.avi
      ├── subject2_video_scrambled.avi
      └── ...               
```

## How to Use the Dataset in Python
The VideoManagerArray class provides a unified interface to load, and visualize and use behavioral video 
trials from multiple subjects. It automatically loads videos from upright/ and scrambled/ subfolders 
inside a chosen dataset folder.
``` python
from video_lib import VideoManagerArray
vmh = VideoManagerArray() # if the folder is in the same directory
```
or
``` python
from video_lib import VideoManagerArray
vmh = VideoManagerArray(folder="/path/to/my_dataset") # to specify the dir
```

In the class a trial is defined as a tuple:
``` python
trl = (sub,trial_num, intensity, if_phase_scrambled,if_flipped)
``` 
The trial (0,4,4, False, False) means subject at index 0, trial number 4, intensity level 4 (500uA), not phase scrambled, not inverted
The trial (1,3,3, True, False) means subject at index 1, trial number 3, intensity level 3 (300uA), phase scrambled applied, not inverted
The trial (3,5,0, False, True) means subject at index 0, trial number 5, intensity level 3 (0uA), not phase scrambled,  vertically inverted
``` python
movie = vmh.get_trial_movie((3,5,0, False, True))

``` 


## OS Compatibility
Tested only on Windows 10

## Citation

### Zenodo 
Caldarelli, M., Papini, E. M., Pizzorusso, T., & Mazziotti, R. (2025). Direct Emotional Response Videos for Vicarious Emotional Processing in Mice [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17661945

### Preprint (BioRxiv): 
[https://doi.org/10.1101/2024.11.20.624327 ](https://doi.org/10.1101/2024.11.20.624327)

## Contact

raffaelemario.mazziotti [at] unifi [dot] it
