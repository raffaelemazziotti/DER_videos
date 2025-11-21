# Direct Emotional Response (DER) Video Dataset
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17661945.svg)](https://doi.org/10.5281/zenodo.17661945)
[![bioRxiv Preprint](https://img.shields.io/badge/bioRxiv-10.1101%2F2024.11.20.624327-red.svg)](https://doi.org/10.1101/2024.11.20.624327)

A curated collection of high-resolution behavioral recordings from head-fixed mice undergoing the Direct Emotional Response (DER) paradigm. Each video contains embedded stimulation timing and detailed metadata.

## Overview

This dataset includes side-view recordings of mice exposed to controlled emotional stimulation (tail shock). Each recording provides:

- A continuous session >30 minutes (habituation + stimulation protocol)
- Stable side-view imaging of facial and body movements
- Five shock intensities, each repeated six times (1-minute intertrial interval)
- 30 total trials per subject
- Frame-level trigger signals embedded in the video and as `.npy` files
- Additional processed materials: GIF previews, motion matrices, trigger plots, and structured metadata

All metadata for each session are stored in `video_info.json`.

## Subjects and Recording Structure

- **Subjects:** 6
- **Trials per subject:** 30
- **Trial design:**
  - 5 stimulation intensities
  - 6 repetitions per intensity
- **Recording duration:** approximately 30 minutes per subject
- **Spatial alignment:** All videos are registered so that the mouse’s eye appears at a fixed 
position across recordings.

## Trigger Encoding

Stimulation timing is encoded in the **top-left pixel** of each video frame:

- Pixel value > 200 → **trigger ON**
- Pixel value ≤ 200 → **trigger OFF**

A matching `.npy` file containing the trigger vector is also included for each recording.

## Folder Structure and Contents

The `.zip` dataset contains two primary folders:

### upright/
Unaltered videos recorded during the DER experiment.

### scrambled/
Fourier phase-scrambled versions of the same videos.
These preserve luminance and motion statistics but remove semantic visual content.

Each folder contains:

- `.avi` files: the video recordings
- `.npy` files: extracted trigger timings
- An `.xlsx` file: trial-level quality annotations (valid or invalid, based on movements thresholds during the prestimulus)

## Metadata (`video_info.json`)

Metadata fields include:

- fps
- total_frames
- resolution
- duration_sec
- duration_min
- num_trials
- first_trigger_frame
- last_trigger_frame
- intensity_counts
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

## Directory Structure

```
res/
│
├── upright/
│     ├── subject1.avi
│     ├── subject1_triggers.npy
│     └── ...
│
└── scrambled/
      ├── subject1_scrambled.avi
      ├── subject1_scrambled_triggers.npy
      └── ...
```

## Using the Dataset in Python

The `VideoManagerArray` class provides a unified interface for loading, previewing, and retrieving trials.
It automatically detects `upright/` and `scrambled/` subfolders.

```python
from video_lib import VideoManagerArray
vmh = VideoManagerArray()
```

or:

```python
vmh = VideoManagerArray(folder="/path/to/my_dataset")
```

A trial is represented as:

```python
(sub, trial_num, intensity, is_scrambled, is_flipped)
```

Examples:

- `(0, 4, 4, False, False)` → subject 0, trial 4, intensity 4 (500 µA), upright video
- `(1, 3, 3, True, False)` → subject 1, trial 3, intensity 3 (300 µA), scrambled video
- `(3, 5, 0, False, True)` → subject 3, trial 5, baseline intensity, flipped vertically

Example usage:

```python
movie, trl = vmh.get_trial_movie((3, 5, 0, False, True))
vmh.preview(movie)
```

## Example Script

`psychopy_example.py` contains a minimal working example for using the videos as stimuli for the Vicarious Emotional Response task.
It requires Psychopy: https://www.psychopy.org/download.html

## OS Compatibility

Tested on **Windows 10**.

## Citation

### Zenodo
Caldarelli, M., Papini, E. M., Pizzorusso, T., & Mazziotti, R. (2025). *Direct Emotional Response Videos for Vicarious Emotional Processing in Mice* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17661945

### Preprint
https://doi.org/10.1101/2024.11.20.624327

