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

```
graphs/trigger.svg
```

During preprocessing, the first 10×10 pixels were removed from the analysis region to eliminate the trigger encoding.

## Provided Files

### 1. GIF Previews (`GIFs/`)

For each subject:

```
GIFs/<recording_name>.gif
```

A compact visualization summarizing all trials.

Additional example with trigger pixel visible:

```
GIFs/trial_4_full.gif
```

### 2. Movement Matrices (`graphs/`)

For each subject:

```
graphs/<recording_name>_movement_matrix.svg
```

These SVG plots show frame-by-frame movement estimation across all trials, aligned to detected triggers.

### 3. Trigger Overview

```
graphs/trigger.svg
```

Plot of the upper-left pixel intensity across frames, with triggers marked by X.

### 4. Metadata (`video_info.json`)

For each recording, the following information is stored:

- fps
- total_frames
- resolution
- duration_sec
- duration_min
- num_trials

Example:

```json
{
  "2024-02-23_08-51-34-ECO_3_000_M-der": {
    "fps": 60,
    "total_frames": 112000,
    "resolution": [1280, 1024],
    "duration_min": 31.1,
    "num_trials": 30
  }
}
```

## Folder Structure

```
├── index.html               
├── style.css                
├── script.js                
├── video_info.json          
│
├── GIFs/
│     ├── <recording>.gif
│     ├── trial_4_full.gif
│
├── graphs/
│     ├── <recording>_movement_matrix.svg
│     ├── trigger.svg
│
└── README.md                
```

## How to Use the Dataset

### Interactive Browser

A GitHub Pages site allows users to browse:

- Recording previews
- Movement matrices
- Metadata
- Trigger visualization

### Programmatic Access

```python
import json

with open("video_info.json", "r") as f:
    info = json.load(f)

print(info.keys())
```

## Data Availability

Zenodo (full dataset): DOI placeholder  
GitHub (browser + tools): Repository placeholder

## Citation

Mazziotti, R. M., Direct Emotional Response Behavioral Dataset, Zenodo (2024).  
DOI: placeholder

Preprint (BioRxiv): DOI placeholder

## Contact

raffaelemario.mazziotti [at] unifi [dot] it
