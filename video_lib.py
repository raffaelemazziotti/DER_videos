"""
video_lib
=========
A collection of tools for loading, managing, preprocessing, and presenting the
Direct Emotional Response (DER) behavioral videos.

Includes:
- TrgManager: Save/load trigger arrays (.npy)
- VideoManager: Low-level per-video movie extraction utilities
- VideoManagerArray: High-level experimental trial sequencing and access

This module is intended for behavioral neuroscience experiments involving
vicarious emotional response, upright vs. phase-scrambled stimuli, and
per-trial movie extraction.
"""

import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from pathlib import Path
import os


class TrgManager:
    """
        TrgManager
        ----------
        Handle saving and loading of trigger arrays associated with video files.

        This utility class provides a simple interface for:
        - Creating a `.npy` filename associated with a video or base name.
        - Saving a 1-dimensional NumPy array to disk.
        - Loading a previously saved trigger array.
        - Checking whether the trigger file exists.

        Typical usage
        -------------
        >>> trg = TrgManager("res/upright/video1.avi")
        >>> if trg.file_exists():
        ...     trigger = trg.load_array()
        ... else:
        ...     trg.save_array(my_trigger_vector)

        Notes
        -----
        The class enforces that the saved and loaded arrays must be monodimensional.
        This ensures consistency with trigger vectors extracted from videos.
        """
    def __init__(self, filename):
        """
        Construct a TrgManager and derive the `.npy` filename.

        Parameters
        ----------
        filename : str
            The base filename or path associated with the trigger array.
            If the input contains an extension, it is removed and replaced
            with `.npy`. Otherwise, `.npy` is appended.

        Examples
        --------
        >>> TrgManager("movie.avi").filename
        'movie.npy'

        >>> TrgManager("session").filename
        'session.npy'
        """

        if '.' in filename:
            self.filename = '.'.join(filename.split('.')[:-1]) + '.npy'
        else:
            self.filename = filename + '.npy'

    def save_array(self, array):
        """
        Save a 1-dimensional NumPy array to the trigger file.

        Parameters
        ----------
        array : numpy.ndarray
            The monodimensional array to be saved as `.npy`.

        Raises
        ------
        ValueError
            If the array is not monodimensional.

        Notes
        -----
        Arrays are saved using `numpy.save`. The file is overwritten if it exists.
        """
        if array.ndim != 1:
            raise ValueError("The array is not monodimensional.")
        np.save(self.filename, array)

    def load_array(self):
        """
        Load a monodimensional trigger array from disk.

        Returns
        -------
        numpy.ndarray
            The loaded 1-dimensional array.

        Raises
        ------
        ValueError
            If the loaded array is not monodimensional.
        FileNotFoundError
            If the trigger file does not exist.

        Notes
        -----
        Files are loaded using `numpy.load`. The method enforces array shape.
        """
        array = np.load(self.filename)
        if array.ndim != 1:
            raise ValueError("The loaded array is not monodimensional.")
        return array

    def file_exists(self):
        """
        Check whether the trigger `.npy` file exists.

        Returns
        -------
        bool
            True if the file exists, otherwise False.

        Examples
        --------
        >>> trg = TrgManager("video.avi")
        >>> trg.file_exists()
        False
        """
        return os.path.exists(self.filename)


class VideoManagerArray:

    def __init__(self, folder="res", pre_sec=20, post_sec=20, shuffle_trials=True):
        """
        VideoManagerArray
        -----------------
        A high-level container that loads multiple VideoManager objects from a dataset
        organized into `upright/` and `scrambled/` folders, generates an experiment-specific
        trial sequence, and provides utilities for retrieving full trial movies across subjects.

        The class automates:
        - Loading all upright and phase-scrambled videos in a given dataset folder.
        - Creating paired VideoManager instances (upright and scrambled).
        - Computing pre- and post-trial window sizes in frames, based on video FPS.
        - Building the fixed set of trial combinations used in the experiment.
        - Randomizing the trial order (optional).
        - Providing access to (trial → movie segment) retrieval.
        - Iterating the predefined stimulus sequence using `next_trial()`.
        - Looking up valid trials for each subject.

        Typical usage
        -------------
        >>> vma = VideoManagerArray("res")
        >>> t = vma.next_trial()
        >>> movie, trg = vma.get_trial_movie(t)
        >>> VideoManagerArray.preview(movie)

        Dataset Structure
        -----------------
        folder/
            upright/
                <subject videos .avi>
            scrambled/
                <phase-scrambled videos .avi>

        Each subject is represented by one upright video and one scrambled video.
        """

        # --- Construct paths dynamically ---
        upright_path = f"{folder}/upright"
        scrambled_path = f"{folder}/scrambled"

        # --- Load lists of video files ---
        self.upright = VideoManager.list_videos(pth=upright_path, verbosity=0)
        self.scrambled = VideoManager.list_videos(pth=scrambled_path, verbosity=0)

        self.vms = []
        self.vms_ps = []

        print("### VideoManagerArray ### Number of videos:", len(self.upright))
        print("### VideoManagerArray ### Loading Upright Videos:")

        fps = []

        for i, v in enumerate(self.upright):
            print(f"### VideoManagerArray ### {i} - {v}")
            vm = VideoManager(v)
            fps.append(vm.fps)
            self.vms.append(vm)

        print("### VideoManagerArray ### Loading Phase Scrambled Videos:")
        for i, v in enumerate(self.scrambled):
            print(f"\t### VideoManagerArray ### Phase Scrambled: {i} - {v}")
            vm = VideoManager(v)
            fps.append(vm.fps)
            self.vms_ps.append(vm)

        # Store config
        self.folder = folder
        self.shuffle_trials = shuffle_trials
        self.pre = pre_sec
        self.post = post_sec
        self.fps = np.median(fps)

        # Build the trial sequence
        self.__new_sequence()
        self.sequence_current = -1

    def __new_sequence(self):
        """
        Construct the predefined experiment trial sequence.

        The experiment uses only a specific subset of subjects and trial indices.
        These combinations are explicitly defined in the list below and include:
            - Upright trials
            - Flipped versions
            - Phase-scrambled versions

        Each entry in the sequence is a tuple:
            (subject_index, intensity, trial_number, is_phase_scrambled, is_flipped)

        Rules
        -----
        - The *first trial* is always (0, 0, 0, False, False) to ensure a consistent start.
        - If `self.shuffle_trials` is True:
              The remaining trials are randomly shuffled.
        - After building the list:
              `self.sequence`        contains all trials
              `self.sequence_n`      stores the count
              `self.sequence_current` resets to -1

        Returns
        -------
        None
        """

        first_trial = [(0, 0, 0, False,
                        False)]  # the first trial of the sequence is always 0 intensity
        self.sequence = [(0, 0, 15, False, False), (1, 0, 0, False, False), (1, 0, 5, False, False),
                         (1, 0, 10, False, False), (1, 0, 15, False, False),
                         (0, 4, 4, False, False), (0, 4, 14, False, False), (0, 4, 24, False, False),
                         (1, 4, 4, False, False), (1, 4, 16, False, False), (1, 4, 24, False, False),
                         # flipped
                         (0, 0, 0, False, True), (0, 0, 15, False, True), (1, 0, 0, False, True), (1, 0, 5, False, True),
                         (1, 0, 10, False, True),
                         (1, 0, 15, False, True),
                         (0, 4, 4, False, True), (0, 4, 14, False, True), (0, 4, 24, False, True),
                         (1, 4, 4, False, True), (1, 4, 16, False, True),
                         (1, 4, 24, False, True),
                         # phase scrambled
                         (0, 0, 0, True, False), (0, 0, 15, True, False), (1, 0, 0, True, False),
                         (1, 0, 5, True, False), (1, 0, 10, True, False), (1, 0, 15, True, False),
                         (0, 4, 4, True, False), (0, 4, 14, True, False), (0, 4, 24, True, False),
                         (1, 4, 4, True, False), (1, 4, 16, True, False), (1, 4, 24, True, False)
                         ]

        if self.shuffle_trials:
            np.random.shuffle(self.sequence)

        self.sequence = first_trial + self.sequence
        self.sequence_n = len(self.sequence)
        self.sequence_current = -1

    def get_trial_movie(self, trial):
        """
        Extract the full movie segment for a given trial descriptor.

        Parameters
        ----------
        trial : tuple
            A 5-element trial description:
                (subject_index, intensity, trial_number,
                 is_phase_scrambled, is_flipped)

        Returns
        -------
        trl : numpy.ndarray
            A 4D array (frames × H × W × channels) containing the movie segment.
            The trigger pixel region (10×10) is automatically cropped out.
        trg : numpy.ndarray
            A 1D array indicating the trigger time within the segment.

        Notes
        -----
        - pre/post time windows are automatically converted from seconds to frames.
        - If `is_phase_scrambled` is True, frames are extracted from `self.vms_ps`.
        - If `is_flipped` is True, frames are vertically flipped after cropping.
        """

        subject = trial[0]
        instensity = trial[1]
        trial_num = trial[2]
        if_phase_scrambled = trial[3]
        if_flipped = trial[4]

        if if_phase_scrambled:
            trl, trg = self.vms_ps[subject].get_trial(trial_num, pre=int(self.pre * self.vms_ps[subject].fps),
                                                      post=int(self.post * self.vms_ps[subject].fps))
        else:
            trl, trg = self.vms[subject].get_trial(trial_num, pre=int(self.pre * self.vms[subject].fps),
                                                   post=int(self.post * self.vms[subject].fps))

        trl = trl[:, 10:, 10:, :]  # excluding trigger # TODO remove the trigger in VideoManager?

        if if_flipped:
            trl = trl[:, ::-1, :, :]

        return trl, trg

    def __getitem__(self, item):
        """
        Provide quick access to VideoManagers or raw trial windows.

        Parameters
        ----------
        item : int or tuple
            - If int → returns the VideoManager at that index.
            - If tuple → interpreted as (subject, trial_index[, ...]) and returns
              raw frames via VideoManager.get_trial().

        Returns
        -------
        VideoManager or (frames, trg)
        """

        # TODO extraction of upside down and phase scrambled?
        if type(item) is int:
            return self.vms[item]
        elif type(item) is tuple:
            trl, trg = self.vms[item[0]].get_trial(item[1], pre=int(self.pre * self.vms[item[0]].fps),
                                                   post=int(self.post * self.vms[item[0]].fps))
            return trl, trg

    def get_valid_trials(self, sub, shuffle=True):
        """
        Return all valid trials for a given subject.

        Parameters
        ----------
        sub : int
            Subject index.
        shuffle : bool
            If True, shuffle the order of returned valid trials.

        Returns
        -------
        list of tuples
            Each tuple is of the form (subject, intensity, trial_number).
        """
        temp = [(sub, 0, t) for t in self.vms[sub].get_valid_trials(0)] + [(sub, 4, t) for t in
                                                                           self.vms[sub].get_valid_trials(4)]
        if shuffle:
            np.random.shuffle(temp)
        return temp

    def next_trial(self):
        """
        Advance the internal pointer and return the next trial.

        Returns
        -------
        tuple
            The trial descriptor (subject, intensity, trial_number,
            is_scrambled, is_flipped).
        """
        self.sequence_current += 1
        current_trial = self.sequence[self.sequence_current]
        return current_trial

    def has_next_trial(self):
        """
        Check whether more trials are available in the experiment sequence.

        Returns
        -------
        bool
            True if there is at least one more trial, False otherwise.
        """
        return self.sequence_current < len(self.sequence) - 1

    def reset_trial(self, reshuffle=True, shuffle_trials=None):
        """
        Reset the trial iterator and optionally rebuild/shuffle the sequence.

        Parameters
        ----------
        reshuffle : bool
            If True, the sequence is rebuilt and optionally reshuffled.
        shuffle_trials : bool or None
            Overrides the current shuffle setting if provided.

        Returns
        -------
        None
        """
        if reshuffle:
            if shuffle_trials is not None:
                self.shuffle_trials = shuffle_trials
            self.__new_sequence()
        self.sequence_current = 0

    @staticmethod
    def preview(frames, window='Frames', fps=20, repeat=False):
        """
        Preview an array of frames using VideoManager.preview().
        """

        VideoManager.preview(frames, window=window, fps=fps, repeat=repeat)

    def __len__(self):
        """
        Number of upright VideoManagers loaded.

        Returns
        -------
        int
        """
        return len(self.upright)

    def close(self):
        """
        Close all VideoManagers and release video resources.

        Returns
        -------
        None
        """
        for vm in self.vms:
            vm.close()
        if self.phase_scrambled:
            for vm in self.vms_ps:
                vm.close()


class VideoManager:
    """
        VideoManager
        ------------
        A high-level interface for loading, processing, analyzing, and previewing
        behavioral videos recorded during stimulation experiments.

        The class supports:
        - Video loading and property extraction (fps, resolution, frame count)
        - Automatic trigger extraction from the first pixel or stored .npy file
        - Trial detection, segmentation, and intensity assignment
        - Habituation extraction
        - Movement computation and visualization
        - Previewing individual trials or full trial mosaics
        - Generating videos or GIFs summarizing all trials
        - Phase scrambling utilities for stimulus generation

        Parameters
        ----------
        video_path : str
            Full path to the input video file.
        pattern : list of int, optional
            A cyclic list of stimulation intensity labels. Each detected trigger
            is assigned an intensity based on this pattern.

        Notes
        -----
        Trigger extraction:
            The trigger is encoded as the BGR value of the first pixel in each frame.
            Values > 100 are considered "trigger ON".

        Trial definition:
            The index of each detected trigger is stored in `self.trigger_timestamp`.
            Trials are segmented relative to those timestamps.

        Usage Example
        -------------
        >>> vm = VideoManager("my_video.avi")
        >>> info = vm.info()
        >>> frames, trg = vm.get_trial(3)
        >>> vm.preview_trial(3)
    """

    def __init__(self, video_path, pattern=[0, 1, 2, 3, 4, 0, 4, 3, 2, 1]):
        """
        Initialize the VideoManager, extract video metadata, load or compute
        the trigger vector, detect trials, and build the intensity database.

        Parameters
        ----------
        video_path : str
            Path to the .avi video file.
        pattern : list of int
            A list representing the stimulation intensity pattern assigned
            cyclically to each detected trigger.

        Behavior
        --------
        - Loads video file using OpenCV.
        - Computes metadata: fps, resolution, total frames.
        - Loads trigger array from disk if available, otherwise computes it by
          reading pixel (0,0) of each frame.
        - Detects trigger timestamps via peak detection.
        - Assigns stimulation intensities.
        - Loads trial validity information from recordings.xlsx.
        - Prepares helper attributes for visualization and preview.

        Side Effects
        ------------
        - Creates a `.npy` trigger file if not already present.
        - Reads `recordings.xlsx` from the video's directory.
    """

        self.video_path = video_path
        self.rec = Path(video_path).stem
        self.cap = cv2.VideoCapture(self.video_path)  # video_folder + '\\' + videos[0] )
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (self.frame_height, self.frame_width)
        self.size_ratio = self.frame_width / self.frame_height

        trgMan = TrgManager(filename=self.video_path)
        if trgMan.file_exists():
            self.trigger = trgMan.load_array()
        else:
            self.trigger = list()
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                # Assuming the video is color and extracting the BGR value of the first pixel
                val = frame[0, 0, 0]
                self.trigger.append(val)
            self.trigger = np.array(self.trigger)
            trgMan.save_array(self.trigger)

        self.trigger_norm = self.trigger > 100
        # simple artifact rejection
        # pks = find_peaks(self.trigger_norm,distance=np.median(np.diff(np.where(self.trigger_norm)[0]))-np.std(np.diff(np.where(self.trigger_norm)[0]))*5)
        pks = find_peaks(self.trigger_norm, distance=np.median(np.diff(np.where(self.trigger_norm)[0])) - 100)
        self.trigger_timestamp = pks[0]
        self.trigger_tot = len(self.trigger_timestamp)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.gotoframe(0)

        self.pattern = pattern
        self.stim_intensity = list()
        for i in range(0, self.trigger_tot):
            self.stim_intensity.append(self.pattern[i % len(self.pattern)])
        self.stim_intensity = np.array(self.stim_intensity)
        self.stimuli = np.unique(self.stim_intensity, return_counts=True)

        self.recs = pd.read_excel(Path(video_path).parent.as_posix() + '/recordings.xlsx', index_col=0)
        if self.rec in self.recs.index:
            self.valid = self.recs.loc[self.rec].values
        else:
            self.valid = np.ones_like(self.stimuli)

        self.stim_db = pd.DataFrame(dict(intensity=self.stim_intensity, valid=self.valid))
        self.habituation_inds = [0, self.trigger_timestamp[0]]

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.thickness = 2
        self.color = (255, 255, 255)
        self.color_stim = (255, 0, 0)

    def plot_trigger(self, ax=None):
        """
        Plot the normalized trigger signal and detected trigger events.

        The trigger is derived from the first pixel of each frame. This method
        shows the binarized trigger vector and marks detected trigger peaks.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes on which to plot the trigger signal. If None, a new figure
            and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the trigger plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.trigger_norm)
        plt.plot(self.trigger_timestamp, self.trigger_norm[self.trigger_timestamp], "x")

    def plot_movement(self, trial_num, pre=60, post=60, ax=None):
        """
        Plot the movement index across a single trial.

        Movement is quantified as frame-to-frame difference using
        `frame_movements`. This method extracts a trial window, computes
        movement indices for all frames, and plots them.

        Parameters
        ----------
        trial_num : int
            Trial index (0-based) to analyze.
        pre : int, optional
            Number of frames before the trigger to include.
        post : int, optional
            Number of frames after the trigger to include.
        ax : matplotlib.axes.Axes, optional
            Axes on which to plot the movement index. If None, a new figure
            and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the movement trace for the selected trial.
        """
        trial, trg = self.get_trial(trial_num, pre, post)
        movement_index = list()
        ax.axvline(np.where(np.array(trg) > 0)[0], color='k', linestyle='--')
        for f in range(0, trial.shape[0]):
            frame = trial[f, :, :, :]
            if f > 0:
                _, mov_ind = self.frame_movements(frame_prev, frame)
                movement_index.append(mov_ind)
            else:
                movement_index.append(0)
            frame_prev = frame.copy()

        if ax == None:
            fig, ax = plt.subplots()
        ax.plot(movement_index)


    def plot_movement_matrix(self, pre=60, post=60):
        """
        Plot a 2D matrix of movement traces for all trials.

        The resulting figure has intensity on the rows and repetition number
        on the columns. Each subplot shows the movement index for one trial.

        Parameters
        ----------
        pre : int, optional
            Frames before the trigger to include in each trial window.
        post : int, optional
            Frames after the trigger to include in each trial window.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the movement matrix.
        """
        fig, ax = plt.subplots(5, 6, figsize=(5, 5), sharey=True)
        fig.supylabel('Intensity', fontsize=16)
        fig.suptitle('Trial', fontsize=16)
        rep = 0
        for t, inten in enumerate(self.stim_intensity):
            if t > 0 and t % 5 == 0:
                rep += 1
            self.plot_movement(t, ax=ax[inten, rep], pre=pre, post=post)
            if rep == 0:
                ax[inten, rep].set_ylabel(inten)
            if inten == 0:
                ax[inten, rep].set_title(rep)
            ax[inten, rep].set_yticklabels([])
            ax[inten, rep].set_xticklabels([])

            if self.valid[t] == 0:
                ax[inten, rep].spines['top'].set_color('magenta')
                ax[inten, rep].spines['right'].set_color('magenta')
                ax[inten, rep].spines['left'].set_color('magenta')
                ax[inten, rep].spines['bottom'].set_color('magenta')

            ax[inten, rep].set_title(f'T: {t}', fontsize=8)
        plt.tight_layout()
        return fig

    def save_movement_matrix(self, pre=60, post=60,
                             outdir="graphs",
                             fmt="svg"):
        """
        Compute and save the movement matrix for all trials to disk.

        This is similar to `plot_movement_matrix` but writes the figure
        directly to a file and closes the figure instead of displaying it.

        Parameters
        ----------
        pre : int, optional
            Frames before trigger to include in each trial window.
        post : int, optional
            Frames after trigger to include in each trial window.
        outdir : str, optional
            Output directory for the figure. Created if it does not exist.
        fmt : str, optional
            Image format for saving (for example 'svg', 'png', 'pdf', 'jpg').

        Returns
        -------
        str
            Path of the saved file.
        """

        # Create output folder if needed
        Path(outdir).mkdir(parents=True, exist_ok=True)

        # File name: <video_stem>_movement_matrix.svg
        outfile = Path(outdir) / f"{self.rec}_movement_matrix.{fmt}"

        # Create the figure
        fig, ax = plt.subplots(5, 6, figsize=(6, 6), sharey=True)
        fig.supylabel('Intensity', fontsize=14)
        fig.supxlabel('Trial', fontsize=14)
        fig.suptitle(f'{self.rec}', fontsize=14)

        rep = 0
        for t, inten in enumerate(self.stim_intensity):

            if t > 0 and t % 5 == 0:
                rep += 1

            self.plot_movement(t, ax=ax[inten, rep], pre=pre, post=post)

            # Labels
            if rep == 0:
                ax[inten, rep].set_ylabel(inten)
            if inten == 0:
                ax[inten, rep].set_title(f"Rep {rep}")

            # No ticks
            ax[inten, rep].set_yticklabels([])
            ax[inten, rep].set_xticklabels([])

            # Invalid trial highlighting
            if self.valid[t] == 0:
                for side in ["top", "right", "left", "bottom"]:
                    ax[inten, rep].spines[side].set_color("magenta")

            ax[inten, rep].set_title(f"T:{t}", fontsize=7)

        plt.tight_layout()

        # Save without showing
        fig.savefig(outfile, format=fmt, dpi=300)
        plt.close(fig)

        return str(outfile)

    def get_frame(self, i, flip_vertical=False):
        """
        Retrieve a single frame by index.

        This performs a seek to frame i and then reads a single frame. It is
        relatively slow for many random accesses.

        Parameters
        ----------
        i : int
            Index of the frame to retrieve (0-based).
        flip_vertical : bool, optional
            If True, the frame is flipped vertically.

        Returns
        -------
        numpy.ndarray or None
            The requested frame as a HxWx3 array, or None if reading fails.
        """
        self.gotoframe(i)
        ret, frame = self.cap.read()
        if not ret:
            return None
        if flip_vertical:
            frame = frame[::-1, :, :]
        return frame

    def get_frames(self, from_frame, to_frame, to_norm=False, flip_vertical=False):
        """
        Retrieve a sequence of frames as a 3D or 4D array.

        Parameters
        ----------
        from_frame : int
            Index of the first frame to retrieve (inclusive).
        to_frame : int
            Index of the last frame to retrieve (exclusive).
        to_norm : bool, optional
            If True, frames are converted to normalized grayscale and mean-centered.
        flip_vertical : bool, optional
            If True, each frame is flipped vertically.

        Returns
        -------
        numpy.ndarray
            A 4D array of shape (N, H, W, 3) for RGB frames, or
            (N, H, W) for normalized grayscale frames, where N is
            to_frame - from_frame.
        """
        if to_norm:
            res = np.zeros((to_frame - from_frame, self.height, self.width), dtype=np.float32)
        else:
            res = np.zeros((to_frame - from_frame, self.height, self.width, 3), dtype=np.uint8)

        self.gotoframe(to_frame)
        for i, f in enumerate(range(from_frame, to_frame)):
            frame = self.get_next_frame(flip_vertical=flip_vertical)
            if to_norm:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype('float32') / 255
                frame = frame - np.mean(frame)
            res[i] = frame
        return res

    def get_habituation(self, to_norm=True, flip_vertical=False):
        """
        Retrieve all frames belonging to the habituation phase.

        The habituation segment is defined by `self.habituation_inds`, which
        is set at initialization based on the first trigger.

        Parameters
        ----------
        to_norm : bool, optional
            If True, frames are converted to normalized grayscale and mean-centered.
        flip_vertical : bool, optional
            If True, frames are flipped vertically.

        Returns
        -------
        numpy.ndarray
            Frames from the habituation phase as either RGB or normalized
            grayscale, depending on `to_norm`.
        """
        return self.get_frames(
            self.habituation_inds[0],
            self.habituation_inds[1],
            to_norm=to_norm,
            flip_vertical=flip_vertical
        )

    def get_next_frame(self, flip_vertical=False):
        """
        Read the next frame in sequence from the video.

        Parameters
        ----------
        flip_vertical : bool, optional
            If True, the frame is flipped vertically.

        Returns
        -------
        numpy.ndarray or None
            The next frame as an array, or None if reading fails.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        if flip_vertical:
            frame = frame[::-1, :, :]

        return frame

    def get_trial(self, trial, pre=60, post=60, to_norm=False, flip_vertical=False):
        """
        Extract a time window around a given trigger (trial).

        The trial window is centered on the trigger timestamp and extends
        `pre` frames before and `post` frames after.

        Parameters
        ----------
        trial : int
            Index of the trigger (trial) to extract.
        pre : int, optional
            Number of frames before the trigger to include.
        post : int, optional
            Number of frames after the trigger to include.
        to_norm : bool, optional
            If True, frames are converted to normalized grayscale and mean-centered.
        flip_vertical : bool, optional
            If True, frames are flipped vertically.

        Returns
        -------
        res : numpy.ndarray
            Extracted frames with shape (pre + post, H, W, 3) for RGB, or
            (pre + post, H, W) if `to_norm` is True.
        trg : numpy.ndarray
            A 1D array (length equal to number of frames) with a single 1
            at the trigger index and 0 elsewhere.
        """

        i = self.trigger_timestamp[trial] - pre  # Example start frame
        z = self.trigger_timestamp[trial] + post  # Example end frame
        res = np.zeros((z - i, self.height, self.width, 3), dtype=np.uint8)
        trg = np.zeros((z - i, 1))
        trg[pre + 1] = 1
        self.gotoframe(i)
        for i, f in enumerate(range(i, z)):
            # frame = self.cap.read()[1]
            frame = self.get_next_frame(flip_vertical=flip_vertical)
            if to_norm:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype('float32') / 255
                frame = frame - np.mean(frame)
            res[i, :, :, :] = frame
        return res, trg

    def preview_trial(self, trial_num, pre=60, post=60, showlabels=True, repeat=False, window='Frame', movements=True,
                      movements_overlay=False, ylims=(0, 60000)):
        """
        Preview a single trial as an OpenCV window animation.

        The method extracts a trial window, optionally overlays labels,
        highlights movement, and allows repeated playback until 'q' is pressed.

        Parameters
        ----------
        trial_num : int
            Index of the trial to preview.
        pre : int, optional
            Frames before the trigger to include.
        post : int, optional
            Frames after the trigger to include.
        showlabels : bool, optional
            If True, shows trial, intensity, and PRE/POST labels on the frame.
        repeat : bool, optional
            If True, loops the trial playback.
        window : str, optional
            Name of the OpenCV window.
        movements : bool, optional
            If True, computes and optionally overlays movement information.
        movements_overlay : bool, optional
            If True, overlays motion mask on the video frames.
        ylims : tuple, optional
            Y-axis limits for the movement trace drawn on the frame.

        Returns
        -------
        None
            The function opens an OpenCV window and blocks until playback ends
            or 'q' is pressed.
        """
        trial, trg = self.get_trial(trial_num, pre, post, )
        text_stage = 'PRE'
        f = 0
        if movements:
            movement_indices = list()
        prev_image = []
        while True:  # for f in range(0,trial.shape[0]):
            image = trial[f, :, :, :]

            if showlabels:
                text = f'Trial: {trial_num}'
                text_intensity = f'Intensity: {self.stim_intensity[trial_num]}'

                text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
                text_x = image.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
                text_y = text_size[1] + 10  # 10 pixels from the top

                text_size_int = cv2.getTextSize(text_intensity, self.font, self.font_scale, self.thickness)[0]
                text_x_int = image.shape[1] - text_size_int[0] - 10  # 10 pixels from the right edge
                text_y_int = text_size[1] + text_size_int[1] + 20  # 10 pixels from the top

                text_size_stage = cv2.getTextSize(text_stage, self.font, self.font_scale, self.thickness)[0]
                text_x_stage = image.shape[1] - text_size_stage[0] - 10  # 10 pixels from the right edge
                text_y_stage = text_size[1] + text_size_stage[1] + text_size_stage[1] + 30  # 10 pixels from the top

                if trg[f]:
                    cv2.putText(image, text, (text_x, text_y), self.font, self.font_scale, self.color_stim,
                                self.thickness)
                    cv2.putText(image, text_intensity, (text_x_int, text_y_int), self.font, self.font_scale,
                                self.color_stim, self.thickness)
                    text_stage = 'POST'
                else:
                    cv2.putText(image, text, (text_x, text_y), self.font, self.font_scale, self.color, self.thickness)
                    cv2.putText(image, text_intensity, (text_x_int, text_y_int), self.font, self.font_scale, self.color,
                                self.thickness)

                cv2.putText(image, text_stage, (text_x_stage, text_y_stage), self.font, self.font_scale, self.color,
                            self.thickness)

            if movements:
                if len(prev_image):
                    thresh_frame, movement_index = self.frame_movements(prev_image[10:, 10:, :], image[10:, 10:, :])
                    movement_indices.append(movement_index)
                    alpha = 0.5
                    overlay = image.copy()
                    overlay[10:, 10:, :][thresh_frame != 0] = [0, 0, 255]
                    if movements_overlay:
                        combined_frame = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                        combined_frame = self.draw_plot_on_frame(combined_frame, movement_indices, 50, ymin=ylims[0],
                                                                 ymax=ylims[1])
                    else:
                        combined_frame = self.draw_plot_on_frame(image, movement_indices, 50, ymin=ylims[0],
                                                                 ymax=ylims[1])
                else:
                    combined_frame = image
                cv2.imshow(window, combined_frame)
            else:
                cv2.imshow(window, image)

            prev_image = image
            if cv2.waitKey(int((1 / self.fps) * 1000)) & 0xFF == ord('q'):
                break

            if f < trial.shape[0] - 1:
                f += 1
            else:
                if repeat:
                    f = 0
                    text_stage = 'PRE'
                    prev_image = []
                    if movements:
                        movement_indices = list()
                else:
                    break
        cv2.destroyAllWindows()

    def previewAllTrials(self, pre=100, post=100, frame_height=700):
        """
        Preview all trials in a single mosaic played as a video.

        Each trial is resized and arranged in a grid with:
        - rows = stimulation intensities
        - columns = repetitions

        The mosaic is then animated over time to show all trials in parallel.

        Parameters
        ----------
        pre : int, optional
            Frames before each trigger to include in the trial segments.
        post : int, optional
            Frames after each trigger to include.
        frame_height : int, optional
            Height of the mosaic preview window.

        Returns
        -------
        None
            Opens an OpenCV window and animates the mosaic until 'q' is pressed.
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        color_stim = (255, 0, 0)

        frame_width = frame_height * (self.resolution[1] // self.resolution[0])
        big_frame_res = (frame_height, frame_width)
        rows, cols = len(self.stimuli[0]), np.max(self.stimuli[1])
        resize_to = big_frame_res[0] // rows, frame_width // cols

        big_frame = np.zeros((pre + post, big_frame_res[0], big_frame_res[1], 3), dtype=np.uint8)
        col = 0
        row_poses = np.arange(0, big_frame_res[0], resize_to[0])
        col_poses = np.arange(0, big_frame_res[1], resize_to[1])
        col_pos = 0
        row_pos = 0
        for i, row in enumerate(self.stim_intensity):
            if i > 0 and row == 0:
                # print('entra',i,row,col)
                col += 1
                col_pos = col_poses[col]
            row_pos = row_poses[row]

            trl, trg = self.get_trial(i, pre, post)

            trl_resized = np.zeros((trl.shape[0], resize_to[0], resize_to[1], 3), dtype=np.uint8)
            text = f'T:{i} I:{row}'
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = resize_to[1] - text_size[0]
            text_y = text_size[1]
            txt_color = color
            for i, frame in enumerate(trl):
                resiz = cv2.resize(frame, (resize_to[1], resize_to[0]), interpolation=cv2.INTER_LINEAR)
                if trg[i]:
                    txt_color = color_stim
                resiz = cv2.putText(resiz, text, (text_x, text_y), font, font_scale, txt_color, thickness)
                trl_resized[i, :, :, :] = resiz

            for f in range(0, trl.shape[0]):
                big_frame[f, row_pos:row_pos + resize_to[0], col_pos:col_pos + resize_to[1], :] = trl_resized[f, :, :,
                                                                                                  :]

        i = 0
        while True:
            # cv2.imshow('frame2', trial2[i,:,:,:])
            image = big_frame[i, :, :, :]

            cv2.imshow('frame1', image)
            # Press Q on keyboard to exit early
            if cv2.waitKey(int((1 / self.fps) * 1000)) & 0xFF == ord('q'):
                close = True
                break
            if i == big_frame.shape[0] - 1:
                i = 0
            else:
                i = i + 1

        cv2.destroyAllWindows()

    def previewAllTrials2Video(self, pre=100, post=100, frame_height=700, video_filename=None):
        """
        Generate and save a mosaic video of all trials.

        Similar to `previewAllTrials`, but instead of showing the mosaic live,
        this method writes it to a video file on disk.

        Parameters
        ----------
        pre : int, optional
            Frames before each trigger to include in the trial segments.
        post : int, optional
            Frames after each trigger to include.
        frame_height : int, optional
            Height of the mosaic frames in the saved video.
        video_filename : str, optional
            Output video filename. If None, uses "<rec>.mp4".

        Returns
        -------
        None
            Writes a video file with the mosaic of all trials.
        """

        if video_filename is None:
            video_filename = self.rec + '.mp4'

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        color_stim = (255, 0, 0)
        print('### VideoManager ### creating the video...')
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = frame_height * (self.resolution[1] // self.resolution[0])
        out = cv2.VideoWriter(video_filename, fourcc, self.fps, (frame_width, frame_height))

        big_frame_res = (frame_height, frame_width)
        rows, cols = len(self.stimuli[0]), np.max(self.stimuli[1])
        resize_to = big_frame_res[0] // rows, frame_width // cols

        big_frame = np.zeros((pre + post, big_frame_res[0], big_frame_res[1], 3), dtype=np.uint8)
        col = 0
        row_poses = np.arange(0, big_frame_res[0], resize_to[0])
        col_poses = np.arange(0, big_frame_res[1], resize_to[1])
        col_pos = 0
        row_pos = 0

        for i, row in enumerate(self.stim_intensity):
            if i > 0 and row == 0:
                col += 1
                col_pos = col_poses[col]
            row_pos = row_poses[row]

            trl, trg = self.get_trial(i, pre, post)

            trl_resized = np.zeros((trl.shape[0], resize_to[0], resize_to[1], 3), dtype=np.uint8)
            text = f'T:{i} I:{row}'
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = resize_to[1] - text_size[0]
            text_y = text_size[1]
            txt_color = color

            for j, frame in enumerate(trl):
                resiz = cv2.resize(frame, (resize_to[1], resize_to[0]), interpolation=cv2.INTER_LINEAR)
                if trg[j]:
                    txt_color = color_stim
                resiz = cv2.putText(resiz, text, (text_x, text_y), font, font_scale, txt_color, thickness)
                trl_resized[j, :, :, :] = resiz

            for f in range(trl.shape[0]):
                big_frame[f, row_pos:row_pos + resize_to[0], col_pos:col_pos + resize_to[1], :] = trl_resized[f, :, :,
                                                                                                  :]

        for i in range(big_frame.shape[0]):
            out.write(big_frame[i])

        out.release()
        print('### VideoManager ### creating the video...done')

    def previewAllTrials2GIF(self, pre=100, post=100, frame_height=700,
                             gif_filename=None,
                             resize_factor=1.0,
                             dither=False,
                             palette=256):
        """
        Generate and save a GIF preview of the full trial mosaic.

        This method builds the same mosaic representation used by
        `previewAllTrials`, but then exports it as an optimized GIF using PIL.

        Parameters
        ----------
        pre : int, optional
            Frames before each trigger to include in each trial window.
        post : int, optional
            Frames after each trigger to include.
        frame_height : int, optional
            Height of the mosaic in pixels.
        gif_filename : str, optional
            Output GIF filename. If None, uses "<rec>.gif".
        resize_factor : float, optional
            Scale factor applied to the final frames (1.0 = full size, 0.5 = half).
        dither : bool, optional
            If True, enables Floyd-Steinberg dithering in the palette reduction.
        palette : int, optional
            Size of the color palette (max 256).

        Returns
        -------
        None
            Writes an animated GIF to disk.
        """

        from PIL import Image

        if gif_filename is None:
            gif_filename = self.rec + '.gif'

        frame_width = frame_height * (self.resolution[1] // self.resolution[0])
        big_frame_res = (frame_height, frame_width)
        rows, cols = len(self.stimuli[0]), np.max(self.stimuli[1])
        resize_to = big_frame_res[0] // rows, frame_width // cols

        big_frame = np.zeros((pre + post, big_frame_res[0], big_frame_res[1], 3),
                             dtype=np.uint8)

        col = 0
        row_poses = np.arange(0, big_frame_res[0], resize_to[0])
        col_poses = np.arange(0, big_frame_res[1], resize_to[1])
        col_pos = 0
        row_pos = 0

        # Fill mosaic
        for i, row in enumerate(self.stim_intensity):
            if i > 0 and row == 0:
                col += 1
                col_pos = col_poses[col]

            row_pos = row_poses[row]
            trl, trg = self.get_trial(i, pre, post)

            trl_resized = np.zeros((trl.shape[0], resize_to[0], resize_to[1], 3),
                                   dtype=np.uint8)
            text = f'T:{i} I:{row}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, 1)[0]
            text_x = resize_to[1] - text_size[0]
            text_y = text_size[1]

            for j, frame in enumerate(trl):
                resiz = cv2.resize(frame, (resize_to[1], resize_to[0]),
                                   interpolation=cv2.INTER_LINEAR)

                color = (255, 0, 0) if trg[j] else (255, 255, 255)
                resiz = cv2.putText(resiz, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 1)

                trl_resized[j] = resiz

            for f in range(trl.shape[0]):
                big_frame[f,
                row_pos:row_pos + resize_to[0],
                col_pos:col_pos + resize_to[1]] = trl_resized[f]

        # Convert to PIL images with quantization and optimization
        frames = []
        for i in range(big_frame.shape[0]):
            im = Image.fromarray(cv2.cvtColor(big_frame[i], cv2.COLOR_BGR2RGB))

            # optional resize
            if resize_factor != 1.0:
                new_w = int(im.width * resize_factor)
                new_h = int(im.height * resize_factor)
                im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # reduce palette (massive file size reduction)
            dither_flag = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
            im = im.quantize(colors=palette, method=Image.MEDIANCUT, dither=dither_flag)

            frames.append(im)

        # Save optimized GIF
        frames[0].save(
            gif_filename,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=int(1000 / self.fps),
            loop=0,
            disposal=2
        )

        print("GIF saved:", gif_filename)

    def reset(self):
        """
        Reset the video position to the first frame.

        This is useful when restarting analysis or preview from the beginning.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def gotoframe(self, i):
        """
        Seek to a specific frame index in the video.

        Parameters
        ----------
        i : int
            Frame index to seek to.

        Returns
        -------
        None
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    def close(self):
        """
        Release the underlying OpenCV VideoCapture resource.

        Call this when the VideoManager instance is no longer needed.
        """
        self.cap.release()

    def get_valid_trials(self, intensity):
        """
        Get indices of valid trials for a given stimulation intensity.

        Parameters
        ----------
        intensity : int
            Stimulation intensity label for which to retrieve valid trials.

        Returns
        -------
        list of int
            Indices of trials that match the requested intensity and are marked
            as valid in `stim_db`.
        """
        valids = self.stim_db[self.stim_db['valid'] == 1]
        temp = list(valids[valids['intensity'] == intensity].index)
        return temp

    @staticmethod
    def frame_movements(prev_frame, curr_frame, gfilt=5, thr=10):
        """
        Compute movement between two frames using absolute difference.

        Both frames are blurred, subtracted, thresholded, and the number of
        non-zero pixels in the thresholded image is used as a movement index.

        Parameters
        ----------
        prev_frame : numpy.ndarray
            Previous frame (HxW or HxWx3).
        curr_frame : numpy.ndarray
            Current frame (HxW or HxWx3).
        gfilt : int, optional
            Gaussian kernel size used for blurring.
        thr : int, optional
            Threshold for detecting motion in the difference image.

        Returns
        -------
        thresh_frame : numpy.ndarray
            Thresholded difference frame, same size as input frames.
        movement_index : int
            Number of non-zero pixels in `thresh_frame`, indicating motion level.
        """

        if len(curr_frame.shape) > 2:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if len(prev_frame.shape) > 2:
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        curr_frame = cv2.GaussianBlur(curr_frame, (gfilt, gfilt), 0)
        prev_frame = cv2.GaussianBlur(prev_frame, (gfilt, gfilt), 0)

        diff_frame = cv2.absdiff(prev_frame, curr_frame)
        _, thresh_frame = cv2.threshold(diff_frame, thr, 255, cv2.THRESH_BINARY)
        movement_index = cv2.countNonZero(thresh_frame)
        return thresh_frame, movement_index

    @staticmethod
    def list_videos(pth='res\\upright', verbosity=0, ext='avi'):
        """
        List all video files in a given directory with a given extension.

        Parameters
        ----------
        pth : str, optional
            Path to the directory containing video files.
        verbosity : int, optional
            If non-zero, prints the found videos to stdout.
        ext : str, optional
            File extension to filter for (for example 'avi').

        Returns
        -------
        list of str
            List of full paths to video files matching the extension.
        """
        videos = os.listdir(pth)
        videos = [os.path.join(pth, v) for v in videos]
        videos = list(filter(lambda x: x.split('.')[-1] == ext, videos))
        if verbosity:
            print(f'{len(videos)} found:')
            for i, v in enumerate(videos):
                print(f'{i} - {v}')
        return videos

    @staticmethod
    def phase_scramble_image(frame, phase_scrambling_matrix=None, seed=None, to_rgb=False):
        """
        Apply Fourier phase scrambling to an image.

        The magnitude spectrum is preserved while phase is randomized or
        shuffled, which preserves low-level statistics but disrupts semantic
        content.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image (HxWx3, BGR).
        phase_scrambling_matrix : numpy.ndarray, optional
            Precomputed phase matrix to reuse for temporal consistency.
        seed : int, optional
            Random seed for reproducibility when generating new phase matrices.
        to_rgb : bool, optional
            If True, the output is converted back to 3-channel RGB.

        Returns
        -------
        scrambled_frame_normalized : numpy.ndarray
            Phase-scrambled image in uint8.
        phase_scrambling_matrix : numpy.ndarray
            Phase matrix used for scrambling, which can be reused for other frames.
        """

        if seed is not None:
            np.random.seed(seed)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        f_transform = np.fft.fft2(gray_frame)
        f_shift = np.fft.fftshift(f_transform)
        if phase_scrambling_matrix is None:
            #    phase_scrambling_matrix = np.exp(1j * np.random.uniform(0, 2 * np.pi, gray_frame.shape))
            phase_scrambling_matrix = np.exp(1j * VideoManager.shuffle_2d_array(np.angle(f_shift)))

        magnitude = np.abs(f_shift)
        scrambled_f_shift = magnitude * phase_scrambling_matrix
        scrambled_f_transform = np.fft.ifftshift(scrambled_f_shift)

        # Inverse Fourier
        scrambled_frame = np.fft.ifft2(scrambled_f_transform)
        scrambled_frame = np.abs(scrambled_frame)

        # Normalize the image for display
        scrambled_frame_normalized = np.uint8(np.clip(scrambled_frame, 0, 255))

        if to_rgb:
            scrambled_frame_normalized = cv2.cvtColor(scrambled_frame_normalized, cv2.COLOR_GRAY2BGR)

        return scrambled_frame_normalized, phase_scrambling_matrix

    @staticmethod
    def shuffle_2d_array(arr):
        """
        Randomly shuffle the elements of a 2D array.

        This is used to randomize the phase map while preserving the global
        distribution of phase values.

        Parameters
        ----------
        arr : numpy.ndarray
            Input 2D array.

        Returns
        -------
        numpy.ndarray
            Shuffled array with the same shape as input.
        """
        # Flatten the array
        flat_arr = arr.flatten()

        # Shuffle the flattened array
        np.random.shuffle(flat_arr)

        # Reshape the array back to its original 2D shape
        shuffled_arr = flat_arr.reshape(arr.shape)

        return shuffled_arr

    @staticmethod
    def preview(frames, window='Frames', fps=20, repeat=False):
        """
        Preview an array of frames in an OpenCV window.

        Frames can be either 3D (N, H, W) or 4D (N, H, W, C). Playback runs
        until the sequence ends or the user presses 'q'.

        Parameters
        ----------
        frames : numpy.ndarray
            Array of frames to display.
        window : str, optional
            Name of the OpenCV window.
        fps : int, optional
            Frames per second for playback.
        repeat : bool, optional
            If True, loops playback when it reaches the end.

        Returns
        -------
        None
        """
        i = 0
        print('Press Q to close the window')
        while True:
            if len(frames.shape) > 3:
                image = frames[i, :, :, :]
            else:
                image = frames[i, :, :]

            cv2.imshow(window, image)
            if cv2.waitKey(int((1 / fps) * 1000)) & 0xFF == ord('q'):
                break

            if i == frames.shape[0] - 1:
                if repeat:
                    i = 0
                else:
                    break
            else:
                i = i + 1

        cv2.destroyAllWindows()

    @staticmethod
    def calculate_movement_indices(frames):
        """
        Compute movement indices for a sequence of frames.

        Movement between each pair of consecutive frames is computed using
        `frame_movements`. The resulting 1D list can be used as a time series
        of motion energy.

        Parameters
        ----------
        frames : numpy.ndarray
            4D array of frames with shape (N, H, W, C).

        Returns
        -------
        list of int
            Movement index for each frame pair (length N, with the first
            element set to 0).
        """

        # Calculate movement indices
        movement_indices = []
        movement_indices.append(0)
        for i in range(1, frames.shape[0]):
            _, mi = VideoManager.frame_movements(frames[i - 1, :, :, :], frames[i, :, :, :])
            movement_indices.append(mi)

        return movement_indices

    @staticmethod
    def draw_plot_on_frame(frame, data_points, max_data_length=50, plot_dimensions=(200, 100), margin=10, offset=None,
                           ymin=None, ymax=None):
        """
            Draw a simple line plot of data directly on a video frame.

            This is used to overlay movement traces on top of video frames using
            pure OpenCV operations.

            Parameters
            ----------
            frame : numpy.ndarray
                Current video frame (HxWx3).
            data_points : list of float or int
                Data values to plot.
            max_data_length : int, optional
                Maximum number of recent data points to display.
            plot_dimensions : tuple of int, optional
                Size of the plot (width, height) in pixels.
            margin : int, optional
                Margin inside the plotting area.
            offset : tuple of int, optional
                (x, y) position of the plot in the frame. If None, the plot is
                placed in the bottom right corner.
            ymin, ymax : float, optional
                Explicit minimum and maximum values for scaling. If None, they
                are inferred from data_points.

            Returns
            -------
            numpy.ndarray
                Frame with the plot drawn on it.
        """

        frame = frame.copy()
        plot_img = np.zeros((plot_dimensions[1], plot_dimensions[0], 3), dtype=np.uint8)

        # Scale the data to fit the plot area
        if data_points:
            if ymax is None:
                max_value = max(data_points)
            else:
                max_value = ymax

            if ymin is None:
                min_value = min(data_points)
            else:
                min_value = ymin

            if max_value == min_value:
                max_value += 1  # Avoid division by zero

            scaled_data = [(data - min_value) / (max_value - min_value) * (plot_dimensions[1] - 2 * margin) for data in
                           data_points[-max_data_length:]]
            for i in range(1, len(scaled_data)):
                pt1 = (int((i - 1) * plot_dimensions[0] / max_data_length),
                       plot_dimensions[1] - margin - int(scaled_data[i - 1]))
                pt2 = (int(i * plot_dimensions[0] / max_data_length), plot_dimensions[1] - margin - int(scaled_data[i]))
                cv2.line(plot_img, pt1, pt2, (0, 255, 0), 2)

        if offset is None:
            frame[frame.shape[0] - plot_img.shape[0]:frame.shape[0],
            frame.shape[1] - plot_img.shape[1]:frame.shape[1]] = plot_img
        else:
            x_offset, y_offset = offset  # Position of the plot on the frame
            frame[y_offset:y_offset + plot_img.shape[0], x_offset:x_offset + plot_img.shape[1]] = plot_img

        return frame

    def info(self):
        """
        Return a dictionary with summary information about the recording.

        The dictionary contains basic video metadata, trigger/trial information,
        and a count of valid and invalid trials per `recordings.xlsx`.

        Returns
        -------
        dict
            Information dictionary with keys:
            - file
            - recording_name
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
        """

        dur_sec = self.total_frames / self.fps
        dur_min = dur_sec / 60

        trial_count = self.trigger_tot
        valid_trials = int(np.sum(self.valid))
        invalid_trials = int(len(self.valid) - valid_trials)

        info_dict = {
            "file": self.rec,
            "recording_name": self.rec,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "resolution": (self.frame_width, self.frame_height),
            "duration_sec": round(dur_sec, 2),
            "duration_min": round(dur_min, 2),
            "num_trials": trial_count,
            "first_trigger_frame": int(self.trigger_timestamp[0]) if trial_count > 0 else None,
            "last_trigger_frame": int(self.trigger_timestamp[-1]) if trial_count > 0 else None,
            "intensity_counts": dict(zip(self.stimuli[0], self.stimuli[1])),
            "valid_trials": valid_trials,
            "invalid_trials": invalid_trials,
        }

        return info_dict




