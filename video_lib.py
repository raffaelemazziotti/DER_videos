import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from pathlib import Path
import os


class TrgManager:
    def __init__(self, filename):
        """
        filename (str): The filename or path where the array will be saved or loaded from.
        """

        if '.' in filename:
            self.filename = '.'.join(filename.split('.')[:-1]) + '.npy'
        else:
            self.filename = filename + '.npy'

    def save_array(self, array):
        """
        Parameters:
        array (numpy.ndarray): The monodimensional array to be saved.

        Raises:
        ValueError: If the array is not monodimensional.
        """
        if array.ndim != 1:
            raise ValueError("The array is not monodimensional.")
        np.save(self.filename, array)

    def load_array(self):
        """

        Returns:
        numpy.ndarray: The loaded monodimensional array.

        Raises:
        ValueError: If the loaded array is not monodimensional.
        """
        array = np.load(self.filename)
        if array.ndim != 1:
            raise ValueError("The loaded array is not monodimensional.")
        return array

    def file_exists(self):
        """
        Check if the file specified during initialization exists.

        Returns:
        bool: True if the file exists, False otherwise.
        """
        return os.path.exists(self.filename)


class VideoManagerArray:

    def __init__(self, folder="res", pre_sec=20, post_sec=20, shuffle_trials=True):
        """
        Initializes a VideoManagerArray instance to manage and access video trial data.

        Parameters:
            folder (str): Path to the dataset root containing 'upright' and 'scrambled' subfolders.
                          Default is 'res', in the current folder.
            pre_sec (int): Seconds to include before each trial starts. Default 20.
            post_sec (int): Seconds to include after each trial ends. Default 20.
            shuffle_trials (bool): Shuffle the order of trials across all videos. Default True.
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
            Generates the trial sequence used by the experiment.

            This method constructs the ordered list of trials that will be
            served by `next_trial()` during stimulus presentation.

            IMPORTANT
            ---------
            This experiment does **not** use all subjects or all trials available
            in the dataset.
            Instead, it uses a fixed subset of subjects, intensities,
            and trial indices that were selected for the experimental protocol.
            These combinations appear explicitly in the list below.

            Each trial in the sequence is represented as a 5-tuple:

                (subject_index, intensity, trial_number, is_phase_scrambled, is_flipped)

            Trial types included:
                • upright videos
                • vertically flipped videos
                • phase-scrambled versions of the same videos

            The first trial is always forced to be:
                (0, 0, 0, False, False)
            This ensures a consistent starting condition before any randomization.

            If `self.shuffle_trials` is True:
                The trial list (excluding the first forced trial) is shuffled.

            After construction:
                -self.sequence contains the full ordered list
                -self.sequence_n stores the number of trials
                -self.sequence_current resets to -1 (so first next_trial() returns index 0)
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
            Retrieves a full stimulus movie segment for a given trial specification.

            Parameters
            ----------
            trial : tuple
                A 5-element tuple describing the trial to extract, with the format:
                    (subject_index, intensity, trial_number, phase_scrambled_flag, flipped_flag)

                Components:
                    subject_index (int)
                        Index of the subject/video in the VideoManagerArray.
                    intensity (int)
                        Shock intensity code. Currently not used for movie extraction.
                    trial_number (int)
                        Index of the trial within the selected subject’s recording.
                    phase_scrambled_flag (bool)
                        If True, the trial is extracted from the phase-scrambled videos (self.vms_ps).
                        If False, the upright/original video is used (self.vms).
                    flipped_flag (bool)
                        If True, the extracted frames are vertically flipped.

            Returns
            -------
            trl : np.ndarray
                A 4-D matrix of shape (frames, height, width, channels)
                containing the extracted movie segment for the requested trial.
                The first 10×10 pixels are removed to exclude the trigger-encoding region.
            trg : np.ndarray
                A 1-D binary array (frames,) indicating whether each frame contains
                a stimulus trigger (1) or not (0).

            Notes
            -----
            • The function automatically computes pre- and post-windows (in frames)
              using the subject-specific FPS, based on the global pre_sec and post_sec
              parameters defined in the VideoManagerArray.

            • The trigger pixels at the top-left corner are cropped out:
                  trl = trl[:, 10:, 10:, :]

            • Flipping is applied **after** cropping and applies to the spatial axis:
                  trl = trl[:, ::-1, :, :]

            • Intensity (trial[1]) is stored in the trial structure for completeness,
              but does not influence which frames are extracted.
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
        Allows for retrieving a specific video manager or trial data using indexing or a tuple.

        Parameters:
            item (int or tuple): If an int, returns the video manager at that index.
            If a tuple, returns trial data for the specified [subject_index, trial_index, if_phase_scrambled (optional)],
            adjusted for pre and post loading times based on fps.

        Returns:
            VideoManager or (frames,trg) trial , depending on the type of 'item'.
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
        Returns a list of valid trials for a specified subject, optionally shuffled.

        Parameters:
            sub (int): Index of the subject to retrieve valid trials for.
            shuffle (bool): If True, shuffles the order of returned trials. Default is True.

        Returns:
            List of tuples representing valid trials for the specified subject.
        """
        temp = [(sub, 0, t) for t in self.vms[sub].get_valid_trials(0)] + [(sub, 4, t) for t in
                                                                           self.vms[sub].get_valid_trials(4)]
        if shuffle:
            np.random.shuffle(temp)
        return temp

    def next_trial(self):
        """
        Advances to and returns the next trial in the sequence.

        Returns:
            Tuple representing the current trial in the sequence (subject, intensity, trial_num).
        """
        self.sequence_current += 1
        current_trial = self.sequence[self.sequence_current]
        return current_trial

    def has_next_trial(self):
        """
        Checks if there is a next trial available in the sequence.

        Returns:
            bool: True if there is a next trial available; False otherwise.
        """
        return self.sequence_current < len(self.sequence) - 1

    def reset_trial(self, reshuffle=True, shuffle_trials=None):
        if reshuffle:
            if shuffle_trials is not None:
                self.shuffle_trials = shuffle_trials
            self.__new_sequence()
        self.sequence_current = 0

    @staticmethod
    def preview(frames, window='Frames', fps=20, repeat=False):
        VideoManager.preview(frames, window=window, fps=fps, repeat=repeat)

    def __len__(self):
        """
        Returns the number of video managers contained within the array.

        Returns:
            int: The number of video managers.
        """
        return len(self.upright)

    def close(self):
        """
       Closes all video managers in the array, releasing any resources they hold.
       """
        for vm in self.vms:
            vm.close()
        if self.phase_scrambled:
            for vm in self.vms_ps:
                vm.close()


class VideoManager:

    def __init__(self, video_path, pattern=[0, 1, 2, 3, 4, 0, 4, 3, 2, 1]):
        """
        Initializes the VideoManager with a specified video file.

        Parameters:
        - video_path (str): The path to the video file to be managed.
        - pattern (list, optional): A list representing the stimulation intensity pattern to apply for each detected trigger event.

        This constructor sets up the video capture, calculates video properties (total frames, fps, dimensions), detects trigger events based on the first pixel's BGR value, and initializes variables for managing stimuli intensity and habituation indices.
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
        Plots the trigger signal used to detect events in the video.

        Parameters:
        - ax (matplotlib.axes.Axes, optional): The axes on which to plot the trigger signal. If None, a new figure and axes are created.

        This method visualizes the normalized trigger signal and marks the detected triggers.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.trigger_norm)
        plt.plot(self.trigger_timestamp, self.trigger_norm[self.trigger_timestamp], "x")

    def plot_movement(self, trial_num, pre=60, post=60, ax=None):
        """
        Plots the movement index for a specified trial.

        Parameters:
        - trial_num (int): The trial number to analyze.
        - pre (int, optional): The number of frames before the trigger to include in the analysis.
        - post (int, optional): The number of frames after the trigger to include in the analysis.
        - phase_scrambled (bool, optional,deprecated): If True, applies phase scrambling to the video frames before analysis.
        - ax (matplotlib.axes.Axes, optional): The axes on which to plot the movement index. If None, a new figure and axes are created.

        This method calculates and plots the movement index across the specified trial.
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
        Plots a matrix of movement indices for multiple trials.

        Parameters:
        - pre (int, optional): The number of frames before the trigger to include in the analysis.
        - post (int, optional): The number of frames after the trigger to include in the analysis.
        - phase_scrambled (bool, optional, deprecated): If True, applies phase scrambling to the video frames before analysis.

        This method visualizes the movement index for multiple trials in a matrix form, facilitating the comparison across different stimuli intensities and repetitions.
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
        Saves the movement matrix plot without showing it.

        Parameters:
            pre (int): Frames before trigger.
            post (int): Frames after trigger.
            outdir (str): Output directory (auto-created if needed).
            fmt (str): File format ('svg', 'png', 'pdf', 'jpg', ...).

        Returns:
            str: Path of the saved file.
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
        Retrieves a specific frame from the video.

        Parameters:
        - i (int): The index of the frame to retrieve.
        - flip_vertical (bool, optional): If True, the frame is flipped vertically. Defaults to False.

        Returns:
        - frame (numpy.ndarray): The requested frame as an array.

        This method is relatively slow for random access and is not recommended for frequent use.
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
        Retrieves a sequence of frames from the video in RGB format.

        Parameters:
        - from_frame (int): The index of the first frame in the sequence to retrieve.
        - to_frame (int): The index of the last frame in the sequence to retrieve, exclusive.
        - to_norm (bool, optional): If True, frames are converted to normalized grayscale. Defaults to False.
        - flip_vertical (bool, optional): If True, each frame is flipped vertically. Defaults to False.

        Returns:
        - res (numpy.ndarray): A 4D array containing the retrieved frames.
          Shape: (number_of_frames, height, width, 3) for RGB or (number_of_frames, height, width) if normalized.

        This method extracts a sequence of frames from the video, optionally normalizes or flips them.
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
        Retrieves the frames corresponding to the habituation phase of the video.

        Parameters:
        - to_norm (bool, optional): If True, frames are converted to normalized grayscale. Defaults to True.
        - flip_vertical (bool, optional): If True, frames are flipped vertically. Defaults to False.

        Returns:
        - res (numpy.ndarray): Frames from the habituation phase.
          Shape: (number_of_frames, height, width, 3) for RGB or (number_of_frames, height, width) if normalized.

        This method extracts the frames from the habituation phase using the stored indices.
        """
        return self.get_frames(
            self.habituation_inds[0],
            self.habituation_inds[1],
            to_norm=to_norm,
            flip_vertical=flip_vertical
        )

    def get_next_frame(self, flip_vertical=False):
        """
        Retrieves the next frame from the video.

        Parameters:
        - flip_vertical (bool, optional): If True, the frame is flipped vertically. Defaults to False.

        Returns:
        - frame (numpy.ndarray or None): The next frame as an array, or None if reading fails.

        This method simplifies sequential frame retrieval during video playback or analysis.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        if flip_vertical:
            frame = frame[::-1, :, :]

        return frame

    def get_trial(self, trial, pre=60, post=60, to_norm=False, flip_vertical=False):
        """
        Extracts a trial from the video with a specified pre- and post-interval around the trigger.

        Parameters:
        - trial (int): The index of the trigger event around which to extract the trial.
        - pre (int, optional): The number of frames before the trigger to include in the trial.
        - post (int, optional): The number of frames after the trigger to include in the trial.
        - to_norm (bool, optional): If True, frames are converted to normalized grayscale. Defaults to False.

        Returns:
        - res (numpy.ndarray): The extracted frames as an array.
        - trg (numpy.ndarray): An array indicating the trigger point within the trial.

        This method allows for the extraction of segments from the video based on detected trigger events, useful for analyzing specific events within the video.
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
           Displays a preview of a specific trial.

           Parameters:
           - trial_num (int): The trial index to preview.
           - pre (int, optional): Number of frames before the trigger to show.
           - post (int, optional): Number of frames after the trigger to show.
           - showlabels (bool, optional): Whether to display text labels.
           - repeat (bool, optional): Whether to loop the preview.
           - window (str, optional): The name of the window to display the preview.
           - movements (bool, optional): Whether to highlight movements.
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
        Previews all trials in a single window.

        Parameters:
        - pre (int, optional): Frames before each trigger to include.
        - post (int, optional): Frames after each trigger to include.
        - frame_height (int, optional): Height of the preview window.
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
        Generates and saves a video with the video on the hard disk.

        Parameters:
        - pre (int, optional): Frames before each trigger to include.
        - post (int, optional): Frames after each trigger to include.
        - frame_height (int, optional): Height of the preview window.
        - video_filename (str, optional): Name of the output video file.
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
        Generates and saves an optimized GIF for the full preview mosaic.

        Parameters:
            pre, post : trial window
            frame_height : mosaic height
            gif_filename : output name (.gif)
            resize_factor : scale output (1.0 = full size, 0.5 = half)
            dither : enable/disable Pillow dithering
            palette : number of colors (max 256)
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
        Resets the video to the first frame.

        This method is useful for restarting the analysis or preview from the beginning of the video.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def gotoframe(self, i):
        """
        Sets the current video position to a specified frame.

        Parameters:
        - i (int): The frame number to jump to.

        This method enables direct access to any frame in the video, adjusting the internal pointer to the specified frame number.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    def close(self):
        """
       Releases the video capture object.

       This method should be called when the VideoManager instance is no longer needed, to properly release system resources.
       """
        self.cap.release()

    def get_valid_trials(self, intensity):
        valids = self.stim_db[self.stim_db['valid'] == 1]
        temp = list(valids[valids['intensity'] == intensity].index)
        return temp

    @staticmethod
    def frame_movements(prev_frame, curr_frame, gfilt=5, thr=10):
        """
       Calculates movement between two frames.

       Parameters:
       - prev_frame (numpy.ndarray): The previous frame.
       - curr_frame (numpy.ndarray): The current frame.
       - gfilt (int, optional): Gaussian filter size.
       - thr (int, optional): Threshold for movement detection.

       Returns:
       - thresh_frame (numpy.ndarray): Thresholded frame showing movement.
       - movement_index (int): The quantity of movement detected.
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
        Lists all videos in a specified directory.

        Parameters:
        - pth (str, optional): The path to the directory containing video files. Defaults to 'res'.
        - verbosity (int, optional): The level of verbosity for output. A non-zero value prints the list of video files.

        Returns:
        - videos (list of str): A list of video file paths.

        This static method provides a utility function to retrieve and optionally print a list of video files in a given directory.
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
        Applies phase scrambling to an image.

        Parameters:
        - frame (numpy.ndarray): Image to scramble.
        - phase_scrambling_matrix (numpy.ndarray, optional): Pre-generated phase matrix.
        - seed (int, optional): Seed for reproducibility.
        - to_rgb (bool, optional): Whether to convert the output to RGB.

        Returns:
        - scrambled_frame_normalized (numpy.ndarray): Scrambled image.
        - phase_scrambling_matrix (numpy.ndarray): Used or generated phase matrix.

        For techniques like phase scrambling, maintaining the same phase shifts or randomization patterns for moving objects across frames can help preserve temporal coherence. This approach ensures that while the content may be scrambled, its movement remains smooth and consistent across frames.
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
        Displays a preview of frames.

        Parameters:
        - frames (numpy.ndarray): Frames to display.
        - window (str, optional): Window name.
        - fps (int, optional): Frames per second for playback.
        - repeat (bool, optional): Whether to loop the playback.
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
        Calculates movement indices for a sequence of frames.

        This method computes the movement index for each pair of consecutive frames
        in a given 4D matrix of frames. The movement index is calculated as the sum
        of absolute differences between consecutive frames, which indicates the amount
        of change or movement between them.

        Parameters:
        - frames (numpy.ndarray): A 4D numpy array of frames with shape (num_frames, height, width, channels).

        Returns:
        - movement_indices (list of int): A list containing the movement index for each pair of consecutive frames.
        """
        # Convert frames to grayscale if they are in color
        # if frames.shape[-1] == 3:
        #    gray_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
        # else:
        #    gray_frames = frames

        # Calculate movement indices
        movement_indices = []
        movement_indices.append(0)
        for i in range(1, frames.shape[0]):
            # Calculate the absolute difference between consecutive frames
            # diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            # Sum all differences to get the movement index
            # movement_index = np.sum(diff)
            # movement_indices.append(movement_index)
            _, mi = VideoManager.frame_movements(frames[i - 1, :, :, :], frames[i, :, :, :])
            movement_indices.append(mi)

        return movement_indices

    @staticmethod
    def draw_plot_on_frame(frame, data_points, max_data_length=50, plot_dimensions=(200, 100), margin=10, offset=None,
                           ymin=None, ymax=None):
        """
        Draw a simple line plot directly on a video frame using OpenCV.

        :param frame: The current video frame.
        :param data_points: List of data points to plot.
        :param max_data_length: Maximum number of data points to display in the plot.
        :param plot_dimensions: Tuple of the plot size (width, height).
        :param margin: Margin around the plot inside the plotting area.
        :param offset: position (x,y) of the plot inside the frame (default is lower right).
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
        Returns a dictionary with information about the video.
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




