"""
===========================================================
 Minimal PsychoPy Example using VideoManagerArray
 - shows habituation (a loop of a mouse during habituation, no shocks delivered, both at the beginning and during intertrials)
 - iterates through all trials
 - SPACE starts each trial
 - Prints the trigger
 - ESC quits
===========================================================
"""

from psychopy import visual, core, event, monitors
import numpy as np
import cv2
from video_lib import VideoManagerArray   # adjust path if needed


# -------------------------------------------
# USER PARAMETERS
# -------------------------------------------
folder = "res"
pre_sec = 3 # seconds before the trigger
post_sec = 6 # seconds after the trigger

bg_lum = 0
height_deg = 60
img_pos = (0, 0)

# -------------------------------------------
# LOAD VIDEOS
# -------------------------------------------
vms = VideoManagerArray(
    folder=folder,
    pre_sec=pre_sec,
    post_sec=post_sec,
    shuffle_trials=True
)
print('Trial sequence with subjects and trials we used (shuffled for all conditions, always showing intensity 0 in the first trial):')
print(vms.sequence)

print("Loaded subjects:", len(vms))
print("FPS =", vms.fps)

movie_hab = vms[0].get_habituation(flip_vertical=False, to_norm=False)

# stimulus size preserving aspect ratio
imsize_deg = (height_deg, height_deg / vms[0].size_ratio)


# -------------------------------------------
# SET UP WINDOW
# -------------------------------------------
mon = monitors.Monitor('testMonitor')
mon.setDistance(12)
mon.setWidth(30)
mon.setSizePix((800, 800))

win = visual.Window(
    size=(800, 800),
    color=(bg_lum, bg_lum, bg_lum),
    units='deg',
    monitor=mon
)

win.mouseVisible = False

# Main stimulus (we will update .image every frame)
image_stim = visual.ImageStim(
    win,
    size=imsize_deg,
    units='deg',
    pos=img_pos,
    flipVert=True   # ALWAYS flip because OpenCV vs PsychoPy
)

# Label (H or S)
text_stim = visual.TextStim(
    win,
    text="",
    pos=(0, 60),
    color=(1, -1, -1),
    height=8
)

frame_interval = 1.0 / vms.fps


# -------------------------------------------
# HABITUATION LOOP (always showing a loop of a mouse during habituation both at the beginning and during intertrials)
# -------------------------------------------
print("Habituation: press SPACE to begin first trial.")

h = 0
frame_clock = core.Clock()

while True:

    # frame timing
    if frame_clock.getTime() >= frame_interval:
        frame_clock.reset()

        # remove 10×10 trigger pixel block
        frame = movie_hab[h, 10:, 10:, :]

        # grayscale + L norm
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype('float32') / 255.0
        frame = frame - np.mean(frame)

        image_stim.image = frame
        text_stim.text = "H"
        text_stim.draw()
        image_stim.draw()
        win.flip()

        h = (h + 1) % movie_hab.shape[0]

    keys = event.getKeys()

    if 'escape' in keys:
        core.quit()

    if 'space' in keys:
        break


# ========================================================
# TRIALS SEQUENCE
# ========================================================
print("Starting trials. Press SPACE to start each trial. ESC to quit.")


# Loop
while vms.has_next_trial():

    # Fetch next trial tuple
    trl = vms.next_trial()
    print("Next trial:", trl)
    text_stim.text = "H"
    # Wait for SPACE to show next trial stimulus
    print("Press SPACE to start this trial.")
    while True:

        if frame_clock.getTime() >= frame_interval:
            frame_clock.reset()

            # remove 10×10 trigger pixel block
            frame = movie_hab[h, 10:, 10:, :]

            # grayscale + L norm
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype('float32') / 255.0
            frame = frame - np.mean(frame)

            image_stim.image = frame
            text_stim.text = "H"
            text_stim.draw()
            image_stim.draw()
            win.flip()

            h = (h + 1) % movie_hab.shape[0]

        keys = event.getKeys()
        if 'escape' in keys:
            core.quit()
        if 'space' in keys:
            break

    print("Trial Started")
    # Load trial movie & triggers
    movie, trg = vms.get_trial_movie(trl)

    # -------------------------
    # TRIAL LOOP
    # -------------------------
    i = 0
    frame_clock.reset()

    while True:

        if frame_clock.getTime() >= frame_interval:
            frame_clock.reset()

            # grayscale + normalization
            frame = cv2.cvtColor(movie[i], cv2.COLOR_RGB2GRAY).astype('float32') / 255.0
            frame = frame - np.mean(frame)

            image_stim.image = frame

            text_stim.text = "S"
            text_stim.draw()
            image_stim.draw()

            win.flip()

            if trg[i] == 1:
                print(f"Trigger {i} frame")

            if i < movie.shape[0] - 1:
                i += 1
            else:
                print("Trial complete.")
                break

        keys = event.getKeys()
        if 'escape' in keys:
            core.quit()


# -------------------------------------------
# END EXPERIMENT
# -------------------------------------------
print("No more trials.")
win.close()
core.quit()
