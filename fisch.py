import os
import mss
import cv2
import time
import pickle
import random
import pyautogui
import subprocess
import numpy as np
from PIL import Image, ImageGrab

# Q-learning variables
alpha = 0.1
gamma = 0.9
epsilon = 0.1

#Threshold's for the control bar detection
white_threshold = np.array([240, 240, 240])
dark_green_threshold_low = np.array([0, 50, 0])
dark_green_threshold_high = np.array([100, 150, 100])
dark_red_threshold_low = np.array([50, 0, 0])
dark_red_threshold_high = np.array([150, 100, 100])

#Autohotkey clicks (Cause ahk normal clicks are op for macros and couldnt find a library that could do it like them)
click_down = "clicks/click_down.ahk"
click_up = "clicks/click_up.ahk"
click = "clicks/click.ahk"
ahk_source_path = "Ahk/AutoHotkey.exe"
def secret_click(path):
    subprocess.run([ahk_source_path, path])


# Find the nearest multiple of a value
def nearest_multiple(value, multiple=4):
    return (value + (multiple - 1)) // multiple * multiple

#Deterministic approach to take an action
#Based on the position of the control bar/Q-learning
def take_action(Q, x, start, end):
    # Use tuples for state representation in Q
    nearest_start = nearest_multiple(start)
    nearest_end = nearest_multiple(end)

    if (x, nearest_start, nearest_end) not in Q:
        Q[(x, nearest_start, nearest_end)] = np.zeros(3)

    if x > end:
        action = 2
        secret_click(click_down)
    elif start <= x <= end:
        action = 1
        secret_click(click_down)
        time.sleep(0.25)
        secret_click(click_up)
    else:
        action = 0
        secret_click(click_up)

    return action, Q


# Load Q-table from file (if it exists)
def load_q_table(filename="train data/q_learning_table.pkl"):
    try:
        with open(filename, "rb") as file:
            Q = pickle.load(file)
    except FileNotFoundError:
        print("No saved Q-table found, initializing new Q-table.")
        Q = {}  # Initialize a new Q-table as an empty dictionary
    return Q


# Save Q-table to file
def save_q_table(Q, filename="train data/q_learning_table.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(Q, file)


# Q-learning (Cause it's model-free) with a deterministic approach
# While the reward system is incremental
def learn(start, end, x, Q, previous_state, previous_action, streak):
    if previous_state[0] is not None and previous_state[1] is not None and previous_state[2] is not None:
        if previous_state[1] <= previous_state[0] <= previous_state[2]:
            reward = 1
            if streak > 1:
                reward += streak
            streak += 1
        else:
            reward = -10
            streak = 0

        near_prev = (previous_state[0], nearest_multiple(previous_state[1]), nearest_multiple(previous_state[2]))
        near_curr = (x, nearest_multiple(start), nearest_multiple(end))

        if near_prev not in Q:
            Q[near_prev] = np.zeros(3)
        if near_curr not in Q:
            Q[near_curr] = np.zeros(3)

        Q[near_prev][previous_action] += alpha * (reward + gamma * np.max(Q[near_curr]) - Q[near_prev][previous_action])

    current_action, Q = take_action(Q, x, start, end)

    previous_state = (x, start, end)
    previous_action = current_action

    save_q_table(Q)

    return Q, previous_state, previous_action, streak

#Getting the first and last position of our row's biggest sequencee (in this case our image)
#While gaps of max_gap length and disregarding sequences of min_seq length
def get_position(row, min_seq, max_gap):
    max_seq = []
    curr_seq = []
    gaps = 0

    for i, value in enumerate(row):
        if gaps and (len(curr_seq) < min_seq or gaps >= max_gap):
            curr_seq = []
            gaps = 0

        if value:
            curr_seq.append(i)
            if len(curr_seq) > len(max_seq):
                max_seq = curr_seq.copy()
        else:
            gaps += 1

    if len(curr_seq) > len(max_seq):
        max_seq = curr_seq.copy()

    if len(max_seq) > 1:
        return max_seq[0], max_seq[-1]
    else:
        return None, None

#Captures the screen
def capture_screen(monitor=None):
    with mss.mss() as sct:
        if monitor is None:
            monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite('captured_screen.png', img)
    return 'captured_screen.png'

# Finds the center coordinates of an image based around a template
# Uses cv2 template matching with thresholding
def coords(image, template, threshold):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        h, w = image.shape
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        return center_x, center_y
    else:
        return None, None

#Just clicks and throws based on delay
#Clicks twice in order to access the tab and restart the shake minigame in case of an error
def throw(rod_t, delay=2):
    secret_click(click_down)
    secret_click(click_down)
    time.sleep(rod_t)
    secret_click(click_up)
    time.sleep(delay)

#Uses the coords function in order to find the shake buttons and clicks them
def shake(image, delay, threshold, initial_delay=2):
    time.sleep(initial_delay)
    while True:
        x, y = coords(image, capture_screen(), threshold)
        if x is None or y is None:
            break
        else:
            pyautogui.moveTo(x, y)
            secret_click(click)
            time.sleep(delay)

#Finds the target (fish icon) using the coords function
#Uses masking for the varying colors of the control bar and gets the position of it
#Processes the Q-learning
def reel(param_coords, param_image, param_sequence):
    x_start, x_end, y_start, y_fish_height = param_coords
    width = x_end - x_start
    height = 1
    monitor = {"top": y_start - y_fish_height, "left": x_start, "width": width, "height": y_fish_height}
    not_set = True
    count_x = 0

    Q = {}
    Q = load_q_table()
    prev_state = (0, 0, 0)
    prev_act = None
    streak = 0

    for _ in range(150):
        x, _ = coords(param_image[0], capture_screen(monitor), param_image[1])

        if x is None:
            count_x += 1
            if count_x >= param_image[3]:
                break
            else:
                continue

        screenshot = np.array(ImageGrab.grab(bbox=(x_start, y_start, x_start + width, y_start + height)))
        if not_set:
            left_extension = np.tile(screenshot[:, 0, :], (1, param_image[2], 1))
            right_extension = np.tile(screenshot[:, -1, :], (1, param_image[2], 1))
            not_set = False
        screenshot = np.concatenate((left_extension, screenshot, right_extension), axis=1)

        white_mask = np.all(screenshot >= white_threshold, axis=-1)
        dark_green_mask = np.logical_and(
            np.all(screenshot >= dark_green_threshold_low, axis=-1),
            np.all(screenshot <= dark_green_threshold_high, axis=-1)
        )
        dark_red_mask = np.logical_and(
            np.all(screenshot >= dark_red_threshold_low, axis=-1),
            np.all(screenshot <= dark_red_threshold_high, axis=-1)
        )

        combined_mask = np.logical_or(white_mask, np.logical_or(dark_green_mask, dark_red_mask))

        x += param_image[2]
        start, end = get_position(combined_mask[0], param_sequence[0], param_sequence[1])

        if start is None or end is None:
            count_x += 0.5
            continue

        Q, prev_state, prev_act, streak = learn(start, end, x, Q, prev_state, prev_act, streak)


