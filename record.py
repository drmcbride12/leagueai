# recorder_v2.py
import time
import uuid
import os
import cv2
import numpy as np
from mss import mss
from pynput import mouse, keyboard

# --- Configuration ---
OUTPUT_DIR = "lol_recorded_data_v2"
CAPTURE_RESOLUTION = (1920, 1080) # Your game resolution
TARGET_RESOLUTION = (256, 144)    # Downsampled resolution for the model
FPS_TARGET = 16
# SEQUENCE_LENGTH = 8 # Not used directly in recorder, but good to keep in mind for trainer

# --- Action Definitions ---
# This vector will be saved. Trainer will split it.
# [mouse_x, mouse_y,                             # 2 elements
#  click_type (0:None, 1:Left, 2:Right, 3:Shift+Right), # 1 element
#  Q, W, E, R, D, F,                             # 6 elements
#  Key_1, Key_2, ..., Key_0,                     # 10 elements
#  TAB, GRAVE,                                   # 2 elements
#  CTRL_Q, CTRL_W, CTRL_E, CTRL_R]              # 4 elements
# Total elements = 2 + 1 + 6 + 10 + 2 + 4 = 25
RAW_ACTION_VECTOR_SIZE = 25

# Indices for raw action vector
MOUSE_X_IDX, MOUSE_Y_IDX = 0, 1
CLICK_TYPE_IDX = 2
KEYS_START_IDX = 3 # Q is the first key after click_type

# Mapping for simple keys (Q, W, E, R, D, F, 1-0, TAB, GRAVE)
# Value is their offset from KEYS_START_IDX
KEY_MAPPING = {
    'q': 0, 'w': 1, 'e': 2, 'r': 3, 'd': 4, 'f': 5,
    '1': 6, '2': 7, '3': 8, '4': 9, '5': 10,
    '6': 11, '7': 12, '8': 13, '9': 14, '0': 15,
    '`': 16, # Grave accent / backtick
    keyboard.Key.tab: 17,
}
# Indices for Ctrl+Ability keys, offset from KEYS_START_IDX
CTRL_Q_IDX_OFFSET = 18
CTRL_W_IDX_OFFSET = 19
CTRL_E_IDX_OFFSET = 20
CTRL_R_IDX_OFFSET = 21

# --- Global Variables for Input State ---
recording = False
mouse_listener = None
keyboard_listener = None

# Mouse state
current_mouse_pos = (0, 0)
# For click_type: 0=None, 1=Left, 2=Right, 3=Shift+Right
# This will be determined *at the moment of capture* based on button state and shift state
# We need to track raw button presses to determine click_type later
_current_left_pressed = False
_current_right_pressed = False

# Keyboard state
_current_keys_pressed = set() # Stores pynput.keyboard.Key or char
_ctrl_pressed = False
_shift_pressed = False


# --- Event Handlers for Input Recording ---
def on_move(x, y):
    global current_mouse_pos
    current_mouse_pos = (x,y)

def on_click(x, y, button, pressed):
    global _current_left_pressed, _current_right_pressed
    if button == mouse.Button.left:
        _current_left_pressed = pressed
    elif button == mouse.Button.right:
        _current_right_pressed = pressed

def on_press(key):
    global _current_keys_pressed, _ctrl_pressed, _shift_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        _ctrl_pressed = True
    elif key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
        _shift_pressed = True
    else:
        _current_keys_pressed.add(key)

def on_release(key):
    global _current_keys_pressed, _ctrl_pressed, _shift_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        _ctrl_pressed = False
    elif key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
        _shift_pressed = False
    else:
        try:
            _current_keys_pressed.remove(key)
        except KeyError:
            pass # Key might have been released already or wasn't tracked

# --- Main Recording Loop ---
def start_recording():
    global recording, mouse_listener, keyboard_listener
    recording = True
    print("Starting recording... Press Ctrl+C in this terminal to stop.")

    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    mouse_listener.start()
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()

    sct = mss()
    monitor = sct.monitors[1] # Adjust if needed for primary monitor

    frame_buffer_paths = []
    action_buffer_vectors = []
    session_id = str(uuid.uuid4())
    session_path = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(os.path.join(session_path, "frames"), exist_ok=True)

    frame_count = 0
    last_capture_time = time.time()

    try:
        while recording:
            current_time = time.time()
            if (current_time - last_capture_time) < (1.0 / FPS_TARGET):
                time.sleep(0.001)
                continue
            last_capture_time = current_time

            # 1. Capture screen
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 2. Construct Action Vector
            raw_action_vector = np.zeros(RAW_ACTION_VECTOR_SIZE, dtype=np.float32)

            # Mouse Coords
            raw_action_vector[MOUSE_X_IDX] = np.clip(current_mouse_pos[0] / CAPTURE_RESOLUTION[0], 0.0, 1.0)
            raw_action_vector[MOUSE_Y_IDX] = np.clip(current_mouse_pos[1] / CAPTURE_RESOLUTION[1], 0.0, 1.0)

            # Click Type (0:None, 1:Left, 2:Right, 3:Shift+Right)
            click_val = 0 # None
            if _current_left_pressed:
                click_val = 1 # Left
            elif _current_right_pressed:
                if _shift_pressed:
                    click_val = 3 # Shift+Right
                else:
                    click_val = 2 # Right
            raw_action_vector[CLICK_TYPE_IDX] = float(click_val)

            # Keys (Q,W,E,R,D,F, 1-0, TAB, GRAVE)
            for key_val, idx_offset in KEY_MAPPING.items():
                # pynput uses key.char for simple keys, or direct key objects for special ones
                processed_key_val = key_val.char if hasattr(key_val, 'char') else key_val
                if processed_key_val in _current_keys_pressed or key_val in _current_keys_pressed:
                    raw_action_vector[KEYS_START_IDX + idx_offset] = 1.0
            
            # Handle single char keys not covered by special Key objects if they are in _current_keys_pressed
            # (e.g. if '`' was pressed directly as char)
            # This part is a bit tricky due to pynput's representation.
            # The KEY_MAPPING should ideally store the exact values pynput uses in _current_keys_pressed.
            # For simplicity, the current KEY_MAPPING uses chars for simple keys and Key objects for tab.
            # This might need refinement based on how pynput stores specific chars like '`'.
            # A common way is to check key.char if available:
            for pressed_key_obj in _current_keys_pressed:
                char_version = None
                if hasattr(pressed_key_obj, 'char'):
                    char_version = pressed_key_obj.char
                
                if char_version == '`' and '`' in KEY_MAPPING: # Check for grave specifically
                     raw_action_vector[KEYS_START_IDX + KEY_MAPPING['`']] = 1.0


            # Ctrl + Ability Keys
            if _ctrl_pressed:
                if 'q' in _current_keys_pressed or (hasattr(keyboard.KeyCode(char='q'), 'vk') and keyboard.KeyCode(char='q') in _current_keys_pressed): # Check char 'q'
                    raw_action_vector[KEYS_START_IDX + CTRL_Q_IDX_OFFSET] = 1.0
                if 'w' in _current_keys_pressed or (hasattr(keyboard.KeyCode(char='w'), 'vk') and keyboard.KeyCode(char='w') in _current_keys_pressed):
                    raw_action_vector[KEYS_START_IDX + CTRL_W_IDX_OFFSET] = 1.0
                if 'e' in _current_keys_pressed or (hasattr(keyboard.KeyCode(char='e'), 'vk') and keyboard.KeyCode(char='e') in _current_keys_pressed):
                    raw_action_vector[KEYS_START_IDX + CTRL_E_IDX_OFFSET] = 1.0
                if 'r' in _current_keys_pressed or (hasattr(keyboard.KeyCode(char='r'), 'vk') and keyboard.KeyCode(char='r') in _current_keys_pressed):
                    raw_action_vector[KEYS_START_IDX + CTRL_R_IDX_OFFSET] = 1.0
            
            # 3. Save frame and action
            frame_filename_rel = os.path.join("frames", f"frame_{frame_count:07d}.png")
            frame_filename_abs = os.path.join(session_path, frame_filename_rel)
            
            processed_frame = cv2.resize(frame_bgr, TARGET_RESOLUTION, interpolation=cv2.INTER_AREA)
            cv2.imwrite(frame_filename_abs, processed_frame)
            
            frame_buffer_paths.append(frame_filename_rel) # Save relative path for portability
            action_buffer_vectors.append(raw_action_vector)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Recorded {frame_count} frames...")

    except KeyboardInterrupt:
        print("Recording stopped by user.")
    finally:
        recording = False
        if mouse_listener: mouse_listener.stop()
        if keyboard_listener: keyboard_listener.stop()
        print("Cleaning up listeners...")

        if action_buffer_vectors:
            actions_np = np.array(action_buffer_vectors, dtype=np.float32)
            np.save(os.path.join(session_path, "actions_raw.npy"), actions_np)
            with open(os.path.join(session_path, "frame_paths.txt"), "w") as f:
                for p in frame_buffer_paths:
                    f.write(p + "\n")
        print(f"Recording saved to {session_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    start_recording()