# inference_v2.py
import time
import numpy as np
import tensorflow as tf
import cv2
from mss import mss
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

# --- Configuration ---
MODEL_PATH = "lol_bc_model_v2.keras"
IMG_HEIGHT, IMG_WIDTH = 144, 256
SEQUENCE_LENGTH = 8
FPS_TARGET = 16 # Inference FPS
CAPTURE_RESOLUTION = (1920, 1080) # Your screen resolution for denormalizing mouse

# Action Thresholds
KEY_ACTION_THRESHOLD = 0.5 # For sigmoid outputs of keys

# --- Action Definitions (must match recorder & trainer) ---
# Click Types (model output is one-hot, we'll use argmax)
# 0:None, 1:Left, 2:Right, 3:Shift+Right
CLICK_TYPE_MAP = {0: "None", 1: "Left", 2: "Right", 3: "ShiftRight"}

# Other Keys (order must match the model's 'other_keys_output' head)
# Q, W, E, R, D, F, 1-0, TAB, GRAVE, CTRL_Q, CTRL_W, CTRL_E, CTRL_R
OTHER_KEYS_ORDER = [
    'q', 'w', 'e', 'r', 'd', 'f',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    Key.tab, '`',  # pynput Key.tab, char '`'
    "ctrl_q", "ctrl_w", "ctrl_e", "ctrl_r" # Custom labels for ctrl combos
]
# For pynput, map custom labels to actual key presses
PINPUT_KEY_MAP = {
    'q':'q', 'w':'w', 'e':'e', 'r':'r', 'd':'d', 'f':'f',
    '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', 
    '6':'6', '7':'7', '8':'8', '9':'9', '0':'0',
    Key.tab: Key.tab, '`':'`',
    # Ctrl combos will be handled specially by pressing Ctrl then the key
}
BASE_ABILITY_KEYS_FOR_CTRL = ["ctrl_q", "ctrl_w", "ctrl_e", "ctrl_r"]
TARGET_KEYS_FOR_CTRL = ['q','w','e','r']


# --- Global Variables ---
model = None
mouse_ctl = MouseController()
kb_ctl = KeyboardController()
frame_sequence_buffer = []
running = False
# Keep track of keys AI is holding down to manage releases
ai_held_keys = set()
ai_held_ctrl = False
ai_held_shift = False


def release_all_ai_keys():
    global ai_held_keys, ai_held_ctrl, ai_held_shift
    print("Releasing all AI controlled keys...")
    for key_obj in list(ai_held_keys): # Iterate over a copy
        try:
            kb_ctl.release(key_obj)
        except Exception as e: # Some keys might not be releasable if not pressed by pynput (e.g. by user)
            # print(f"Minor issue releasing {key_obj}: {e}")
            pass
    ai_held_keys.clear()
    if ai_held_ctrl:
        kb_ctl.release(Key.ctrl)
        ai_held_ctrl = False
    if ai_held_shift:
        kb_ctl.release(Key.shift)
        ai_held_shift = False


def execute_actions(mouse_pred, click_type_pred_probs, keys_pred):
    global ai_held_keys, ai_held_ctrl, ai_held_shift

    # 1. Mouse Movement (mouse_pred: [x, y])
    target_x = int(mouse_pred[0] * CAPTURE_RESOLUTION[0])
    target_y = int(mouse_pred[1] * CAPTURE_RESOLUTION[1])
    mouse_ctl.position = (target_x, target_y)

    # 2. Click Type (click_type_pred_probs: softmax output)
    predicted_click_idx = np.argmax(click_type_pred_probs)
    # print(f"Click pred: {CLICK_TYPE_MAP.get(predicted_click_idx, 'Unknown')}, Probs: {click_type_pred_probs}")

    # Manage Shift for Shift+Right Click
    # If model wants Shift+Right, but Shift is not held by AI, press it.
    # If model doesn't want Shift+Right, but Shift is held by AI (for this purpose), release it.
    is_shift_right_click_pred = (predicted_click_idx == 3)
    
    if is_shift_right_click_pred and not ai_held_shift:
        kb_ctl.press(Key.shift)
        ai_held_shift = True
        # print("AI Pressing SHIFT for click")
    elif not is_shift_right_click_pred and ai_held_shift:
        # Only release shift if it was pressed FOR a shift-click and no longer needed for it.
        # More complex logic might be needed if Shift is used for other AI keys.
        # For now, assume Shift is mainly for Shift+Right Click from this head.
        kb_ctl.release(Key.shift)
        ai_held_shift = False
        # print("AI Releasing SHIFT after click")


    if predicted_click_idx == 1: # Left Click
        mouse_ctl.click(Button.left, 1) # Single quick click
    elif predicted_click_idx == 2: # Right Click
        mouse_ctl.click(Button.right, 1)
    elif predicted_click_idx == 3: # Shift+Right Click
        # Shift should be handled above. Now just right click.
        mouse_ctl.click(Button.right, 1)
    # else: click_idx == 0 (None), do nothing.


    # 3. Other Keys (keys_pred: sigmoid outputs for each key)
    # Determine if any Ctrl+Ability is predicted
    any_ctrl_ability_pred = False
    for i, key_label in enumerate(OTHER_KEYS_ORDER):
        if key_label in BASE_ABILITY_KEYS_FOR_CTRL and keys_pred[i] > KEY_ACTION_THRESHOLD:
            any_ctrl_ability_pred = True
            break
    
    # Manage Ctrl press/release based on Ctrl+Ability predictions
    if any_ctrl_ability_pred and not ai_held_ctrl:
        kb_ctl.press(Key.ctrl)
        ai_held_ctrl = True
        # print("AI Pressing CTRL")
    elif not any_ctrl_ability_pred and ai_held_ctrl:
        kb_ctl.release(Key.ctrl)
        ai_held_ctrl = False
        # print("AI Releasing CTRL")

    # Handle individual key presses/releases
    for i, key_label in enumerate(OTHER_KEYS_ORDER):
        key_active = keys_pred[i] > KEY_ACTION_THRESHOLD
        
        pynput_key = None
        is_ctrl_combo_key = False

        if key_label in PINPUT_KEY_MAP:
            pynput_key = PINPUT_KEY_MAP[key_label]
        elif key_label in BASE_ABILITY_KEYS_FOR_CTRL: # e.g. "ctrl_q"
            is_ctrl_combo_key = True
            # The actual key to press is the base ability key ('q', 'w', 'e', 'r')
            # Ctrl itself is handled above.
            base_key_idx = BASE_ABILITY_KEYS_FOR_CTRL.index(key_label)
            pynput_key = TARGET_KEYS_FOR_CTRL[base_key_idx]
        
        if pynput_key is None:
            continue # Should not happen if mappings are correct

        if key_active:
            if pynput_key not in ai_held_keys:
                # print(f"AI Pressing: {pynput_key} (Ctrl: {ai_held_ctrl if is_ctrl_combo_key else 'N/A'})")
                kb_ctl.press(pynput_key)
                ai_held_keys.add(pynput_key)
        else: # Not active
            if pynput_key in ai_held_keys:
                # print(f"AI Releasing: {pynput_key}")
                kb_ctl.release(pynput_key)
                ai_held_keys.remove(pynput_key)


def start_inference():
    global model, running, frame_sequence_buffer, ai_held_keys, ai_held_ctrl, ai_held_shift
    if model is None:
        print("Loading model...")
        # When loading a model with custom architecture or layers,
        # sometimes you might need to provide custom_objects, but for standard layers it's often fine.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False) # compile=False speeds up loading if not retraining
        print("Model loaded.")

    running = True
    sct = mss()
    monitor = sct.monitors[1]

    last_inference_time = time.time()
    
    print("Starting inference in 3 seconds. Switch to the League of Legends game window.")
    print("Press Ctrl+C in this terminal to stop.")
    time.sleep(3)

    try:
        while running:
            current_time = time.time()
            if (current_time - last_inference_time) < (1.0 / FPS_TARGET):
                time.sleep(0.001)
                continue
            last_inference_time = current_time

            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            frame_resized = cv2.resize(frame_rgb, (IMG_WIDTH, IMG_HEIGHT))
            frame_normalized = frame_resized / 255.0

            frame_sequence_buffer.append(frame_normalized)
            if len(frame_sequence_buffer) > SEQUENCE_LENGTH:
                frame_sequence_buffer.pop(0)
            
            if len(frame_sequence_buffer) == SEQUENCE_LENGTH:
                input_sequence = np.expand_dims(np.array(frame_sequence_buffer), axis=0)
                
                # Model prediction returns a list of outputs, one for each head
                predictions = model.predict(input_sequence, verbose=0) 
                mouse_coords_pred = predictions[0][0]      # Shape (2,)
                click_type_probs_pred = predictions[1][0]  # Shape (NUM_CLICK_TYPES,)
                other_keys_probs_pred = predictions[2][0]  # Shape (NUM_OTHER_KEYS,)
                
                execute_actions(mouse_coords_pred, click_type_probs_pred, other_keys_probs_pred)

    except KeyboardInterrupt:
        print("Inference stopped by user.")
    finally:
        running = False
        release_all_ai_keys()
        print("Cleanup complete.")

if __name__ == "__main__":
    start_inference()