# trainer_v2.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Dropout, Flatten, Dense, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

# --- Configuration ---
DATA_DIR = "lol_recorded_data_v2"
MODEL_SAVE_PATH = "lol_bc_model_v2.keras" # Using .keras format
IMG_HEIGHT, IMG_WIDTH = 144, 256
SEQUENCE_LENGTH = 8 # Number of frames per sequence
BATCH_SIZE = 32
EPOCHS = 50 # Adjust as needed

# --- Action Vector Details (from recorder_v2.py) ---
# RAW_ACTION_VECTOR_SIZE = 25
# MOUSE_X_IDX, MOUSE_Y_IDX = 0, 1
# CLICK_TYPE_IDX = 2
# KEYS_START_IDX = 3 # Q is the first key after click_type
# Number of distinct click types (0:None, 1:Left, 2:Right, 3:Shift+Right)
NUM_CLICK_TYPES = 4
# Number of other keys (Q,W,E,R,D,F, 0-9, TAB, GRAVE, CtrlQ,CtrlW,CtrlE,CtrlR)
# 6 + 10 + 2 + 4 = 22 keys
NUM_OTHER_KEYS = 22


# --- Data Loading and Preprocessing ---
def load_and_prepare_data_from_session(session_path):
    frame_paths_file = os.path.join(session_path, "frame_paths.txt")
    actions_file = os.path.join(session_path, "actions_raw.npy")

    if not os.path.exists(frame_paths_file) or not os.path.exists(actions_file):
        return [], [], [], []

    with open(frame_paths_file, "r") as f:
        # frame_paths are relative to session_path
        frame_paths_rel = [line.strip() for line in f.readlines()]
        frame_paths_abs = [os.path.join(session_path, rel_path) for rel_path in frame_paths_rel]

    raw_actions = np.load(actions_file) # Shape: (num_total_frames, RAW_ACTION_VECTOR_SIZE)

    # Split raw_actions into targets for each head
    mouse_xy_targets = raw_actions[:, 0:2] # Mouse X, Y
    
    click_type_targets_int = raw_actions[:, 2].astype(int) # Click Type (0, 1, 2, 3)
    click_type_targets_onehot = to_categorical(click_type_targets_int, num_classes=NUM_CLICK_TYPES)
    
    other_keys_targets = raw_actions[:, 3:] # All other keys

    return frame_paths_abs, mouse_xy_targets, click_type_targets_onehot, other_keys_targets

def create_sequences(frame_paths, mouse_targets, click_targets, key_targets, sequence_length):
    sequences_X = []
    sequences_y_mouse = []
    sequences_y_click = []
    sequences_y_keys = []

    for i in range(len(frame_paths) - sequence_length + 1): # Ensure we can form a full sequence
        sequence_frames_data = []
        valid_sequence = True
        for j in range(sequence_length):
            img_path = frame_paths[i+j]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                valid_sequence = False
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0 # Normalize
            sequence_frames_data.append(img)
        
        if not valid_sequence or len(sequence_frames_data) != sequence_length:
            continue

        sequences_X.append(np.array(sequence_frames_data))
        # Action corresponding to the *last* frame in the sequence
        action_idx = i + sequence_length - 1
        sequences_y_mouse.append(mouse_targets[action_idx])
        sequences_y_click.append(click_targets[action_idx])
        sequences_y_keys.append(key_targets[action_idx])

    return (np.array(sequences_X), 
            np.array(sequences_y_mouse), 
            np.array(sequences_y_click), 
            np.array(sequences_y_keys))


def build_multi_head_model(sequence_length, img_height, img_width, 
                           num_click_types, num_other_keys):
    input_shape = (sequence_length, img_height, img_width, 3)
    inp = Input(shape=input_shape, name="input_frames")

    # Shared base (ConvLSTM)
    x = ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False)(x) # Last sequence layer
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    shared_embedding = Dense(256, activation='relu')(x)
    shared_embedding = Dropout(0.4)(shared_embedding)

    # Head 1: Mouse Coordinates (Regression)
    mouse_output = Dense(128, activation='relu')(shared_embedding)
    mouse_output = Dense(2, activation='sigmoid', name='mouse_xy_output')(mouse_output) # 0-1 for x,y

    # Head 2: Click Type (Classification - mutually exclusive)
    click_output = Dense(64, activation='relu')(shared_embedding)
    click_output = Dense(num_click_types, activation='softmax', name='click_type_output')(click_output)

    # Head 3: Other Keys (Multi-label Classification)
    keys_output = Dense(128, activation='relu')(shared_embedding)
    keys_output = Dense(num_other_keys, activation='sigmoid', name='other_keys_output')(keys_output)
    
    model = Model(inputs=inp, outputs=[mouse_output, click_output, keys_output])

    losses = {
        'mouse_xy_output': 'mse',
        'click_type_output': 'categorical_crossentropy',
        'other_keys_output': 'binary_crossentropy'
    }
    loss_weights = { # Adjust if one task is harder to learn or more important
        'mouse_xy_output': 1.0, 
        'click_type_output': 1.0, 
        'other_keys_output': 1.0
    }
    metrics = {
        'mouse_xy_output': ['mae'], # Mean Absolute Error for coords
        'click_type_output': ['accuracy'],
        'other_keys_output': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    }

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics=metrics)
    model.summary()
    return model

if __name__ == "__main__":
    all_frame_paths_global = []
    all_mouse_targets_global = []
    all_click_targets_global = []
    all_key_targets_global = []

    for session_folder in os.listdir(DATA_DIR):
        session_path = os.path.join(DATA_DIR, session_folder)
        if os.path.isdir(session_path):
            print(f"Loading data from session: {session_folder}")
            f_paths, m_targets, c_targets, k_targets = load_and_prepare_data_from_session(session_path)
            if f_paths:
                all_frame_paths_global.extend(f_paths)
                all_mouse_targets_global.append(m_targets)
                all_click_targets_global.append(c_targets)
                all_key_targets_global.append(k_targets)

    if not all_frame_paths_global:
        print("No data found. Please record data first.")
        exit()

    all_mouse_targets_global = np.concatenate(all_mouse_targets_global, axis=0)
    all_click_targets_global = np.concatenate(all_click_targets_global, axis=0)
    all_key_targets_global = np.concatenate(all_key_targets_global, axis=0)

    print(f"Total frames paths: {len(all_frame_paths_global)}")
    print(f"Mouse targets shape: {all_mouse_targets_global.shape}")
    print(f"Click targets shape: {all_click_targets_global.shape}")
    print(f"Key targets shape: {all_key_targets_global.shape}")

    X_seq, y_mouse_seq, y_click_seq, y_keys_seq = create_sequences(
        all_frame_paths_global, all_mouse_targets_global, all_click_targets_global, all_key_targets_global, SEQUENCE_LENGTH
    )
    
    if X_seq.shape[0] == 0:
        print("Not enough data to form sequences. Record more data or check sequence length.")
        exit()

    print(f"Number of sequences: {X_seq.shape[0]}")
    print(f"X_sequences shape: {X_seq.shape}")
    print(f"y_mouse_seq shape: {y_mouse_seq.shape}")
    print(f"y_click_seq shape: {y_click_seq.shape}")
    print(f"y_keys_seq shape: {y_keys_seq.shape}")

    # Split data
    (X_train, X_val, 
     y_mouse_train, y_mouse_val,
     y_click_train, y_click_val,
     y_keys_train, y_keys_val) = train_test_split(
        X_seq, y_mouse_seq, y_click_seq, y_keys_seq, test_size=0.2, random_state=42
    )
    
    # Prepare y_train and y_val as lists of arrays for Keras multi-output
    y_train_dict = {'mouse_xy_output': y_mouse_train, 'click_type_output': y_click_train, 'other_keys_output': y_keys_train}
    y_val_dict = {'mouse_xy_output': y_mouse_val, 'click_type_output': y_click_val, 'other_keys_output': y_keys_val}


    model = build_multi_head_model(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, NUM_CLICK_TYPES, NUM_OTHER_KEYS)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)
    ]

    print("Starting training...")
    history = model.fit(
        X_train, y_train_dict,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val_dict),
        callbacks=callbacks
    )
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")