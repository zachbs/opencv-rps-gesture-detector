import cv2
import tensorflow as tf
import numpy as np
import time
import mediapipe as mp
import os
import glob
import math

# --- PRE-FLIGHT CHECKLIST ---
MODEL_PATH = 'image_classification_model2.h5'
CLASS_LABELS = ['paper', 'rock', 'scissors']
IMG_SIZE = (150, 150)
HAND_MODEL_PATH = 'hand_landmarker.task'
SEGMENT_MODEL_PATH = 'selfie_multiclass_256x256.tflite'
PADDING_FACTOR = 0.1
SEGMENTATION_THRESHOLD = 0.5
TARGET_CATEGORY_INDEX = 2

# --- NEW: Frame Rate Optimization Settings ---
process_every_n_frames = 3  # Process every 3rd frame. Increase for faster FPS, more lag.
frame_count = 0

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE, # Still use IMAGE mode for synchronous detection
    num_hands=1
)

segment_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=SEGMENT_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    output_confidence_masks=True
)

# --- TIMING SUMMARY ---
def print_timing_summary(timings):
    print("\n================ TIMING SUMMARY (ms) ================")
    for name, values in timings.items():
        if not values:
            print(f"{name:<22}: no samples")
            continue
        arr = np.array(values) * 1000.0
        print(f"{name:<22}: median {np.median(arr):7.1f} | mean {arr.mean():7.1f} | "
              f"min {arr.min():7.1f} | max {arr.max():7.1f} | n={len(arr)}")

# Timing samples (seconds), recorded only on processed frames (every Nth frame)
timings = {"hand_detection": [], "segmentation": [], "model_prediction": [], "processed_frame_total": []}

# --- LOAD THE TRAINED MODEL ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

def calculate_bounding_box(image_shape, landmarks):
    h, w = image_shape[:2]
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    box_w_px = (x_max - x_min) * w
    box_h_px = (y_max - y_min) * h
    x_min_px = int(x_min * w)
    y_min_px = int(y_min * h)
    padding_w = int(box_w_px * PADDING_FACTOR)
    padding_h = int(box_h_px * PADDING_FACTOR)
    x_min_pad = max(0, x_min_px - padding_w)
    y_min_pad = max(0, y_min_px - padding_h)
    x_max_pad = min(w, x_min_px + int(box_w_px) + padding_w)
    y_max_pad = min(h, y_min_px + int(box_h_px) + padding_h)
    return x_min_pad, y_min_pad, x_max_pad, y_max_pad

# --- INITIALIZE WEBCAM ---
cap = cv2.VideoCapture(0)
# Optional: Set lower resolution here if needed
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nWebcam opened. Press 'q' to quit.")
startTime = time.time()
timeLapsed = 0.0 # Use float for more precision
randomPrediction = None

# --- NEW: Variables to store the last successful prediction ---
last_predicted_label = "Starting..."
last_confidence = 0.0
last_hand_cutout_display = np.zeros((150, 150, 3), dtype=np.uint8) # Default black display
# ---

with HandLandmarker.create_from_options(hand_options) as landmarker, \
     ImageSegmenter.create_from_options(segment_options) as segmenter:

    while True:
        # 1. READ A FRAME
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        # Flip the frame
        frame = cv2.flip(frame, 1)
       
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1 # Increment frame counter

        # --- NEW: Determine if we process this frame ---
        run_processing = (frame_count % process_every_n_frames == 0)

        # --- 2. CONDITIONAL PROCESSING BLOCK ---
        if run_processing:
            # --- START HEAVY PROCESSING ---
            current_predicted_label = "No Hand" # Default for this frame
            current_confidence = 0.0
            current_hand_cutout_display = np.zeros((150, 150, 3), dtype=np.uint8)
            totalTime = time.time() # For performance measurement
            t0 = time.time() # For performance measurement
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
            detection_result = landmarker.detect(mp_image_full)
            detection_time = time.time() - t0
            timings["hand_detection"].append(detection_time)
            print(f"Hand detection time: {detection_time:.4f} seconds")
            t0 = time.time()
            
            if detection_result.hand_landmarks:
                try:
                    landmarks = detection_result.hand_landmarks[0]
                    x_min, y_min, x_max, y_max = calculate_bounding_box(frame.shape, landmarks)

                    if x_min < x_max and y_min < y_max:
                        cropped_frame = frame[y_min:y_max, x_min:x_max]
                        if cropped_frame.size > 0:
                            cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                            cropped_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_rgb)
                            segmentation_result = segmenter.segment(cropped_mp_image)
                            segmentation_time = time.time() - t0
                            timings["segmentation"].append(segmentation_time)
                            print(f"Segmentation time: {segmentation_time:.4f} seconds")
                            

                            if hasattr(segmentation_result, 'confidence_masks') and segmentation_result.confidence_masks:
                                if TARGET_CATEGORY_INDEX < len(segmentation_result.confidence_masks):
                                    confidence_mask = segmentation_result.confidence_masks[TARGET_CATEGORY_INDEX].numpy_view()
                                    condition = confidence_mask > SEGMENTATION_THRESHOLD

                                    if condition.shape != cropped_frame.shape[:2]:
                                        condition = cv2.resize(condition.astype(np.uint8), (cropped_frame.shape[1], cropped_frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

                                    hand_cutout = np.zeros(cropped_frame.shape, dtype=np.uint8)
                                    condition_3c = np.stack((condition,) * 3, axis=-1)
                                    hand_cutout[condition_3c] = cropped_frame[condition_3c]
                                    current_hand_cutout_display = cv2.resize(hand_cutout, (150, 150)) # Update display

                                    # Preprocess for Classifier
                                    img_resized_for_model = cv2.resize(hand_cutout, IMG_SIZE)
                                    # Assuming Rescaling layer is first in model, no manual normalization needed
                                    img_batch = np.expand_dims(img_resized_for_model, axis=0)

                                    # Make Prediction
                                    t0 = time.time()
                                    predictions = model.predict(img_batch)
                                    prediction_time = time.time() - t0
                                    timings["model_prediction"].append(prediction_time)
                                    print(f"Model Prediction time: {prediction_time:.4f} seconds")
                                    predicted_class_index = np.argmax(predictions[0])
                                    current_predicted_label = CLASS_LABELS[predicted_class_index]
                                    current_confidence = np.max(predictions[0])
                                else: print("Target index error.")
                            else: print("Segmentation fail.")
                        else: print("Crop empty.")
                    else: print("Invalid BBox.")
                except Exception as e:
                    print(f"Error processing hand: {e}")
                    current_predicted_label = "Error"
                    current_confidence = 0.0
            # else: Hand not detected, defaults remain "No Hand"

            # --- Store the results from this processing run ---
            last_predicted_label = current_predicted_label
            last_confidence = current_confidence
            last_hand_cutout_display = current_hand_cutout_display
            endTime = time.time()
            print(f"Total loop time: {endTime - totalTime:.4f} seconds")
            if current_predicted_label not in ("No Hand", "Error"):
                timings["processed_frame_total"].append(endTime - totalTime)
            # --- END HEAVY PROCESSING ---
        # --- End conditional processing block ---


        # --- 3. ALWAYS UPDATE TIMER AND DISPLAY (Using stored results) ---
        currentTime = time.time()
        timeLapsed = currentTime - startTime

        # Game Logic uses 'last_predicted_label' and 'last_confidence'
        if timeLapsed < 3: # Countdown phase
             if randomPrediction is None: # Generate computer's choice only once per round
                 randomPrediction = CLASS_LABELS[np.random.randint(0, len(CLASS_LABELS))]
             text = f"Shoot in: {3 - timeLapsed:.1f}s"
             cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
             # Display the prediction from the last processed frame
             cv2.putText(frame, f"Prediction: {last_predicted_label} ({last_confidence:.2f})", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif 3 <= timeLapsed < 6: # Result phase
            player_choice = last_predicted_label # Use the prediction held during countdown
            computer_choice = randomPrediction

            result_text = "Show Hand!"
            if player_choice != "No Hand" and player_choice != "Starting..." and player_choice != "Error":
                if player_choice == computer_choice:
                    result_text = "Tie!"
                    color = (0, 255, 255) # Yellow
                elif (player_choice == 'rock' and computer_choice == 'scissors') or \
                     (player_choice == 'scissors' and computer_choice == 'paper') or \
                     (player_choice == 'paper' and computer_choice == 'rock'):
                    result_text = "You Win!"
                    color = (0, 255, 0) # Green
                else:
                    result_text = "You Lose!"
                    color = (0, 0, 255) # Red
            else:
                 color = (255, 255, 255) # White if no hand or error

            display_text = f"You: {player_choice} | Comp: {computer_choice} | {result_text}"
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        elif timeLapsed >= 6: # Reset for next round
            startTime = time.time()
            randomPrediction = None # Reset computer choice
            # Don't reset last_predicted_label here, let the next processing update it


        # Display the segmented hand cutout (from last processed frame)
        cv2.imshow('Hand Cutout (Debug)', last_hand_cutout_display)

        # Show the main frame
        cv2.imshow('Rock Paper Scissors', frame)

        


        # QUIT
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
print_timing_summary(timings)