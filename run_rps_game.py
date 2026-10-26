import cv2
import tensorflow as tf
import numpy as np
import time

# --- PRE-FLIGHT CHECKLIST ---
# 1. Set the correct MODEL_PATH
MODEL_PATH = 'image_classification_model.h5' # Or 'my_final_model.keras', etc.

# 2. Set the correct CLASS_LABELS in alphabetical order
CLASS_LABELS = ['paper', 'rock', 'scissors'] 

# 3. Set the correct IMG_SIZE
IMG_SIZE = (150, 150)

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

# Timing samples (seconds). Model-running loops are only during the countdown.
timings = {"model_prediction": [], "loop_with_prediction": []}

# --- LOAD THE TRAINED MODEL ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# --- INITIALIZE WEBCAM ---

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # Use CAP_MSMF for Windows to avoid warnings

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nWebcam opened. Press 'q' to quit.")
print("Show your hand in the green box.")
startTime = time.time()
timeLapsed = 0
randomPrediction = None

while True:
    # 1. READ A FRAME
    ret, frame = cap.read()
    if not ret:
        break
    totalTime = time.time()  # For performance measurement
    
    # Flip the frame for a mirror-like effect
    frame = cv2.flip(frame, 1)
    
    frame = cv2.resize(frame, (640, 480))

    

    # 2. DEFINE REGION OF INTEREST (ROI)
    roi_x1, roi_y1 = 100, 100
    roi_x2, roi_y2 = 400, 400
    
    # Draw the ROI for user guidance
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    
    # Extract the ROI from the frame
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # 3. PREPROCESS THE ROI FOR THE MODEL
    # a. Resize to the model's expected input size
    img_resized = cv2.resize(roi_rgb, IMG_SIZE)
    
    # c. Expand dimensions to create a "batch" of 1
    img_batch = np.expand_dims(img_resized, axis=0)

    
    

    # 4. MAKE PREDICTION
    ran_prediction = False
    if timeLapsed < 3  : # Predict every 3 seconds
        t0 = time.time()
        predictions = model.predict(img_batch)
        prediction_time = time.time() - t0
        timings["model_prediction"].append(prediction_time)
        ran_prediction = True
        print(f"Model Prediction time: {prediction_time:.4f} seconds")
        randomPrediction = CLASS_LABELS[np.random.randint(0, len(CLASS_LABELS))]
        print(f"Predictions: {predictions} | Random: {randomPrediction}")
        text = f"Rock Paper Scissor Shoot Time: {3 - timeLapsed:.2f}s"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        
    elif timeLapsed < 6:
        # 5. INTERPRET PREDICTION
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = CLASS_LABELS[predicted_class_index]
        confidence = np.max(predictions[0])
        print(f"TimeLapsed: {timeLapsed}")

        # 6. DISPLAY THE RESULT
        if (predicted_label == randomPrediction):
            text = f"Prediction: {predicted_label} ({confidence:.2f}) - Tie!"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif (predicted_label == 'rock' and randomPrediction == 'scissors') or \
             (predicted_label == 'scissors' and randomPrediction == 'paper') or \
                (predicted_label == 'paper' and randomPrediction == 'rock'):
            text = f"Prediction: {predicted_label} ({confidence:.2f}) - You Win!"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            text = f"Prediction: {predicted_label} ({confidence:.2f}) - You Lose!"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    elif timeLapsed >= 6:
        startTime = time.time()
        randomPrediction = None
        timeLapsed = 0

    endTime = time.time()
    print(f"Total loop time: {endTime - totalTime:.4f} seconds")
    if ran_prediction:
        timings["loop_with_prediction"].append(endTime - totalTime)
    
        
       

    # Show the final frame
    cv2.imshow('Rock Paper Scissors', frame)

    timeLapsed = int(time.time() - startTime)

    # 7. QUIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
print_timing_summary(timings)