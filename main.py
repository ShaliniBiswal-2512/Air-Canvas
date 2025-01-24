import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Initialize Canvas Variables
canvas = None
x_prev, y_prev = 0, 0
drawing = False  # Flag to toggle drawing mode

# Default Color and Thickness
color = (255, 0, 0)  # Default color: Blue
thickness = 5

# Define Colors
colors = {
    'b': (255, 0, 0),    # Blue
    'g': (0, 255, 0),    # Green
    'r': (0, 0, 255),    # Red
    'y': (0, 255, 255),  # Yellow
    'w': (255, 255, 255) # White
}

# Webcam Input
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural interaction
    h, w, c = frame.shape

    # Initialize canvas if not already
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert BGR to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates for the index finger tip (landmark 8)
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            if drawing:
                if x_prev == 0 and y_prev == 0:  # First point
                    x_prev, y_prev = x, y
                # Draw on the canvas
                cv2.line(canvas, (x_prev, y_prev), (x, y), color, thickness)
                x_prev, y_prev = x, y
            else:
                x_prev, y_prev = 0, 0

    # Combine Canvas with Frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display Instructions
    cv2.putText(combined, "Press 'D' to Toggle Drawing Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Press 'C' to Clear Canvas", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Press 'B/G/R/Y/W' to Change Color", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Press 'Q' to Quit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show Frame and Canvas
    cv2.imshow("Air Canvas", combined)

    # Key Controls
    key = cv2.waitKey(1)
    if key == ord('q'):  # Quit
        break
    elif key == ord('d'):  # Toggle drawing mode
        drawing = not drawing
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key in [ord(c) for c in colors.keys()]:  # Change color
        color = colors[chr(key)]

# Release Resources
cap.release()
cv2.destroyAllWindows()
