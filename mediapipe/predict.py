# import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model

# def min_max_scale_sample(X):
#     min_vals = X.min(axis=1, keepdims=True)
#     max_vals = X.max(axis=1, keepdims=True)
#     scaled_X = (X - min_vals) / (max_vals - min_vals)
#     return scaled_X

# model = load_model('hand_model_left.h5')

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     min_detection_confidence=0.9,
#     min_tracking_confidence=0.9,
#     max_num_hands=1
# )

# cap = cv2.VideoCapture(0)

# unique_labels = ['Ascend', 'Backward', 'Descent', 'Forward', 'Right', 'Left', 'Stop']

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame = cv2.flip(frame, 1)
    
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     results = hands.process(image_rgb)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             hand_coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            
#             flattened_landmarks = hand_coords.flatten()
            
#             normalized_input = min_max_scale_sample(flattened_landmarks.reshape(1, -1))
            
#             prediction = model.predict(normalized_input)
#             predicted_class = np.argmax(prediction)
#             confidence = prediction[0][predicted_class]

#             if confidence > 0.95:
#                 if predicted_class < len(unique_labels):
#                     predicted_label = unique_labels[predicted_class]
#                 else:
#                     predicted_label = 'Unknown'

#                 cv2.putText(frame, f"Prediction: {predicted_label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 cv2.putText(frame, f"Confidence: {confidence*100:.2f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
#     cv2.imshow('Hand Landmark Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

def min_max_scale_sample(X):
    min_vals = X.min(axis=1, keepdims=True)
    max_vals = X.max(axis=1, keepdims=True)
    scaled_X = (X - min_vals) / (max_vals - min_vals)
    return scaled_X

# Load the models
left_hand_model = load_model('hand_model_left.h5')
right_hand_model = load_model('hand_model_right.h5')

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
    max_num_hands=2  # Allow detection of two hands
)

cap = cv2.VideoCapture(0)

# Unique labels for each hand
left_labels = ['Ascend', 'Backward', 'Descent', 'Forward', 'Right', 'Left', 'Stop']
right_labels = ['Ascend', 'Backward', 'Descent', 'Forward', 'Left', 'Right', 'Stop']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirrored view
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Determine if the hand is left or right
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            hand_coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

            # Flatten the landmarks for model input
            flattened_landmarks = hand_coords.flatten()

            # Normalize the landmarks
            normalized_input = min_max_scale_sample(flattened_landmarks.reshape(1, -1))

            # Use the appropriate model and labels for the hand
            if hand_label == "Left":
                prediction = left_hand_model.predict(normalized_input)
                labels = left_labels
            else:
                prediction = right_hand_model.predict(normalized_input)
                labels = right_labels

            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]

            if confidence > 0.95:
                if predicted_class < len(labels):
                    predicted_label = labels[predicted_class]
                else:
                    predicted_label = 'Unknown'

                x_coord = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y_coord = int(hand_landmarks.landmark[0].y * frame.shape[0])

                cv2.putText(frame, f"{hand_label} Hand: {predicted_label}", (x_coord, y_coord - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence*100:.2f}%", (x_coord, y_coord - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Landmark Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
