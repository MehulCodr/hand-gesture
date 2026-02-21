import cv2
import os

# Parameters
SAVE_PATH = "dataset/fist"  # Change this for each gesture
IMAGE_SIZE = (128, 128)  # Resize images for CNN
REGION_SIZE = (300, 300)  # The region to capture (width, height)

# Create directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

# Get the width of the webcam frame
frame_width = int(cap.get(3))  # Get frame width

count = 0  # Image counter
print("Press SPACE to capture an image, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Define the region in the **top-right** corner
    x_start = frame_width - REGION_SIZE[0] - 10  # Adjust x coordinate for right-side placement
    y_start = 10  # Keep it at the top

    roi = frame[y_start:y_start + REGION_SIZE[1], x_start:x_start + REGION_SIZE[0]]

    # Show the ROI in a separate window
    cv2.imshow("Hand Region", roi)

    # Draw a rectangle on the webcam feed to indicate the capture region
    cv2.rectangle(frame, (x_start, y_start), (x_start + REGION_SIZE[0], y_start + REGION_SIZE[1]), (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)

    # Capture image when the spacebar is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Spacebar key
        img_path = os.path.join(SAVE_PATH, f"img_{count}.jpg")
        roi_resized = cv2.resize(roi, IMAGE_SIZE)
        cv2.imwrite(img_path, roi_resized)
        count += 1
        print(f"Saved: {img_path}")

    elif key == ord('q'):  # Quit on 'q'
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
