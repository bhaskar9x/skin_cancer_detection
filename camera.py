import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("skin_cancer_model.h5")

IMG_SIZE = 128

# Open camera
cap = cv2.VideoCapture(0)

print("Press SPACE to capture image")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press SPACE to Capture", frame)

    key = cv2.waitKey(1)

    # Press SPACE to capture
    if key == 32:
        captured_image = frame
        break

# Release camera
cap.release()
cv2.destroyAllWindows()

# -----------------------
# Preprocess captured image
# -----------------------
img = cv2.resize(captured_image, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)[0][0]
confidence = prediction * 100

# Result
if prediction > 0.30:
    result = f"CANCER DETECTED ({confidence:.2f}%)"
else:
    result = f"NO CANCER ({100-confidence:.2f}%)"

print("\nFinal Result:", result)

# Show final image with result
cv2.putText(captured_image, result,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            
            1,
            (0, 0, 255) if prediction > 0.30 else (0, 255, 0),
            2)

cv2.imshow("Prediction Result", captured_image)
cv2.waitKey(0)
cv2.destroyAllWindows()