# enroll.py
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# Init model
app = FaceAnalysis(name='buffalo_l')  # ResNet100 + RetinaFace
app.prepare(ctx_id=0)

# Create embedding folder
os.makedirs("embeddings", exist_ok=True)

# Capture image from webcam
cap = cv2.VideoCapture(0)
print("Press SPACE to capture face...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Enrollment", frame)
    key = cv2.waitKey(1)
    if key == 32:  # SPACE to capture
        break
cap.release()
cv2.destroyAllWindows()

# Get face and embedding
faces = app.get(frame)
if not faces:
    print("No face detected!")
    exit()

# Use first face
embedding = faces[0].normed_embedding

# Save embedding
user_name = input("Enter name to register: ")
np.save(f"embeddings/{user_name}.npy", embedding)
print(f"✅ Saved embedding for {user_name}.")
