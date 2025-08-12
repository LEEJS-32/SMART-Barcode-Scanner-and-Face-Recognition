# verify.py
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# Init model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

# Load embeddings
def load_embeddings():
    db = {}
    for file in os.listdir("embeddings"):
        if file.endswith(".npy"):
            name = file[:-4]
            vec = np.load(os.path.join("embeddings", file))
            db[name] = vec
    return db

db = load_embeddings()
if not db:
    print("No registered users. Run enroll.py first.")
    exit()

# Capture from webcam
cap = cv2.VideoCapture(0)
print("Press SPACE to verify...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Verification", frame)
    key = cv2.waitKey(1)
    if key == 32:  # SPACE
        break
cap.release()
cv2.destroyAllWindows()

# Get face embedding
faces = app.get(frame)
if not faces:
    print("No face detected!")
    exit()

embedding = faces[0].normed_embedding

# Match against database
def cosine_similarity(a, b):
    return np.dot(a, b)

best_match = None
best_score = -1
for name, ref_vec in db.items():
    score = cosine_similarity(embedding, ref_vec)
    if score > best_score:
        best_score = score
        best_match = name

# Decision threshold (adjustable)
threshold = 0.5
if best_score >= threshold:
    print(f"✅ Verified: {best_match} (Score: {best_score:.4f})")
else:
    print(f"❌ Not recognized (Best Score: {best_score:.4f})")
