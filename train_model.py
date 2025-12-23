import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

faces = []
ids = []

for file in os.listdir(path):
    img = Image.open(os.path.join(path, file)).convert("L")
    img_np = np.array(img, "uint8")
    id = int(file.split(".")[1])

    faces.append(img_np)
    ids.append(id)

recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")

print("Training Completed Successfully")
