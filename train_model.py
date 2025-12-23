import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def get_images(path):
    faces, ids = [], []
    for file in os.listdir(path):
        img = Image.open(os.path.join(path, file)).convert('L')
        img_np = np.array(img, 'uint8')
        id = int(file.split('.')[1])
        faces.append(img_np)
        ids.append(id)
    return faces, ids

faces, ids = get_images(path)
recognizer.train(faces, np.array(ids))
recognizer.save('trainer.yml')

print("Training completed")
