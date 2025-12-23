import cv2
import os

face_id = input("Enter ID: ")

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

os.makedirs("dataset", exist_ok=True)
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg",
                    gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('Dataset Creation', img)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 40:
        break

cam.release()
cv2.destroyAllWindows()

