import cv2
import pandas as pd
from datetime import datetime
import os

# Load users properly (force int)
users = pd.read_csv("users.csv")
users["ID"] = users["ID"].astype(int)
user_dict = dict(zip(users.ID, users.Name))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)
attendance = {}

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 55 and id in user_dict:
            name = user_dict[id]
            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%H:%M:%S")

            attendance[(id, date)] = [id, name, date, time]

            cv2.putText(img, name, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.putText(img, "Unknown", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Attendance", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

df = pd.DataFrame(attendance.values(),
                  columns=["ID","Name","Date","Time"])

df.to_csv("attendance.csv", index=False)
print("✅ Attendance Saved Correctly")
