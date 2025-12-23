from flask import Flask, render_template, request, redirect, session, url_for, Response
import cv2
import pandas as pd
import os
from datetime import datetime

print(">>> RUNNING FINAL app.py <<<")

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

# ================= GLOBAL FLAGS =================
attendance_active = False
last_detected_name = "None"
present_names = set()
marked_today = set()

capture_user = False
new_user_name = ""
new_user_id = None
capture_count = 0

# ================= CAMERA =================
camera = cv2.VideoCapture(0)

# ================= FACE RECOGNITION =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

users_df = pd.read_csv("users.csv")
user_dict = dict(zip(users_df.ID, users_df.Name))


# ================= HELPER FUNCTIONS =================
def get_new_user_id():
    if not os.path.exists("users.csv"):
        return 1
    df = pd.read_csv("users.csv")
    return int(df["ID"].max()) + 1


def mark_attendance(name):
    global marked_today

    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if name in marked_today:
        return

    user_id = None
    for uid, uname in user_dict.items():
        if uname == name:
            user_id = uid
            break

    if user_id is None:
        return

    file_exists = os.path.exists("attendance.csv")

    with open("attendance.csv", "a") as f:
        if not file_exists:
            f.write("ID,Name,Date,Time\n")
        f.write(f"{user_id},{name},{today},{time_now}\n")

    marked_today.add(name)


# ================= VIDEO FRAMES =================
def gen_frames():
    global attendance_active, last_detected_name, present_names
    global capture_user, capture_count

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            # -------- ADD USER MODE --------
            if capture_user:
                capture_count += 1
                face_img = gray[y:y+h, x:x+w]
                os.makedirs("dataset", exist_ok=True)
                cv2.imwrite(
                    f"dataset/User.{new_user_id}.{capture_count}.jpg",
                    face_img
                )

                cv2.putText(frame, f"Capturing {capture_count}/40",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)

                if capture_count >= 40:
                    capture_user = False

            # -------- RECOGNITION MODE --------
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 60 and id_ in user_dict:
                name = user_dict[id_]
                last_detected_name = name

                if attendance_active:
                    present_names.add(name)
                    mark_attendance(name)

                cv2.putText(frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ================= AUTH =================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin":
            session["admin"] = True
            return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    if "admin" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")


# ================= USERS =================
@app.route("/users")
def users():
    if "admin" not in session:
        return redirect(url_for("login"))
    data = pd.read_csv("users.csv") if os.path.exists("users.csv") else []
    return render_template("users.html", users=data)


# ================= ATTENDANCE PAGE =================
@app.route("/attendance_page")
def attendance_page():
    if "admin" not in session:
        return redirect(url_for("login"))
    records = pd.read_csv("attendance.csv").to_dict(orient="records") if os.path.exists("attendance.csv") else []
    return render_template("attendance.html", records=records)


# ================= CAMERA =================
@app.route("/camera")
def camera_page():
    if "admin" not in session:
        return redirect(url_for("login"))
    return render_template("camera.html")


@app.route("/video_feed")
def video_feed():
    if "admin" not in session:
        return redirect(url_for("login"))
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ================= ADD USER FROM WEB =================
@app.route("/add_user_web", methods=["POST"])
def add_user_web():
    global capture_user, new_user_name, new_user_id, capture_count

    new_user_name = request.form.get("username")
    new_user_id = get_new_user_id()
    capture_count = 0
    capture_user = True

    with open("users.csv", "a") as f:
        if os.stat("users.csv").st_size == 0:
            f.write("ID,Name\n")
        f.write(f"{new_user_id},{new_user_name}\n")

    user_dict[new_user_id] = new_user_name

    return redirect(url_for("camera_page"))


# ================= START / STOP ATTENDANCE =================
@app.route("/start_attendance_btn")
def start_attendance_btn():
    global attendance_active
    attendance_active = True
    return redirect(url_for("camera_page"))


@app.route("/stop_attendance_btn")
def stop_attendance_btn():
    global attendance_active
    attendance_active = False
    present_names.clear()
    marked_today.clear()
    return redirect(url_for("camera_page"))


# ================= STATUS =================
@app.route("/status")
def status():
    return {
        "attendance": attendance_active,
        "last_detected": last_detected_name,
        "present": len(present_names)
    }


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
