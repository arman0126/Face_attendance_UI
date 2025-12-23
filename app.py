from flask import Flask, render_template, redirect
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/add_user", methods=["POST"])
def add_user():
    os.system("python dataset_creator.py")
    return redirect("/")

@app.route("/train")
def train():
    os.system("python train_model.py")
    return redirect("/")

@app.route("/attendance")
def attendance():
    os.system("python attendance.py")
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
