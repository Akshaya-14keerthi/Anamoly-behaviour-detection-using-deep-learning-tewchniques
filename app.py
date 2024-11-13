from __future__ import division, print_function
import os
import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template
from keras.preprocessing.image import load_img, img_to_array
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
VIDEO_UPLOAD_FOLDER = 'static/uploads/videos/'

# Allow files of specific types
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
ALLOWED_VIDEO_EXTENSIONS = set(['mp4', 'avi', 'mov'])

# Function to check the file extension
def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

static_path = 'static'
app.config['IMAGE_UPLOADS'] = static_path

CTS = load_model('xce.h5')

def model_predict_image(image_path, model):
    print("Predicting Image")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    result = np.argmax(model.predict(image))
    print("result", result)
    if result == 0:
        return "abnormal behavior"
    elif result == 1:
        return "normal behavior"
    elif result == 2:
        return "abnormal behavior"
    elif result == 3:
        return "normal face"

def model_predict_video(video_path, model):
    print("Predicting Video")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128))
        frame = img_to_array(frame)
        frame = frame / 255
        frame = np.expand_dims(frame, axis=0)
        result = np.argmax(model.predict(frame))
        predictions.append(result)
    cap.release()
    result = max(set(predictions), key=predictions.count)
    if result == 0:
        return "abnormal behavior"
    elif result == 1:
        return "normal behavior"
    elif result == 2:
        return "abnormal behavior"
    elif result == 3:
        return "normal face"

# Routes
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        file = request.files['my_image']
        if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            pred = model_predict_image(file_path, CTS)
            return render_template('after.html', pred_output=pred, img_src=UPLOAD_FOLDER + filename)
    return render_template('home.html')

@app.route("/submit_video", methods=['GET', 'POST'])
def get_video():
    if request.method == 'POST':
        file = request.files['my_video']
        if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            filename = file.filename
            file_path = os.path.join(VIDEO_UPLOAD_FOLDER, filename)
            file.save(file_path)
            pred = model_predict_video(file_path, CTS)
            return render_template('after_vedio.html', pred_output=pred, video_src=VIDEO_UPLOAD_FOLDER + filename)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
