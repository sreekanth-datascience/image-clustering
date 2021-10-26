import numpy as np
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
from controller import post_upload

# run this in cmd before running below code: find . -name ".DS_Store" -delete
app=Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads' + '/')
UPLOAD_FOLDER

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')

def get_files(path_to_files, size = (224, 224)):
    try:
        fn_imgs = []
        files = [file for file in os.listdir(path_to_files)]
        for file in files:
            img = cv2.resize(cv2.imread(path_to_files + file), size)
            fn_imgs.append([file, img])
    except:
        Exception
    return dict(fn_imgs)


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        flash('File(s) successfully uploaded')
    post_upload()
    return redirect('/')

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
