import numpy as np
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
from uploadfile import get_files, app

def feature_vector(img_arr, model):
    if img_arr.shape[2] == 1:
        img_arr = img_arr.repeat(3, axis = 2) # (1, 224, 224, 3)
    return model.predict(preprocess_input(np.expand_dims(img_arr, axis = 0)))[0,:]

def feature_vectors(imgs_dict, model):
    f_vect = {}
    for fn, img in imgs_dict.items():
        f_vect[fn] = feature_vector(img, model)
    return f_vect

# method to find Optimal K: silhouette score
def silhouette_scores(X):
    scores = {}
    for i in range(2, 25):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit_predict(X)
        score = silhouette_score(X, km.labels_, metric='euclidean')
        score_dict = {i: score}
        scores.update(score_dict)
        sil_sco, cluster_no = max(zip(scores.values(), scores.keys()))
    return cluster_no, sil_sco


def post_upload():
    base_model = ResNet50(weights = 'imagenet', include_top = True)
    model = Model(inputs = base_model.input, outputs = base_model.get_layer('avg_pool').output)
    filepath = app.config['UPLOAD_FOLDER']
    imgs_dict = get_files(filepath, (224, 224))
    img_feature_vector = feature_vectors(imgs_dict, model) # feed images through the model and extract feature vector
    images = list(img_feature_vector.values())
    cluster_no, _ = silhouette_scores(images)
    cluster_path = filepath
    path_to_files = filepath
    kmeans = KMeans(n_clusters = cluster_no, init = 'k-means++')
    kmeans.fit(images)
    y_kmeans = kmeans.predict(images)
    file_names = list(imgs_dict.keys())
    
    for c in range(0, cluster_no):
        if not os.path.exists(cluster_path + 'cluster_' + str(c)):
            os.mkdir(cluster_path + 'cluster_' + str(c))
    
    for fn, cluster in zip(file_names, y_kmeans):
        image = cv2.imread(path_to_files + fn)
        cv2.imwrite(cluster_path + 'cluster_' + str(cluster) + '/' + fn, image)