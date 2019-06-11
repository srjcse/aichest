#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

# import magic

import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import json

ALLOWED_EXTENSIONS = set([
    'txt',
    'pdf',
    'png',
    'jpg',
    'jpeg',
    'gif',
    ])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() \
        in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the files part

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                get_filename = os.path.join(app.config['UPLOAD_FOLDER'
                        ], filename)
                file.save(get_filename)
                new_model = load_model('./content/chest-xray-pneumonia.h5')
                img = image.load_img(get_filename, target_size=(224,
                        224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                normal, pneumonia = np.around(new_model.predict(x),
                        decimals=2)[0]
                res1 = str(normal*100)
                res2 = str(pneumonia*100)

        # flash('File(s) successfully uploaded')

        flash({'normal': res1, 'pneumonia': res2})
        return redirect('/')

if __name__ == '__main__':

    # initialize the log handler

    app._static_folder = './static'

    logHandler = RotatingFileHandler('info.log', maxBytes=1000,
            backupCount=1)

    # set the log handler level

    logHandler.setLevel(logging.INFO)

    # set the app logger level

    app.logger.setLevel(logging.INFO)

    app.logger.addHandler(logHandler)
    app.run()
