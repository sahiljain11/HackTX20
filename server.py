from flask import Flask, jsonify, request, render_template, abort, session, send_from_directory, url_for, send_file
import flask
import os
from werkzeug import secure_filename

from findlie import RunModel
import cv2

app = Flask(__name__)
app.config.from_object(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exists_ok=True)


@app.route("/save_video", methods=["POST"])
def save_video():

    video = request.files['video']
    video.save(os.path.join(uploads_dir, secure_filename(video.filename)))

    vidcap = cv2.VideoCapture(uploads_dir, secure_filename(video.filename))
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

    return jsonify({"finished" : "yay"})

@app.route("/process_lie")
def process_lie():

    # process jpgs into tensor


    # run model
    lstm = RunModel()
    return_val = lstm.get_result()

    return jsonify({"lie": lstm.get_result()})

@app.route("/")
def main():
    return render_template('main.html')

if __name__ == "__main__":
    app.run()