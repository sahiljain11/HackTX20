from flask import Flask, jsonify, request, render_template, abort, session, send_from_directory, url_for, send_file
import flask


app = Flask(__name__)
app.config.from_object(__name__)

@app.route("/")
def main():
    return render_template('main.html')
