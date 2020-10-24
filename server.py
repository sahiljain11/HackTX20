from flask import Flask, jsonify, request, render_template, abort, session, send_from_directory, url_for, send_file
import flask

from pokemon import Pokemon

app = Flask(__name__)
app.config.from_object(__name__)

@app.route("/get_pokemon", methods=["POST"])
def get_pokemon():
    return jsonify({})


@app.route("/")
def main():
    return render_template('main.html')
