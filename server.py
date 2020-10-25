from flask import Flask, jsonify, request, render_template, abort, session, send_from_directory, url_for, send_file
import flask


app = Flask(__name__)
app.config.from_object(__name__)

@app.route("/process_lie")
def process_lie():


    return_val = #something

    return jsonify({"lie": return_val})

@app.route("/")
def main():
    return render_template('main.html')

if __name__ == "__main__":
    app.run()