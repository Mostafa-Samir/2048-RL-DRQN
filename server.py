#!/usr/bin/python

from flask import Flask, send_from_directory, jsonify, request
from flask_socketio import SocketIO
import os

server = Flask(__name__, static_url_path='', static_folder='game')
#websocket = SocketIO(server)


@server.route('/')
def index():
    return send_from_directory('game', 'index.html')

@server.route('/dfnn/experience', methods=['POST'])
def retrieve_experience():
    experience = request.get_json();
    return jsonify({'success': True})

if __name__ == "__main__":
    server.run(debug=True)

    tensorflow_logdir = os.path.join(os.path.dirname(__file__), "tflog")
    if os.path.exists(tensorflow_logdir):
        # clear all the logs if any
        for entry in os.listdir(tensorflow_logdir):
            entry_path = os.path.join(tensorflow_logdir, entry)
            if(os.path.isfile(entry_path)):
                os.unlink(entry_path)
    else:
        #craete the directory
        os.mkdir(tensorflow_logdir)
