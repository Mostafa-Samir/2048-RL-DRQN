#!/usr/bin/python

from flask import Flask, send_from_directory, jsonify, request
from flask_socketio import SocketIO

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
