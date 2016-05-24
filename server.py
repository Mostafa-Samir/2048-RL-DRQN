#!/usr/bin/python

from flask import Flask, send_from_directory, jsonify, request
from flask_socketio import SocketIO
from ai.dfnn import DFCNN
from ai.dqn import DQN

import tensorflow as tf
import os

tensorflow_logdir = os.path.join(os.path.dirname(__file__), "tflogs")
if os.path.exists(tensorflow_logdir):
    # clear all the logs if any
    for entry in os.listdir(tensorflow_logdir):
        entry_path = os.path.join(tensorflow_logdir, entry)
        if(os.path.isfile(entry_path)):
            os.unlink(entry_path)
else:
    #craete the directory
    os.mkdir(tensorflow_logdir)

graph = tf.Graph()
with graph.as_default():
    session = tf.InteractiveSession(graph=graph)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    summary_writer = tf.train.SummaryWriter(os.path.dirname(__file__) + "/tflogs")

    qnn = DFCNN([16, 100, 50, 4])
    controller = DQN(qnn, optimizer, session, 16, 4)

    tf.initialize_all_variables().run()

server = Flask(__name__, static_url_path='', static_folder='game')
#websocket = SocketIO(server)


@server.route('/')
def index():
    return send_from_directory('game', 'index.html')

@server.route('/dfnn/experience', methods=['POST'])
def record_and_train():
    data = request.get_json()
    controller.remember(data['state'], data['action'], data['reward'], data['nextstate'])
    controller.train()
    return jsonify({'success': True})

@server.route('/dfnn/action', methods=['POST'])
def get_action():
    data = request.get_json()
    print data
    action = controller.get_action(data['state'], data['legalActions'], data['playMode'])
    return jsonify({'success': True, 'action': action})

if __name__ == "__main__":
    server.run()
