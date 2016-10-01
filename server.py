#!/usr/bin/python

from flask import Flask, send_from_directory, jsonify, request
from flask_socketio import SocketIO
from ai.dfnn import DFCNN
from ai.dqn import DQN

import tensorflow as tf
import os
import re

dirname = os.path.dirname(__file__)

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
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    summary_report = tf.train.SummaryWriter(os.path.dirname(__file__) + "/tflogs")

    qnn = DFCNN([16, 64 , 32, 4])
    controller = DQN(qnn, optimizer, session, 16, 4, final_exploration_probability=0.05, exploration_period=1000, minibatch_size=128, summary_writer=summary_report)

    tf.initialize_all_variables().run()

server = Flask(__name__, static_url_path='', static_folder='game')
#websocket = SocketIO(server)


@server.route('/')
def index():
    return send_from_directory('game', 'index.html')

@server.route('/dfnn/experience', methods=['POST'])
def record_and_train():
    data = request.get_json()
    controller.remember(data['state'], data['action'], data['reward'], data['nextstate'], data['nextLegalActions'])
    outcome = controller.train()
    if outcome is None:
        return jsonify({'success': True})
    else:
        return jsonify({'success': True, 'loss': float(outcome[0]), 'step':outcome[1]})

@server.route('/dfnn/action', methods=['POST'])
def get_action():
    data = request.get_json()
    action = controller.get_action(data['state'], data['legalActions'], data['playMode'])
    return jsonify({'success': True, 'action': action})

@server.route('/dfnn/save', methods=['POST'])
def save_qnn():
    data = request.get_json()
    qnn.save(session, dirname + "/saved-models/" + data['filename'])
    return jsonify({'success': True})

@server.route('/dfnn/load', methods=['POST'])
def load_qnn():
    data = request.get_json()
    qnn.restore(session, dirname + "/saved-models/" + data['filename'])
    return jsonify({'sucess': True})

@server.route('/dfnn/saved-models', methods=['GET'])
def list_saved_qnns():
    saved_qnns = [];
    for entry in os.listdir(dirname + "/saved-models"):
        if(re.match(r'.*\.ckpt$', entry)):
            saved_qnns.append(entry)
    return jsonify({'success': True, 'models': saved_qnns})

if __name__ == "__main__":
    server.run(debug=True)
