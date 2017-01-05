#!/usr/bin/python

from flask import Flask, send_from_directory, jsonify, request
import tensorflow as tf
import models
import getopt
import json
import sys
import os

qnn, trainer, state_represntation, reward_name, session = None, None, None, None, None

dirname = os.path.dirname(__file__)

def init_model(model_name):
    """
    initializes the chosen model

    Parameters:
    ----------
    model_name: string
        the name of the model to be initialized
    """
    global qnn, trainer, state_represntation, reward_name, session

    graph = tf.Graph()
    with graph.as_default():
        session = tf.InteractiveSession(graph=graph)
        model = getattr(models, model_name)
        qnn, trainer, state_represntation, reward_name = model(session)

        tf.initialize_all_variables().run()

server = Flask(__name__, static_url_path='', static_folder='game')


@server.route('/')
def index():
    return send_from_directory('game', 'index.html')

@server.route('/ai/experience', methods=['POST'])
def record_and_train():
    data = request.get_json()
    trainer.remember(
        data['state'][state_represntation],
        data['action'],
        data['reward'][reward_name],
        data['nextstate'][state_represntation],
        data['nextLegalActions'],
        data['lastTransition']
    )
    outcome = trainer.train()
    if outcome is None:
        return jsonify({'success': True})
    else:
        return jsonify({'success': True, 'loss': float(outcome[0]), 'step':outcome[1]})

@server.route('/ai/action', methods=['POST'])
def get_action():
    data = request.get_json()
    action = trainer.get_action(
        data['state'][state_represntation],
        data['legalActions'],
        data['playMode']
    )
    return jsonify({'success': True, 'action': action})

@server.route('/ai/save', methods=['POST'])
def save_checkpoint():
    data = request.get_json()
    checkpoint_path = dirname + "/checkpoints/" + data["checkpoint"]
    os.mkdir(checkpoint_path)

    try:
        qnn.save(session, checkpoint_path + "/model.ckpt")
        controller_fdata = open(checkpoint_path + "/controller.json", 'w+')
        trainer_fdata = open(checkpoint_path + "/trainer.json", 'w+')

        json.dump(data["data"], controller_fdata)
        json.dump(trainer.serialize(), trainer_fdata)

        controller_fdata.close()
        trainer_fdata.close()
    except Exception as e:
        raise e
        return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': True})

@server.route('/ai/load', methods=['POST'])
def load_checkpoint():
    data = request.get_json()
    checkpoint_path = dirname + "/checkpoints/" + data["checkpoint"]
    controller_data = None

    try:
        qnn.restore(session, checkpoint_path + "/model.ckpt")
        controller_fdata = open(checkpoint_path + "/controller.json", 'r')
        trainer_fdata = open(checkpoint_path + "/trainer.json", 'r')

        controller_data = json.load(controller_fdata)
        trainer_data = json.load(trainer_fdata)

        trainer.restore(trainer_data)

        controller_fdata.close()
        trainer_fdata.close()
    except Exception as e:
        raise e
        return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': True, 'data': controller_data})

@server.route('/ai/checkpoints', methods=['GET'])
def list_saved_checkpoints():
    saved_chkpts = [];
    for entry in os.listdir(dirname + "/checkpoints"):
        saved_chkpts.append(entry)
    return jsonify({'success': True, 'models': saved_chkpts})

if __name__ == "__main__":

    debug_mode = False
    selected_model = None
    options,_ = getopt.getopt(sys.argv[1:], '', ['debug', 'model='])

    for opt in options:
        if opt[0] == '--debug':
            debug_mode = True
        elif opt[0] == '--model':
            selected_model = opt[1]

    if selected_model is None:
        raise ValueError("You must specify a model via --model option")
    else:
        init_model(selected_model)
        server.run(debug=debug_mode, host='0.0.0.0')
