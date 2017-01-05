'use strict';

// container and control variables for the AI controller
let AI = {
    controlVariables: {
        train: {
            targetEpisodes: 0,
            episodesDone: 0,
            callback: null,
            manuallyStopped: false,
            averageLoss: 0
        },

        play: {
            targetEpisodes: 0,
            episodesDone: 0,
            manuallyStopped: false,
            callback: null
        },
    },

    GameManager: null
};

/**
* initializes the AI controller in a clean state
*/
AI.init = function() {
    this.GameManager = new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
    graphs.init();
};

/**
 * signals the AI core to retrive a saved checkpoint data
 * and loads the control variables and the graphs from these data
 * @param {String} checkpoint - the name of the checkpoint
*/
AI.load = function(checkpoint) {
    return $http.post('/ai/load', {checkpoint: checkpoint})
    .then((response) => {
        utility.overwrite(AI.controlVariables, response.data.controlVariables);
        utility.overwrite(graphs, response.data.graphs);

        logger.log('successfully loaded checkpoint');
    })
    .catch((response) => {
        logger.error('failed to load a checkpoint');
        logger.error(response.error);
    });
};

/**
 * serializes the controller and graph variables and send them
 * to server to be saved alongside the core model
 * @param {String} checkpoint - checkpoint name
*/
AI.save = function(checkpoint) {
    let serializedState = {
        controlVariables: JSON.parse(JSON.stringify(AI.controlVariables)),
        graphs: graphs.serialize()
    };

    return $http.post('/ai/save', {checkpoint: checkpoint, data: serializedState})
    .then(() => logger.log('successfully saved checkpoint'))
    .catch((response) => {
        logger.error('failed to save a checkpoint');
        logger.error(response.error);
    });
};

/**
 * serializes current game information and returns it
 * as the current state of the game
 * @return {Object}
*/
AI.state = function() {
    let stateInfo = this.GameManager.serialize();
    stateInfo.legalActions = utility.listLegalActions(this.GameManager);

    return stateInfo;
};

/**
 * encodes the given state with all the encodings defined
 * in state-represntations.js
 * @param {Object} state
 * @return {Object} encodings
*/
AI.prepareState = function(state) {
    let encodings = {};

    for(let key in stateRepresenters) {
        if(stateRepresenters.hasOwnProperty(key) && typeof(stateRepresenters[key] === 'function')) {
            let encodingName = key.replace('encode', '');
            encodings[encodingName] = stateRepresenters[key](state);
        }
    }

    return encodings;
};

/**
 * calculates the transition reward from oldstate to newstate
 * using every reward function defined in reward-functions.js
 * @param {Object} oldstate
 * @param {Object} newstate
 * @return {Object}
*/
AI.calculateRewards = function(oldstate, newstate) {
    let rewards = {};

    for(let key in rewardFunctions) {
        if(rewardFunctions.hasOwnProperty(key) && typeof(rewardFunctions[key] === 'function')) {
            rewards[key] = rewardFunctions[key](oldstate, newstate);
        }
    }

    return rewards;
};

/**
 * transitions form a state to another by moving tiles in the given
 * direction, returning transition info
 * @param {Number} direction
 * @return {Object}
*/
AI.makeAMove = function(direction) {
    let oldstate = this.state();
    this.GameManager.move(direction);
    let newstate = this.state();

    return {
        state: AI.prepareState(oldstate),
        action: direction,
        reward: AI.calculateRewards(oldstate, newstate),
        nextstate: AI.prepareState(newstate),
        nextLegalActions: newstate.legalActions,
        lastTransition: newstate.over
    };
};

/**
 * recursively runs the training episodes till completion or forced stop
*/
AI._recursiveTrain = function() {
    if(this.controlVariables.train.manuallyStopped) {
        return;
    }

    let state = this.state();

    if(state.over) {
        this.controlVariables.train.episodesDone++;
        logger.log(`Episodes: ${this.controlVariables.train.episodesDone}/${this.controlVariables.train.targetEpisodes}`);

        graphs.scoresGraph.x.push(this.controlVariables.train.episodesDone);
        graphs.scoresGraph.y.push(state.score);

        let maxCell = state.grid.maxCell;
        if(!graphs.maxTilesGraph.data[maxCell]) {
            graphs.maxTilesGraph.data[maxCell] = 0;
        }
        graphs.maxTilesGraph.data[maxCell]++;
        graphs.draw()

        if(this.controlVariables.train.episodesDone === this.controlVariables.train.targetEpisodes) {
            this.controlVariables.train.callback();
            this.controlVariables.status = 'stand-by';
            return;
        }

        this.GameManager.restart();
        state = this.state();
    }

    if(state.won) {
        this.GameManager.keepPlaying()
    }

    $http.post('/ai/action', {
        state: AI.prepareState(state),
        playMode:false,
        legalActions: state.legalActions
    })
    .then((response) => {
        let action = response.action;
        let experience = this.makeAMove(action, true);

        return $http.post('/ai/experience', experience);
    })
    .then((response) => {
        if(response.success) {
            if(response.hasOwnProperty('loss')) {
                graphs.lossGraph.x.push(response.step - 1);
                graphs.lossGraph.y.push(response.loss);

                if(response.step === 1) {
                    graphs.avgLossGraph.x.push(response.step - 1);
                    graphs.avgLossGraph.y.push(response.loss);
                }

                this.controlVariables.train.averageLoss += response.loss;
                if(response.step % 100 === 0) {
                    graphs.avgLossGraph.x.push(response.step - 1);
                    graphs.avgLossGraph.y.push(this.controlVariables.train.averageLoss / 100);

                    this.controlVariables.train.averageLoss = 0;
                }
                graphs.draw();
            }

            // advance to the next step
            this._recursiveTrain();
        }
    });
};

/**
 * recursively runs the playing episodes till completion or forced stop
*/
AI._recursivePlay = function() {
    if(this.controlVariables.play.manuallyStopped) {
        return;
    }

    let state = this.state();

    if(state.over) {
        this.controlVariables.play.episodesDone++;
        logger.log(`Episodes: ${this.controlVariables.play.episodesDone}/${this.controlVariables.play.targEpisodes}`);

        graphs.scoresGraph.x.push(this.controlVariables.play.episodesDone);
        graphs.scoresGraph.y.push(state.score);

        let maxCell = state.grid.maxCell;
        if(!graphs.maxTilesGraph.data[maxCell]) {
            graphs.maxTilesGraph.data[maxCell] = 0;
        }
        graphs.maxTilesGraph.data[maxCell]++;
        graphs.draw()

        if(this.controlVariables.play.episodesDone === this.controlVariables.play.targEpisodes) {
            this.controlVariables.play.callback();
            this.controlVariables.status = 'stand-by';
            return;
        }

        this.GameManager.restart();
        state = this.state();
    }

    if(state.won) {
        this.GameManager.keepPlaying()
    }

    $http.post('/ai/action', {
        state: AI.prepareState(state),
        playMode:true,
        legalActions: state.legalActions
    })
    .then((response) => {
        let action = response.action;
        this.makeAMove(action, true);

        this._recursivePlay();
    });
};

/**
 * starts the a training phase
 * @param {Number} episodes
 * @param {function} callbak
*/
AI.train = function(episodes, callback) {
    if(!this.controlVariables.train.manuallyStopped) {
        this.controlVariables.train.targetEpisodes = episodes;
        this.controlVariables.train.episodesDone = 0;
        this.controlVariables.train.callback = callback || function() { };
    }
    else {
        this.controlVariables.train.manuallyStopped = false;
    }

    this.GameManager.restart();  // restart the game first
    this._recursiveTrain();
};

/**
 * starts the a playing phase
 * @param {Number} episodes
 * @param {function} callbak
*/
AI.play = function(episodes, callback) {
    if(!this.controlVariables.play.manuallyStopped) {
        this.controlVariables.play.targetEpisodes = episodes;
        this.controlVariables.play.episodesDone = 0;
        this.controlVariables.play.callback = callback || function() { };
    }
    else {
        this.controlVariables.play.manuallyStopped = false;
    }

    this.GameManager.restart();  // restart the game first
    this._recursivePlay();
};
