'use strict';

// container object for the AI controls
var AI = {};

AI.trainingEpisodes = 1000;
AI.playedGames = 0;

AI.init = function() {
    this.GameManager = new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
};

AI.move = function(direction) {
    this.GameManager.move(direction);
};

AI.restart = function() {
    this.GameManager.restart();
};

AI.state = function() {
    let state = this.GameManager.serialize();
    state.grid._2d = this.GameManager.grid.getNumericalRepresentation();
    state.grid._1d = this.GameManager.grid.getNumericalLinearization();

    return state;
};

AI.listLegalActions = function() {
    let legalActions = [];
    let self = this;

    self.GameManager.prepareTiles();

    for(let direction = 0; direction < 4; ++direction) {
        let vector = self.GameManager.getVector(direction);
        let traversals = self.GameManager.buildTraversals(vector);

        traversals.x.forEach(function (x) {
            traversals.y.forEach(function(y) {

                let cell = {x: x, y:y};
                let tile = self.GameManager.grid.cellContent(cell);

                if(tile) {
                    let position = self.GameManager.findFarthestPosition(cell, vector);
                    let next = self.GameManager.grid.cellContent(position.next);

                    if((next && next.value === tile.value && !next.mergedFrom) ||
                    (!self.GameManager.positionsEqual(cell, position.farthest))) {
                        if(legalActions.indexOf(direction) === -1) {
                            legalActions.push(direction);
                        }
                    }
                }
            });
        });
    }

    return legalActions;
};

AI.getAction = function(state) {
    // random for now
    var availableMoves = this.listLegalActions();
    var randomMoveIndx = Math.floor(Math.random() * availableMoves.length)

    return availableMoves[randomMoveIndx];
}

AI.makeAMove = function(direction, training) {
    let mode = training ? 'train' : 'play';
    let state = this.state();

    if(state.over && mode === 'train') {
        this.trainingEpisodes;
        this.playedGames++;
        console.log("Episodes: %d/%d", this.playedGames, this.trainingEpisodes)

        if(this.trainingEpisodes - this.playedGames > 0) {
            this.restart();
            state = this.state();
        }
        else {
            return false;
        }
    }
    else if(state.over && mode === 'play') {
        return false;
    }

    if(state.won) {
        this.GameManager.keepPlaying()
    }

    this.move(direction);
    let newstate = this.state();

    return {
        state: state.grid._1d,
        action: direction,
        reward: newstate.score - state.score,
        nextstate: newstate.grid._1d
    }
}

AI._recursiveTrain = function() {
    let state = this.state();

    $http.post('/dfnn/action', {state: state.grid._1d, playMode:false})
    .then((response) => {
        let action = response.action;
        let experience = this.makeAMove(action, true);

        return experience;
    })
    .then((experience) => {
        if(!experience) {
            return 'stop';
        }
        else {
            return $http.post('/dfnn/experience', experience);
        }
    })
    .then((control) => {
        if(control === 'stop') {
            return;
        }
        else if(control.success === 'true') {
            AI._recursiveTrain();
        }
    });
}

AI._recursivePlay = function() {
    let state = this.state();

    $http.post('/dfnn/action', {state: state.grid._1d, playMode:true})
    .then((response) => {
        let action = response.action;
        let outcome = this.makeAMove(action);

        if(!outcome) {
            return;
        }
        else {
            setTimeout(this._recursivePlay.bind(this), 500);
        }
    });
}

AI.train = function(episodes) {
    this.restart();  // restart the game first
    this.trainingEpisodes = episodes;
    this.playedGames = 0;

    this._recursiveTrain();
}

AI.play = function() {
    this._recursivePlay();
}
