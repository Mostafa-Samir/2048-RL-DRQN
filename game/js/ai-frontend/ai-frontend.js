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

AI.makeAMove = function(direction) {
    let state = this.state();

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

    if(state.over) {
        this.playedGames++;
        console.log("Episodes: %d/%d", this.playedGames, this.trainingEpisodes);

        if(this.playedGames === this.trainingEpisodes) {
            return;
        }
        this.restart();
        state = this.state();
    }

    if(state.won) {
        this.GameManager.keepPlaying()
    }

    let availableMoves = this.listLegalActions();

    $http.post('/dfnn/action', {
        state: state.grid._1d,
        playMode:false,
        legalActions: availableMoves
    })
    .then((response) => {
        let action = response.action;
        let experience = this.makeAMove(action, true);

        return $http.post('/dfnn/experience', experience);
    })
    .then((response) => {
        if(response.success) {
            this._recursiveTrain();
        }
    });
}

AI._recursivePlay = function() {
    let state = this.state();

    if(state.over) {
        return;
    }

    if(state.won) {
        this.GameManager.keepPlaying()
    }

    let availableMoves = this.listLegalActions();

    $http.post('/dfnn/action', {
        state: state.grid._1d,
        playMode:true,
        legalActions: availableMoves
    })
    .then((response) => {
        let action = response.action;
        let outcome = this.makeAMove(action);

        setTimeout(this._recursivePlay.bind(this), 500);
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
