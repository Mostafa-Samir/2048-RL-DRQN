'use strict';

// container object for the AI controls
var AI = {};

AI.experience = [];

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

AI.collectExperience = function(size) {
    var lastScore = 0;

    while(this.experience.length < size) {
        var gameState = this.state();

        if(gameState.over) {
            lastScore = 0;
            this.GameManager.restart();
        }

        var direction = this.getAction(gameState.grid._1d);
        this.move(direction);

        var newState = this.state();

        this.experience.push({
            state: gameState.grid._1d,
            action: direction,
            reward: newState.score - gameState.score
        });
    }
}

AI.train = function() {
    this.collectExperience(256);
    $http.post('/dfnn/experience', {e: this.experience})
    .then((res) => console.log(res));
}
