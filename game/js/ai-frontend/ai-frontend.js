'use strict';

// container object for the AI controls
var AI = {};

AI.trainingEpisodes = 1000;
AI.playedGames = 0;
AI.afterTrainCallback = null;
AI.manuallyStoppedTraining = false;
AI.avg_loss = 0;

AI.prepare = function(state) {
    let input = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];

    for(let x = 0; x < 4; ++x) {
        for(let y = 0; y < 4; ++y) {
            let channels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            let indx = state[x][y] === 0 ? 0 : Math.log2(state[x][y])

            channels[indx] = 1;

            input[x][y] = channels;
        }
    }

    return input;
}

AI.init = function(learningCurveCharter, scoresCharter, reporter) {
    this.GameManager = new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
    /*this.learningCurveCharter = learningCurveCharter;
    this.scoresCharter = scoresCharter;
    this.reporter = reporter;*/
};

AI.move = function(direction) {
    this.GameManager.move(direction);
};

AI.restart = function() {
    this.GameManager.restart();
};

AI.state = function() {
    let state = this.GameManager.serialize();
    state.grid = this.GameManager.grid;
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
    let grid = AI.prepare(state.grid.getNumericalRepresentation());

    this.move(direction);
    let newstate = this.state();
    let newgrid = AI.prepare(newstate.grid.getNumericalRepresentation());
    let nextLegalActions = this.listLegalActions();

    let plainReward = newstate.score - state.score;
    let log2Reward = plainReward !== 0 ? Math.log2(plainReward) : 0;

    return {
        state: grid,
        action: direction,
        reward:  log2Reward,
        nextstate: newgrid,
        nextLegalActions: nextLegalActions,
        lastTransition: newstate.over
    }
}

AI._recursiveTrain = function() {
    let graphsInView = document.querySelector('div.graphs').style.display !== "none";
    if(AI.manuallyStoppedTraining) {
        return;
    }

    let state = this.state();

    if(state.over) {
        this.playedGames++;
        logger.log(`Episodes: ${this.playedGames}/${this.trainingEpisodes}`);


        graphs.scoresGraph.x.push(this.playedGames);
        graphs.scoresGraph.y.push(state.score);

        let maxCell = state.grid.maxCellValue();

        if(!graphs.maxTilesGraph.data[maxCell]) {
            graphs.maxTilesGraph.data[maxCell] = 0;
        }
        graphs.maxTilesGraph.data[maxCell]++;

        if(graphsInView) {
            graphs.draw()
        }

        if(this.playedGames === this.trainingEpisodes) {
            this.afterTrainCallback();
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
        state: AI.prepare(state.grid.getNumericalRepresentation()),
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
            if(response.hasOwnProperty('loss')) {
                graphs.lossGraph.x.push(response.step - 1);
                graphs.lossGraph.y.push(response.loss);

                if(response.step === 1) {
                    graphs.avgLossGraph.x.push(response.step - 1);
                    graphs.avgLossGraph.y.push(response.loss);
                }

                this.avg_loss += response.loss;
                if(response.step % 100 === 0) {
                    graphs.avgLossGraph.x.push(response.step - 1);
                    graphs.avgLossGraph.y.push(this.avg_loss / 100);

                    this.avg_loss = 0;
                }

                if(graphsInView) {
                    graphs.draw();
                }
            }
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
        state: state.grid._2d,
        playMode:true,
        legalActions: availableMoves
    })
    .then((response) => {
        let action = response.action;
        let outcome = this.makeAMove(action);

        this._recursivePlay();
    });
}

AI.train = function(episodes, callback) {
    if(!this.manuallyStoppedTraining) {
        this.restart();  // restart the game first
        this.trainingEpisodes = episodes;
        this.playedGames = 0;
        this.afterTrainCallback = callback || function() { };
    }
    else {
        this.manuallyStoppedTraining = false;
    }

    this._recursiveTrain();
}

AI.play = function() {
    this._recursivePlay();
}
