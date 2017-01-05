'use strict';

let utility = {};

/**
 * overwrites proprties form in the first object from proprties in the second one
 * @param {Object} tagret - the object to be overwritten
 * @param {Object} source - the object to overwrite from
*/
utility.overwrite = function(target, source) {
    for(let key in source) {

        let targetKeyExists = target.hasOwnProperty(key);
        let sourceType = typeof(source[key]);
        let targetType = typeof(target[key]);

        if(targetKeyExists && sourceType === targetType) {
            if(sourceType === "object" && !(source[key] instanceof Array)) {
                utility.overwrite(target[key], source[key]);
            }
            else {
                target[key] = source[key];
            }
        }
    }
};

/**
 * given an 2048 GameManager Object, it returns the list of legal
 * actions that can be performed at that particular step
 * @param {Object} manager
 * @return {Array}
*/
utility.listLegalActions = function(manager) {
    let legalActions = [];

    manager.prepareTiles();

    for(let direction = 0; direction < 4; ++direction) {
        let vector = manager.getVector(direction);
        let traversals = manager.buildTraversals(vector);

        traversals.x.forEach(function (x) {
            traversals.y.forEach(function(y) {
                let cell = {x: x, y:y};
                let tile = manager.grid.cellContent(cell);
                if(tile) {
                    let position = manager.findFarthestPosition(cell, vector);
                    let next = manager.grid.cellContent(position.next);
                    if((next && next.value === tile.value && !next.mergedFrom) || (!manager.positionsEqual(cell, position.farthest))) {
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
