'use strict';

let rewardFunctions = {};

/**
 * computes the reward as a log2 of the difference between the states' scores
 * @param {Object} oldstate
 * @param {Object} newstate
 * @return {Number}
*/
rewardFunctions.log2Plain = function(oldstate, newstate) {
    let diff = newstate.score - oldstate.score;

    return diff === 0 ? 0 : Math.log2(diff);
};

/**
 * computes the reward as the log2 of the difference between the max tiles in each state
 * @param {Object} oldstate
 * @param {Object} newstate
 * @return {Number}
*/
rewardFunctions.log2MaxTileEmptyDiff = function(oldstate, newstate) {
    let diff = newstate.grid.maxCell - oldstate.grid.maxCell;
    diff = (diff === 0) ? 0 : Math.log2(diff);

    return diff + (newstate.grid.emptyCellsCount - oldstate.grid.emptyCellsCount);
};
