'use strict';

let stateRepresenters = {};

/**
 * encodes the grid as 1 dimensional array of 16 elements
 * each of which is a log2 of the tile value
 * @param {Object}: the state to encode
 * @return {Array}
*/
stateRepresenters.encode16x1 = function(state) {
    let linearization = state.grid.numerical1D;

    return linearization.map(cell => cell === 0 ? 0 : Math.log2(cell))
};

/**
 * encodes the grid as 2 dimensional array of 4x4x1 elements
 * each of which is a log2 of the tile value
 * @param {Object}: the state to encode
 * @return {Array}
*/
stateRepresenters.encode4x4x1 = function(state) {
    let grid = state.grid.numerical2D;

    return grid.map(row => row.map(cell => cell === 0 ? 0 : Math.log2(cell)))
};

/**
 * encodes the grid as 3 dimensional array of 4x4x16 elements
 * each of which is a 16 elements array x0, ..., x15
 * where xj = 1 , j = log2(cell), xi = 0 for all i not equal to j
 * @param {Object}: the state to encode
 * @return {Array}
*/
stateRepresenters.encode4x4x16 = function(state) {
    let grid = state.grid.numerical2D;
    let channels = (cell) => {
        let zeros = Array(16).fill(0);
        let indx = cell === 0 ? 0 : Math.log2(cell);

        zeros[indx] = 1;

        return zeros;
    }

    return grid.map(row => row.map(cell => channels(cell)))
};
