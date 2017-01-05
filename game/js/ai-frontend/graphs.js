'use strict';

let graphs = {

    lossGraph: {
        name: 'Loss',
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: 'rgba(0, 0, 255, 0.2)',
        }
    },

    avgLossGraph: {
        name: 'Average Loss',
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: 'blue',
            width: 2
        }
    },

    scoresGraph: {
        name: 'Scores',
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: 'red',
            width: 2
        }
    },

    maxTilesGraph: {
        name: 'Maximum Tiles',
        data: {},
        type: 'bar',
        marker: {
            color: 'rgba(255, 153, 0, 0.6)',
            line: {
                color: 'rgba(255, 153, 0, 1)',
                width: 2
            }
        }
    },

    commonLayout: {
        xaxis: {showgrid: false},
        yaxis: {showgrid: false},
        hovermode: 'closest',
        autosize:false,
        margin: {t: 40, b: 40, l: 40, r: 40},
        paper_bgcolor: 'rgba(255,255,255, 0)',
        plot_bgcolor: 'rgba(255,255,255, 0)'
    },

    commonProprties: {
        showLink: false,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud', 'hoverCompareCartesian']
    }
};

/**
 * initializes all graphs to a clean state
*/
graphs.init = function() {
    graphs.lossGraph.x = [];
    graphs.lossGraph.y = [];
    graphs.avgLossGraph.x = [];
    graphs.avgLossGraph.y = [];
    graphs.scoresGraph.x = [];
    graphs.scoresGraph.y = [];
    graphs.maxTilesGraph.data = {};
};

/**
 * renders all graphs
*/
graphs.draw = function () {

    if(document.querySelector('.graphs').style.display === 'none') {
        // do not render any thing while the graphgs are invisible
        return;
    }

    let maxHeight = document.querySelector('.graphs div').clientHeight;
    let maxWidth = document.querySelector('.graphs div').clientWidth;

    graphs.commonLayout.width = maxWidth;
    graphs.commonLayout.height = maxHeight;

    let lossLayout = Object.create(graphs.commonLayout);
    lossLayout.title = 'Loss';
    Plotly.newPlot('loss-graph', [graphs.lossGraph, graphs.avgLossGraph], lossLayout, graphs.commonProprties);

    let scoresLayout = Object.create(graphs.commonLayout);
    scoresLayout.title = 'Scores';
    Plotly.newPlot('score-graph', [graphs.scoresGraph], scoresLayout, graphs.commonProprties);

    let maxTilesLayout = Object.create(graphs.commonLayout);
    maxTilesLayout.title = 'Maximum Tile';
    graphs.maxTilesGraph.x = [];
    graphs.maxTilesGraph.y = [];

    for(let tile in graphs.maxTilesGraph.data) {
        graphs.maxTilesGraph.x.push(tile);
        graphs.maxTilesGraph.y.push(graphs.maxTilesGraph.data[tile]);
    }

    Plotly.newPlot('maxtile-graph', [graphs.maxTilesGraph], maxTilesLayout, graphs.commonProprties);
}

/**
 * serializes graph data for checkpoint saving
*/
graphs.serialize = function() {
    return {
        lossGraph: {
            x: graphs.lossGraph.x,
            y: graphs.lossGraph.y
        },
        avgLossGraph: {
            x: graphs.avgLossGraph.x,
            y: graphs.avgLossGraph.y
        },
        scoresGraph: {
            x: graphs.scoresGraph.x,
            y: graphs.scoresGraph.y
        },
        maxTilesGraph: {
            data: graphs.maxTilesGraph.data
        }
    };
};
