/**
 * constructs a wrraper around CanavasJS.Chart
 * @param conatiner {String}: id of the container element
 * @param type {String}: type of the chart
 * @param title {String}: title of the chart
 * @param maxDatapoints {Number}: maximum number of data points to chart (default: 10000)
 * @return {Object}: a Charter instance
**/
function Charter(container, type, title, maxDatapoints) {

    this.container = container;
    this.type = type;
    this.title = title;
    this.maxDatapoints = this.maxDatapoints || 5000;

    this.datapoints = [{x: 0, y: 0}];

    this.chart = new CanvasJS.Chart(this.container, {
        title: {
            text: this.title,
            fontSize: 15,
            fontColor: "white"
        },
        backgroundColor: null,
        data: [{
            type: this.type,
            dataPoints: this.datapoints
        }]
    });

    this.chart.render();
}

/**
 * update the chart with a new data point
 * @param newdatapoint {Object}: the new datapoint {x, y}
**/
Charter.prototype.update = function (newdatapoint) {
    this.datapoints.push(newdatapoint);
    if(this.datapoints.length > this.maxDatapoints) {
        this.datapoints.shift();
    }

    this.chart.render();
};
