"use strict";

/**
 * Constructs a Progress reporter that writes progress to given container
 * @param conatiner {String}: a valid css selector for the container element
**/
function ProgressReporter(container) {

    this.element = document.querySelector(container);
}

/**
 * reports the given status into the defined element
 * @param current {Number}: the current state
 * @param goal {Number}: the goal state
**/
ProgressReporter.prototype.report = function (current, goal) {
    let reportMessage = current + "/" + goal;
    this.element.innerText = reportMessage;
};
