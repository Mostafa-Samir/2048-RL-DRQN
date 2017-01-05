"use strict";

let logger = {};

/**
 * logs a regular message to the on screen logger
 * @param {String} msg - the message to be logged
 */
logger.log = function(msg) {
    let msgP = document.createElement('p');
    msgP.className = "log";
    let time = "[" + (new Date()).toLocaleString() + "]";
    msgP.innerText = time + '> ' + msg;

    let log = document.querySelector('div.ai-logs-container');
    log.appendChild(msgP);
    log.scrollTop = log.scrollHeight;
}

/**
 * logs a warning message to the on screen logger
 * @param {String} msg - the message to be logged
 */
logger.warn = function(msg) {
    let msgP = document.createElement('p');
    msgP.className = "log warning";
    let time = "[" + (new Date()).toLocaleString() + "]";
    msgP.innerText = time + '> ' + msg;

    let log = document.querySelector('div.ai-logs-container');
    log.appendChild(msgP);
    log.scrollTop = log.scrollHeight;
}

/**
 * logs an error message to the on screen logger
 * @param {String} msg - the message to be logged
 */
logger.error = function(msg) {
    let msgP = document.createElement('p');
    msgP.className = "log error";
    let time = "[" + (new Date()).toLocaleString() + "]";
    msgP.innerText = time + '> ' + msg;

    let log = document.querySelector('div.ai-logs-container');
    log.appendChild(msgP);
    log.scrollTop = log.scrollHeight;
}
