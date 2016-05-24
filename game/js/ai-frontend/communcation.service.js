// container for the http sevice methods
var $http = {}

/**
 * sends a GET request to the specified url
 * @param url {String}: the url to which the request will be sent
 * @return {Promise}
 **/
$http.get = function(url) {
    var req = new XMLHttpRequest();
    req.open('GET', url, true);
    req.send();

    return new Promise(function(resolve, reject) {
        req.onreadystatechange = function() {
            if(req.readyState === XMLHttpRequest.DONE) {
                if(req.status === 200) {
                    resolve(JSON.parse(req.responseText));
                }
                else {
                    reject(JSON.parse(req.responseText));
                }
            }
        }
    });
};

/**
 * sends a POST request with the given data to the specified url
 * @param url {String}: the url to which the request will be sent
 * @param data {Object}: the data to be sent with the request
 * @return {Promise}
 **/
$http.post = function(url, data) {
    var req = new XMLHttpRequest();
    req.open('POST', url, true);

    req.setRequestHeader('Content-Type', 'application/json');
    req.send(JSON.stringify(data))

    return new Promise(function(resolve, reject) {
        req.onreadystatechange = function() {
            if(req.readyState === XMLHttpRequest.DONE) {
                if(req.status === 200) {
                    resolve(JSON.parse(req.responseText));
                }
                else {
                    reject(JSON.parse(req.responseText));
                }
            }
        }
    });
};

//var $socket = io.connect('http://' + document.domain + ':' + location.port);
