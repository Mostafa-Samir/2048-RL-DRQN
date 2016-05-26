"use strict";

// Wait till the browser is ready to render the game (avoids glitches)
window.requestAnimationFrame(function () {
  new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);

  $http.get("/dfnn/saved-models")
  .then((response) => {

      let modelsSelect = document.querySelector('#trained-model')

      response.models.forEach((model) => {
          let optElement = document.createElement('option');
          let txtNode = document.createTextNode(model);

          optElement.appendChild(txtNode);
          optElement.value = model;

          modelsSelect.appendChild(optElement);
      });
  });

  let learnCurve = new Charter("learning-curve", "line", "Learning Curve");
  let scoreChart  = new Charter("score-chart", "line", "Final Scores");
  let gamesReporter = new ProgressReporter(".progress");

  AI.init(learnCurve, scoreChart, gamesReporter);
});


// UI control for the dashboard
document.querySelector("span.show-hide").addEventListener('click', function(e) {
    let self = e.target;
    let dashboardContent = document.querySelector('.dashboard-content');
    let state = dashboardContent.style.display;

    if(state === 'none') {
        dashboardContent.style.display = 'block';
        self.innerText = 'Hide';
    }
    else {
        dashboardContent.style.display = 'none';
        self.innerText = 'Show';
    }
});

document.querySelector("#start-training").addEventListener('click', function(e) {
    let self = e.target;
    let episodesCounter = document.querySelector("input[name='train-episodes']");
    let modelLoader = document.querySelector("#load-model");
    let modelSaver = document.querySelector("#save-model");
    let episodesCount = parseInt(episodesCounter.value);

    episodesCounter.disabled = true;
    modelLoader.disabled = true;
    modelSaver.disabled = true;
    self.disabled = true;


    AI.train(episodesCount, () => {
        episodesCounter.disabled = false;
        modelLoader.disabled = false;
        modelSaver.disabled = false;
        self.disabled = false;
    });
});

document.querySelector("#load-model").addEventListener('click', function(e) {
    let filename = document.querySelector("#trained-model").value;

    if(filename) {
        $http.post("/dfnn/load", {filename: filename});
    }
});

document.querySelector("#save-model").addEventListener('click', function(e) {
    let datestr = (new Date()).toLocaleString().replace(/[\/, \:]/g, '.');
    let filename = window.prompt("Enter Model's name:", "model-" + datestr + ".ckpt");

    if(filename) {
        $http.post("/dfnn/save", {filename: filename});
    }
});
