"use strict";

// Wait till the browser is ready to render the game (avoids glitches)
window.requestAnimationFrame(function () {
  new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);

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
    let episodesCount = parseInt(episodesCounter.value);

    episodesCounter.disabled = true;
    self.disabled = true;

    AI.train(episodesCount, () => {
        episodesCounter.disabled = false;
        self.disabled = false;
    });
})
