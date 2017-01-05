"use strict";

/**
 * queries the server for available saved checkpoints
*/
function loadAvailableCheckpoints() {

    return $http.get("/ai/checkpoints")
    .then((response) => {

        let modelsSelect = document.querySelector('.load-dialog ul');
        while(modelsSelect.firstChild) {
            modelsSelect.removeChild(modelsSelect.firstChild);
        }

        response.models.forEach((chckpt) => {
            let item = document.createElement('li');
            item.className = 'saved-model';
            item.dataset.name = chckpt;
            item.innerText = chckpt;

            item.addEventListener('click', function(e) {
                let chckpt = e.target.dataset.name;
                AI.load(chckpt)
                .then(() => {
                    document.querySelector('.save-load-container').style.display = 'none';
                    document.querySelector('.load-dialog').style.display = 'none';
                    document.querySelector('#load-spinner').style.visibility = 'hidden';

                    document.querySelector('#training-episodes').value = AI.controlVariables.train.targetEpisodes;
                    document.querySelector('#playing-episodes').value = AI.controlVariables.play.targetEpisodes;
                });

                document.querySelector('#load-spinner').style.visibility = 'visible';
            });
            modelsSelect.appendChild(item);
        });
    });
}

// Wait till the browser is ready to render the game (avoids glitches)
window.requestAnimationFrame(function () {
  new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
  AI.init();
  loadAvailableCheckpoints();
});


// UI control for the dashboard
document.querySelector(".view-stats").addEventListener('click', function(e) {
    let gameContainer = document.querySelector('div.container');
    let graphsContainer = document.querySelector('div.graphs');

    Array.prototype.slice.call(document.querySelectorAll('html, body'))
    .forEach(element => element.style.height='100%');

    gameContainer.style.display = "none";
    graphsContainer.style.display = "flex";

    graphs.draw();
});

document.querySelector(".view-game").addEventListener('click', function(e) {
    let gameContainer = document.querySelector('div.container');
    let graphsContainer = document.querySelector('div.graphs');

    Array.prototype.slice.call(document.querySelectorAll('html, body'))
    .forEach(element => element.style.height='auto');

    gameContainer.style.display = "block";
    graphsContainer.style.display = "none";
});

document.querySelector('.train-toggle').addEventListener('click', function(e) {
    let btn = document.querySelector('.train-toggle');

    if(btn.dataset.disabled === "true") {
        return;
    }

    if(btn.title === 'Start') {

        let count = parseInt(document.querySelector("#training-episodes").value);
        AI.train(count, ()=> {
            btn.title = 'Start';
            btn.querySelector('i').className = 'fa fa-play';

            Array.prototype.slice.call(document.querySelectorAll("input[type='text']"))
            .forEach(function(element) { element.disabled = false });
            Array.prototype.slice.call(document.querySelectorAll(".model-controller span"))
            .forEach(function(element) { element.dataset.disabled = "false" });
            document.querySelector('.train-toggle').dataset.disabled = "false";
        });

        btn.title = 'Stop';
        btn.querySelector('i').className = 'fa fa-pause';

        Array.prototype.slice.call(document.querySelectorAll("input[type='text']"))
        .forEach(function(element) { element.disabled = true });
        Array.prototype.slice.call(document.querySelectorAll(".model-controller span"))
        .forEach(function(element) { element.dataset.disabled = "true" });
        document.querySelector('.play-toggle').dataset.disabled = "true";
    }
    else {
        AI.controlVariables.train.manuallyStopped = true;
        logger.warn("Training was manually stopped");
        btn.title = 'Start';
        btn.querySelector('i').className = 'fa fa-play';

        Array.prototype.slice.call(document.querySelectorAll("input[type='text']"))
        .forEach(function(element) { element.disabled = false });
        Array.prototype.slice.call(document.querySelectorAll(".model-controller span"))
        .forEach(function(element) { element.dataset.disabled = "false" });
        document.querySelector('.play-toggle').dataset.disabled = "false";
    }
});

document.querySelector('.play-toggle').addEventListener('click', function(e) {
    let btn = document.querySelector('.play-toggle');

    if(btn.dataset.disabled === "true") {
        return;
    }

    if(btn.title === 'Start') {
        let count = parseInt(document.querySelector("#playing-episodes").value);

        AI.play(count, () => {
            Array.prototype.slice.call(document.querySelectorAll("input[type='text']"))
            .forEach(function(element) { element.disabled = false });
            Array.prototype.slice.call(document.querySelectorAll(".model-controller span"))
            .forEach(function(element) { element.dataset.disabled = "false" });
            document.querySelector('.train-toggle').dataset.disabled = "false";
        });

        btn.title = 'Stop';
        btn.querySelector('i').className = 'fa fa-pause';

        Array.prototype.slice.call(document.querySelectorAll("input[type='text']"))
        .forEach(function(element) { element.disabled = true });
        Array.prototype.slice.call(document.querySelectorAll(".model-controller span"))
        .forEach(function(element) { element.dataset.disabled = "true" });
        document.querySelector('.train-toggle').dataset.disabled = "true";
    }
    else {
        AI.controlVariables.play.manuallyStopped = true;
        btn.title = 'Start';
        btn.querySelector('i').className = 'fa fa-play';

        Array.prototype.slice.call(document.querySelectorAll("input[type='text']"))
        .forEach(function(element) { element.disabled = false });
        Array.prototype.slice.call(document.querySelectorAll(".model-controller span"))
        .forEach(function(element) { element.dataset.disabled = "false" });
        document.querySelector('.train-toggle').dataset.disabled = "false";
    }
});

document.querySelector('.load-model').addEventListener('click', function(e) {
    let btn = document.querySelector('.load-model');
    console.log(btn.dataset.disabled);
    if(btn.dataset.disabled !== "true") {
        loadAvailableCheckpoints()
        .then(() => {
            document.querySelector('.save-load-container').style.display = 'flex';
            document.querySelector('.load-dialog').style.display = 'block';
        });
    }
});

document.querySelector('.save-model').addEventListener('click', function(e) {
    let btn = document.querySelector('.save-model');
    console.log(btn.dataset.disabled);
    if(btn.dataset.disabled !== "true") {
        document.querySelector('.save-load-container').style.display = 'flex';
        document.querySelector('.save-dialog').style.display = 'block';

        document.querySelector("#saved-model-name").value = "checkpoint-" + (new Date()).toJSON();
        document.querySelector("#saved-model-name").select();
    }
});

document.querySelector('.save-dialog #save').addEventListener('click', function(e) {
    let fname = document.querySelector("#saved-model-name").value;
    if(fname.trim() !== '') {
        AI.save(fname.trim())
        .then(() => {
            document.querySelector('.save-load-container').style.display = 'none';
            document.querySelector('.save-dialog').style.display = 'none';
            document.querySelector('#save-spinner').style.visibility = 'hidden';
        });

        document.querySelector('#save-spinner').style.visibility = 'visible';
    }
});

document.querySelector('.save-load-container').addEventListener('click', function(e) {
    document.querySelector('.save-load-container').style.display = 'none';
    document.querySelector('.save-dialog').style.display = 'none';
    document.querySelector('.load-dialog').style.display = 'none';
});

Array.prototype.slice.call(document.querySelectorAll('.save-load-container > div'))
.forEach((element) => {
    element.addEventListener('click', function(e) {
        e.stopPropagation();
    });
});

Array.prototype.slice.call(document.querySelectorAll('input[type="text"]'))
.forEach((element) => {
    element.addEventListener('keydown', function(e) {
        e.stopPropagation();
    });
});
