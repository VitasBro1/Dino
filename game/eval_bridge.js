(function () {
  "use strict";

  // Real keyboard input from the user. Synthetic events sent by the agent
  // are isTrusted=false and are ignored here.
  window.__dinoUserStarted = false;
  window.__dinoManual = { jump: false, duck: false };
  document.addEventListener("keydown", function (event) {
    if (!event.isTrusted) return;
    if (event.code === "Space" || event.keyCode === 32) {
      window.__dinoUserStarted = true;
      window.__dinoManual.jump = true;
    }
    if (event.code === "ArrowDown" || event.keyCode === 40) {
      window.__dinoManual.duck = true;
    }
  });
  document.addEventListener("keyup", function (event) {
    if (!event.isTrusted) return;
    if (event.code === "ArrowDown" || event.keyCode === 40) {
      window.__dinoManual.duck = false;
    }
  });

  function keyEvent(type, code) {
    const ev = new KeyboardEvent(type, { bubbles: true, cancelable: true });
    Object.defineProperty(ev, "keyCode", { get: () => code });
    Object.defineProperty(ev, "which", { get: () => code });
    document.dispatchEvent(ev);
  }

  function activation(name, x) {
    switch (name) {
      case "tanh":
      case "tanh_activation": return Math.tanh(x);
      case "sigmoid":
      case "sigmoid_activation": return 1 / (1 + Math.exp(-x));
      case "relu":
      case "relu_activation": return x > 0 ? x : 0;
      case "identity":
      case "identity_activation": return x;
      case "clamped":
      case "clamped_activation": return Math.max(0, Math.min(1, x));
      default: return Math.tanh(x);
    }
  }

  function activateNetwork(net, inputs) {
    const values = {};
    for (let i = 0; i < net.input_keys.length; i++) {
      values[net.input_keys[i]] = inputs[i] || 0;
    }
    for (let j = 0; j < net.node_evals.length; j++) {
      const ne = net.node_evals[j];
      let s = 0;
      for (let k = 0; k < ne.links.length; k++) {
        s += (values[ne.links[k][0]] || 0) * ne.links[k][1];
      }
      values[ne.node] = activation(ne.act, ne.bias + ne.response * s);
    }
    return net.output_keys.map((k) => values[k] || 0);
  }

  function obstacleFeatures(ob, dino) {
    if (!ob) return [1, 0, 0, 0, 0];
    const dx = ob.xPos - dino.xPos;
    const type = ob.typeConfig && ob.typeConfig.type ? ob.typeConfig.type : "";
    const isPtero = type.indexOf("Ptero") !== -1;
    const w = (ob.typeConfig && ob.typeConfig.width) || ob.width || 0;
    const h = (ob.typeConfig && ob.typeConfig.height) || ob.height || 0;
    const vec = [
      dx / 600.0,
      w / 100.0,
      h / 100.0,
      (ob.yPos || 0) / 150.0,
      isPtero ? 1 : 0
    ];
    for (let i = 0; i < vec.length; i++) {
      if (!isFinite(vec[i])) vec[i] = 0;
      vec[i] = Math.max(-5, Math.min(5, vec[i]));
    }
    return vec;
  }

  function makeObservation(runner, stack) {
    const dino = runner.tRex;
    const obstacles = (runner.horizon && runner.horizon.obstacles) || [];
    obstacles.sort((a, b) => a.xPos - b.xPos);
    const speed = runner.currentSpeed || 6;

    const dinoFeat = [
      (dino.groundYPos - dino.yPos) / 100.0,
      (dino.jumpVelocity || 0) / 10.0,
      dino.jumping ? 1 : 0,
      dino.ducking ? 1 : 0,
      speed / 13.0
    ];
    const vec = dinoFeat
      .concat(obstacleFeatures(obstacles[0], dino))
      .concat(obstacleFeatures(obstacles[1], dino));

    stack.push(vec);
    while (stack.length > 3) stack.shift();
    while (stack.length < 3) stack.unshift(new Array(15).fill(0));

    const flat = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 15; j++) flat.push(stack[i][j]);
    }
    return flat;
  }

  function manualStep(runner) {
    // requestAnimationFrame is disabled and performance.now() is virtualized:
    // every call advances exactly one game frame (deterministic across runs).
    window.__dinoClock += runner.msPerFrame;
    runner.update();
  }

  function releaseKeys(state) {
    if (state.jumpHeld) {
      keyEvent("keyup", 32);
      state.jumpHeld = false;
    }
    if (state.duckHeld) {
      keyEvent("keyup", 40);
      state.duckHeld = false;
    }
  }

  function applyAction(action, runner, state) {
    if (action === 1) {
      if (state.duckHeld) {
        keyEvent("keyup", 40);
        state.duckHeld = false;
      }
      if (!state.jumpHeld && !runner.tRex.jumping && !runner.tRex.ducking) {
        keyEvent("keydown", 32);
        state.jumpHeld = true;
      }
    } else if (action === 2) {
      if (state.jumpHeld) {
        keyEvent("keyup", 32);
        state.jumpHeld = false;
      }
      if (!state.duckHeld) {
        keyEvent("keydown", 40);
        state.duckHeld = true;
      }
    } else {
      releaseKeys(state);
    }
  }

  let visNet = null;
  let visStack = [];
  let visState = null;
  let visSteps = 0;
  let visBest = 0;

  function resetSession(runner, net, seed) {
    window.__dinoRngState = (seed | 0) || 1;
    window.__dinoClock = 1000.0;
    runner.raqId = 0;
    runner.restart();
    runner.playingIntro = false;
    runner.activated = true;
    runner.raqId = 1;
    runner.updatePending = false;
    if (runner.tRex) {
      runner.tRex.playingIntro = false;
      runner.tRex.xPos = 50;
    }
    // Original CSS keeps the runner container 44px wide until the intro
    // finishes. Our bridge skips the intro, so open it up explicitly.
    if (runner.containerEl) {
      runner.containerEl.style.width = runner.dimensions.WIDTH + "px";
      runner.containerEl.style.height = runner.dimensions.HEIGHT + "px";
    }
    visNet = net;
    visStack = [];
    for (let i = 0; i < 3; i++) visStack.push(new Array(15).fill(0));
    visState = { jumpHeld: false, duckHeld: false };
    if (window.__dinoManual) window.__dinoManual.jump = false;
    visSteps = 0;
    visBest = 0;
  }

  function sessionStep(runner) {
    const obs = makeObservation(runner, visStack);
    const out = activateNetwork(visNet, obs);
    let action = 0;
    if (out[1] >= out[0] && out[1] >= out[2]) action = 1;
    else if (out[2] >= out[0] && out[2] >= out[1]) action = 2;

    // A real key press wins for this frame. The network itself is untouched:
    // it keeps running from the current game state on the next frame.
    const manual = window.__dinoManual || {};
    if (manual.jump) action = 1;
    else if (manual.duck) action = 2;
    manual.jump = false;

    applyAction(action, runner, visState);
    manualStep(runner);
    const score = (runner.distanceRan || 0) * 0.025;
    if (score > visBest) visBest = score;
    visSteps++;
    return {
      distance: visBest,
      steps: visSteps,
      crashed: !!runner.crashed,
      action: action,
    };
  }

  window.__dinoStart = function (net, seed) {
    const runner = window.__runner;
    if (!runner) return { error: "not installed" };
    resetSession(runner, net, seed);
    return { distance: 0, steps: 0, crashed: false };
  };

  window.__dinoStepOnce = function () {
    const runner = window.__runner;
    if (!runner || !visNet) return { distance: 0, steps: 0, crashed: true };
    return sessionStep(runner);
  };

  window.__dinoStop = function () {
    const runner = window.__runner;
    if (!runner) return;
    releaseKeys(visState || { jumpHeld: false, duckHeld: false });
    if (runner.crashed || visSteps > 0) {
      runner.stop();
      runner.crashed = false;
      runner.paused = false;
      runner.raqId = 0;
    }
    visNet = null;
  };

  window.__dinoInstall = function (opt) {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) return false;
    if (runner.__dinoInstalled) return true;
    runner.__dinoInstalled = true;

    cancelAnimationFrame(runner.raqId);
    runner.raqId = 0;
    // The game polls window size every 250ms and calls stop() on resize.
    // During long visual episodes that silently resets the runner.
    if (runner.resizeTimerId_) {
      clearInterval(runner.resizeTimerId_);
      runner.resizeTimerId_ = null;
    }
    runner.adjustDimensions = function () {};
    runner.updatePending = false;
    runner.scheduleNextUpdate = function () {};
    runner.onVisibilityChange = function () {};
    runner.loadSounds = function () {};
    runner.playSound = function () {};
    runner.startListening = function () {};
    runner.stopListening = function () {};

    // Seeded RNG: same genome + same seed always sees the same obstacles.
    window.__dinoRngState = 1;
    Math.random = function () {
      window.__dinoRngState = (window.__dinoRngState + 0x6D2B79F5) | 0;
      let t = window.__dinoRngState;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    // Virtual clock: one manual step == exactly one game frame.
    window.__dinoClock = performance.now();
    Object.defineProperty(performance, "now", {
      value: function () { return window.__dinoClock; },
      configurable: true,
    });

    // In fast training mode rendering is disabled; visual mode keeps it.
    if (!opt || !opt.render) {
      CanvasRenderingContext2D.prototype.drawImage = function () {};
      CanvasRenderingContext2D.prototype.clearRect = function () {};
      CanvasRenderingContext2D.prototype.fill = function () {};
    }

    window.__runner = runner;
    return true;
  };

  window.__dinoEval = function (net, maxSteps, seed) {
    const runner = window.__runner;
    if (!runner) return { distance: 0, steps: 0, crashed: true, error: "not installed" };

    window.__dinoRngState = (seed | 0) || 1;
    window.__dinoClock = 1000.0;
    runner.raqId = 0;
    runner.restart();
    runner.playingIntro = false;
    runner.activated = true;
    runner.raqId = 1; // isRunning() must be true so keyup can end jumps
    runner.updatePending = false;

    const stack = [];
    for (let i = 0; i < 3; i++) stack.push(new Array(15).fill(0));
    const state = { jumpHeld: false, duckHeld: false };
    let best = 0;
    let steps = 0;

    for (let s = 0; s < maxSteps; s++) {
      const obs = makeObservation(runner, stack);
      const out = activateNetwork(net, obs);
      let action = 0;
      if (out[1] >= out[0] && out[1] >= out[2]) action = 1;
      else if (out[2] >= out[0] && out[2] >= out[1]) action = 2;

      applyAction(action, runner, state);
      manualStep(runner);

      const score = (runner.distanceRan || 0) * 0.025;
      if (score > best) best = score;
      steps++;

      if (runner.crashed) break;
    }

    releaseKeys(state);
    const crashed = !!runner.crashed;
    runner.stop();
    runner.crashed = false;
    runner.paused = false;
    runner.raqId = 0;

    return { distance: best, score: best, steps: steps, crashed: crashed };
  };
})();
