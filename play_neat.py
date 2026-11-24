"""Play the best NEAT genome and report the real in-game distance."""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

# Let "python play_neat.py" work even if python is not the venv interpreter.
if sys.platform == "win32":
    _venv_py = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if _venv_py.exists() and Path(sys.executable).resolve() != _venv_py.resolve():
        os.execv(str(_venv_py), [str(_venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])

import neat
from playwright.sync_api import sync_playwright

from fast_dino_env import FastDinoEnv

CONFIG_PATH = "neat_config.cfg"
BEST_FILE = "champion.pkl"


def _cfg() -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )


def load_genome(config: neat.Config):
    if os.path.exists(BEST_FILE):
        try:
            with open(BEST_FILE, "rb") as f:
                rec = pickle.load(f)
            genome = rec["genome"]
            if isinstance(genome, bytes):
                genome = pickle.loads(genome)
            seed = rec.get("seed")
            if seed is None:
                seed = int(genome.key) * 7919 + 1
            print(
                f"Loaded {BEST_FILE}: gen={rec.get('gen')}, "
                f"best_score={rec.get('score', 0.0):.2f}, seed={seed}"
            )
            return genome, seed
        except Exception as e:
            print(f"Could not load {BEST_FILE}: {e}")

    ckpts = sorted(
        glob.glob("neat-checkpoint-*"),
        key=lambda p: int(p.split("-")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError("No neat_best_by_distance.pkl and no neat-checkpoint-* found")

    population = neat.Checkpointer.restore_checkpoint(ckpts[-1])
    genomes = [g for g in population.population.values() if g.fitness is not None]
    if not genomes:
        raise RuntimeError(f"No evaluated genomes in {ckpts[-1]}")
    genome = max(genomes, key=lambda g: g.fitness)
    print(f"Loaded best genome from {ckpts[-1]}: fitness={genome.fitness:.2f}")
    return genome, int(genome.key) * 7919 + 1


def action_from_net(net, obs) -> int:
    out = net.activate(obs.tolist() if hasattr(obs, "tolist") else list(obs))
    return max(0, min(2, out.index(max(out))))


def play_fast(genome, config, episodes: int, steps: int, headless: bool, seed: int = 1):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            args=["--disable-gpu", "--disable-extensions", "--mute-audio"],
        )
        context = browser.new_context(viewport={"width": 800, "height": 600})
        env = FastDinoEnv(context)
        try:
            for ep in range(1, episodes + 1):
                score, used = env.evaluate(net, steps, seed)
                print(f"Episode {ep}: steps={used}, distance/score={score:.2f}", flush=True)
        finally:
            env.close()
            context.close()
            browser.close()


def play_visual(genome, config, episodes: int, steps: int, seed: int):
    """Watch the same deterministic policy used during training."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            args=["--mute-audio", "--window-size=1000,400"],
        )
        context = browser.new_context(viewport={"width": 1000, "height": 400})
        env = FastDinoEnv(context, render=True)
        env.page.evaluate(
            "document.body.style.zoom = '1.5'; document.body.style.overflow = 'hidden';"
        )
        env.page.evaluate("""() => {
          const hint = document.createElement("div");
          hint.id = "dino-start-hint";
          hint.textContent = "Press Space to start";
          Object.assign(hint.style, {
            position: "fixed", top: "42%", left: "0", width: "100%",
            textAlign: "center", fontSize: "30px", fontWeight: "bold",
            color: "#fff", zIndex: "9999", pointerEvents: "none"
          });
          document.body.appendChild(hint);
          document.addEventListener("keydown", (event) => {
            if (!event.isTrusted) return;
            if (event.code === "Space" || event.keyCode === 32) {
              hint.remove();
              window.__dinoUserStarted = true;
            }
          });
        }""")
        env.page.bring_to_front()
        print("Press Space in the browser window to start.", flush=True)
        env.page.wait_for_function("window.__dinoUserStarted === true", timeout=0)
        try:
            ep = 1
            # episodes <= 0 means "keep running until Ctrl+C".
            while episodes <= 0 or ep <= episodes:
                env.start_visual(net, seed)
                best = 0.0
                next_tick = time.perf_counter()
                for _ in range(steps):
                    result = env.step_visual()
                    best = max(best, float(result.get("distance", 0.0) or 0.0))
                    if result.get("crashed"):
                        break
                    next_tick += 1.0 / 60.0
                    time.sleep(max(0.0, next_tick - time.perf_counter()))
                env.stop_visual()
                print(f"Episode {ep}: distance/score={best:.2f}", flush=True)
                ep += 1
        finally:
            env.close()
            context.close()
            browser.close()


def main():
    parser = argparse.ArgumentParser(description="Watch the trained Dino (visible window by default)")
    parser.add_argument("--episodes", type=int, default=0, help="0 = run until Ctrl+C")
    parser.add_argument("--steps", type=int, default=60000, help="Frames per episode (60000 ≈ 15500+ score)")
    parser.add_argument("--fast", action="store_true", help="No window, just print the score")
    parser.add_argument("--visual", action="store_true", help="Same as default (kept for old commands)")
    parser.add_argument("--show", action="store_true", help="Show window in --fast mode")
    parser.add_argument("--seed", type=int, default=None, help="Obstacle seed (default: training seed)")
    args = parser.parse_args()

    config = _cfg()
    genome, train_seed = load_genome(config)
    seed = args.seed if args.seed is not None else (train_seed or 1)
    if args.fast:
        play_fast(genome, config, args.episodes, args.steps, headless=not args.show, seed=seed)
    else:
        play_visual(genome, config, args.episodes, args.steps, seed)


if __name__ == "__main__":
    main()
