"""
NEAT trainer for Dino.

Key design:
- one Chromium instance with 15 isolated pages (15 Dino games at once)
- the game is advanced deterministically inside JS (no real-time sleeps)
- fitness = in-game score (distanceRan * 0.025), target 15500+
- best genome ever is saved to champion.pkl
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pickle

import neat
from playwright.async_api import async_playwright

from fast_dino_env import network_to_js
from fast_dino_env_async import AsyncFastDinoEnv


NUM_ENVS = 15
MAX_STEPS_PER_EVAL = 120000
NEAT_CONFIG_PATH = "neat_config.cfg"
CHECKPOINT_PREFIX = "neat-checkpoint-"
BEST_GENOME_FILE = "champion.pkl"
GENERATIONS = 400


def load_config() -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG_PATH,
    )


def latest_checkpoint() -> str | None:
    files = [
        f for f in os.listdir(".")
        if f.startswith(CHECKPOINT_PREFIX)
        and f[len(CHECKPOINT_PREFIX):].isdigit()
    ]
    if not files:
        return None
    files.sort(key=lambda p: int(p[len(CHECKPOINT_PREFIX):]))
    return files[-1]


def load_best_record() -> dict:
    if os.path.exists(BEST_GENOME_FILE):
        try:
            with open(BEST_GENOME_FILE, "rb") as f:
                rec = pickle.load(f)
            if isinstance(rec.get("genome"), bytes):
                rec["genome"] = pickle.loads(rec["genome"])
            return rec
        except Exception:
            pass
    return {"fitness": -1e18, "score": 0.0, "gen": -1, "genome": None}


def save_best_record(rec: dict):
    data = dict(rec)
    if data.get("genome") is not None:
        data["genome"] = pickle.dumps(data["genome"])
    with open(BEST_GENOME_FILE, "wb") as f:
        pickle.dump(data, f)


async def main_async(args):
    config = load_config()
    ckpt = None if args.fresh else latest_checkpoint()
    if ckpt:
        print(f"Resuming from checkpoint: {ckpt}", flush=True)
        population = neat.Checkpointer.restore_checkpoint(ckpt)
    else:
        print("Starting new NEAT population.", flush=True)
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(
        neat.Checkpointer(generation_interval=5, filename_prefix=CHECKPOINT_PREFIX)
    )

    best = load_best_record()
    print(
        f"Saved best: score={best['score']:.2f}, gen={best['gen']}",
        flush=True,
    )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--disable-extensions", "--mute-audio"],
        )
        context = await browser.new_context(viewport={"width": 800, "height": 600})

        envs: list[AsyncFastDinoEnv] = []
        try:
            for i in range(args.envs):
                print(f"Starting Dino env {i + 1}/{args.envs}", flush=True)
                envs.append(await AsyncFastDinoEnv.create(context))

            loop = asyncio.get_running_loop()

            def eval_genomes(genomes, config):
                nonlocal best

                items = list(genomes)
                for start in range(0, len(items), args.envs):
                    chunk = items[start:start + args.envs]
                    net_jsons = []
                    seeds = []
                    for gid, genome in chunk:
                        net = neat.nn.FeedForwardNetwork.create(genome, config)
                        net_jsons.append(network_to_js(net))
                        # Deterministic per genome: retesting gives the same obstacles.
                        seeds.append(int(gid) * 7919 + 1)

                    coros = [
                        envs[i].evaluate(net_json, args.steps, seeds[i])
                        for i, net_json in enumerate(net_jsons)
                    ]
                    futures = [
                        asyncio.run_coroutine_threadsafe(coro, loop) for coro in coros
                    ]
                    results = [f.result() for f in futures]

                    for (gid, genome), (score, steps, crashed), seed in zip(chunk, results, seeds):
                        genome.fitness = score
                        if score > best["fitness"]:
                            best = {
                                "fitness": score,
                                "score": score,
                                "steps": steps,
                                "crashed": crashed,
                                "seed": seed,
                                "gen": population.generation,
                                "genome": genome,
                            }
                            save_best_record(best)
                            print(
                                f"  NEW BEST gen={population.generation}: "
                                f"score={score:.2f}, steps={steps}",
                                flush=True,
                            )

            try:
                await asyncio.to_thread(
                    population.run,
                    eval_genomes,
                    args.generations,
                )
            except KeyboardInterrupt:
                print("\nTraining interrupted. Best genome is saved.", flush=True)

            print(
                f"Final best: score={best['score']:.2f}, gen={best['gen']}",
                flush=True,
            )
        finally:
            for env in envs:
                await env.close()
            await context.close()
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--envs", type=int, default=NUM_ENVS)
    parser.add_argument("--steps", type=int, default=MAX_STEPS_PER_EVAL)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints")
    asyncio.run(main_async(parser.parse_args()))
