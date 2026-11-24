"""Fast deterministic Dino environment driven entirely inside Chromium."""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import BrowserContext

ROOT = Path(__file__).resolve().parent
GAME_FILE = ROOT / "game" / "index.html"
BRIDGE_FILE = ROOT / "game" / "eval_bridge.js"


def network_to_js(net) -> dict:
    """Convert a neat FeedForwardNetwork to a JSON-friendly phenotype."""
    node_evals = []
    for node, act_func, _agg_func, bias, response, links in net.node_evals:
        node_evals.append({
            "node": int(node),
            "act": act_func.__name__,
            "bias": float(bias),
            "response": float(response),
            "links": [[int(inode), float(weight)] for inode, weight in links],
        })
    return {
        "input_keys": [int(k) for k in net.input_nodes],
        "output_keys": [int(k) for k in net.output_nodes],
        "node_evals": node_evals,
    }


class FastDinoEnv:
    """One browser page = one deterministic Dino runner."""

    def __init__(self, context: BrowserContext, render: bool = False):
        self.page = context.new_page()
        self.page.set_default_timeout(0)
        self.page.goto(GAME_FILE.as_uri(), wait_until="domcontentloaded")
        self.page.wait_for_function(
            "window.Runner && window.Runner.instance_ && window.Runner.instance_.tRex"
        )
        self.page.add_script_tag(content=BRIDGE_FILE.read_text(encoding="utf-8"))
        ok = self.page.evaluate("(p) => window.__dinoInstall(p)", {"render": bool(render)})
        if not ok:
            raise RuntimeError("Failed to install Dino evaluation bridge")

    def evaluate(self, net, max_steps: int, seed: int = 1) -> tuple[float, int]:
        """Run one full episode. Returns (score/distance, steps)."""
        result = self.page.evaluate(
            "(p) => window.__dinoEval(p.net, p.maxSteps, p.seed)",
            {"net": network_to_js(net), "maxSteps": int(max_steps), "seed": int(seed)},
        )
        return float(result.get("distance", 0.0)), int(result.get("steps", 0))

    def start_visual(self, net, seed: int = 1):
        self.page.evaluate(
            "(p) => window.__dinoStart(p.net, p.seed)",
            {"net": network_to_js(net), "seed": int(seed)},
        )

    def step_visual(self) -> dict:
        return self.page.evaluate("window.__dinoStepOnce()")

    def stop_visual(self):
        try:
            self.page.evaluate("window.__dinoStop()")
        except Exception:
            pass

    def close(self):
        try:
            self.page.close()
        except Exception:
            pass
