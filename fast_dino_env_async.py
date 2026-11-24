"""Async version of the fast deterministic Dino environment."""
from __future__ import annotations

from pathlib import Path

from playwright.async_api import BrowserContext

ROOT = Path(__file__).resolve().parent
GAME_FILE = ROOT / "game" / "index.html"
BRIDGE_FILE = ROOT / "game" / "eval_bridge.js"


class AsyncFastDinoEnv:
    """One Chromium page with its own deterministic Dino runner."""

    def __init__(self, page):
        self.page = page

    @classmethod
    async def create(cls, context: BrowserContext) -> "AsyncFastDinoEnv":
        page = await context.new_page()
        page.set_default_timeout(0)
        await page.goto(GAME_FILE.as_uri(), wait_until="domcontentloaded")
        await page.wait_for_function(
            "window.Runner && window.Runner.instance_ && window.Runner.instance_.tRex"
        )
        await page.add_script_tag(content=BRIDGE_FILE.read_text(encoding="utf-8"))
        ok = await page.evaluate("window.__dinoInstall()")
        if not ok:
            raise RuntimeError("Failed to install Dino evaluation bridge")
        return cls(page)

    async def evaluate(self, net_json: dict, max_steps: int, seed: int = 1) -> tuple[float, int, bool]:
        result = await self.page.evaluate(
            "(p) => window.__dinoEval(p.net, p.maxSteps, p.seed)",
            {"net": net_json, "maxSteps": int(max_steps), "seed": int(seed)},
        )
        return (
            float(result.get("distance", 0.0) or 0.0),
            int(result.get("steps", 0) or 0),
            bool(result.get("crashed", False)),
        )

    async def close(self):
        try:
            await self.page.close()
        except Exception:
            pass
