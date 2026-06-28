#!/usr/bin/env python3
"""mlx-imagegen — an interactive, local image generator.

Runs FLUX diffusion models on Apple Silicon via mflux (MLX). Type a prompt to
generate a PNG into ./output/<timestamp>.png. Switch models, style modes, and
output resolution on the fly with slash commands:

    /model   choose the generation model
    /mode    choose a style preset (realistic / cartoon / anime / sketch)
    /res     choose output resolution (1080 / 4K / 8K)
    /help    show commands
    /exit    quit

Requires an Apple Silicon Mac. Heavy ML libraries are imported lazily on the
first generation so the prompt appears instantly.
"""

from __future__ import annotations

import random
import sys
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MODELS_DIR = Path(__file__).resolve().parent / "models"  # local quantized-model cache for fast reloads

# FLUX renders natively around 1 megapixel; pushing it past ~2 MP causes
# artifacts and huge memory use. So we always render at this 16:9 base (good
# quality, fast) and upscale to the chosen output resolution.
GEN_BASE_W, GEN_BASE_H = 1536, 864  # 16:9, ~1.3 MP, both divisible by 16

# Selectable output resolutions. FLUX can't render 4K/8K directly, so those are
# upscaled (Lanczos) from the native render; 1080p is a mild upscale.
RESOLUTIONS: dict[str, dict] = {
    "1080": {"size": (1920, 1080), "label": "1080p · 1920×1080", "desc": "Full HD · fast, small files (default)"},
    "4K":   {"size": (3840, 2160), "label": "4K · 3840×2160",    "desc": "Ultra HD · upscaled from native render"},
    "8K":   {"size": (7680, 4320), "label": "8K · 7680×4320",    "desc": "Upscaled · very large files, slow to encode"},
}
DEFAULT_RES = "1080"

# Selectable models. Each maps to an mflux FLUX variant + quantization and the
# generation settings that suit it. `name` is the mflux model name; `quantize`
# is the bit-width (4 or 8). schnell is guidance-distilled so `guidance` is
# effectively ignored for it; dev uses it.
MODELS: dict[str, dict] = {
    "schnell-4bit": {
        "name": "schnell", "quantize": 4, "steps": 4, "guidance": 0.0,
        "label": "FLUX.1 schnell · 4-bit",
        "desc": "Fastest · Apache-2.0 · no login (default)",
    },
    "schnell-8bit": {
        "name": "schnell", "quantize": 8, "steps": 4, "guidance": 0.0,
        "label": "FLUX.1 schnell · 8-bit",
        "desc": "Sharper · a bit more memory",
    },
    "dev-4bit": {
        "name": "dev", "quantize": 4, "steps": 20, "guidance": 3.5,
        "label": "FLUX.1 dev · 4-bit",
        "desc": "Top quality · slower · requires Hugging Face login",
    },
}
DEFAULT_MODEL = "schnell-4bit"

# Style modes. These are prompt presets: the keywords are appended to whatever
# the user types so one model can produce different looks with no extra weights.
MODES: dict[str, dict] = {
    "realistic": {
        "desc": "Photorealistic, natural lighting, sharp detail",
        "keywords": "photorealistic, ultra detailed, natural lighting, sharp focus, "
                    "high dynamic range, lifelike textures, 8k",
    },
    "cartoon": {
        "desc": "Bold outlines, flat vibrant colors",
        "keywords": "cartoon illustration, bold clean outlines, flat vibrant colors, "
                    "cel shaded, playful, stylized",
    },
    "anime": {
        "desc": "Cel-shaded anime key-visual style",
        "keywords": "anime style, detailed anime illustration, cel shading, vibrant "
                    "colors, studio anime key visual, expressive eyes",
    },
    "sketch": {
        "desc": "Hand-drawn graphite pencil sketch",
        "keywords": "pencil sketch, hand drawn, graphite, fine cross-hatching, "
                    "monochrome, detailed line art, sketchbook",
    },
}
DEFAULT_MODE = "realistic"

# --------------------------------------------------------------------------- #
# Tiny terminal styling (no dependencies; disabled when output isn't a TTY)   #
# --------------------------------------------------------------------------- #

_COLOR = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def bold(s: str) -> str: return _c("1", s)
def dim(s: str) -> str: return _c("2", s)
def cyan(s: str) -> str: return _c("36", s)
def green(s: str) -> str: return _c("32", s)
def yellow(s: str) -> str: return _c("33", s)
def red(s: str) -> str: return _c("31", s)


def info(s: str) -> None: print(dim("· ") + s)
def ok(s: str) -> None: print(green("✓ ") + s)
def warn(s: str) -> None: print(yellow("! ") + s)
def err(s: str) -> None: print(red("✗ ") + s)


# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #

def timestamp_filename() -> str:
    """A collision-resistant, sortable filename: 20260627-143022-123.png."""
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}.png"


def build_prompt(user_prompt: str, mode_key: str) -> str:
    """Append the active mode's style keywords to the user's prompt."""
    return f"{user_prompt.strip()}, {MODES[mode_key]['keywords']}"


def model_items() -> list[tuple[str, str, str]]:
    return [(k, v["label"], v["desc"]) for k, v in MODELS.items()]


def mode_items() -> list[tuple[str, str, str]]:
    return [(k, k.capitalize(), v["desc"]) for k, v in MODES.items()]


def res_items() -> list[tuple[str, str, str]]:
    return [(k, v["label"], v["desc"]) for k, v in RESOLUTIONS.items()]


# --------------------------------------------------------------------------- #
# Interactive UI                                                              #
# --------------------------------------------------------------------------- #

def banner() -> None:
    print()
    print("  " + bold(cyan("mlx-imagegen")) + dim("  ·  FLUX diffusion on Apple Silicon (MLX)"))
    print(dim("  Type a prompt to generate an image.  Commands: ") + bold("/model /mode /res /help /exit"))


def show_status(app: "App") -> None:
    spec = MODELS[app.model_key]
    print(
        dim("  model: ") + bold(spec["label"])
        + dim("    mode: ") + bold(app.mode_key)
        + dim("    res: ") + bold(app.res_key)
        + dim(f"    output: {OUTPUT_DIR.name}/")
    )


def show_help() -> None:
    print()
    print(bold("  Commands"))
    print("    " + bold("/model") + dim("   choose the generation model"))
    print("    " + bold("/mode") + dim("    choose a style preset (realistic / cartoon / anime / sketch)"))
    print("    " + bold("/res") + dim("     choose output resolution (1080 / 4K / 8K)"))
    print("    " + bold("/help") + dim("    show this help"))
    print("    " + bold("/exit") + dim("    quit (Ctrl-D also works)"))
    print()
    print(dim("  Anything else you type becomes the image prompt."))


def choose(title: str, items: list[tuple[str, str, str]], current_key: str) -> str:
    """Show a numbered menu and return the selected key (Enter keeps current)."""
    print()
    print(bold("  " + title))
    for i, (key, label, desc) in enumerate(items, 1):
        marker = green("●") if key == current_key else dim("○")
        line = f"   {marker} {i}. {bold(label)}"
        if desc:
            line += dim(f" — {desc}")
        print(line)
    print()
    while True:
        try:
            raw = input(cyan("  select # ") + dim("(Enter to keep current): ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return current_key
        if raw == "":
            return current_key
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return items[int(raw) - 1][0]
        warn("Enter a number between 1 and %d." % len(items))


# --------------------------------------------------------------------------- #
# Application state + generation                                              #
# --------------------------------------------------------------------------- #

class App:
    """Holds the current model/mode/resolution selection and a cached model."""

    def __init__(self) -> None:
        self.model_key = DEFAULT_MODEL
        self.mode_key = DEFAULT_MODE
        self.res_key = DEFAULT_RES
        self._flux = None        # cached mflux Flux1 instance
        self._flux_key = None    # which model_key the cache holds

    def _load_flux(self):
        """Return a loaded Flux1 for the active model, (re)loading if needed.

        The mflux import and model load are deferred to here so startup is
        instant. The first time a model is used, its full weights download from
        Hugging Face (tens of GB) and are quantized; we then save a small
        quantized copy under models/<model_key>/ so later launches load quickly
        without re-downloading or re-quantizing.
        """
        if self._flux_key == self.model_key and self._flux is not None:
            return self._flux

        spec = MODELS[self.model_key]
        # Release any previously loaded model before loading the next one so we
        # don't hold two large models in unified memory at once.
        self._flux = None
        self._flux_key = None

        try:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.flux.variants.txt2img.flux import Flux1
        except Exception as e:  # pragma: no cover - environment/setup issue
            raise RuntimeError(
                f"Could not import mflux (Apple Silicon + MLX required): {e}"
            ) from e

        saved = MODELS_DIR / self.model_key
        flux = None

        # Fast path: load a previously-saved quantized copy (small, no re-quantize).
        if saved.exists():
            try:
                info(f"Loading {spec['label']} from cache " + dim(f"models/{self.model_key}/") + " …")
                flux = Flux1(
                    model_config=ModelConfig.from_name(model_name=spec["name"]),
                    model_path=str(saved),
                    quantize=None,  # quantization level is read from the saved weights' metadata
                )
            except Exception as e:
                warn(f"Local cache unusable ({e}); re-downloading from Hugging Face.")
                flux = None

        # Slow path: download full weights, quantize, then cache a small copy.
        if flux is None:
            info(f"Loading {spec['label']} … "
                 + dim("(first use downloads full weights from Hugging Face — tens of GB — and quantizes)"))
            flux = Flux1.from_name(model_name=spec["name"], quantize=spec["quantize"])
            try:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                info("Saving a quantized copy for fast future launches " + dim(f"→ models/{self.model_key}/"))
                flux.save_model(str(saved))
                ok(f"Cached {spec['label']}. Next launch of this model loads quickly.")
            except Exception as e:
                warn(f"Couldn't save quantized copy (will re-quantize next time): {e}")

        self._flux = flux
        self._flux_key = self.model_key
        return flux

    def generate(self, user_prompt: str) -> Path | None:
        """Generate one image for the prompt and save it. Returns the path."""
        spec = MODELS[self.model_key]
        prompt = build_prompt(user_prompt, self.mode_key)

        try:
            flux = self._load_flux()
        except Exception as e:
            err(str(e))
            if spec["name"] == "dev":
                info("FLUX.1 dev is gated. Accept the license at "
                     "https://huggingface.co/black-forest-labs/FLUX.1-dev and run "
                     + bold("uv run huggingface-cli login"))
            return None

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / timestamp_filename()
        seed = random.randint(0, 2**32 - 1)
        res = RESOLUTIONS[self.res_key]
        target = res["size"]

        info(f"Generating {GEN_BASE_W}×{GEN_BASE_H} render → {res['label']}, "
             f"{spec['steps']} steps, seed {seed} " + dim(f"[mode: {self.mode_key}]"))
        start = time.monotonic()
        try:
            generated = flux.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=spec["steps"],
                height=GEN_BASE_H,
                width=GEN_BASE_W,
                guidance=spec["guidance"],
            )
        except KeyboardInterrupt:
            warn("Generation cancelled.")
            return None
        except Exception as e:
            # mflux raises StopImageGenerationException on Ctrl-C mid-loop.
            if type(e).__name__ == "StopImageGenerationException":
                warn("Generation cancelled.")
            else:
                err(f"Generation failed: {e}")
            return None

        # FLUX renders at the native base size; scale up to the chosen resolution.
        from PIL import Image
        img = generated.image
        if img.size != target:
            info(f"Upscaling {img.size[0]}×{img.size[1]} → {target[0]}×{target[1]} " + dim("(Lanczos)"))
            img = img.resize(target, Image.Resampling.LANCZOS)
        img.save(out_path)

        elapsed = time.monotonic() - start
        ok(f"Saved {bold(str(out_path))} " + dim(f"{target[0]}×{target[1]}") + f"  " + dim(f"({elapsed:.1f}s)"))
        return out_path


# --------------------------------------------------------------------------- #
# REPL                                                                        #
# --------------------------------------------------------------------------- #

def handle_command(app: App, raw: str) -> bool:
    """Handle a /command. Returns False if the program should exit."""
    cmd = raw.split()[0].lower()
    if cmd == "/exit":
        return False
    elif cmd == "/model":
        app.model_key = choose("Select a model:", model_items(), app.model_key)
        show_status(app)
    elif cmd == "/mode":
        app.mode_key = choose("Select a mode:", mode_items(), app.mode_key)
        show_status(app)
    elif cmd in ("/res", "/resolution", "/size"):
        app.res_key = choose("Select an output resolution:", res_items(), app.res_key)
        show_status(app)
    elif cmd in ("/help", "/?"):
        show_help()
    else:
        warn(f"Unknown command: {cmd}   (try /help)")
    return True


def main() -> int:
    banner()
    app = App()
    show_status(app)

    while True:
        try:
            raw = input(cyan(f"\n  [{app.mode_key}] ❯ ")).strip()
        except EOFError:          # Ctrl-D quits
            print()
            break
        except KeyboardInterrupt:  # Ctrl-C cancels the current line, not the app
            print()
            info("(type /exit to quit)")
            continue

        if not raw:
            continue
        if raw.startswith("/"):
            if not handle_command(app, raw):
                break
            continue

        app.generate(raw)

    print(dim("\n  bye 👋\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
