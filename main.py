#!/usr/bin/env python3
"""mlx-imagegen — an interactive, local image generator.

Runs FLUX diffusion models on Apple Silicon via mflux (MLX). Type a prompt to
generate a PNG into ./output/<timestamp>.png. Switch models, style modes, and
output resolution on the fly with slash commands:

    /model   choose the generation model
    /mode    choose a style preset (realistic / cartoon / anime / sketch)
    /res     choose output resolution (1080 / 4K / 8K)
    /person  use an optional reference photo of a person (FLUX Kontext)
    /voice   speak your prompt instead of typing it (local Whisper)
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

# Person/reference generation uses FLUX.1 Kontext-dev (gated): it conditions on
# a single reference photo + the prompt to place that person in a new scene.
# Recommended dev settings are ~28 steps and guidance ~2.5; quantized to 4-bit
# to fit comfortably in memory.
KONTEXT_QUANTIZE = 4
KONTEXT_STEPS = 28
KONTEXT_GUIDANCE = 2.5

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

# Local speech-to-text (mlx-whisper). The model downloads once (~1.6 GB) and
# then runs fully offline — no audio ever leaves the machine.
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
STT_SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono audio

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
    print(dim("  Type a prompt to generate an image.  Commands: ") + bold("/model /mode /res /person /voice /help /exit"))


def show_status(app: "App") -> None:
    if app.ref_image:
        head = dim("  model: ") + bold("FLUX.1 Kontext-dev")
        person = dim("    person: ") + bold(Path(app.ref_image).name)
    else:
        head = dim("  model: ") + bold(MODELS[app.model_key]["label"])
        person = ""
    print(
        head
        + dim("    mode: ") + bold(app.mode_key)
        + dim("    res: ") + bold(app.res_key)
        + person
        + dim(f"    output: {OUTPUT_DIR.name}/")
    )


def show_help() -> None:
    print()
    print(bold("  Commands"))
    print("    " + bold("/model") + dim("   choose the generation model"))
    print("    " + bold("/mode") + dim("    choose a style preset (realistic / cartoon / anime / sketch)"))
    print("    " + bold("/res") + dim("     choose output resolution (1080 / 4K / 8K)"))
    print("    " + bold("/person") + dim("  use a reference photo of a person (FLUX Kontext) · /person clear to turn off"))
    print("    " + bold("/voice") + dim("   speak your prompt instead of typing (local Whisper)"))
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
        self.ref_image: str | None = None  # optional person/reference photo (Kontext)
        self._flux = None        # cached mflux Flux1 instance
        self._flux_key = None    # which model_key the cache holds
        self._kontext = None     # cached mflux Flux1Kontext instance (reference mode)

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

    def _load_kontext(self):
        """Load FLUX.1 Kontext-dev, used for person/reference generation.

        Kontext is a separate (gated) model; loading it frees the text-to-image
        model so only one large model sits in memory at a time. It has no
        quantized-save support, so it loads from the Hugging Face cache and
        re-quantizes on each launch (slower startup than the cached schnell).
        """
        if self._kontext is not None:
            return self._kontext
        # Free the txt2img model so we don't hold two large models at once.
        self._flux = None
        self._flux_key = None
        try:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
        except Exception as e:  # pragma: no cover - environment/setup issue
            raise RuntimeError(f"Could not import mflux Kontext: {e}") from e
        info("Loading FLUX.1 Kontext-dev · 4-bit … "
             + dim("(gated; first use downloads the model, then cached; re-quantizes each launch)"))
        kontext = Flux1Kontext(model_config=ModelConfig.dev_kontext(), quantize=KONTEXT_QUANTIZE)
        self._kontext = kontext
        return kontext

    def generate(self, user_prompt: str) -> Path | None:
        """Generate one image and save it. Uses the person reference (Kontext)
        when one is set, otherwise plain text-to-image. Returns the path."""
        prompt = build_prompt(user_prompt, self.mode_key)
        if self.ref_image:
            return self._generate_with_reference(prompt)
        return self._generate_txt2img(prompt)

    def _generate_txt2img(self, prompt: str) -> Path | None:
        spec = MODELS[self.model_key]
        try:
            flux = self._load_flux()
        except Exception as e:
            err(str(e))
            if spec["name"] == "dev":
                info("FLUX.1 dev is gated. Accept the license at "
                     "https://huggingface.co/black-forest-labs/FLUX.1-dev and run "
                     + bold("uv run huggingface-cli login"))
            return None
        self._kontext = None  # free the reference model if it was loaded

        res = RESOLUTIONS[self.res_key]
        seed = random.randint(0, 2**32 - 1)
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
            if type(e).__name__ == "StopImageGenerationException":
                warn("Generation cancelled.")
            else:
                err(f"Generation failed: {e}")
            return None
        return self._finish_image(generated, start)

    def _generate_with_reference(self, prompt: str) -> Path | None:
        try:
            kontext = self._load_kontext()
        except Exception as e:
            err(str(e))
            info("FLUX.1 Kontext-dev is gated. Accept the license at "
                 "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev and run "
                 + bold("uv run huggingface-cli login"))
            return None

        res = RESOLUTIONS[self.res_key]
        seed = random.randint(0, 2**32 - 1)
        info(f"Generating from reference {bold(Path(self.ref_image).name)} → {res['label']}, "
             f"{KONTEXT_STEPS} steps, seed {seed} " + dim(f"[mode: {self.mode_key}]"))
        start = time.monotonic()
        try:
            generated = kontext.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=KONTEXT_STEPS,
                height=GEN_BASE_H,
                width=GEN_BASE_W,
                guidance=KONTEXT_GUIDANCE,
                image_path=self.ref_image,
            )
        except KeyboardInterrupt:
            warn("Generation cancelled.")
            return None
        except Exception as e:
            if type(e).__name__ == "StopImageGenerationException":
                warn("Generation cancelled.")
            else:
                err(f"Generation failed: {e}")
            return None
        return self._finish_image(generated, start)

    def _finish_image(self, generated, start: float) -> Path:
        """Upscale the native render to the chosen resolution and save it."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / timestamp_filename()
        target = RESOLUTIONS[self.res_key]["size"]
        from PIL import Image
        img = generated.image
        if img.size != target:
            info(f"Upscaling {img.size[0]}×{img.size[1]} → {target[0]}×{target[1]} " + dim("(Lanczos)"))
            img = img.resize(target, Image.Resampling.LANCZOS)
        img.save(out_path)
        elapsed = time.monotonic() - start
        ok(f"Saved {bold(str(out_path))} " + dim(f"{target[0]}×{target[1]}") + "  " + dim(f"({elapsed:.1f}s)"))
        return out_path


# --------------------------------------------------------------------------- #
# Speech input — fully local STT via mlx-whisper                              #
# --------------------------------------------------------------------------- #

def record_and_transcribe() -> str | None:
    """Record from the mic until Enter, then transcribe locally with Whisper.

    Returns the transcribed text, or None if nothing was captured / on error.
    Audio libraries and the Whisper model are imported/loaded lazily; the model
    downloads once (~1.6 GB) and then runs fully offline.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as e:  # pragma: no cover - environment/setup issue
        err(f"Audio libraries unavailable: {e}")
        return None

    frames: list = []

    def _callback(indata, _frame_count, _time_info, _status):  # runs on audio thread
        frames.append(indata.copy())

    print(cyan("  🎤 Recording…") + dim("  speak your prompt, then press Enter to stop"))
    try:
        with sd.InputStream(samplerate=STT_SAMPLE_RATE, channels=1, dtype="float32", callback=_callback):
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                print()
    except Exception as e:
        err(f"Could not access the microphone: {e}")
        info("On macOS, allow mic access for your terminal in "
             "System Settings ▸ Privacy & Security ▸ Microphone.")
        return None

    if not frames:
        return None
    audio = np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)
    if audio.size < STT_SAMPLE_RATE * 0.3:  # less than ~0.3s of audio
        return None

    info(f"Transcribing locally with {WHISPER_MODEL.split('/')[-1]} "
         + dim("(first use downloads the model once, then offline)"))
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL)
    except Exception as e:
        err(f"Transcription failed: {e}")
        return None
    return (result.get("text") or "").strip() or None


def do_voice(app: "App") -> None:
    """Record a spoken prompt, let the user confirm/edit it, then generate."""
    text = record_and_transcribe()
    if not text:
        warn("Didn't catch any speech — try again.")
        return
    # Print the recognized voice prompt as soon as recording stops (Enter pressed).
    print()
    ok("Voice prompt:  " + bold(text))
    print()
    try:
        edited = input(cyan("  [Enter] generate · type to edit · /c cancel ❯ ")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if edited == "/c":
        info("Cancelled.")
        return
    app.generate(edited if edited else text)


# --------------------------------------------------------------------------- #
# Person / reference photo (optional, FLUX Kontext)                           #
# --------------------------------------------------------------------------- #

def set_reference(app: "App", raw: str) -> None:
    """Set or clear the optional person/reference photo used by Kontext.

    Usage: ``/person <path-to-photo>`` to set, ``/person clear`` to turn off.
    With no argument it reports usage.
    """
    parts = raw.split(maxsplit=1)
    arg = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else ""

    if arg.lower() in ("", "clear", "off", "none"):
        if app.ref_image:
            app.ref_image = None
            app._kontext = None  # free the Kontext model
            ok("Reference cleared — back to text-to-image.")
        else:
            info("Usage: " + bold("/person <path-to-photo>") + dim("   ·   /person clear  to turn off"))
        show_status(app)
        return

    path = Path(arg).expanduser()
    if not path.is_file():
        err(f"No such image: {path}")
        return
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
    except Exception:
        err(f"Couldn't read '{path.name}' as an image.")
        return

    app.ref_image = str(path)
    ok(f"Person reference set: {bold(path.name)}")
    info("Prompts now place this person in the scene you describe, via FLUX Kontext "
         + dim("(gated model — first use downloads it).  /person clear to turn off."))
    show_status(app)


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
    elif cmd in ("/person", "/ref", "/face"):
        set_reference(app, raw)
    elif cmd in ("/voice", "/speak", "/mic"):
        do_voice(app)
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
