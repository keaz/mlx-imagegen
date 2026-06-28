# mlx-imagegen

An interactive, terminal-based image generator that runs **FLUX** diffusion models
locally on Apple Silicon via [`mflux`](https://github.com/filipstrand/mflux) (MLX).

Type a prompt, get a PNG. Switch models and style modes on the fly with slash commands.

## Requirements

- Apple Silicon Mac (M1 or newer)
- [`uv`](https://docs.astral.sh/uv/) (manages Python 3.12 + dependencies for you)
- A microphone — optional, only for the `/voice` speech-to-text command
- A Hugging Face login — optional, only for gated models (`/person` Kontext, and FLUX.1 dev)

## Run

```bash
uv run main.py
```

Generated images are written to `output/` with a timestamp filename,
e.g. `output/20260627-143022-123.png`.

### First run & caching

The **first** time you use a model, its full-precision weights download from Hugging Face
(~34 GB for FLUX.1 schnell) and are quantized on load. The app then saves a small quantized
copy under `models/<model>/` (a few GB). **Every later launch loads that copy directly** —
no re-download, no re-quantize, much faster startup.

After the quantized copy exists, the original ~34 GB download under `~/.cache/huggingface`
is no longer needed; you can delete it to reclaim space, e.g.:

```bash
rm -rf ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell
```

## Commands

| Command   | What it does                                                        |
|-----------|---------------------------------------------------------------------|
| *(prompt)* | Type any text and press Enter to generate an image                 |
| `/model`  | Choose the generation model (FLUX.1 schnell / dev, 4-bit / 8-bit)   |
| `/mode`   | Choose a style preset: `realistic`, `cartoon`, `anime`, `sketch`    |
| `/res`    | Choose output resolution: `1080`, `4K`, `8K`                        |
| `/person` | Use an optional reference photo of a person (FLUX Kontext)           |
| `/voice`  | Speak your prompt instead of typing it (local Whisper STT)          |
| `/help`   | Show available commands and current settings                        |
| `/exit`   | Quit the program                                                    |

## Modes

Modes are prompt-style presets — they append style keywords to whatever you type so the
same FLUX model can produce different looks without extra downloads.

| Mode        | Look                                              |
|-------------|---------------------------------------------------|
| `realistic` | Photorealistic, natural lighting, sharp detail    |
| `cartoon`   | Bold outlines, flat vibrant colors                |
| `anime`     | Cel-shaded anime key-visual style                 |
| `sketch`    | Hand-drawn graphite pencil sketch                 |

## Resolution

Pick the output size with `/res` (all 16:9):

| Option | Pixels       | Notes                                          |
|--------|--------------|------------------------------------------------|
| `1080` | 1920 × 1080  | Full HD — fast, small files (default)          |
| `4K`   | 3840 × 2160  | Ultra HD — upscaled from the native render     |
| `8K`   | 7680 × 4320  | Upscaled — very large PNGs, slow to encode     |

**How it works:** FLUX renders natively around 1 megapixel — it can't generate 4K or 8K
directly (that causes artifacts and huge memory use). So every image is rendered at a
FLUX-native **1536 × 864** base and then upscaled to the chosen resolution with Lanczos
resampling. You get true 4K/8K *dimensions*, but the detail is that of the ~1.3 MP render
enlarged — Lanczos makes it bigger, it doesn't invent new detail. For genuinely detailed
high-res output you'd add a super-resolution pass (e.g. mflux's SeedVR2 upscaler), which
isn't wired in yet.

## Person reference (optional)

Want images of a specific person? Type `/person path/to/photo.jpg`, then prompt as usual —
the prompt describes the scene/style and the person comes from the photo, generated with
[FLUX.1 Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev). Type
`/person clear` to turn it off and return to plain text-to-image.

- **Optional:** with no reference set, generation works exactly as before.
- **One photo, no training:** a single clear, front-facing photo works best.
- **Modes still apply:** e.g. a reference + `anime` mode renders that person as an anime character.
- **Likeness is approximate:** Kontext keeps the subject recognizably consistent, but it is *not*
  a pixel-perfect face match — this stack has no dedicated face-identity adapter (e.g. PuLID).
- **Gated model:** FLUX.1 Kontext-dev needs a one-time Hugging Face login — accept the license on
  its model page, then run `uv run huggingface-cli login`. The first `/person` generation downloads
  it (cached afterward; re-quantizes on each launch).

## Voice input (local speech-to-text)

Prefer talking to typing? Type `/voice`, speak your prompt, and press Enter to stop. The
audio is transcribed **locally** with [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
(model `whisper-large-v3-turbo`); the text is shown for you to confirm or edit, then it
generates. Typing still works exactly as before — `/voice` is just an alternative.

- **Fully local:** the Whisper model downloads once (~1.6 GB) and then runs offline — no
  audio ever leaves your machine.
- **First use:** macOS asks permission for your terminal to use the microphone
  (System Settings ▸ Privacy & Security ▸ Microphone).
- After `heard: …`, press **Enter** to generate, **type** to correct the text first, or
  **`/c`** to cancel.

## Models

| Key            | Model           | Notes                                              |
|----------------|-----------------|----------------------------------------------------|
| `schnell-4bit` | FLUX.1 schnell  | Fastest, 4 steps, Apache-2.0, no login (default)   |
| `schnell-8bit` | FLUX.1 schnell  | Sharper, slightly more memory                      |
| `dev-4bit`     | FLUX.1 dev      | Highest quality, ~20 steps, requires Hugging Face login |
