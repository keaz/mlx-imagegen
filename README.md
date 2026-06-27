# mlx-imagegen

An interactive, terminal-based image generator that runs **FLUX** diffusion models
locally on Apple Silicon via [`mflux`](https://github.com/filipstrand/mflux) (MLX).

Type a prompt, get a PNG. Switch models and style modes on the fly with slash commands.

## Requirements

- Apple Silicon Mac (M1 or newer)
- [`uv`](https://docs.astral.sh/uv/) (manages Python 3.12 + dependencies for you)

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

## Models

| Key            | Model           | Notes                                              |
|----------------|-----------------|----------------------------------------------------|
| `schnell-4bit` | FLUX.1 schnell  | Fastest, 4 steps, Apache-2.0, no login (default)   |
| `schnell-8bit` | FLUX.1 schnell  | Sharper, slightly more memory                      |
| `dev-4bit`     | FLUX.1 dev      | Highest quality, ~20 steps, requires Hugging Face login |
