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
| `/train`  | Train a LoRA of a specific person from a photo folder (Z-Image-Turbo)|
| `/lora`   | Load/clear a trained LoRA: `/lora <file.safetensors> [scale]` · `/lora clear` |
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

## Exact same person (LoRA fine-tuning)

`/person` (Kontext) conditions on a single photo and only *loosely* preserves a face — good
for "someone like this", not "this exact person". To reliably get the **same** individual,
train a small **LoRA** on several photos of them, then generate with it.

Because mflux can no longer train FLUX.1, training runs on **Z-Image-Turbo** — a model mflux
*can* fine-tune locally on Apple Silicon, and which then loads the trained LoRA for inference.
The whole loop is on-device: no PyTorch, no cloud, no login (Z-Image-Turbo isn't gated).

### 1. Gather photos

Make a folder with ~10–20 varied photos of **one** person (different angles, lighting,
backgrounds; clear face):

```
person/thejan/
  001.jpg
  002.jpg
  ...
```

Fewer than ~5 works but tends to overfit to a single pose/background.

### 2. Train

```
/train person/thejan thejan
```

- The second argument is the **trigger word** you'll use in prompts (defaults to the folder name).
- A caption `.txt` is auto-written next to each photo (`a photo of thejan`). For better results,
  edit them to describe each scene — caption everything *except* the identity (clothing, setting,
  pose), so the LoRA learns the face rather than the background.
- The first run downloads Z-Image-Turbo. Training runs `~100` epochs; **Ctrl-C** stops early and
  keeps the last checkpoint.
- Output: `loras/thejan.safetensors`, automatically loaded when training finishes.

### 3. Generate

Prompt using the trigger word — the model now renders that specific person:

```
thejan as an astronaut on Mars, cinematic
```

Style `/mode`s and `/res` still apply.

### Loading an existing LoRA

```
/lora loras/thejan.safetensors        # load (use /lora <file> 0.8 to apply it more weakly)
/lora clear                           # turn off — back to FLUX text-to-image
```

- A loaded LoRA **routes generation to Z-Image-Turbo** (not FLUX), since the adapter is
  architecture-specific. `/lora clear` returns to your selected FLUX model.
- Trained LoRAs, training configs and checkpoints live under `loras/` (git-ignored).
- Tuning knobs live near the top of `main.py`: `TRAIN_EPOCHS`, `TRAIN_RANK`, `TRAIN_LEARNING_RATE`,
  `TRAIN_QUANTIZE`. Raising epochs/rank strengthens likeness but risks overfitting.

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

## Troubleshooting

### `/person` generates a mosaic or random-pixel image

**Cause:** The FLUX.1 Kontext-dev transformer weights are split across three shard files
(`diffusion_pytorch_model-00001/00002/00003-of-00003.safetensors`). If the download was
interrupted, one or more shards may be missing. mflux does not error on a missing shard —
it silently loads the model with zeros for those layers, which produces garbage output.

**Diagnose:** Check that all three shards are present in the cache:

```bash
ls ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-Kontext-dev/snapshots/*/transformer/
```

You should see all three `diffusion_pytorch_model-000{01,02,03}-of-00003.safetensors` symlinks.
If any are missing, download the absent shard explicitly:

```bash
uv run hf download black-forest-labs/FLUX.1-Kontext-dev \
  transformer/diffusion_pytorch_model-00002-of-00003.safetensors \
  --repo-type model
```

Replace `00002` with whichever shard number is absent. Each shard is roughly 9 GB, so the
download takes a while on a slow connection. Once complete, relaunch the app and try again.

### First `/person` generation is slow

Expected — Kontext re-quantizes from the full HuggingFace weights on every launch (unlike
the text-to-image models which cache a quantized copy). Subsequent generations in the same
session are fast.
