# ChopShop — WIP audio/video "chop → analyze" toolkit

**Status:** very early **work-in-progress**. It's sharp around the edges and brittle in spots. I'll keep iterating, but expect breaking changes.

ChopShop's goal is to take input media (e.g., video), split it into constituent streams (video/audio/text), and generate rich features per stream. The current focus is **audio** — others will be added at a later point as I need them.

Current processes include:
* Extract each audio stream from a video container to WAV
* Diarize + transcribe multi-speaker audio (wrapping Mahmoud Ashraf's excellent `whisper-diarization`)
* Export a timestamped transcript in multiple formats (CSV/SRT/TXT)
* Build **per-speaker** WAVs from the transcript
* Export **Whisper encoder embeddings** per transcript segment (CTranslate2 / faster-whisper)

---

## Prerequisites

I'll admit, I'm not a guru here. The below items should be taken as "well, it works on my machine..." advice. Key specs that may or may not be important to match include:

* **OS:** Linux (tested on Ubuntu 22.04)
* **Python:** 3.10
* **GPU stack:** **CUDA 12.4** + **cuDNN 9**
* **FFmpeg binary:** must be available on your `PATH`

  * Ubuntu: `sudo apt-get install ffmpeg`
  * macOS: `brew install ffmpeg`
  * Windows: install FFmpeg and add it to PATH

> I **strongly recommend** using a fresh virtual environment (e.g., `python -m venv venv && source venv/bin/activate`) for everything below.

To verify your CUDA runtime: `nvidia-smi` should show the driver and CUDA version. I'm only testing against **CUDA 12.4** right now. It is unlikely that I will target any other CUDA versions in the foreseeable future. In theory, this will only impact which version of the dependencies you install, but your mileage may vary.

---

## Install

As a note, I'm putting in some effort to get all of this packaged up nicely as a `pip` installable library. At the moment, you can try this:

`pip install "chopshop[diarization,cuda]"`

...followed by the three following `pip` installs:

```
pip install git+https://github.com/MahmoudAshraf97/demucs.git
pip install git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
```

...and lastly, followed by this:

```
pip install --force-reinstall --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Also, again, be sure that you have ffpmeg installed if not already:
`sudo apt-get update && sudo apt-get install -y ffmpeg`

### Manual Installation of Dependencies

> It is highly likely that the order matters to avoid dependency pinning conflicts.

```bash
# (Recommended) fresh virtualenv
python -m venv venv-chopshop
source venv-chopshop/bin/activate

# 1) Core ASR/diarization deps
pip install "faster-whisper>=1.1.0"
pip install "nemo-toolkit[asr]>=2.dev"
pip install git+https://github.com/MahmoudAshraf97/demucs.git
pip install git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# 2) cuDNN user-space libs for CUDA 12
pip install -U nvidia-cudnn-cu12

# (Optional) some systems also install the OS package:
# sudo apt-get -y install cudnn9-cuda-12

# 3) PyTorch built for CUDA 12.4 (exact versions I tested)
pip install --force-reinstall --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

If you hit cuDNN errors, it usually means the CUDA/cuDNN runtime and the PyTorch wheels don't match. Keep them consistent (CUDA **12.4** with the `cu124` wheel index, and cuDNN 9 from either the `nvidia-cudnn-cu12` wheel or your OS package).

---

## Quick usage example

> The filenames are intentionally generic. Adjust as needed.

```python
from pathlib import Path
from chopshop.ChopShop import ChopShop

cs = ChopShop()

# 1) Split audio streams from a video/container
wav_list = cs.split_audio_streams(
    "sample_video.mp4",
    output_dir="audio/",
    sample_rate=48000,
    bit_depth=16,
    overwrite=True,
)
diar_input_audio = str(wav_list[0])  # pick a stream

# 2) Diarize + transcribe (wrapper around whisper-diarization)
tp = cs.diarize_with_thirdparty(
    input_audio=diar_input_audio,
    out_dir="transcripts/",
    repo_dir="path/to/whisper-diarization",  # if you have a local clone; otherwise leave default
    whisper_model="base",   # use whatever one you like; just be consistent
    language="en",
    device="cuda",
    batch_size=0,
    no_stem=False,
    suppress_numerals=False,
    parallel=False,
    use_custom=True,   # use my custom entry point that also writes a CSV
    keep_temp=False,   # if set to False, it will clean up after itself (good idea)
    num_speakers=2,    # optional hint; can be omitted to let diarization infer
)

# 3) Make per-speaker WAVs using the transcript CSV
cs.split_wav_by_speaker(
    source_wav=diar_input_audio,
    transcript_csv=tp.raw_files["csv"],
    out_dir="audio_split/",
    time_unit="ms",    # the transcript CSV uses milliseconds
    silence_ms=500,   # add 0.5s before/after each clip (1s between clips)
)

# 4) Export Whisper encoder embeddings per transcript segment
cs.export_embeddings(
    transcript_csv=tp.raw_files["csv"],
    source_wav=diar_input_audio,
    output_dir="whisper_embed/",
    model_name="base",
    device="cuda",
    compute_type="float16",  # good default on GPU
    time_unit="ms",
    run_in_subprocess=True,  # isolates the encoder to avoid cuDNN conflicts
)
```

**Outputs (typical):**

* `transcripts/…` — CSV (timestamped), SRT, and TXT
* `audio_split/…` — one WAV per detected speaker
* `whisper_embed/<source_stem>_embeddings.csv` — one row per transcript span with `e0..eN` encoder features

---

## Troubleshooting

* **`Unable to load any of {libcudnn_ops.so…}`**
  The process can't find cuDNN shared libs. Ensure `nvidia-cudnn-cu12` is installed and that your `LD_LIBRARY_PATH` includes its `lib` directory.

* **`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`**
  Version mismatch between runtime cuDNN and the PyTorch wheels. Reinstall the PyTorch wheels for **your** CUDA minor version (e.g., `cu124`) and keep cuDNN aligned.

* **Diarization is slow / OOM**

  * Keep `batch_size=0` (pipeline default)
  * Skip source separation with `no_stem=True`
  * Provide `num_speakers` if you know it (e.g., `2`)

* **Embeddings CSV is empty**

  * Make sure you pass the correct `time_unit` for your transcript (I use `ms` which should be the default)
  * Enable extractor debug logs by setting `CHOPSHOP_DEBUG=1` in your environment to see row counts and shapes

---

## Acknowledgments

Huge thanks to **Mahmoud Ashraf** for the outstanding [`whisper-diarization`](https://github.com/MahmoudAshraf97/whisper-diarization) project, which I modify and wrap for diarization.

---

## Roadmap

TBD. One of the big ones is to unify everything with defaults into a single function for the `ChopShop()` class.